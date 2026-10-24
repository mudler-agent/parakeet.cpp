#!/usr/bin/env python3
"""Reader for ``fermion-five-value-parakeet-v1`` weight containers (Phonon-2.5).

The Phonon-2.5 release is a ``.bps.tar.zst`` archive that holds ``model.fermion``
(the weights), ``packed_manifest.json`` and ``bps_manifest.json`` (SHA-256 of
every member) and a NeMo ``config.json``. This module opens the archive,
checks the SHA-256 sums and decodes the container. It needs only numpy.
Zstandard comes from the standard library (Python 3.14), the ``zstandard``
package or the ``zstd`` command, whichever is found first.

Container layout, all little endian::

    u64 header_len | header_len bytes of JSON | records, in index order

The JSON header has ``format`` and ``index``. Each index entry has ``n`` (the
tensor name), ``k`` (the kind), ``shape`` and ``b`` (the record size in bytes).

====================  ======================================================
kind                  record bytes
====================  ======================================================
``five_value``        O*ceil(I/5) bytes of trits (5 per byte, base 3, least
                      significant digit first, digit 1 = zero, 0 = -1,
                      2 = +1), then ceil(nnz/8) bytes of "hi" flags (one bit
                      per nonzero weight, row-major order, least significant
                      bit first), then fp16 ``lo[O]``, then fp16 ``hi[O]``.
                      A weight is 0, +-lo[row] or +-hi[row].
``int6`` / ``int8``   packed signed integers (6 bits: 4 values per 3 bytes,
                      offset 32) followed by one fp16 scale per row.
``fp16``              plain fp16
====================  ======================================================

Every ``five_value`` matrix therefore equals ``lo * A + hi * B`` where A and B
are ternary matrices that never share a nonzero position.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import numpy as np

FORMAT = "fermion-five-value-parakeet-v1"
_MAX_MEMBER = 1 << 30   # no member of a real release is near 1 GiB

# trit digits of every byte value, least significant digit first
_TRITS = np.array([[(b // 3 ** k) % 3 for k in range(5)] for b in range(256)], dtype=np.uint8)


class FermionError(ValueError):
    """A malformed archive or container, or a failed checksum."""


@dataclass
class FiveValue:
    sign: np.ndarray    # int8 [O, I], -1 / 0 / +1
    is_hi: np.ndarray   # bool [O, I], True where a nonzero weight uses hi (False elsewhere)
    lo: np.ndarray      # float16 [O]
    hi: np.ndarray      # float16 [O]


@dataclass
class Record:
    name: str
    kind: str
    shape: Tuple[int, ...]
    blob: memoryview


def row_bytes(i: int) -> int:
    return (i + 4) // 5


# --- five_value --------------------------------------------------------------

def split_five_value(blob, shape) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cut one record into (trit bytes [O, ceil(I/5)], hi-flag bytes, lo, hi)
    without decoding it. The bytes are the upstream bytes, unchanged."""
    if len(shape) != 2:
        raise FermionError(f"five_value record needs a 2-D shape, got {tuple(shape)}")
    o, i = int(shape[0]), int(shape[1])
    rb = row_bytes(i)
    buf = np.frombuffer(blob, dtype=np.uint8)
    if buf.size < o * rb + 4 * o:
        raise FermionError("five_value record is shorter than its trits and scales")
    trits = buf[: o * rb].reshape(o, rb)
    if (trits >= 243).any():
        raise FermionError("five_value record has a trit byte >= 3**5")
    # nonzero count: digits of the real columns that are not 1 (pad digits are ignored)
    codes = _TRITS[trits].reshape(o, rb * 5)[:, :i]
    nnz = int((codes != 1).sum())
    nbit = (nnz + 7) // 8
    if buf.size != o * rb + nbit + 4 * o:
        raise FermionError(f"five_value record has {buf.size} bytes, expected {o * rb + nbit + 4 * o}")
    hibits = buf[o * rb: o * rb + nbit]
    sc = np.frombuffer(blob, dtype=np.float16, count=2 * o, offset=o * rb + nbit)
    return trits, hibits, sc[:o].copy(), sc[o:].copy()


def decode_five_value(blob, shape) -> FiveValue:
    trits, hibits, lo, hi = split_five_value(blob, shape)
    o, i = int(shape[0]), int(shape[1])
    codes = _TRITS[trits].reshape(o, -1)[:, :i]
    nz = codes != 1
    nnz = int(nz.sum())
    bits = np.unpackbits(hibits, bitorder="little")[:nnz].astype(bool)
    is_hi = np.zeros((o, i), dtype=bool)
    is_hi[nz] = bits
    return FiveValue(sign=codes.astype(np.int8) - 1, is_hi=is_hi, lo=lo, hi=hi)


def five_value_to_dense(fv: FiveValue) -> np.ndarray:
    """float32 [O, I]: sign * (hi if is_hi else lo). Exact: every value is an fp16."""
    mag = np.where(fv.is_hi, fv.hi.astype(np.float32)[:, None], fv.lo.astype(np.float32)[:, None])
    return fv.sign.astype(np.float32) * mag


def five_value_planes(fv: FiveValue) -> Tuple[np.ndarray, np.ndarray]:
    """The two ternary planes (int8, -1/0/+1) with W = lo * A + hi * B."""
    a = np.where(fv.is_hi, 0, fv.sign).astype(np.int8)
    b = np.where(fv.is_hi, fv.sign, 0).astype(np.int8)
    return a, b


# --- intN and fp16 -----------------------------------------------------------

def decode_int(blob, shape, bits: int) -> np.ndarray:
    o = int(shape[0])
    total = int(np.prod(shape, dtype=np.int64))
    ncol = total // o if o else 0
    buf = np.frombuffer(blob, dtype=np.uint8)
    if buf.size < 2 * o:
        raise FermionError("int record is shorter than its scales")
    body, scales = buf[: buf.size - 2 * o], np.frombuffer(blob, dtype=np.float16, count=o, offset=buf.size - 2 * o)
    if bits == 8:
        if body.size != total:
            raise FermionError(f"int8 record has {body.size} value bytes, expected {total}")
        q = body.view(np.int8).astype(np.int32)
    elif bits == 6:
        want = -(-total // 4) * 3
        if body.size != want:
            raise FermionError(f"int6 record has {body.size} value bytes, expected {want}")
        t = body.reshape(-1, 3).astype(np.uint32)
        packed = t[:, 0] | (t[:, 1] << 8) | (t[:, 2] << 16)
        u = np.stack([(packed >> s) & 0x3F for s in (0, 6, 12, 18)], axis=1).ravel()
        q = u[:total].astype(np.int32) - 32
    else:
        raise FermionError(f"unsupported integer width {bits}")
    w = q.reshape(o, ncol).astype(np.float32) * scales.astype(np.float32)[:, None]
    return w.reshape(shape)


def decode_fp16(blob, shape) -> np.ndarray:
    n = int(np.prod(shape, dtype=np.int64))
    if len(blob) != 2 * n:
        raise FermionError(f"fp16 record has {len(blob)} bytes, expected {2 * n}")
    return np.frombuffer(blob, dtype=np.float16).reshape(shape).astype(np.float32)


# --- container ---------------------------------------------------------------

def iter_records(data) -> Iterator[Record]:
    """Yield every record of a container held in memory (bytes or memoryview)."""
    mv = memoryview(data)
    if len(mv) < 8:
        raise FermionError("container is shorter than its header length")
    n = int.from_bytes(mv[:8], "little")
    if n <= 0 or 8 + n > len(mv):
        raise FermionError("container header length is out of range")
    try:
        header = json.loads(bytes(mv[8: 8 + n]))
    except ValueError as e:
        raise FermionError(f"container header is not JSON: {e}")
    if header.get("format") != FORMAT:
        raise FermionError(f"unknown container format {header.get('format')!r}, expected {FORMAT!r}")
    off = 8 + n
    seen = set()
    for e in header["index"]:
        size = int(e["b"])
        if size < 0 or off + size > len(mv):
            raise FermionError(f"record {e['n']} runs past the end of the container")
        if e["n"] in seen:
            raise FermionError(f"duplicate record {e['n']}")
        seen.add(e["n"])
        yield Record(e["n"], e["k"], tuple(int(x) for x in e["shape"]), mv[off: off + size])
        off += size
    if off != len(mv):
        raise FermionError(f"{len(mv) - off} trailing bytes after the last record")


def decode_record(rec: Record) -> np.ndarray:
    """float32 tensor for any record kind (a five_value record is the dense matrix)."""
    if rec.kind == "five_value":
        return five_value_to_dense(decode_five_value(rec.blob, rec.shape))
    if rec.kind.startswith("int"):
        return decode_int(rec.blob, rec.shape, int(rec.kind[3:]))
    if rec.kind == "fp16":
        return decode_fp16(rec.blob, rec.shape)
    raise FermionError(f"unknown record kind {rec.kind!r}")


# --- release archive ---------------------------------------------------------

def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_sums(path) -> Dict[str, str]:
    """Parse a ``sha256sum`` style file: ``<hex>  <name>`` per line."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([0-9a-fA-F]{64})\s+\*?(.+?)\s*$", line)
        if m:
            out[m.group(2)] = m.group(1).lower()
    return out


def _open_tar_zst(path: Path):
    """Return an open tarfile for a .tar.zst. Falls back from the standard
    library (3.14) to the zstandard package to the zstd command."""
    try:
        return tarfile.open(path, mode="r:zst")   # Python 3.14+
    except (tarfile.ReadError, tarfile.CompressionError, ValueError, AttributeError, OSError):
        pass
    try:
        import zstandard
        raw = zstandard.ZstdDecompressor().stream_reader(open(path, "rb"))
        return tarfile.open(fileobj=raw, mode="r|")
    except ImportError:
        pass
    exe = shutil.which("zstd")
    if not exe:
        raise FermionError("cannot read .zst: need Python 3.14, the zstandard package or the zstd command")
    proc = subprocess.run([exe, "-dc", str(path)], stdout=subprocess.PIPE, check=True)
    return tarfile.open(fileobj=io.BytesIO(proc.stdout), mode="r:")


@dataclass
class Release:
    container: bytes
    meta: Dict[str, object]     # packed_manifest.json, {} if absent
    nemo_config: Dict[str, object]  # config.json inside the archive, {} if absent
    archive_sha256: Optional[str]
    container_sha256: str


def load_release(path, *, sums: Optional[str] = None, sha256: Optional[str] = None,
                 verify: bool = True) -> Release:
    """Open a release archive (.tar.zst), a directory holding the unpacked files,
    or a bare ``model.fermion``.

    With ``verify`` (the default) an archive must match its published SHA-256,
    taken from ``sha256`` or from ``sums`` (default: ``SHA256SUMS.txt`` next to the
    archive), and every member must match the digests in the archive's own
    manifests. A bare container or directory is checked against the manifests
    when they exist.
    """
    p = Path(path)
    files: Dict[str, bytes] = {}
    arch_sha = None
    if p.is_dir():
        for n in ("model.fermion", "packed_manifest.json", "bps_manifest.json", "config.json"):
            if (p / n).is_file():
                files[n] = (p / n).read_bytes()
    elif p.name.endswith(".tar.zst") or p.name.endswith(".tzst"):
        if verify:
            arch_sha = sha256_file(p)
            want = (sha256 or "").lower() or None
            if want is None:
                sp = Path(sums) if sums else p.with_name("SHA256SUMS.txt")
                if not sp.is_file():
                    raise FermionError(f"no checksum to verify {p.name}: pass --sha256 or --sums, "
                                       f"or put SHA256SUMS.txt next to it (or pass --no-verify)")
                want = read_sums(sp).get(p.name)
                if want is None:
                    raise FermionError(f"{sp.name} has no entry for {p.name}")
            if arch_sha != want:
                raise FermionError(f"SHA-256 of {p.name} is {arch_sha}, expected {want}")
        with _open_tar_zst(p) as tf:
            for m in tf:
                base = m.name.split("/")[-1]
                if not m.isreg() or base not in ("model.fermion", "packed_manifest.json",
                                                 "bps_manifest.json", "config.json"):
                    continue
                if m.size > _MAX_MEMBER:
                    raise FermionError(f"archive member {m.name} is too large")
                files[base] = tf.extractfile(m).read()
    else:
        files["model.fermion"] = p.read_bytes()
    if "model.fermion" not in files:
        raise FermionError(f"{p} holds no model.fermion")

    def parse(name):
        try:
            return json.loads(files[name]) if name in files else {}
        except ValueError as e:
            raise FermionError(f"{name} is not valid JSON: {e}")
    meta, bps = parse("packed_manifest.json"), parse("bps_manifest.json")
    csha = hashlib.sha256(files["model.fermion"]).hexdigest()
    if verify:
        for f in bps.get("files", []):
            if f["path"] in files:
                got = hashlib.sha256(files[f["path"]]).hexdigest()
                if got != f["original_sha256"]:
                    raise FermionError(f"{f['path']} does not match bps_manifest.json: {got}")
        if meta.get("container_sha256") and meta["container_sha256"] != csha:
            raise FermionError(f"model.fermion SHA-256 {csha} does not match packed_manifest.json")
        if meta.get("container_bytes") and int(meta["container_bytes"]) != len(files["model.fermion"]):
            raise FermionError("model.fermion size does not match packed_manifest.json")
        if not (bps or meta) and not p.name.endswith(".tar.zst"):
            print("fermion_reader: warning: no manifest to verify the container against", file=sys.stderr)
    return Release(files["model.fermion"], meta, parse("config.json"), arch_sha, csha)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="List the records of a Phonon container")
    ap.add_argument("archive")
    ap.add_argument("--sums")
    ap.add_argument("--sha256")
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()
    rel = load_release(a.archive, sums=a.sums, sha256=a.sha256, verify=not a.no_verify)
    kinds: Dict[str, int] = {}
    for r in iter_records(rel.container):
        kinds[r.kind] = kinds.get(r.kind, 0) + 1
    print(json.dumps({"container_sha256": rel.container_sha256, "records": kinds}, indent=1))
