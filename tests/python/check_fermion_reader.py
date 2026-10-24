#!/usr/bin/env python3
"""Round-trip test for scripts/fermion_reader.py.

Builds a small container with an independent encoder (written here from the
format description in the reader's docstring), then checks that the reader
returns exactly the matrices that went in: five_value (including row lengths
that are not a multiple of 5 and hi-flag counts that are not a multiple of 8),
int6, int8 and fp16 records. Also checks the archive path (tar.zst plus
SHA256SUMS.txt) and that a wrong digest, a corrupt trit byte, a truncated
record and trailing bytes are refused.

Optional, against the real release: set PARAKEET_TEST_FERMION_ARCHIVE to a
.bps.tar.zst and PARAKEET_TEST_FERMION_SUMS to its SHA256SUMS.txt; then the
reader must decode all 264 five_value records into planes that rebuild the
dense matrices exactly.

Exit codes (ctest convention): 0 = pass, 77 = skip, 1 = fail.
"""
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile

try:
    import numpy as np
except ImportError:
    print("check_fermion_reader: numpy not installed; skipping", file=sys.stderr)
    sys.exit(77)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts"))
import fermion_reader as fr  # noqa: E402

rng = np.random.default_rng(7)


# --- independent encoder -----------------------------------------------------

def enc_five_value(sign, is_hi, lo, hi):
    o, i = sign.shape
    rb = (i + 4) // 5
    codes = (sign + 1).astype(np.int64)
    pad = np.ones((o, rb * 5), dtype=np.int64)   # pad digits are 1 (zero)
    pad[:, :i] = codes
    by = np.zeros((o, rb), dtype=np.int64)
    for k in range(5):
        by += pad[:, k::5] * 3 ** k
    nz = sign != 0
    bits = is_hi[nz].astype(np.uint8)
    nbytes = (len(bits) + 7) // 8
    packed = np.zeros(nbytes * 8, dtype=np.uint8)
    packed[: len(bits)] = bits
    hib = np.packbits(packed, bitorder="little")
    return by.astype(np.uint8).tobytes() + hib.tobytes() + lo.tobytes() + hi.tobytes()


def enc_int(q, scale, bits):
    o = q.shape[0]
    flat = q.reshape(-1).astype(np.int64)
    if bits == 8:
        body = flat.astype(np.int8).tobytes()
    else:
        u = (flat + 32).astype(np.uint32)
        u = np.concatenate([u, np.zeros(-len(u) % 4, dtype=np.uint32)]).reshape(-1, 4)
        p = u[:, 0] | (u[:, 1] << 6) | (u[:, 2] << 12) | (u[:, 3] << 18)
        body = np.stack([p & 255, (p >> 8) & 255, (p >> 16) & 255], axis=1).astype(np.uint8).tobytes()
    return body + scale.tobytes()


def make_container():
    index, blobs, want = [], [], {}

    def add(name, kind, shape, blob):
        index.append({"n": name, "k": kind, "shape": list(shape), "b": len(blob)})
        blobs.append(blob)

    for name, (o, i) in {"a.lin": (6, 13), "b.lin": (4, 10), "c.lin": (3, 128), "d.lin": (5, 1)}.items():
        sign = rng.integers(-1, 2, size=(o, i)).astype(np.int8)
        is_hi = (rng.random((o, i)) < 0.3) & (sign != 0)
        lo = rng.uniform(0.01, 0.2, o).astype(np.float16)
        hi = rng.uniform(0.2, 0.9, o).astype(np.float16)
        add(name, "five_value", (o, i), enc_five_value(sign, is_hi, lo, hi))
        mag = np.where(is_hi, hi.astype(np.float32)[:, None], lo.astype(np.float32)[:, None])
        want[name + ".weight"] = (sign.astype(np.float32) * mag, sign, is_hi, lo, hi)
    for name, shape, bits in (("e.t", (5, 7), 6), ("f.t", (4, 8), 6), ("g.t", (3, 9), 8)):
        lim = 31 if bits == 6 else 127
        q = rng.integers(-lim, lim + 1, size=shape)
        sc = rng.uniform(0.001, 0.05, shape[0]).astype(np.float16)
        add(name, f"int{bits}", shape, enc_int(q, sc, bits))
        want[name] = q.astype(np.float32) * sc.astype(np.float32)[:, None]
    v = rng.normal(size=(2, 3)).astype(np.float16)
    add("h.norm", "fp16", v.shape, v.tobytes())
    want["h.norm"] = v.astype(np.float32)
    s = np.float16(3.0)
    add("i.count", "fp16", (), s.tobytes())
    want["i.count"] = np.asarray(s, dtype=np.float32)
    hdr = json.dumps({"format": fr.FORMAT, "index": index}).encode()
    return len(hdr).to_bytes(8, "little") + hdr + b"".join(blobs), want


def check_equal(rel_container, want):
    got = {}
    for r in fr.iter_records(rel_container):
        got[r.name + (".weight" if r.kind == "five_value" else "")] = r
    assert set(got) == set(want), (sorted(got), sorted(want))
    for k, r in got.items():
        w = want[k]
        dense = fr.decode_record(r)
        if r.kind == "five_value":
            exp, sign, is_hi, lo, hi = w
            assert dense.dtype == np.float32 and np.array_equal(dense, exp), k
            fv = fr.decode_five_value(r.blob, r.shape)
            assert np.array_equal(fv.sign, sign) and np.array_equal(fv.is_hi, is_hi), k
            assert np.array_equal(fv.lo, lo) and np.array_equal(fv.hi, hi), k
            a, b = fr.five_value_planes(fv)
            assert not ((a != 0) & (b != 0)).any(), "planes overlap"
            rebuilt = a.astype(np.float32) * lo.astype(np.float32)[:, None] + b.astype(np.float32) * hi.astype(np.float32)[:, None]
            assert np.array_equal(rebuilt, exp), f"planes do not rebuild {k}"
            trits, hib, lo2, hi2 = fr.split_five_value(r.blob, r.shape)
            assert trits.shape == (r.shape[0], (r.shape[1] + 4) // 5) and np.array_equal(lo2, lo)
        else:
            assert np.array_equal(dense, w), k


def expect_error(fn, what):
    try:
        fn()
    except fr.FermionError:
        return
    raise AssertionError(f"{what}: no FermionError")


def zstd_pack(tar_bytes, dst):
    exe = shutil.which("zstd")
    if exe:
        subprocess.run([exe, "-q", "-f", "-o", dst], input=tar_bytes, check=True)
        return True
    try:
        import zstandard
    except ImportError:
        return False
    with open(dst, "wb") as f:
        f.write(zstandard.ZstdCompressor().compress(tar_bytes))
    return True


def main():
    container, want = make_container()
    check_equal(container, want)

    # corruption is refused
    c = bytearray(container)
    hl = int.from_bytes(c[:8], "little")
    first = 8 + hl                       # first record: five_value a.lin, 6 rows x 3 trit bytes
    bad = bytes(c[:first]) + b"\xff" + bytes(c[first + 1:])
    expect_error(lambda: list(map(fr.decode_record, fr.iter_records(bad))), "trit byte 255")
    expect_error(lambda: list(fr.iter_records(container[:-3])), "truncated")
    expect_error(lambda: list(fr.iter_records(container + b"\0")), "trailing bytes")
    expect_error(lambda: list(fr.iter_records(container.replace(fr.FORMAT.encode(), b"other-format-" + b"x" * (len(fr.FORMAT) - 13), 1))), "format")

    # archive path
    with tempfile.TemporaryDirectory() as d:
        pm = json.dumps({"container_sha256": hashlib.sha256(container).hexdigest(),
                         "container_bytes": len(container)}).encode()
        bps = json.dumps({"files": [{"path": "model.fermion", "original_bytes": len(container),
                                     "original_sha256": hashlib.sha256(container).hexdigest()}]}).encode()
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tf:
            for n, b in (("bps_manifest.json", bps), ("model.fermion", container), ("packed_manifest.json", pm)):
                ti = tarfile.TarInfo(n)
                ti.size = len(b)
                tf.addfile(ti, io.BytesIO(b))
        arch = os.path.join(d, "t.bps.tar.zst")
        if zstd_pack(buf.getvalue(), arch):
            sums = os.path.join(d, "SHA256SUMS.txt")
            with open(sums, "w") as f:
                f.write(f"{hashlib.sha256(open(arch, 'rb').read()).hexdigest()}  t.bps.tar.zst\n")
            rel = fr.load_release(arch)                     # SHA256SUMS.txt next to it
            check_equal(rel.container, want)
            with open(sums, "w") as f:
                f.write(f"{'0' * 64}  t.bps.tar.zst\n")
            expect_error(lambda: fr.load_release(arch), "wrong archive digest")
            os.remove(sums)
            expect_error(lambda: fr.load_release(arch), "no checksum source")
            check_equal(fr.load_release(arch, verify=False).container, want)
            # a container that no longer matches the manifest inside the archive
            buf2 = io.BytesIO()
            tampered = bytearray(container)
            tampered[-1] ^= 1
            with tarfile.open(fileobj=buf2, mode="w") as tf:
                for n, b in (("bps_manifest.json", bps), ("model.fermion", bytes(tampered)), ("packed_manifest.json", pm)):
                    ti = tarfile.TarInfo(n)
                    ti.size = len(b)
                    tf.addfile(ti, io.BytesIO(b))
            arch2 = os.path.join(d, "u.bps.tar.zst")
            zstd_pack(buf2.getvalue(), arch2)
            expect_error(lambda: fr.load_release(arch2, sha256=hashlib.sha256(open(arch2, "rb").read()).hexdigest()),
                         "member digest")
        else:
            print("check_fermion_reader: no zstd available, skipped the archive part", file=sys.stderr)

    # the real release, when given
    real, sums = os.environ.get("PARAKEET_TEST_FERMION_ARCHIVE"), os.environ.get("PARAKEET_TEST_FERMION_SUMS")
    if real:
        rel = fr.load_release(real, sums=sums)
        n5 = 0
        for r in fr.iter_records(rel.container):
            if r.kind != "five_value":
                continue
            fv = fr.decode_five_value(r.blob, r.shape)
            a, b = fr.five_value_planes(fv)
            rebuilt = a.astype(np.float32) * fv.lo.astype(np.float32)[:, None] + b.astype(np.float32) * fv.hi.astype(np.float32)[:, None]
            assert np.array_equal(rebuilt, fr.five_value_to_dense(fv)), r.name
            n5 += 1
        assert n5 > 0
        print(f"check_fermion_reader: real release ok, {n5} five_value records")
    print("check_fermion_reader: ok")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"check_fermion_reader: FAIL: {e}", file=sys.stderr)
        sys.exit(1)
