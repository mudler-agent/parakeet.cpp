#!/usr/bin/env python3
"""Convert an HF-format Parakeet TDT checkpoint (safetensors) to GGUF.

Targets checkpoints that follow the transformers ``ParakeetForTDT`` layout and
are derived from ``nvidia/parakeet-tdt-0.6b-v3``, for example
``moondream/parakeet-ultra`` (post-trained, F16) and ``moondream/parakeet-redux``
(ternary encoder, packed ``qweight`` + ``scales``).

There is no NeMo dependency. The HF repos carry neither the mel filterbank and
window nor a SentencePiece vocab, so the KV metadata, the tokenizer pieces and
the two featurizer buffers are taken from a GGUF of the teacher model that was
produced by ``convert_parakeet_to_gguf.py`` (``--template``). The HF
``config.json`` is cross-checked against the template and the run aborts on any
mismatch, so a checkpoint that is not v3-shaped cannot be converted silently.

HF tensor names are mapped back to the verbatim NeMo names the C++ loader
expects (the inverse of transformers' ``convert_nemo_to_hf.py``). Ternary
weights are dequantized by default (``w = scales[row, col // group] * (code - 1)``),
so the engine sees ordinary linear weights and needs no new kernel.

The optional ``vad_head.*`` tensors are written as F32 together with the
``parakeet.vad.*`` KVs unless ``--vad drop`` is given. With ``--ternary keep``
the ternary linears stay packed (see docs/ternary.md) instead of dequantized.

Phonon-2.5 (FermionResearch/Phonon-2.5 and Phonon-2.5-English) ships its weights
as a ``fermion-five-value-parakeet-v1`` container inside a ``.bps.tar.zst``
archive instead of safetensors. Pass the archive with ``--fermion`` (and the
HF ``config.json`` of nvidia/parakeet-tdt-0.6b-v3 with ``--config``). The
archive SHA-256 is checked against ``SHA256SUMS.txt`` next to it, or against
``--sha256``, and every member against the manifests inside the archive. The
five-value, int6 and fp16 records are expanded to float32, and the rest of the
conversion is the same as for a safetensors checkpoint. With ``--five-value keep``
the 264 five-value linears are not expanded: their upstream bytes are written
unchanged as ``<base>.fvcodes`` (I8), ``<base>.fvhi`` (I8) and ``<base>.fvscale``
(F16, shape [2, N]: lo then hi), together with ``parakeet.fivevalue.present``;
the engine splits them into two ternary planes at load (docs/phonon.md).
See ``scripts/fermion_reader.py``.
"""
import argparse
import json
import pathlib
import re
import struct
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import fermion_reader  # noqa: E402

try:
    import gguf
except ImportError as e:  # pragma: no cover - env guard
    print(f"converter: missing dependency 'gguf': {e}", file=sys.stderr)
    sys.exit(2)

# HF name -> NeMo name. Order matters only where patterns could overlap; every
# rule is anchored, so the first match wins.
_RENAMES = [
    (r"^encoder\.subsampling\.layers\.(\d+)\.(weight|bias)$", r"encoder.pre_encode.conv.\1.\2"),
    (r"^encoder\.subsampling\.linear\.(weight|bias)$", r"encoder.pre_encode.out.\1"),
    (r"^encoder\.layers\.(\d+)\.conv\.norm\.", r"encoder.layers.\1.conv.batch_norm."),
    (r"^encoder\.layers\.(\d+)\.self_attn\.q_proj\.", r"encoder.layers.\1.self_attn.linear_q."),
    (r"^encoder\.layers\.(\d+)\.self_attn\.k_proj\.", r"encoder.layers.\1.self_attn.linear_k."),
    (r"^encoder\.layers\.(\d+)\.self_attn\.v_proj\.", r"encoder.layers.\1.self_attn.linear_v."),
    (r"^encoder\.layers\.(\d+)\.self_attn\.o_proj\.", r"encoder.layers.\1.self_attn.linear_out."),
    (r"^encoder\.layers\.(\d+)\.self_attn\.relative_k_proj\.", r"encoder.layers.\1.self_attn.linear_pos."),
    (r"^encoder\.layers\.(\d+)\.self_attn\.bias_([uv])$", r"encoder.layers.\1.self_attn.pos_bias_\2"),
    (r"^decoder\.embedding\.", "decoder.prediction.embed."),
    (r"^decoder\.lstm\.", "decoder.prediction.dec_rnn.lstm."),
    (r"^encoder_projector\.", "joint.enc."),
    (r"^decoder\.decoder_projector\.", "joint.pred."),
    (r"^joint\.head\.", "joint.joint_net.2."),
]
_RENAMES = [(re.compile(a), b) for a, b in _RENAMES]

# Unchanged names (norms, FFN, conv, pointwise convs, ...) pass through.
_PASSTHROUGH = re.compile(r"^encoder\.layers\.\d+\.")

# Linear weights that go through ggml_mul_mat and may be quantized. Mirrors the
# ASR entries of _QUANTIZABLE_PATTERNS in convert_parakeet_to_gguf.py and the
# policy in docs/quantization.md.
_QUANTIZABLE = [re.compile(p) for p in (
    r"^encoder\.layers\.\d+\.feed_forward[12]\.linear[12]\.weight$",
    r"^encoder\.layers\.\d+\.self_attn\.linear_(q|k|v|out|pos)\.weight$",
    r"^encoder\.pre_encode\.out\.weight$",
    r"^joint\.enc\.weight$",
    r"^joint\.pred\.weight$",
)]

_VAD = re.compile(r"^vad_head\.")


def hf_to_nemo(name):
    for rx, rep in _RENAMES:
        if rx.search(name):
            return rx.sub(rep, name)
    if _PASSTHROUGH.match(name):
        return name
    raise KeyError(f"no NeMo name for HF tensor {name!r}")


def read_safetensors(path):
    """Yield (name, numpy array or (dtype, shape, bytes)) without needing torch."""
    dt = {"F32": np.float32, "F16": np.float16, "I64": np.int64, "U8": np.uint8}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(n))
        base = 8 + n
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            if meta["dtype"] not in dt:
                raise ValueError(f"{name}: unsupported dtype {meta['dtype']}")
            a, b = meta["data_offsets"]
            f.seek(base + a)
            buf = f.read(b - a)
            yield name, np.frombuffer(buf, dtype=dt[meta["dtype"]]).reshape(meta["shape"])


def check_packed_bytes(q, what="ternary payload"):
    """Five base-3 digits give bytes 0..242. Anything larger is corrupt; the C++
    loader rejects it too, so --ternary keep must not write such a file."""
    bad = np.asarray(q) >= 3 ** 5
    if bad.any():
        idx = tuple(int(i) for i in np.argwhere(bad)[0])
        raise ValueError(f"{what} has byte {int(np.asarray(q)[idx])} at {idx}, which is >= 3**5 = 243")


def unpack_ternary(q, scales, in_features, group):
    """Dequantize a thrush-ternary-v2 tensor to F32 [out, in].

    ``q`` is uint8 [out, ceil(in/5)]: element i of a row is base-3 digit i%5 of
    byte i//5, least significant digit first. code in {0,1,2}, weight is
    scales[row, col // group] * (code - 1).
    """
    check_packed_bytes(q)
    out, nbytes = q.shape
    assert nbytes == -(-in_features // 5), (q.shape, in_features)
    digits = np.empty((out, nbytes, 5), dtype=np.int8)
    v = q.astype(np.int32)
    for d in range(5):
        digits[:, :, d] = v % 3
        v //= 3
    codes = digits.reshape(out, nbytes * 5)[:, :in_features]
    if (v != 0).any() or codes.max() > 2:
        raise ValueError("ternary payload has a byte >= 3**5")
    s = np.repeat(scales.astype(np.float32), group, axis=1)[:, :in_features]
    return s * (codes.astype(np.float32) - 1.0)


_POINTWISE = re.compile(r"\.conv\.pointwise_conv[12]\.weight$")


class FiveValueRaw(tuple):
    """(module name, trit bytes [O, ceil(I/5)], hi-flag bytes, lo fp16 [O], hi fp16 [O])."""


def read_fermion(args, tkv, tv):
    """Yield (HF name, float32 array) for every record of a Phonon release.

    Five-value matrices come out as the dense matrix named ``<module>.weight``;
    the 1x1 convolutions are reshaped to [out, in, 1] like the safetensors export.
    With ``--five-value keep`` they come out as (module name, FiveValueRaw) instead.
    """
    rel = fermion_reader.load_release(args.fermion, sums=args.sums, sha256=args.sha256,
                                      verify=not args.no_verify)
    check_fermion_config(rel.nemo_config, tkv, tv)
    for rec in fermion_reader.iter_records(rel.container):
        if rec.kind == "five_value" and args.five_value == "keep":
            trits, hibits, lo, hi = fermion_reader.split_five_value(rec.blob, rec.shape)
            yield rec.name, FiveValueRaw((rec.name, trits, hibits, lo, hi))
            continue
        arr = fermion_reader.decode_record(rec)
        name = rec.name + (".weight" if rec.kind == "five_value" else "")
        if arr.ndim == 2 and _POINTWISE.search(name):
            arr = arr[:, :, None]
        yield name, arr


def check_fermion_config(nemo, tkv, tv):
    """Cross-check the NeMo config inside the archive against the template."""
    if not nemo:
        return
    enc, pre = nemo.get("encoder", {}), nemo.get("preprocessor", {})
    checks = {
        "parakeet.encoder.d_model": enc.get("d_model"),
        "parakeet.encoder.n_layers": enc.get("n_layers"),
        "parakeet.encoder.n_heads": enc.get("n_heads"),
        "parakeet.encoder.feat_in": enc.get("feat_in"),
        "parakeet.encoder.conv_kernel": enc.get("conv_kernel_size"),
        "parakeet.encoder.subsampling_factor": enc.get("subsampling_factor"),
        "parakeet.preprocessor.n_fft": pre.get("n_fft"),
    }
    for k, want in checks.items():
        if want is not None and k in tv and int(tv[k]) != int(want):
            sys.exit(f"archive config mismatch vs template: {k} template={tv[k]} archive={want}")
    if "parakeet.encoder.xscaling" in tv and "xscaling" in enc and bool(tv["parakeet.encoder.xscaling"]) != bool(enc["xscaling"]):
        sys.exit("archive config mismatch vs template: xscaling")
    vocab = nemo.get("joint", {}).get("vocabulary")
    pieces = tkv.get("parakeet.tokenizer.pieces")
    if vocab and pieces is not None and [str(p) for p in pieces.contents()] != list(vocab):
        sys.exit("archive vocabulary differs from the template tokenizer; use the template of the same tokenizer")


def load_template(path):
    r = gguf.GGUFReader(path)
    kv = {}
    for k, f in r.fields.items():
        if k.startswith("GGUF."):
            continue
        kv[k] = f
    tensors = {t.name: t for t in r.tensors}
    return r, kv, tensors


def field_value(f):
    return f.contents()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hf", help="local dir holding model.safetensors + config.json (+ ternary.json)")
    ap.add_argument("--fermion", help="Phonon-2.5 release (.bps.tar.zst, an unpacked directory or a bare model.fermion) "
                    "instead of --hf; needs --config")
    ap.add_argument("--config", help="HF config.json of nvidia/parakeet-tdt-0.6b-v3 (default: <--hf>/config.json)")
    ap.add_argument("--sums", help="SHA256SUMS.txt that lists the --fermion archive (default: the one next to it)")
    ap.add_argument("--sha256", help="expected SHA-256 of the --fermion archive, instead of --sums")
    ap.add_argument("--no-verify", action="store_true", help="skip the --fermion checksum checks")
    ap.add_argument("--five-value", choices=["dequant", "keep"], default="dequant", dest="five_value",
                    help="--fermion only. keep: store the five-value linears as their upstream bytes "
                    "(<name>.fvcodes/.fvhi/.fvscale, needs the CPU engine); dequant: ordinary weights")
    ap.add_argument("--template", required=True,
                    help="GGUF of nvidia/parakeet-tdt-0.6b-v3 from convert_parakeet_to_gguf.py (KV, vocab, mel buffers)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--dtype", choices=["f32", "f16", "q8_0"], default="f32")
    ap.add_argument("--name", default=None, help="general.name (default: the --hf directory name)")
    ap.add_argument("--ternary", choices=["dequant", "keep"], default="dequant",
                    help="keep: store ternary linears packed as <name>.qweight + <name>.scales (needs the native ternary kernel); dequant: ordinary weights")
    ap.add_argument("--vad", choices=["keep", "drop"], default="keep",
                    help="keep the vad_head.* tensors and parakeet.vad.* KVs when the checkpoint has them")
    args = ap.parse_args()

    if bool(args.hf) == bool(args.fermion):
        sys.exit("give exactly one of --hf and --fermion")
    if args.five_value == "keep" and not args.fermion:
        sys.exit("--five-value keep needs --fermion")
    if args.fermion:
        if not args.config:
            sys.exit("--fermion needs --config (the HF config.json of nvidia/parakeet-tdt-0.6b-v3)")
        src = pathlib.Path(args.fermion)
        cfg = json.load(open(args.config))
        tern = None
    else:
        src = pathlib.Path(args.hf)
        cfg = json.load(open(args.config or src / "config.json"))
        tern_path = src / "ternary.json"
        tern = json.load(open(tern_path)) if tern_path.exists() else None

    _, tkv, ttensors = load_template(args.template)
    tv = {k: field_value(f) for k, f in tkv.items() if "pieces" not in k}

    # --- cross-check HF config against the template -------------------------
    enc = cfg["encoder_config"]
    checks = {
        "parakeet.encoder.d_model": enc["hidden_size"],
        "parakeet.encoder.n_layers": enc["num_hidden_layers"],
        "parakeet.encoder.n_heads": enc["num_attention_heads"],
        "parakeet.encoder.ff_dim": enc["intermediate_size"],
        "parakeet.encoder.conv_kernel": enc["conv_kernel_size"],
        "parakeet.encoder.feat_in": enc["num_mel_bins"],
        "parakeet.encoder.subsampling_factor": enc["subsampling_factor"],
        "parakeet.encoder.subsampling_conv_channels": enc["subsampling_conv_channels"],
        "parakeet.decoder.pred_hidden": cfg["decoder_hidden_size"],
        "parakeet.decoder.pred_rnn_layers": cfg["num_decoder_layers"],
        "parakeet.vocab_size": cfg["vocab_size"] - 1,
        "parakeet.blank_id": cfg["blank_token_id"],
    }
    for k, want in checks.items():
        if int(tv[k]) != int(want):
            sys.exit(f"config mismatch vs template: {k} template={tv[k]} hf={want}")
    if [int(x) for x in tv["parakeet.tdt.durations"]] != cfg["durations"]:
        sys.exit("config mismatch vs template: tdt durations")

    # --- KV: copy the template, HF checkpoints are TDT-only (no CTC head) ----
    w = gguf.GGUFWriter(args.output, "parakeet")
    w.add_string("general.name", args.name or (src.name.split(".")[0] if args.fermion else src.name))
    for k, f in tkv.items():
        if k in ("general.architecture", "general.name", "general.quantization_version",
                 "general.file_type"):
            continue
        if k == "parakeet.arch":
            w.add_string(k, "tdt")
        elif k == "parakeet.tokenizer.pieces":
            w.add_array(k, [str(p) for p in f.contents()])
        else:
            _copy_kv(w, k, f)

    # --- tensors ------------------------------------------------------------
    keep_tern = args.ternary == "keep"
    if keep_tern and not tern:
        sys.exit("--ternary keep needs ternary.json next to model.safetensors")
    tens = {}   # float32 tensors; allowlisted linears are quantized per --dtype
    raw = {}    # written as-is: ternary qweight (int8 view) and scales (f16), vad_head.*
    qmods = {m["name"]: m for m in (tern["quantized_modules"] if tern else [])}
    group = (tern or {}).get("quant", {}).get("group_size", 0)
    pending_q = {}
    if args.fermion:
        try:
            tensor_stream = read_fermion(args, tkv, tv)
        except fermion_reader.FermionError as e:
            sys.exit(f"fermion: {e}")
    else:
        tensor_stream = read_safetensors(src / "model.safetensors")
    for name, arr in tensor_stream:
        if isinstance(arr, FiveValueRaw):
            _, trits, hibits, lo, hi = arr
            base = hf_to_nemo(name + ".weight")[: -len(".weight")]
            raw[base + ".fvcodes"] = np.ascontiguousarray(trits).view(np.int8)
            raw[base + ".fvhi"] = np.ascontiguousarray(hibits).view(np.int8)
            raw[base + ".fvscale"] = np.ascontiguousarray(np.stack([lo, hi]), dtype=np.float16)
            continue
        if arr.ndim == 0:
            continue
        if _VAD.match(name):
            if args.vad == "keep":
                raw[name] = np.ascontiguousarray(arr, dtype=np.float32)
            continue
        if name.endswith(".qweight"):
            pending_q.setdefault(name[:-8], {})["q"] = arr
            continue
        if name.endswith(".scales"):
            pending_q.setdefault(name[:-7], {})["s"] = arr
            continue
        tens[hf_to_nemo(name)] = np.ascontiguousarray(arr, dtype=np.float32)

    for mod, d in pending_q.items():
        m = qmods[mod]
        g = m["group_size"] or group
        base = hf_to_nemo(mod + ".weight")[: -len(".weight")]
        if keep_tern:
            if g != 128 or m["in_features"] % g:
                sys.exit(f"--ternary keep needs group 128 and in_features % 128 == 0 "
                         f"(got group {g}, in {m['in_features']} for {mod})")
            try:
                check_packed_bytes(d["q"], f"{mod}.qweight")
            except ValueError as e:
                sys.exit(str(e))
            raw[base + ".qweight"] = np.ascontiguousarray(d["q"]).view(np.int8)
            raw[base + ".scales"] = np.ascontiguousarray(d["s"], dtype=np.float16)
            continue
        wt = unpack_ternary(d["q"], d["s"], m["in_features"], g)
        if m.get("as_conv1d") or mod.rsplit(".", 1)[-1] in ("pointwise_conv1", "pointwise_conv2"):
            wt = wt[:, :, None]  # NeMo stores 1x1 convs as [out, in, 1]
        tens[base + ".weight"] = np.ascontiguousarray(wt, dtype=np.float32)

    # mel featurizer buffers are not in the HF repo; lift them from the template
    for k in ("preprocessor.featurizer.fb", "preprocessor.featurizer.window"):
        tens[k] = np.ascontiguousarray(ttensors[k].data, dtype=np.float32)

    # The tensor set must equal the template's (minus the weights replaced by
    # qweight/scales), with identical shapes.
    replaced = {n[: -len(".qweight")] + ".weight" for n in raw if n.endswith(".qweight")}
    replaced |= {n[: -len(".fvcodes")] + ".weight" for n in raw if n.endswith(".fvcodes")}
    expect = {n for n in ttensors
              if not n.startswith("ctc_decoder.") and not n.startswith("decoder.decoder_layers")} - replaced
    got = set(tens)
    if expect != got:
        sys.exit(f"tensor set differs from template: missing={sorted(expect - got)[:5]} "
                 f"extra={sorted(got - expect)[:5]}")
    for n, a in tens.items():
        want = tuple(int(x) for x in ttensors[n].shape[::-1])
        if a.ndim > 1 and a.shape != want and n != "preprocessor.featurizer.fb":
            sys.exit(f"shape mismatch {n}: hf={a.shape} template(numpy order)={want}")

    if keep_tern:
        w.add_bool("parakeet.ternary.present", True)
        w.add_uint32("parakeet.ternary.group_size", 128)
    if any(n.endswith(".fvcodes") for n in raw):
        w.add_bool("parakeet.fivevalue.present", True)
    if any(n.startswith("vad_head.") for n in raw):
        proj, ctxw = raw["vad_head.proj.weight"], raw["vad_head.ctx.weight"]
        hop, sub, sr = (int(tv["parakeet.preprocessor.hop_length"]),
                        int(tv["parakeet.encoder.subsampling_factor"]),
                        int(tv["parakeet.preprocessor.sample_rate"]))
        w.add_bool("parakeet.vad.present", True)
        w.add_uint32("parakeet.vad.d_in", int(proj.shape[1]))
        w.add_uint32("parakeet.vad.hidden", int(proj.shape[0]))
        w.add_uint32("parakeet.vad.kernel", int(ctxw.shape[2]))
        w.add_float32("parakeet.vad.frame_sec", hop * sub / sr)

    for n, a in raw.items():
        w.add_tensor(n, a)

    written = quantized = 0
    for n, a in tens.items():
        ne = list(a.shape[::-1])
        qt = None
        if args.dtype != "f32" and any(rx.match(n) for rx in _QUANTIZABLE) and len(ne) >= 2 and ne[0] >= 32 and ne[1] >= 32:
            if args.dtype == "f16":
                qt = gguf.GGMLQuantizationType.F16
            elif ne[0] % 32 == 0:
                qt = gguf.GGMLQuantizationType.Q8_0
        if qt is None:
            w.add_tensor(n, a)
        else:
            qa = gguf.quantize(a, qt)
            w.add_tensor(n, qa, raw_shape=qa.shape, raw_dtype=qt)
            quantized += 1
        written += 1

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.output}: arch=tdt tensors={written + len(raw)} dtype={args.dtype} "
          f"quantized={quantized} ternary={'keep' if keep_tern else 'dequant'} "
          f"five-value={args.five_value if args.fermion else 'n/a'} raw={len(raw)}")


def _copy_kv(w, key, f):
    """Re-emit one template KV with its original GGUF type."""
    t = f.types
    v = f.contents()
    T = gguf.GGUFValueType
    if t[0] == T.ARRAY:
        w.add_array(key, list(v))
        return
    {T.UINT32: w.add_uint32, T.INT32: w.add_int32, T.FLOAT32: w.add_float32,
     T.BOOL: w.add_bool, T.STRING: w.add_string, T.UINT64: w.add_uint64,
     T.INT64: w.add_int64, T.UINT8: w.add_uint8, T.UINT16: w.add_uint16}[t[0]](key, v)


if __name__ == "__main__":
    main()
