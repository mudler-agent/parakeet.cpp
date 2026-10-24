#!/usr/bin/env python3
"""Reference run of a Phonon-2.5 release on the stock transformers ParakeetForTDT.

The weights come from the release archive through scripts/fermion_reader.py
(five-value, int6 and fp16 records expanded to float32), are loaded into an
unmodified ``ParakeetForTDT`` built from the config of nvidia/parakeet-tdt-0.6b-v3,
and the model runs in fp32 on the CPU. For every wav it writes into --out:

  ref.json   {"<stem>": {"text": ..., "token_ids": [...]}}  (blank ids removed)
  ref.npz    mel_<stem>  [T, n_mels]  the features the processor produced
             len_<stem>  number of mel frames the processor's attention mask marks as
                         audio (the frames after it are padding for the encoder)
             enc_<stem>  [T', d_model]  the encoder output (the last frames can be
                         padding positions; see len_)

``scripts/compare_hf_reference.py`` compares these with parakeet.cpp.
Needs torch, transformers (with ParakeetForTDT), soundfile and librosa (only
for clips that are not 16 kHz).

  python3 scripts/hf_phonon_reference.py --release phonon-2.5.bps.tar.zst \\
      --base-dir <dir with config.json, generation_config.json, processor files> \\
      --out ref/ clip1.wav clip2.wav
"""
import argparse
import json
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import fermion_reader as fr  # noqa: E402

_POINTWISE = re.compile(r"\.conv\.pointwise_conv[12]\.weight$")


def state_dict(release):
    import torch
    sd = {}
    for rec in fr.iter_records(release.container):
        arr = fr.decode_record(rec)
        name = rec.name + (".weight" if rec.kind == "five_value" else "")
        if name.endswith("num_batches_tracked"):
            sd[name] = torch.tensor(0, dtype=torch.int64)   # BatchNorm in eval mode never reads it
            continue
        if arr.ndim == 2 and _POINTWISE.search(name):
            arr = arr[:, :, None]                             # stock layout: [O, I, 1]
        sd[name] = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    return sd


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--release", required=True)
    ap.add_argument("--sums")
    ap.add_argument("--sha256")
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("wavs", nargs="+")
    a = ap.parse_args()

    import soundfile as sf
    import torch
    from transformers import AutoProcessor, GenerationConfig, ParakeetForTDT, ParakeetTDTConfig
    torch.set_num_threads(a.threads)
    rel = fr.load_release(a.release, sums=a.sums, sha256=a.sha256)
    cfg = ParakeetTDTConfig.from_pretrained(a.base_dir)
    model = ParakeetForTDT(cfg)
    missing, unexpected = model.load_state_dict(state_dict(rel), strict=False)
    if missing or unexpected:
        sys.exit(f"weights do not fit the model: missing {list(missing)[:5]} unexpected {list(unexpected)[:5]}")
    model.eval()
    model.generation_config = GenerationConfig.from_pretrained(a.base_dir)
    proc = AutoProcessor.from_pretrained(a.base_dir)
    blank = cfg.blank_token_id

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, arrays = {}, {}
    for w in a.wavs:
        stem = pathlib.Path(w).stem
        x, sr = sf.read(w, dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        if sr != 16000:
            import librosa
            x = librosa.resample(x, orig_sr=sr, target_sr=16000).astype(np.float32)
        inp = proc([x], sampling_rate=16000, return_tensors="pt", padding=True)
        feats = inp["input_features"]
        with torch.no_grad():
            enc = model.encoder(input_features=feats, attention_mask=inp.get("attention_mask"))
            seq = model.generate(input_features=feats, attention_mask=inp.get("attention_mask"))
        seq = getattr(seq, "sequences", seq)
        text = proc.batch_decode(seq, skip_special_tokens=True)[0].strip()
        rows[stem] = {"text": text, "token_ids": [int(t) for t in seq[0].tolist() if int(t) != blank]}
        arrays["mel_" + stem] = feats[0].float().numpy()
        am = inp.get("attention_mask")
        arrays["len_" + stem] = np.array(int(am[0].sum()) if am is not None else feats.shape[1])
        arrays["enc_" + stem] = enc.last_hidden_state[0].float().numpy()
        print(f"{stem:>20} {text[:80]}", flush=True)
    json.dump(rows, open(out / "ref.json", "w"), indent=1, ensure_ascii=False)
    np.savez(out / "ref.npz", **arrays)


if __name__ == "__main__":
    main()
