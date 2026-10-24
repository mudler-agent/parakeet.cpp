#!/usr/bin/env python3
"""Compare parakeet.cpp with the transformers reference run of hf_phonon_reference.py.

The reference side is the ``ref.json`` and ``ref.npz`` that
``scripts/hf_phonon_reference.py`` wrote. The parakeet.cpp side is a GGUF, run
through ``build/tests/dump_encoder`` (log-mel and encoder output) and
``parakeet-cli transcribe --json`` (token ids and text).

For every clip it prints

  enc(hf mel)   largest absolute encoder difference when parakeet.cpp runs the
                encoder on the reference's own log-mel features, cut to the frames
                the reference's attention mask marks as audio: this isolates the
                weights and the graph from the front end (the number to compare
                with the 1e-4 target). The reference encodes one trailing mel
                frame of some clips as padding, which parakeet.cpp has no reason
                to do, so that frame is left out of the input and the output,
  mel           largest absolute difference of the two log-mel front ends,
  enc(own mel)  the same encoder difference with parakeet.cpp's own front end,
  tokens/text   whether the whole pipeline emits the reference's token ids and text.

The exit status is 1 when a clip misses --enc-tol on the first column or has
different token ids (unless --no-fail).

  python3 scripts/compare_hf_reference.py --ref ref/ --model phonon-2.5-f32.gguf \
      --build build --work scratch/ clip1.wav clip2.wav
"""
import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ref", required=True, help="directory with ref.json and ref.npz")
    ap.add_argument("--model", required=True)
    ap.add_argument("--build", default="build", help="CMake build directory")
    ap.add_argument("--work", required=True, help="directory for the dumps")
    ap.add_argument("--enc-tol", type=float, default=1e-3)
    ap.add_argument("--no-fail", action="store_true")
    ap.add_argument("wavs", nargs="+")
    a = ap.parse_args()

    ref = json.load(open(pathlib.Path(a.ref) / "ref.json"))
    arr = np.load(pathlib.Path(a.ref) / "ref.npz")
    work = pathlib.Path(a.work)
    own, hfm, hfe = work / "own", work / "hfmel", work / "hfmel_out"
    for d in (own, hfm, hfe):
        d.mkdir(parents=True, exist_ok=True)
    stems = [pathlib.Path(w).stem for w in a.wavs]
    for s in stems:
        n = int(arr["len_" + s]) if "len_" + s in arr else arr["mel_" + s].shape[0]
        arr["mel_" + s][:n].T.astype(np.float32).tofile(hfm / f"{s}.mel.f32")   # [n_mels, n]
    dump = str(pathlib.Path(a.build) / "tests" / "dump_encoder")
    subprocess.run([dump, a.model, str(own), *a.wavs], check=True, stdout=subprocess.DEVNULL)
    subprocess.run([dump, "--mel-dir", str(hfm), a.model, str(hfe), *a.wavs], check=True, stdout=subprocess.DEVNULL)

    bad, worst = 0, 0.0
    print(f"{'clip':<18}{'frames':>7}{'enc(hf mel)':>13}{'mel':>10}{'enc(own mel)':>14}{'max|ref|':>10}  tokens  text")
    for w, s in zip(a.wavs, stems):
        e = arr["enc_" + s]                      # [T', d]
        m = arr["mel_" + s]                      # [T, n_mels]
        o1 = np.fromfile(hfe / f"{s}.enc.f32", dtype=np.float32).reshape(e.shape[1], -1)
        e1 = e[: o1.shape[1]]                    # frames the reference encodes as audio
        o2 = np.fromfile(own / f"{s}.enc.f32", dtype=np.float32).reshape(e.shape[1], -1)
        mel_own = np.fromfile(own / f"{s}.mel.f32", dtype=np.float32).reshape(m.shape[1], -1)
        d1 = float(np.abs(o1.T - e1).max())
        d2 = float(np.abs(o2.T - e).max()) if o2.shape[1] == e.shape[0] else float("nan")   # all frames
        k = min(mel_own.shape[1], m.shape[0])
        dm = float(np.abs(mel_own[:, :k].T - m[:k]).max())
        worst = max(worst, d1)
        out = subprocess.run([str(pathlib.Path(a.build) / "examples" / "cli" / "parakeet-cli"), "transcribe",
                              "--model", a.model, "--input", w, "--json"],
                             check=True, capture_output=True, text=True).stdout
        j = json.loads(out)
        same_ids = [t["id"] for t in j["tokens"]] == ref[s]["token_ids"]
        same_text = j["text"].strip() == ref[s]["text"].strip()
        print(f"{s:<18}{e1.shape[0]:>7}{d1:>13.2e}{dm:>10.1e}{d2:>14.2e}{float(np.abs(e1).max()):>10.2f}  "
              f"{'same' if same_ids else 'DIFF':<6}  {'same' if same_text else 'DIFF'}")
        if d1 > a.enc_tol or not same_ids:
            bad += 1
    print(f"clips={len(a.wavs)} worst_encoder_max_abs_diff_on_reference_mel={worst:.3e} failing={bad}")
    if bad and not a.no_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
