# Phonon-2.5

## What it is

[Phonon-2.5](https://huggingface.co/FermionResearch/Phonon-2.5) by Fermion Research
is `nvidia/parakeet-tdt-0.6b-v3` with weights retrained for 25 languages and
compressed. [Phonon-2.5-English](https://huggingface.co/FermionResearch/Phonon-2.5-English)
is the English-only sibling. Both have the architecture of the v3 model: the
24-layer FastConformer encoder (d_model 1024, 8 heads, feed-forward 4096,
convolution kernel 9, subsampling x8), the 2-layer LSTM prediction network, the
TDT joint with durations 0 to 4, the same 8192-piece SentencePiece vocabulary and
the same 128-bin log-mel front end. The engine needs no code change to run them
once the weights are ordinary tensors.

What changed is the file format. Fermion ships the weights in a
`fermion-five-value-parakeet-v1` container inside a `.bps.tar.zst` archive
(about 163 MB):

| Record kind | Tensors | Content |
|---|---|---|
| `five_value` | 264: all 11 linear layers of every encoder block | every weight is 0, +-`lo[row]` or +-`hi[row]`: base-3 trits (5 per byte), one flag bit per nonzero weight that picks `hi`, and two fp16 scales per row |
| `int6` | 35: subsampling, depthwise convolutions, LSTM, embedding, projectors, joint | 6-bit signed integers with one fp16 scale per row |
| `fp16` | 424: norms, biases, BatchNorm statistics | plain |

A five-value matrix is `W = lo * A + hi * B`, where A and B are ternary matrices
that never share a nonzero position. `scripts/fermion_reader.py` documents the
layout and reads it with numpy only.

## Files

Converted here, not trained. Credit Fermion Research and NVIDIA, see [Licence](#licence).

| File | Source | Form | Runs on |
|---|---|---|---|
| `phonon-2.5-f16.gguf`, `-q8_0`, `-q6_k`, `-q4_k` | Phonon-2.5 (25 languages) | the container expanded to ordinary weights, then F16, Q8_0 or a K-quant | any backend |
| `phonon-2.5-english-f16.gguf`, `-q8_0`, `-q6_k`, `-q4_k` | Phonon-2.5-English | same | any backend |

The sizes and SHA-256 sums of the published files are in
[models/MANIFEST.md](../models/MANIFEST.md).

## Converting

```
python3 scripts/convert_hf_parakeet_to_gguf.py \
    --fermion phonon-2.5.bps.tar.zst --sums SHA256SUMS.txt \
    --config parakeet-tdt-0.6b-v3/config.json \
    --template tdt-0.6b-v3-f16.gguf \
    --name phonon-2.5 --dtype f16 --output phonon-2.5-f16.gguf
```

- `--fermion` takes the archive, a directory with the unpacked files, or a bare
  `model.fermion`. The archive SHA-256 is checked against `--sums` (or
  `SHA256SUMS.txt` next to the archive, or `--sha256`), and every member against the
  manifests inside the archive. A wrong digest stops the run; `--no-verify` skips the checks.
- `--config` is the transformers `config.json` of `nvidia/parakeet-tdt-0.6b-v3`. It is checked
  against the template and against the NeMo config inside the archive (encoder sizes, FFT size,
  `xscaling`, the whole vocabulary). Any difference aborts.
- `--template` is a v3 GGUF made by `convert_parakeet_to_gguf.py` (for example the
  published `tdt-0.6b-v3-f16.gguf`). It supplies the KVs, the vocabulary and the mel filterbank
  and window, as for the other HF checkpoints (see [conversion.md](conversion.md)).
- Zstandard is read with the standard library (Python 3.14), the `zstandard` package or the
  `zstd` command.
- K-quants come from `parakeet-cli quantize phonon-2.5-f32.gguf phonon-2.5-q4_k.gguf q4_k`, as
  for the other models.
- The 1x1 convolutions are reshaped to `[out, in, 1]` and the BatchNorm batch counters are
  dropped, as in the Redux converter.

`tests/python/check_fermion_reader.py` is a round-trip test of the reader: it encodes a small
container with an independent writer (rows whose length is not a multiple of 5, flag counts
that are not a multiple of 8, int6 and int8, fp16, a scalar), reads it back and requires the
exact matrices. It also checks the planes, the archive path and that a bad archive digest, a
bad member digest, a trit byte of 243 or more, a truncated record and trailing bytes are
refused. With `PARAKEET_TEST_FERMION_ARCHIVE` and `PARAKEET_TEST_FERMION_SUMS` it also reads a
real release.

## Parity with the transformers reference

There is no NeMo checkpoint for Phonon-2.5, so the reference is the stock
`transformers` `ParakeetForTDT` in fp32 on the CPU, loaded with the same container
(`scripts/hf_phonon_reference.py`; transformers 5.20.2, torch 2.14.1). It writes the
log-mel features the processor made, the encoder output and the token ids.
`scripts/compare_hf_reference.py` then runs parakeet.cpp on the F32 GGUF through
`build/tests/dump_encoder` (log-mel and encoder output) and `parakeet-cli transcribe --json`.

```
python3 scripts/hf_phonon_reference.py --release phonon-2.5.bps.tar.zst --sums SHA256SUMS.txt \
    --base-dir parakeet-tdt-0.6b-v3-hf --out ref/ clip1.wav clip2.wav
python3 scripts/compare_hf_reference.py --ref ref/ --model phonon-2.5-f32.gguf \
    --build build --work work/ clip1.wav clip2.wav
```

`--base-dir` is a copy of `nvidia/parakeet-tdt-0.6b-v3` with `config.json`,
`generation_config.json` and the processor and tokenizer files (revision `541d1f99`).

Phonon-2.5, F32 GGUF, 25 clips: `tests/fixtures/speech.wav` and the first 4 clips of the FLEURS
test split in English, German, French, Italian, Spanish (es_419) and Polish.

| Check | Result |
|---|---|
| Encoder output on the reference's own log-mel input, largest absolute difference over all frames and 1024 channels | 1.0e-6 worst clip (median clip 1.9e-7); the output values reach about 0.26 |
| Token ids and text, full parakeet.cpp pipeline with its own front end | identical on 25 of 25 clips |
| Log-mel of the two front ends, largest absolute difference | 1.6e-5 to 4.9e-4 |

Phonon-2.5-English, F32 GGUF, 5 clips (`speech.wav` and the first 4 FLEURS English clips): encoder
difference 1.3e-6 at worst on the reference's input, identical token ids and text on 5 of 5. The
English container has the same format, tensor names and shapes as the multilingual one, and its
NeMo config matches the v3 template in every field the converter checks, so it needed no code.

The input to the first row is the reference's log-mel cut to the number of frames its attention
mask marks as audio. The reference treats the last mel frame of most clips as padding.
parakeet.cpp has no reason to and encodes it. If the encoder is run on the same input without
the cut, the largest difference per clip is between 1.8e-3 and 1.1e-1 (3.4e-6 on `speech.wav`,
which has no such frame), and the token ids stay the same. Cutting the input in the comparison
removes the difference: on one clip it goes from 8.6e-2 to 2.7e-7. This convention is a property
of the engine and the reference, not of the Phonon weights; it was not checked against the v3
weights.

## WER on a FLEURS subset

These are our own measurements on a small subset. They are not the Open ASR Leaderboard
numbers, and they cannot be compared with the figures on the upstream model cards (full test
sets, another harness and normalizer).

- Data: the first 30 utterances of the FLEURS test split (`google/fleurs`, in dataset order, no
  filtering) in English (`en_us`), German, French, Italian, Spanish (`es_419`) and Polish: 180
  utterances, 4,204 reference words, 2,117.6 s of audio. Fetched with
  `scripts/fetch_fleurs_subset.py --n 30`.
- Scoring: `scripts/eval_manifest_wer.py` (`parakeet-cli bench`, greedy TDT). Both sides go
  through `scripts/asr_metrics.normalize` (lowercase, punctuation removed). Corpus WER per
  language, in percent; MEAN is the unweighted average of the six languages.
- Model files: Phonon-2.5 converted with this converter, the quantized files from the F32 GGUF,
  and `tdt-0.6b-v3-f16.gguf` (the published NeMo v3 conversion) for comparison.

| Language (words) | Phonon-2.5 F16 | Q8_0 | Q4_K | v3 F16 |
|---|---:|---:|---:|---:|
| English (670) | 4.78 | 4.78 | 4.78 | 5.67 |
| German (681) | 5.29 | 5.58 | 5.43 | 5.29 |
| French (859) | 6.87 | 6.75 | 7.10 | 4.89 |
| Italian (747) | 3.75 | 3.75 | 4.15 | 1.61 |
| Spanish (749) | 3.60 | 3.60 | 3.47 | 3.34 |
| Polish (498) | 9.24 | 9.44 | 10.64 | 7.83 |
| MEAN | 5.59 | 5.65 | 5.93 | 4.77 |
| pooled over all 4,204 words | 5.42 | 5.47 | 5.71 | 4.57 |

With 30 utterances per language one word is 0.15 to 0.2 points of a language, so single cells
are noisy. Paired bootstrap over the 180 utterances (2,000 resamples, pooled WER, 95% interval):

| Difference | Points | Interval |
|---|---:|---|
| Q8_0 minus F16 | +0.05 | -0.05 to +0.18 |
| Q4_K minus F16 | +0.29 | -0.09 to +0.70 |
| Phonon-2.5 F16 minus v3 F16 | +0.86 | +0.12 to +1.60 |

Q8_0 and Q4_K are not distinguishable from F16 on this sample. Phonon-2.5 is worse than v3 on
this subset (French, Italian and Polish carry the difference; English is better on these 30
utterances). The upstream card reports the same direction for French, Italian and Polish on
the full test sets, and a better 25-language average for Phonon-2.5. This subset does not test
that claim: it has 6 of the 25 languages.

Phonon-2.5-English on the 30 English utterances: F16 5.67, Q8_0 5.67, Q4_K 5.37 (670 words;
the multilingual model scores 4.78 on the same utterances and v3 5.67). That is a difference
of 6 words, so it says nothing about which model is better.
