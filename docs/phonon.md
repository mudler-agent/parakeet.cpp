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
| `phonon-2.5-f16.gguf`, `-q8_0`, `-q4_k` | Phonon-2.5 (25 languages) | the container expanded to ordinary weights, then F16, Q8_0 or Q4_K | any backend |
| `phonon-2.5-packed.gguf` | Phonon-2.5 | the five-value encoder kept packed, see [Packed files](#packed-files-five-value-encoder) | any backend (a GPU gets dense weights, see [below](#packed-files-on-a-gpu)), offline only |
| `phonon-2.5-english-f16.gguf`, `-q8_0`, `-q4_k` | Phonon-2.5-English | same as the multilingual ones | any backend |
| `phonon-2.5-english-packed.gguf` | Phonon-2.5-English | same | any backend, offline only |

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
  for the other models (the F32 file is the converter's default `--dtype f32`).
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
the cut, the largest difference per clip is between 1.8e-3 and 1.1e-1 (3.4e-6 on `speech.wav`), and the token ids stay the same. Cutting the input in the comparison
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

## Packed files (five-value encoder)

`--five-value keep` keeps the 264 five-value linears as the bytes Fermion ships, instead of
expanding them (`phonon-2.5-packed.gguf`, `phonon-2.5-english-packed.gguf`). The converter also writes
`parakeet.fivevalue.present`. The engine splits each matrix into two ternary planes at load
and runs both on the Redux kernels. Format, runtime path, tests and limits are in
[ternary.md](ternary.md#five-value-encoders-phonon-25).

```
python3 scripts/convert_hf_parakeet_to_gguf.py \
    --fermion phonon-2.5.bps.tar.zst --sums SHA256SUMS.txt \
    --config parakeet-tdt-0.6b-v3/config.json --template tdt-0.6b-v3-f16.gguf \
    --name phonon-2.5 --dtype f16 --five-value keep --output phonon-2.5-packed.gguf
```

| File | Size | Resident weights |
|---|---:|---|
| `phonon-2.5-packed.gguf` | 241.7 MB | the file, plus about 300 MB of repacked planes (2 planes x 2 bits x 603 M weights) |
| `phonon-2.5-q8_0.gguf` | 940.7 MB | the file |
| `phonon-2.5-f16.gguf` | 1,441.0 MB | the file |

The 242 MB is 160 MB of five-value bytes (2.1 bits per weight) and 82 MB of other tensors. Those are
F32, as for every GGUF here, except the three allowlisted linears that `--dtype f16` stores as F16. A
smaller packed file would need those tensors in a lower precision, which this converter does not do.

What was checked, with the F16 file as the reference:

- All 264 matrices of Phonon-2.5: the planes rebuild the weights of the F16 file bit for bit
  (0 mismatching weights of 603 million; `test_fivevalue_model`). The converter's output holds the
  upstream bytes unchanged (compared tensor by tensor).
- `tests/fixtures/speech.wav` and `two_speakers.wav` give the same transcripts as the F16 file; a batch
  of the two equals the two single runs.
- The int8 activation quantization is not part of the upstream model, so the encoder output is not
  the F16 output. On 4 clips the relative L2 difference of the encoder output is 7.9% to 11.2%
  (Redux, same test on 2 clips: 4.4% and 4.6%, measured against its Q8_0 file). By layer on `speech.wav`: 1.1% after
  layer 0, 4% to 6% in the middle, 11% after layer 23. Transcripts and WER are not visibly affected,
  see below.
- WER on the FLEURS subset above: packed 5.65 (MEAN) and 5.47 pooled, against F16 5.59 and 5.42. The
  paired bootstrap difference packed minus F16 is +0.05 points, interval -0.17 to +0.28. For
  Phonon-2.5-English the packed file scores 5.67 on the 30 English utterances, like its F16 file.

### Packed files on a GPU

On a GPU backend the engine expands the packed matrices to the weights of the F16 file while it uploads
them, so the GPU holds 1.4 GB and runs the F16 graph. The packed file is then only a smaller download.
Checked on two CUDA devices (Arm SoCs with an integrated GPU, compute capability 8.7 and 11.0), with the
log-mel features of the CPU run as input to both:

- Encoder output of the packed file and of the F16 file on the same device: bit identical (maximum
  difference 0) on `speech.wav` and `two_speakers.wav`. Same token ids.
- Packed file on the GPU against the packed file on the CPU: relative L2 difference 8.6% to 10.4%, the
  int8 activation quantization of the CPU path (see above). Token ids identical on both clips.
- F16 file on the GPU against the F16 file on the CPU: 0.6% to 1.9% (F16 activations in the GPU matrix
  products). Token ids identical.
- A ROCm device was not tested.

`PARAKEET_DEVICE=cpu` runs the packed file on the CPU kernels instead. Details in
[ternary.md](ternary.md#five-value-files-on-a-gpu).

### Speed

**These timings are not reliable.** The host was shared and busy for the whole session: the one-minute load
average was between 9 and 26 on 12 logical CPUs (AMD Ryzen 5 7430U, 6 cores, AVX2, no AVX-512, so the
`avx2` ternary kernel ran). No run kept a load below 6. Read the numbers as
indicative; repeat them on an idle machine before quoting them.

One thread, pinned to one logical CPU (`taskset -c 5 parakeet-cli bench --threads 1`), the first 30 utterances of
`benchmarks/librispeech_manifest.tsv` (241.7 s of audio). RTF is audio seconds over the summed per-utterance
`proc_ms`; two passes, packed, Q8_0 and F16 one after the other. Load is the one-minute average at the start of the run.

| File | RTF pass 1 (load) | RTF pass 2 (load) | Peak RSS |
|---|---|---|---:|
| `phonon-2.5-packed.gguf` | 4.40 (20.1) | 4.18 (13.4) | 755 MB |
| `phonon-2.5-q8_0.gguf` | 3.44 (10.8) | 2.66 (9.8) | 1,099 MB |
| `phonon-2.5-f16.gguf` | 3.49 (12.0) | 2.67 (25.5) | 1,590 MB |

Eight threads, all 100 utterances (901 s), the three files in turn, three passes, `taskset` not used. RTF per pass:
packed 6.6, 6.6, 7.5; Q8_0 5.2, 4.7, 7.0; F16 6.6, 8.6, 6.5. The load average at the start of the runs was 5.0 to 26
(only the two first packed runs started below 6, and the load rose at once). That is too noisy to rank the
files.

The kernel alone (`build/tests/bench_ternary N K 100 10`, one thread, load about 16): one plane of
4096 x 1024 takes 4.1 ms, ggml Q8_0 11.6 ms and F16 9.3 ms; for 1024 x 4096 4.7 ms, 11.7 ms and 8.9 ms. The packed
linear runs two planes on the same quantized activations, which is about twice the single-plane time, so it
should stay below Q8_0 on this CPU, and the end-to-end runs above agree.

WER on the 100 LibriSpeech utterances of that manifest (plain normalizer, same run as the 8-thread timings):
F16 1.78, Q8_0 1.78, packed 1.91 (about 2 words more).

### Where the time goes, and the fused kernel

Profile first. One thread pinned to one logical CPU, packed file, the first 10 utterances of the manifest
(70 s of audio), AVX2 kernel, with counters around the three custom ops of a five-value linear
(the counters were temporary and are not in the tree). The one-minute load was 9 to 10 when the runs started, so read the
shares, not the milliseconds:

| Part | Time | Share of the run |
|---|---:|---:|
| Matmul, plane A (scale lo) | 6.4 s | 40% |
| Matmul, plane B (scale hi) | 6.4 s | 40% |
| Activation quantization (once per linear) | 0.09 s | 0.6% |
| Sum of the two planes | 0.22 s | 1.4% |
| Everything else (attention, convolutions, subsampling, mel, decoder) | about 2.9 s | 18% |

Two runs gave the same split. The cost is the matmul, and each plane costs the same, so the packed linear is
two Redux linears. Quantizing the activations once is already free, and the plane sum is small.
What a fused kernel can save is the sum (1.4%), the intermediate output and part of the loop and
activation-load work.

`PARAKEET_TERNARY_FUSED=1` switches on a fused AVX2 kernel: one pass over the activations and both planes, the
integer group sums and the float accumulator of each plane kept apart, one output write. Each plane is scaled and
accumulated exactly as in the single-plane kernel and the two results are added at the end, so the output is
bit-identical to two planes plus an add. It is not one shared accumulator, because that would change the
float rounding. `test_fivevalue` checks the fused kernels against the reference on fixed and random shapes (40
random N, K, T, with partial row ranges, an all-zero activation row and an outlier): every output float equal, bytes
outside the range untouched. The encoder output of `speech.wav` and `two_speakers.wav` (mel, layers 0, 11, 23 and the final
output) is byte-identical with and without the switch, and `test_fivevalue_model` passes with it.

It is not faster end to end, so it is off by default:

- `bench_ternary 4096 1024 100`, 1 thread, interleaved best of 3, load 11 to 15, so indicative only: 9.1 to 9.7 ms for
  two planes plus add, 8.7 to 9.1 ms fused with two activation rows per tile, 9.7 to 10.5 ms with one or
  three rows (one earlier run at load 4.5: 9.8 ms and 8.9 ms). 1024 x 4096: 8.0 ms and 8.2 ms. The register file (16 ymm) holds the four int16 accumulators of two
  planes for two rows and not much more; the work per MAC is about the same as in two passes.
- End to end, one thread, three rounds (table below): fused 4.07, 4.41, 4.61 RTF against 4.29, 4.53, 4.78 for two planes,
  that is 3% to 5% slower each time; at 8 threads 8.93, 8.28, 10.96 against 8.27, 8.28, 10.52, within the noise.

Most likely the kernel is limited by vector ALU work (decoding the 2-bit codes and the `maddubs` accumulation), not
by memory traffic or the sum, which would explain why sharing the activation loads does not help. No hardware
counters were available to confirm it. A saving would have to come from the decode itself, which means a different
weight layout, and the layout is shared with Redux.

### Benchmark: packed, Q8_0, F16

Same setup as above, `parakeet-cli bench --decoder tdt` on the first 10 utterances of the manifest (70.0 s of audio),
one thread pinned to one logical CPU, and 8 threads on CPUs 0 to 7. RTF is audio seconds over the summed `proc_ms` of the
9 utterances after the first. Three rounds, each round runs every configuration once, in the order of the table.
Load is the one-minute load average read just before the run. **The host was never quiet: every run started at a load of 3.6
or more, and all are above the limit of 2 for a usable number.** The figures are for comparing the configurations that ran side by side,
not for quoting.

| Configuration | Threads | RTF round 1 / 2 / 3 | Load at start, round 1 / 2 / 3 |
|---|---:|---|---|
| packed, two planes (default) | 1 | 4.29 / 4.53 / 4.78 | 9.3 / 7.8 / 7.0 |
| packed, fused (`PARAKEET_TERNARY_FUSED=1`) | 1 | 4.07 / 4.41 / 4.61 | 10.4 / 6.5 / 6.0 |
| Q8_0 | 1 | 3.49 / 3.53 / 3.78 | 8.1 / 5.4 / 4.8 |
| F16 | 1 | 4.02 / 4.17 / 4.38 | 7.5 / 6.2 / 3.6 |
| packed, two planes (default) | 8 | 8.27 / 8.28 / 10.52 | 7.1 / 7.6 / 6.1 |
| packed, fused | 8 | 8.93 / 8.28 / 10.96 | 5.2 / 5.1 / 7.2 |
| Q8_0 | 8 | 6.64 / 6.68 / 8.45 | 7.4 / 7.3 / 4.5 |
| F16 | 8 | 8.20 / 9.74 / 3.10 | 8.8 / 7.9 / 14.2 |

Reading it: every row rises from round to round while the load fell, so compare configurations within a round, not across
rounds. At one thread the order is packed, then F16, then Q8_0, in all three rounds. Packed beats Q8_0 by 23% to 28% and F16 by
7% to 9% in the same round. At 8 threads packed is ahead of Q8_0 in every round; against F16 it ties in round 1 (8.27 against
8.20) and loses in round 2 (8.28 against 9.74); F16's round 3 ran at a load of 14 to 31 and says nothing. So the earlier
statement stands, that the two-plane version is not slower than Q8_0 on this CPU, and the F16 comparison is open at 8 threads.

Nothing here was measured on a quiet host. The raw per-run logs and the profile counters are not in the repository.

### Accelerators

- **AVX-512 VNNI and NEON dot product: not changed.** The existing single-plane kernels run both planes
  there. A fused version was not written: the machine used for this work (Ryzen 5 7430U, AVX2 only) cannot run
  VNNI, and has neither an aarch64 cross compiler nor qemu-user. Code that cannot be compiled and run here is not merged.
  `TernaryKernel::fn2` is the hook: a kernel that sets it is used when `PARAKEET_TERNARY_FUSED=1` is set. The AVX2 result says to measure
  before expecting a gain. The VNNI kernel does more MACs per instruction (see the single-thread table in
  [ternary.md](ternary.md)), so the shared loads and loop work could be a larger share there; that is untested.
- **GPU: no path, by design.** The packed linears are custom ops with host-side kernels over repacked host
  memory; the model loader refuses a GPU backend for them, and ggml's CUDA, Metal and Vulkan backends have no
  op that reads this layout. A clean GPU path means a device kernel for the 2-bit plane layout, planes
  uploaded as tensors and an op registered in the vendored ggml, which this repository patches only for small fixes.
  The profile above says what that would buy: the matmul is 80% of the CPU run, and the packed file is 242 MB
  against 941 MB for Q8_0, so a GPU kernel could only win on memory-bound devices, while the dense files already run
  on the GPU backends. No GPU was used for this work and no kernel was written for one.

## Licence

The Phonon-2.5 weights are released by Fermion Research under CC-BY-4.0, and the model is derived from
NVIDIA's [parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), also CC-BY-4.0. Fermion's
code is Apache-2.0; none of it is in this repository (`scripts/fermion_reader.py` is a new implementation of the
format). When you use the GGUF files, credit Fermion Research and NVIDIA, link the
[licence](https://creativecommons.org/licenses/by/4.0/) and say that the files are converted (and quantized, or
unpacked) copies. Nothing was trained or fine-tuned here.

The upstream `NOTICE` says that the training audio includes CHiME-6, distributed under CC-BY-SA-4.0, and
SPGISpeech, used under Kensho's public terms. The Hugging Face repo keeps that notice, unchanged, in
`NOTICE-phonon-2.5.txt`. Whether the share-alike term of CHiME-6 reaches the trained weights, and so the
converted files, is a legal question that this project does not settle. See also [licenses.md](licenses.md).

## Limits

- Packed files: CPU only, offline only, no streaming; the AVX2, AVX-512 VNNI and NEON dot-product kernels
  are fast, the scalar kernel (MSVC, Windows on ARM) is not. Use a dense file there. Only the AVX2 kernel was run for the
  five-value path on the machine used here; the VNNI and NEON kernels are the Redux kernels, tested on Redux.
- The format of the upstream container is marked v1 and the release is days old. The reader refuses
  anything it does not know; check the archive digest before relying on a newer release.
- The WER figures above come from 30 utterances per language. They do not rank the models.
