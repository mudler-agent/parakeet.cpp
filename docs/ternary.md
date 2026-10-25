# Ternary encoders (moondream/parakeet-redux)

## What Redux is

`moondream/parakeet-redux` is a Parakeet TDT 0.6B v3 derivative whose encoder
linear layers are ternary: every weight is -1, 0 or +1 times a per-group scale.
Its sibling `moondream/parakeet-ultra` has the same shapes but ordinary F16
weights, so it needs no special support and loads like any other v3 GGUF.

In the encoder the ternary layers are the two FFN pairs (`linear1`, `linear2`
in each of `feed_forward1` and `feed_forward2`), the attention projections
(`linear_q/k/v/out/pos`) and the two pointwise convolutions, which is 11
layers per block and 264 in the 24 block encoder. Norms, biases, the
depthwise conv, the subsampling stack, the prediction net and the joint stay
in ordinary float tensors.

## GGUF form

Produce it with `scripts/convert_hf_parakeet_to_gguf.py --ternary keep` (see
`docs/conversion.md`). For each ternary linear `<base>` the file holds:

| Tensor | Type | Content |
|---|---|---|
| `<base>.qweight` | I8 | the upstream bytes, N rows of ceil(K/5) bytes (5 base-3 digits per byte) |
| `<base>.scales` | F16 | one scale per row and 128 input columns, shape N x K/128 |

and no `<base>.weight`. Two KVs mark the file: `parakeet.ternary.present`
(bool) and `parakeet.ternary.group_size` (u32, 128). The loader reads them into
`ParakeetConfig::ternary`. A packed Redux GGUF is 213.3 MB on disk, against
1441.0 MB for the v3 F16 file.

## Runtime path

1. At load time `ternary_prepare` repacks every `qweight` once into a
   kernel layout with 2 bits per weight (the same size as the upstream
   bytes). Rows are grouped in blocks of 16, padded with zeros at the end.
   For each block and each run of 16 columns there are 64 bytes: byte
   4*i + j holds the codes of row i at columns 4p + j of the run, for
   p = 0..3, in its four bit pairs. The kernels keep one output row per
   vector lane, so a group sum for 16 rows comes out in one vector with no
   horizontal reduction. The exact layout is in `TernaryWeight` in
   `src/ternary.hpp`. Nothing is dequantized per call.
   The repack decodes the upstream bytes with a 256 entry table (five
   trits per byte) and weights are repacked in parallel on up to
   min(cores, 8) threads. `PARAKEET_REPACK_THREADS=N` overrides the count; 1
   forces the serial path. The output is byte-identical to the earlier scalar
   code, which `tests/test_ternary.cpp` keeps as `repack_reference`. Measured
   on packed Redux (`parakeet-cli bench`, `load_ms`, `taskset -c 0-7`, 5 runs,
   load average 2.3 to 3.0): 866 ms min / 868 ms median before, 74 ms min /
   76 ms median after; with `PARAKEET_REPACK_THREADS=1`, 204 ms min / 208 ms
   median. In-process, 264 weights of 4096x1024 and 1024x4096 take 1414 ms
   (old code) against 210 ms (table, one thread) and 29 ms (8 threads);
   `bench_ternary repack` reproduces this.
2. At run time the activations of each ternary linear are quantized to int8
   per token (one float scale and one int32 sum per 128-column group). The
   scalar version defines the bytes; the AVX-512 and AVX2 versions write the
   same bytes.
3. The layer is a pair of ggml custom ops: one quantizes the activations, one
   runs the integer matmul and applies the scales. The dot product uses
   `code * int8` sums with the constant offset removed through the group sums.
4. The kernel is picked once per process: AVX-512 VNNI, then NEON dot product,
   then AVX2, then a scalar reference. All kernels are bit-identical to the
   scalar reference. `PARAKEET_TERNARY_KERNEL=scalar|avx2|vnni|neon` forces one
   (an unavailable name falls back to automatic selection with a log line).
   The activation quantizer is picked by CPU features only.

Reference path. Upstream's reference path for Redux is weight-only: the ternary
weights are expanded and the activations stay in floating point. The int8
activation quantization in steps 2 and 3 is our own scheme, chosen so the
integer kernels can run, and it is not part of the upstream model. The
reference for this engine is therefore `--ternary dequant` (ordinary F16 or
Q8_0 weights expanded from the ternary ones). The packed path is checked
against it: same transcripts on the test fixtures and 64 differing words in
22,922 on the long talks (see the long-form section), but it is not
bit-identical to it and is not a reproduction of any upstream runtime. We did
not run or inspect Moondream's own runtime (Photon).

## Limits

- A packed tensor anywhere in the file (any name ending in `.qweight`) while
  `parakeet.ternary.present` is false is refused at load, and so is the flag set with no packed
  tensors. Every tensor name is scanned, not only layer 0.
- CPU only. Loading a packed GGUF with a GPU backend active fails with a
  message; re-convert with `--ternary dequant` to use a GPU.
- The kernel is chosen at run time. x86-64 with AVX2 or AVX-512 VNNI and aarch64 with the
  dot-product extension (dotprod) get SIMD kernels. MSVC builds, Windows on ARM and aarch64 without
  dotprod select the scalar kernel (about 1 GMAC/s); the load logs a warning when that happens.
  Re-convert with `--ternary dequant` on such machines.
- The loader keeps the original packed tensors resident next to the repacked planes, so a packed
  model uses more weight memory than its file size (roughly twice for Redux, not measured).
- No cache-aware streaming. `StreamingEncoder` rejects packed GGUFs; use a
  `--ternary dequant` file for streaming.
- `parakeet-cli quantize` never touches ternary tensors. Running it on a packed
  file copies `.qweight` (I8) and `.scales` (F16) verbatim.

## Measured speed

### How the numbers were measured

The numbers come from this repository's standard build. Its CMake configure
applies the in-tree ggml patches from `third_party/ggml-patches`
(`scripts/apply_ggml_patches.sh`); the ggml submodule pin itself is unchanged.
Results on an unpatched ggml were not measured.

Machine: AMD Ryzen 9 9950X3D (16 cores, Zen 5, AVX-512 VNNI), CPU backend,
2026-09-29. The end-to-end numbers were taken at commit f228376, the microbench
at 8b66c63 (which only changes the compile target of the activation
quantizer). Ultra has the same architecture and shapes as v3, so `ultra-q8_0`
(converted from the HF weights with `--dtype q8_0`) stands in for a Q8_0 v3.
The dequantized Redux was also converted to Q8_0 the same way. Every timed
command was serialized with `flock /tmp/pk-bench.lock`.

### Per utterance, LibriSpeech

```
taskset -c 0-7 build/examples/cli/parakeet-cli bench --model <gguf> \
    --manifest benchmarks/librispeech_manifest.tsv --decoder tdt --threads 8 --json <out.json>
```

100 utterances, 901.1 s of audio, one utterance at a time. RTF is total audio
over the summed per-utterance `proc_ms` of the JSON, median of 3 full passes.
WER is against the manifest text after lowercasing and replacing every
character other than a-z and the apostrophe with a space; it is from a quick
script, not `validate_vs_nemo.py`, and is identical across passes. The kernel
rows set `PARAKEET_TERNARY_KERNEL`; "automatic" leaves it unset, which picks
vnni here.

| Model | Form | RTF pass 1 / 2 / 3 | Median RTF | WER |
|---|---|---|---:|---:|
| parakeet-redux | packed ternary, automatic (vnni) | 74.8 / 75.6 / 78.0 | 75.6 | 1.96% |
| parakeet-redux | packed ternary, avx2 | 59.6 / 59.0 / 58.8 | 59.0 | 1.96% |
| parakeet-redux | dequantized F16 | 46.6 / 46.1 / 45.6 | 46.1 | 1.92% |
| parakeet-redux | dequantized Q8_0 | 42.7 / 42.3 / 42.7 | 42.7 | 1.96% |
| parakeet-ultra | Q8_0 | 42.5 / 43.1 / 44.2 | 43.1 | 1.71% |
| parakeet-redux | packed ternary, previous kernel (b4164da) | 40.3 / 40.2 / 40.1 | 40.2 | 1.96% |

The last row is the kernel before the current one (one row per call, 256-bit
vectors, scalar activation quantization), built from commit b4164da and run in
the same session. The transcripts of the old and the new kernel are identical
for all 100 utterances, as expected from bit-identical kernels.

### Long clip, one utterance

`transcribe --decoder tdt` on `benchmarks/audio/diverse/i_have_a_dream.wav`
(180 s), whole process wall time including model load, median of 5 runs:

```
taskset -c 0-7 build/examples/cli/parakeet-cli transcribe --model <gguf> \
    --input benchmarks/audio/diverse/i_have_a_dream.wav --decoder tdt --threads 8
```

RTF is 180 s divided by the median.

| Model | Form | Size on disk | 8 threads (cores 0-7) | RTF | 16 threads (cores 0-15) |
|---|---|---:|---:|---:|---:|
| parakeet-redux | packed ternary, vnni kernel | 213.3 MB | 7.93 s | 22.7 | 7.45 s |
| parakeet-redux | packed ternary, avx2 kernel | 213.3 MB | 8.52 s | 21.1 | 8.01 s |
| parakeet-redux | dequantized F16 | 1441.9 MB | 8.73 s | 20.6 | 7.75 s |
| parakeet-redux | dequantized Q8_0 | 941.5 MB | 9.04 s | 19.9 | 8.11 s |
| parakeet-ultra | Q8_0 | 941.5 MB | 9.04 s | 19.9 | 8.29 s |
| parakeet-ultra | F16 | 1441.9 MB | 8.63 s | 20.9 | 7.83 s |
| parakeet-tdt-0.6b-v3 | F16 | 1441.0 MB | 32.79 s | 5.5 | 29.29 s |

The v3 F16 row read its model from a network share, and its wall time includes
that model load, so it is not comparable with the other rows.

The scalar kernel is a reference only (116.39 s for this clip in one run at
commit c8499e6, with an earlier and faster form of the reference).

### Single-thread kernel throughput

```
taskset -c 2 build/tests/bench_ternary 4096 1024 200 10
taskset -c 2 build/tests/bench_ternary 1024 4096 200 10
taskset -c 2 build/tests/bench_ternary 1024 4096 1000 10
```

The ggml rows are `ggml_mul_mat` with an F32 activation matrix on one thread,
so they include ggml's own activation quantization; the ternary rows time only
the matmul kernel. `bench_ternary` times `ternary_quant_rows` on its own line.

| N x K, T | scalar | ternary avx2 | ternary vnni | ggml Q8_0 | ggml F16 | ternary quant |
|---|---:|---:|---:|---:|---:|---:|
| 4096 x 1024, T=200 | 1.04 | 184.12 | 546.86 | 87.51 | 121.21 | 0.014 ms |
| 1024 x 4096, T=200 | 1.05 | 184.38 | 553.21 | 91.28 | 129.36 | 0.060 ms |
| 1024 x 4096, T=1000 | 1.05 | 183.27 | 492.23 | 89.85 | 125.12 | 0.408 ms |

Kernel columns are GMAC/s. For comparison, the previous kernel (b4164da)
measured 81.91, 87.01 and 82.57 GMAC/s (vnni) and 78.14, 78.63 and 66.79
(avx2) with the same commands. The quantization of the activations took 1.405 ms
for K=4096, T=200 before it was vectorized, about as long as the new matmul.
Repeated runs of the same commands moved the vnni numbers by up to about 10
percent (the T=1000 shape measured between 492 and 542 over five runs).

### What the data say

- The vnni kernel does 490 to 550 GMAC/s on one core, 6.0 to 6.7 times the
  previous kernel and 3.9 to 6.3 times ggml's Q8_0 and F16 mul_mat (546.86/87.51 = 6.25 at the top, 492.23/125.12 = 3.93 at the bottom). At a
  clock of about 5 GHz (not measured) that is about 110 int8 multiply-adds
  per cycle, close to two 512-bit `vpdpbusd` per cycle. The gain comes from
  keeping one output row per vector lane (no horizontal reduction per group),
  512-bit vectors, 12 independent int32 accumulators, and unpacking the 2-bit
  weights once per 4 activation rows.
- Per utterance on LibriSpeech, packed ternary is now the fastest form: median
  RTF 75.6 against 46.1 for the same model in F16 and 43.1 for Ultra Q8_0,
  1.6 to 1.8 times faster. The three passes of each row are within 5 percent
  of each other, well inside that margin. WER is unchanged at 1.96 percent.
- On the 180 s clip the gain is small: 7.93 s against 8.73 s (F16) and
  9.04 s (Q8_0), 9 to 12 percent less time. The linears are a smaller share of this run,
  which includes model load and attention over a long sequence.
- The avx2 kernel (184 GMAC/s, 59.0 RTF) is also faster than the ggml F16 and
  Q8_0 paths. It accumulates a whole group in int16, which is exact here.
- Moondream reports 113x for its own Photon runtime with these ternary weights
  on 8 Zen 5 cores, against 45x for parakeet.cpp Q8_0. We did not run Photon;
  our 75.6 is not measured under the same conditions and is not a comparison
  with that number.
- The NEON kernel uses the same layout and is bit-identical to the scalar
  reference under `qemu-aarch64`; its speed on real ARM hardware has not been
  measured.
- The v3 F16 row is 3.8 times slower than Ultra and Redux F16 with identical
  shapes on the long clip. That gap was not investigated.

Transcripts on `tests/fixtures/speech.wav` are identical across the scalar,
avx2 and vnni kernels and the dequantized F16 and Q8_0 Redux, and equal the
reference transcript in `AGENTS.md`.

## Five-value encoders (Phonon-2.5)

Phonon-2.5 (see [phonon.md](phonon.md)) stores the same 264 encoder linears as Redux, but each
weight is 0, plus or minus `lo[row]`, or plus or minus `hi[row]`: two scales per output row instead
of one per group of 128 columns, and one flag bit per nonzero weight that picks between them.
Every such matrix is `W = lo * A + hi * B`, where A and B are ternary matrices that never share a
nonzero position. That is two ternary matrices, so the Redux kernels can run it.

### GGUF form

`convert_hf_parakeet_to_gguf.py --fermion ... --five-value keep` writes the upstream bytes
unchanged. For each linear `<base>` the file holds:

| Tensor | Type | Content |
|---|---|---|
| `<base>.fvcodes` | I8 | N rows of ceil(K/5) bytes of trits, as in `qweight` (digit 1 is a zero weight) |
| `<base>.fvhi` | I8 | ceil(nnz/8) bytes, one bit per nonzero weight in row-major order, least significant bit first; 1 selects `hi` |
| `<base>.fvscale` | F16 | shape [N, 2]: `lo[N]` followed by `hi[N]` |

and no `<base>.weight`. The KV `parakeet.fivevalue.present` (bool) marks the file; the loader reads
it into `ParakeetConfig::fivevalue`. The other tensors are ordinary (F32, or F16 for the three
allowlisted linears with `--dtype f16`). A file without the flag loads exactly as before.

### Runtime path

1. Load: `ternary_prepare` validates the three tensors of every linear against the shape the
   model config demands, then splits each matrix into two `TernaryWeight` planes. Plane A holds
   the weights whose flag is 0 with the scale `lo`, plane B those whose flag is 1 with the scale
   `hi`; the row scale is repeated for every group of 128 columns. The split reuses the Redux
   repack, so the planes have the kernel layout described above, and runs on the same thread pool
   (`PARAKEET_REPACK_THREADS`). A flag stream whose length does not match the number of nonzero
   trits, a trit byte of 243 or more and a scale that is not finite are refused at load.
2. Run: the activations of a linear are quantized to int8 once (the same op as Redux), both planes
   run on those bytes, and the two outputs are added. The kernels and their selection are those
   of the Redux path (`PARAKEET_TERNARY_KERNEL`).
3. The result is not bit-identical to the dequantized file: it carries the same int8
   activation quantization as Redux. The reference for this engine is the dequantized file; the
   packed file is checked against it by transcripts.

Limits are those of Redux, except that a GPU backend is supported (next section): offline only
(`StreamingEncoder` refuses the file), on the CPU the original bytes stay resident next to the repacked
planes (4 bits per weight for the two planes, on top of the about 2.1 bits per weight of the file), and
`parakeet-cli quantize` copies the three tensors unchanged.
The scalar kernel is selected on MSVC builds, Windows on ARM and aarch64 without dotprod; use a
dequantized file there.

`tests/test_fivevalue.cpp` checks, without a model, that `lo * A + hi * B` equals the dequantized
weights bit for bit for random matrices (including a row length that is not a multiple of 5, N that
is not a multiple of 16, no hi weights, only hi weights, no nonzero weights), that the planes never
overlap, and that the two-plane product with the quantized activations equals the float product
of the dequantized weights. `tests/test_fivevalue_model.cpp` (`PARAKEET_TEST_GGUF_PHONON_KEEP`,
`PARAKEET_TEST_GGUF_PHONON_DEQ`) requires the same transcripts from the packed and the dequantized
file on the test clips, and batch equal to per-item.

### Five-value files on a GPU

The two-plane kernels exist for the CPU only. When the compute device is a GPU backend, the loader
does not repack the planes. It expands each of the 264 matrices to dense weights while it uploads the
model: F32 for the two pointwise convolutions and F16 for the other nine linears of a block, the types
the dequantized F16 file uses for the same tensors. The packed bytes are not kept, so the GPU holds the
same weights as the F16 file (1.4 GB, not 242 MB) and runs the same graph and kernels. Every value is
exactly representable in F16 (it is `lo` or `hi` of its row, which are F16 numbers), so the expansion is
lossless. The expansion adds 0.1 s to 0.3 s to the load on the devices tried.

Consequences:

- The GPU result is the result of the dequantized file on that device (the encoder output was bit
  identical, same transcripts and token ids). It is not the CPU packed result: that path quantizes the
  activations to int8, which the GPU path does not, so the two differ by the 8% to 11% relative L2
  difference listed in [phonon.md](phonon.md), with identical token ids on the test clips.
- Packed Redux files are still refused on a GPU. Set `PARAKEET_DEVICE=cpu` to keep running a packed
  file on the CPU when the build has a GPU backend.
- A GPU path that keeps the packed bytes on the device would need a new ggml kernel for the two-plane
  layout. It would save memory only. It is not implemented.

`tests/test_fivevalue_gpu.cpp` (same two environment variables) loads the packed file on the GPU
backend, requires the reference transcript for `speech.wav`, and requires the same transcripts as the
dequantized file on the same device. It exits with the skip code 77 when the compute device is the CPU,
so the CPU suite is unchanged.

## VAD head wiring

Ultra and Redux carry a small voice-activity head (`vad_head.proj`, `vad_head.ctx`, `vad_head.out`) on the
subsampler output: 1x1 conv 1024 to 128, conv k=5 128 to 128, 1x1 conv 128 to 1, sigmoid, one probability per
80 ms frame. The checkpoint does not document the activations or whether `ctx` is residual.

Wiring in use: SiLU after `proj`, SiLU after `ctx`, no residual, then `out` and a sigmoid. This is the wiring
Moondream's public behaviour describes. It is described by behaviour only; no upstream code was copied and no
proprietary kernel was inspected.

History. The first version of this engine had to infer the wiring from the shapes and the probe below, and chose ReLU
after both convs. The probe could not tell ReLU from SiLU (the pre-activations are large, so the two are almost the
same function), which is why the change to SiLU moves the numbers by less than 0.01 in the median and not at all in
the AUC (table below). `VadVariant` (`src/vad_head.hpp`) keeps the other choices as debug options
(`parakeet-cli vad-probe --variant N`: 0 is the default; bit 0 ReLU after proj, bit 1 residual, bit 2 ReLU after ctx;
the old default is variant 5).

How the probe worked. Test data: three clips of 12 LibriSpeech utterances each (the first 36 of
`benchmarks/librispeech_manifest.tsv`, joined with no inserted silence), `jfk.wav`, and the first 120 s of
`i_have_a_dream.wav`. Labels came from the model's own TDT word timestamps: a speech frame (80 ms) has its center inside a
word span widened by 0.04 s; a pause frame lies inside an inter-word, leading or trailing gap of at least 0.4 s, minus a
0.16 s margin at each end. Counts: 4289 speech frames, 479 pause frames, 218 ignored, identical for both models. (An
earlier probe with digital silence and white noise clips was discarded: zero padding wrecks the per-feature mel
normalization of the speech part.) Features were dumped once per clip from the F16 Ultra and the dequantized F16 Redux.

Grid, 108 variants: tap (subsampler output, final encoder output) x input treatment (none, per-frame LayerNorm without
affine, per-frame L2 normalize times sqrt(1024)) x activation after proj (none, ReLU, SiLU) x activation after ctx (none,
ReLU, SiLU) x ctx residual (no, yes). Each is scored by the median p on speech frames, the median p on pause frames and
the ROC AUC of p separating the two.

Top rows (speech median, pause median, AUC). All are the subsampler tap with no input treatment:

| proj act, ctx act, residual | Ultra | Redux | mean AUC |
| --- | --- | --- | --- |
| ReLU, ReLU, no (the first default, variant 5) | 0.999, 0.437, 0.930 | 1.000, 0.197, 0.948 | 0.939 |
| ReLU, SiLU, no | 0.999, 0.437, 0.930 | 1.000, 0.197, 0.948 | 0.939 |
| SiLU, ReLU, no | 0.999, 0.437, 0.930 | 1.000, 0.197, 0.948 | 0.939 |
| SiLU, SiLU, no (default now) | 0.999, 0.437, 0.930 | 1.000, 0.197, 0.948 | 0.939 |
| ReLU, ReLU, yes | pause median 1.000 (collapses) | 1.000, 0.200, 0.959 | |

The SiLU rows tie with ReLU because the pre-activations are huge and saturated. No variant met the strict rule (AUC of at
least 0.90, speech median at least 0.8, pause median at most 0.2, on both models); the SiLU default misses it only on the
Ultra pause median (0.437), like the ReLU variant did.

Refuted. The final encoder output as the tap scores AUC 0.17 to 0.67 for every treatment and activation, so it is at
chance or inverted. LayerNorm and L2 input normalization do not help: the medians collapse to about 0.5 for both classes
with equal or lower AUC.

Decision: subsampler tap, no input treatment, SiLU after proj, SiLU after ctx, no residual (the default of `VadVariant`).
The probe does not pick this wiring; it is chosen because it matches Moondream's public behaviour, and the probe only
confirms that it is consistent with the data (it scores the same as the ReLU variant and the other tap and treatments
stay refuted).

Caveats:
- Pause detection is weak. Pause frames with p below 0.5: 58 percent on Ultra, 93 to 94 percent on Redux (the same for
  gaps of 0.8 s and 1.5 s or more). Speech frames above 0.5: 92.8 percent on Ultra, 92.9 to 94.6 percent on Redux.
- In our probe Ultra needs no residual (with it the pause median is 1.000) and Redux scores slightly better with one (AUC
  0.959 against 0.948). We follow the public description (no residual) for both.
- The wiring comes from Moondream's public behaviour, not from a comparison of outputs with their runtime. We did not run
  that runtime, so the per-frame probabilities have not been compared with it.
- The default threshold is 0.5 for both models, as in Moondream's behaviour. In our probe the best thresholds differ
  (Ultra about 0.5 to 0.7, Redux about 0.3). `--vad-threshold` overrides it.
- Long-form validation is in the sections below. The segmenter falls back to a midpoint or hard cut when it finds no
  pause, so a weak pause response degrades gracefully.

## Segmenter rules

`segment_by_vad` (`src/vad_segmenter.cpp`) turns the per-frame probabilities into segments of at most 30 s. The rules
follow Moondream's public behaviour, written from the description and not from their code.

1. Speech is a frame with p >= 0.5 (`--vad-threshold`, default 0.5 for both models).
2. Smoothing: speech gaps shorter than 0.1 s are filled, then speech runs shorter than 0.1 s are removed. With 80 ms
   frames that fills a one-frame dip and removes a one-frame blip.
3. A pause is a silent run of at least 0.2 s (`--vad-min-pause`).
4. Audio of at most 30 s (`--vad-max-seg`) is not cut and goes through whole, with or without speech.
5. Longer audio is cut from the front. For a segment that starts at s the cut is the midpoint of the last pause that lies
   fully inside [s + 1 s, s + 30 s]. If there is none, it is the midpoint of the last pause whose midpoint lies in that
   range. Otherwise it is a hard cut at s + 30 s. A cut is never closer than 1 s to the start of the segment.
6. A segment without speech is dropped, including a trailing remainder. Long audio with no speech at all gives an empty
   transcript. Kept segments are ordered and disjoint but need not touch.
7. The VAD runs on blocks of 120 s, each with its own mel normalization, and the probabilities are joined on the 80 ms
   grid. A trailing block shorter than 5 s is folded into the previous block. The head's conv sees zero padding at each
   block edge, so the one or two frames at an edge are slightly less reliable.

Behaviour changes against the first version of this segmenter (the one that produced the measurements in the sections
below up to the re-measurement):

| | before | now |
|---|---|---|
| minimum pause | 0.32 s | 0.2 s |
| earliest cut inside a segment | 8 s (fallback), last third first | 1 s |
| which pause | the longest in the last third of the window, else the longest in the window | the last one fully inside, else the last one with its midpoint inside |
| smoothing | none | bridge gaps under 0.1 s, drop runs under 0.1 s |
| segments without speech | kept and decoded | dropped; no speech at all gives an empty transcript |
| VAD input | the whole clip, one mel normalization | 120 s blocks, one mel normalization each |
| VAD head activations | ReLU, ReLU | SiLU, SiLU |

Audio of 30 s or less is unchanged (same code path, transcripts identical). The C-API function
`parakeet_capi_transcribe_path_json_vad` and the CLI options keep their names; only their defaults changed.
`SegmenterOpts` gained `bridge_sec` and `min_speech_sec` (additive; no ABI change).

## Using the VAD on its own

The full reference, with the Silero detector that uses the same entry points, is
in [vad.md](vad.md). This section describes the head.

The head also runs without transcribing, like the VAD of whisper.cpp. Three entry points return the same JSON
(`src/vad_json.cpp`):

```
parakeet-cli vad --model ultra.gguf --input audio.wav [--mode speech|segments] [--probabilities] \
    [--threshold 0.5] [--min-pause 0.2] [--min-speech 0.1] [--max-segment 30] [--threads N]

char* parakeet_capi_vad_pcm_json(parakeet_ctx* ctx, const float* samples, int n_samples,
                                 int sample_rate, const char* options_json);
char* parakeet_capi_vad_path_json(parakeet_ctx* ctx, const char* wav_path, const char* options_json);
```

`options_json` is NULL, "" or a flat object with the keys `threshold`, `min_pause`, `min_speech`, `max_segment`, `mode`
and `probabilities`. The result is freed with `parakeet_capi_free_string`; on error it is NULL and
`parakeet_capi_last_error` has the message (`model has no VAD head` for a model without the head, such as v3).

```
{"mode":"speech","duration":21.370,"frame_sec":0.080,"backend":"cpu",
 "segments":[{"start":2.400,"end":6.560},{"start":7.040,"end":9.200}],
 "probabilities":[0.0123, ...]}          // only with "probabilities":true; one value per 80 ms frame from t = 0
```

Two kinds of segment are available:

- `"speech"` (the default) is the speech regions: the frames with p >= threshold after the smoothing of rule 2, with
  regions separated by a silence shorter than `min_pause` merged. It works for audio of any length, is never capped, and
  never includes silence. Use it to find where speech is, to gate a recorder, or to feed another engine.
- `"segments"` is what `transcribe --vad` decodes (segmenter rules 4 to 6): pieces of at most `max_segment` seconds cut at
  pauses, with no-speech pieces dropped. Audio of at most `max_segment` seconds comes back as one segment `[0, duration]`
  even when it holds no speech. Use it to reproduce or to plan a transcription.

Times are on the original timeline: other sample rates are resampled to 16 kHz and the times stay in seconds of the input.
Backend rules are the same as for `--vad`: the head runs on the context's compute backend (the pool of
`parakeet_capi_set_concurrency` when set), a packed Redux file is CPU only, and the JSON reports the device in `backend`.
The call is safe from several threads, on one context or several. It is additive to the C-API: `parakeet_capi_abi_version`
is unchanged. The head is a pause detector with the limits listed above (AUC 0.93 and 0.95): expect soft edges of a
frame or two (80 to 160 ms), and tune `threshold` for your audio.

## Long-form WER with and without the VAD

`--vad` cuts long audio at pauses found by the VAD head and transcribes each segment on its own (see "Segmenter rules"
above). Without it, the model sees the whole clip in one pass.

The tables in this section and in the TED-LIUM section were measured with the first version of the segmenter and the ReLU
head (threshold 0.5, min pause 0.32 s, max segment 30 s, min segment 8 s). The re-measurement with the rules above is in
"Re-measurement with the Moondream-style segmenter" at the end of the long-form validation.

How it was measured. `scripts/make_longform.py` joins the first 90 utterances of `benchmarks/librispeech_manifest.tsv`
into three clips of 30 utterances each (218 to 354 s each), with a known reference
(the joined manifest texts). Utterances are separated by Gaussian noise at about -55 dBFS (fixed seed) rather than
digital zeros, because digital silence distorts the per-feature mel normalization. Three sets: a 0.45 s gap, a 0.16 s gap
(below the 0.32 s minimum pause on purpose), and no inserted gap (only the natural utterance edges). These are synthetic
long-form clips built from LibriSpeech read speech, not TED-LIUM or other real long recordings.
`scripts/eval_vad_longform.py` runs `parakeet-cli transcribe --decoder tdt --threads 8` under `taskset -c 0-7` with and
without `--vad` and scores with `scripts/asr_metrics.py` `wer` (case and punctuation normalized). CPU pinning and the shared bench lock are applied by the caller through the script's `--prefix` option, not by the
script. The command used:

```
python3 scripts/eval_vad_longform.py --model <gguf> --dir <longform dir> --glob 'longform_0p45_*.wav' \
    --prefix 'flock /tmp/pk-bench.lock taskset -c 0-7'
```

(`--threads 8` is the script default; sweep runs add `--skip-plain --vad-arg=--vad-threshold=0.3` and similar.) The models are the
Ultra F16 GGUF and the packed ternary Redux GGUF (`--ternary keep`, native kernel).

WER per clip, plain single pass vs `--vad` (percent, three clips per set, then the mean):

| Model | Gap | Clip 0 | Clip 1 | Clip 2 | Mean |
|---|---|---:|---:|---:|---:|
| Ultra F16, plain | 0.45 s | 0.47 | 2.39 | 2.29 | 1.71 |
| Ultra F16, `--vad` | 0.45 s | 0.47 | 2.21 | 2.39 | 1.69 |
| Ultra F16, plain | 0.16 s | 0.31 | 2.58 | 2.18 | 1.69 |
| Ultra F16, `--vad` | 0.16 s | 0.62 | 2.21 | 2.50 | 1.78 |
| Ultra F16, plain | none | 0.31 | 2.39 | 2.18 | 1.63 |
| Ultra F16, `--vad` | none | 0.62 | 2.39 | 2.50 | 1.84 |
| Redux packed, plain | 0.45 s | 0.62 | 2.58 | 2.72 | 1.97 |
| Redux packed, `--vad` | 0.45 s | 0.47 | 2.58 | 2.72 | 1.92 |
| Redux packed, plain | 0.16 s | 0.62 | 2.39 | 2.83 | 1.95 |
| Redux packed, `--vad` | 0.16 s | 0.31 | 2.03 | 2.72 | 1.69 |
| Redux packed, plain | none | 0.62 | 2.03 | 2.83 | 1.83 |
| Redux packed, `--vad` | none | 0.62 | 2.21 | 2.29 | 1.71 |

Reading. The change in mean WER from plain to `--vad` runs in both directions and stays within about 0.2 points:
Ultra -0.02, +0.09 and +0.21 (0.45 s, 0.16 s, no gap); Redux -0.05, -0.26 and -0.12. The single largest loss (+0.21, Ultra,
no gap) is marginally over the 0.2-point investigation limit. The Redux gains and the Ultra losses are the same size, so
neither direction is a real effect on three clips per set: VAD segmentation does not change WER meaningfully on these
clips. One clip is 645 to 919 words, so 0.1 point is about one word. Inspecting the worst Ultra clip (no gap, clip 0, 10 segments) shows every cut lands between words, none
inside one, and the differing words are spelling variants (`tail`/`tale`, `honour`/`honor`) and rare names that flip
between the two runs. The segment boundaries there were 20.32, 48.48, 70.00, 98.48, 127.12, 156.56, 179.92, 204.16 and
229.28 s. The result is a wash, not a win: on these clips VAD does not measurably help or hurt accuracy. Its benefit is
bounded memory and time on audio of any length (the 30 s cap), not a lower WER.

Parameter sweep on Ultra F16 with the first segmenter version (mean `--vad` WER, percent, one variable at a time from its defaults):

| Setting | 0.45 s gap | 0.16 s gap | no gap | mean of the three |
|---|---:|---:|---:|---:|
| threshold 0.3 | 1.69 | 1.67 | 1.89 | 1.75 |
| threshold 0.5 (default) | 1.69 | 1.78 | 1.84 | 1.77 |
| threshold 0.7 | 1.69 | 1.83 | 1.89 | 1.80 |
| min pause 0.16 s | 1.69 | 1.78 | 1.84 | 1.77 |
| min pause 0.32 s (default) | 1.69 | 1.78 | 1.84 | 1.77 |
| min pause 0.64 s | 1.95 | 1.79 | 1.85 | 1.86 |

Threshold 0.3 is the only setting ahead of the default. On Redux packed it gives 1.97, 1.65 and 1.63 (mean 1.75)
against 1.92, 1.69 and 1.71 (mean 1.77) for the default. Both gains are 0.02 points, under half a word per clip set, and
Redux gets worse on the 0.45 s gap set, so the defaults stay. A min pause of 0.64 s is worse on the 0.45 s gap set for
Ultra (1.95); probably a longer required pause makes the segmenter fall back to hard cuts more often, which was not measured. A min pause of 0.16 s
changes nothing here.

Known limits:
- A 30 s window with no usable pause makes a hard cut, which can land inside a word.
- Offline only: no streaming with `--vad`.
- The VAD head detects pauses weakly (see the caveats above), so many cuts are hard cuts or land at the model's best guess.
- Ternary kernels are CPU only.
- The figures in this section are from synthetic LibriSpeech clips (218 to 354 s); real talks are in the multilingual and long-form section below.

## Multilingual and long-form validation

Everything above is English (LibriSpeech and synthetic long clips built from it). Ultra is a 25-language model, and the
upstream cards report FLEURS and TED-LIUM long-form results, so this section measures our engine on both. Nothing was
tuned, no `src/` code changed, and every number below comes from the commands listed here. The scoring, the subsets and
the caveats matter for how to read the tables, so read those first.

### Scoring and subset rules

- Scoring uses `normalize` from `scripts/asr_metrics.py` on both reference and hypothesis (NFKC, lowercase, punctuation
  replaced by a space, whitespace collapsed). It is the same for every model. It is NOT the Open ASR Leaderboard
  normalizer (no number spelling, no text-normalizer for other languages), so the absolute numbers here are not
  comparable with the upstream model cards. What matters is the direction and size of the differences between our
  models on the same subset.
- Upstream's own figures, from their pipeline on the full test splits, quoted only as theirs: FLEURS average 11.62 for
  v3, 9.55 for Ultra and 10.56 for Redux; TED-LIUM long-form 2.71 for v3, 1.94 for Ultra and 2.51 for Redux.
- FLEURS: the first 50 utterances of the `test` split of each language in dataset (streaming parquet) order, no
  shuffling and no filtering. The reference is the dataset's `raw_transcription` (natural case and punctuation).
  25 languages, 1250 utterances, 26784 reference words, 14580.6 s of audio. Audio is resampled to 16 kHz mono int16.
  Because each language has about 1000 words, one word is about 0.1 point, so differences under roughly 1 point on a
  single language are noise. WER per language is a corpus WER (total edits over total reference words); the mean is the
  unweighted mean over the 25 languages.
- The config ids all exist as listed: bg_bg, hr_hr, cs_cz, da_dk, nl_nl, en_us, et_ee, fi_fi, fr_fr, de_de, el_gr,
  hu_hu, it_it, lv_lv, lt_lt, mt_mt, pl_pl, pt_br, ro_ro, ru_ru, sk_sk, sl_si, es_419, sv_se, uk_ua.
- TED-LIUM: `distil-whisper/tedlium-long-form`, `test` split, 11 full talks (8905 s in total, 2.5 hours). The reference
  is the `text` column with tags such as `<unk>` removed. One of the 11 is a 5.5 s clip with 24 words
  (`DanBarber_2010_S103`), so it moves a per-talk mean by a lot; means are given with and without it.
- Models: v3 F16 (`tdt-0.6b-v3-f16.gguf`), Ultra F16, Redux packed ternary (`--ternary keep`, native kernel) and
  Redux dequantized F16 (`--ternary dequant`, isolates the int8 activation and packed kernel path). All with
  `--decoder tdt`.

### FLEURS-25, WER percent

| FLEURS config | v3 F16 | Ultra F16 | Redux packed | Redux dequantized F16 |
|---|---:|---:|---:|---:|
| bg_bg | 11.24 | 9.23 | 10.37 | 10.98 |
| cs_cz | 11.93 | 9.61 | 10.62 | 9.71 |
| da_dk | 17.96 | 16.18 | 16.44 | 16.71 |
| de_de | 4.75 | 3.67 | 4.92 | 4.83 |
| el_gr | 36.14 | 33.01 | 32.13 | 32.29 |
| en_us | 5.30 | 4.65 | 6.23 | 6.04 |
| es_419 | 3.10 | 2.38 | 3.42 | 3.65 |
| et_ee | 17.83 | 15.21 | 12.47 | 13.34 |
| fi_fi | 13.06 | 9.77 | 11.84 | 12.21 |
| fr_fr | 4.23 | 4.45 | 10.31 | 10.09 |
| hr_hr | 11.80 | 8.90 | 9.67 | 9.19 |
| hu_hu | 16.18 | 12.69 | 18.48 | 18.08 |
| it_it | 1.88 | 2.11 | 3.54 | 3.69 |
| lt_lt | 22.03 | 18.57 | 20.30 | 19.98 |
| lv_lv | 23.56 | 19.08 | 14.39 | 14.61 |
| mt_mt | 21.20 | 16.59 | 15.46 | 15.29 |
| nl_nl | 7.94 | 7.15 | 10.86 | 10.41 |
| pl_pl | 8.71 | 7.69 | 11.54 | 11.43 |
| pt_br | 4.55 | 3.99 | 5.58 | 5.34 |
| ro_ro | 12.73 | 11.31 | 12.56 | 12.90 |
| ru_ru | 5.68 | 5.11 | 8.14 | 7.85 |
| sk_sk | 9.22 | 7.01 | 9.42 | 9.22 |
| sl_si | 24.12 | 19.00 | 20.81 | 20.17 |
| sv_se | 17.39 | 14.87 | 14.57 | 14.07 |
| uk_ua | 6.81 | 5.43 | 8.09 | 7.55 |
| Mean over 25 languages | 12.77 | 10.71 | 12.09 | 11.99 |

Reading:

- Ultra beats v3 on 23 of 25 languages, mean 10.71 against 12.77 (2.07 points lower on average). It loses slightly on
  fr_fr (4.45 against 4.23) and it_it (2.11 against 1.88), which are within noise at this sample size. The size of the
  gain is close to upstream's (11.62 to 9.55), which is a useful sign that the conversion did not lose anything, but
  the two pipelines differ so this is a direction check, not a reproduction.
- Redux is better than v3 on 13 languages (mostly Baltic, Slavic, Uralic and Greek: bg, cs, da, el, et, fi, hr, lt, lv,
  mt, ro, sl, sv) and worse on the rest; the mean is 12.09 against 12.77. It loses to Ultra on 20 of 25. The largest
  gaps to Ultra are fr_fr (10.31 against 4.45, +5.9), hu_hu (+5.8), pl_pl (+3.9), nl_nl (+3.7), ru_ru (+3.0), uk_ua
  (+2.7) and sk_sk (+2.4). French and Polish match what the upstream card says Redux gives up. Redux is ahead of Ultra
  on el_gr, et_ee, lv_lv, mt_mt and sv_se, with lv_lv (14.39 against 19.08) and et_ee (12.47 against 15.21) the biggest.
  I did not check why; it may be a difference in the training mix.
- Int8 activations. Packed Redux (12.09) against dequantized Redux F16 (11.99) is +0.10 points on the mean, with
  per-language differences from -0.87 to +0.91 and no direction (packed is lower on 9 languages, higher on 16). With
  about 1000 words per language that is noise. On this data the packed ternary path costs no measurable WER against the
  dequantized weights, and the speed benefit is in the speed section above.
- el_gr is high for every model (32 to 36). The FLEURS Greek references and the model outputs likely disagree on
  normalization details (accents, final sigma, number forms); I did not investigate, and it does not change the
  comparison between models on the same references.

### TED-LIUM long-form, WER percent

Plain is one single pass over the whole talk; VAD is `--vad` with the options of the first segmenter version
(threshold 0.5, min pause 0.32 s, max segment 30 s, min segment 8 s; ReLU head). The `--vad` option needs the VAD head, which v3
does not have (`model has no VAD head`), so v3 has only the plain column. The segment count was not recorded.

| Talk (length) | v3 plain | Ultra plain | Ultra VAD | Redux packed plain | Redux packed VAD | Redux deq plain | Redux deq VAD |
|---|---:|---:|---:|---:|---:|---:|---:|
| AimeeMullins (1249 s) | 6.65 | 3.44 | 3.41 | 4.18 | 4.34 | 4.08 | 4.21 |
| BillGates (1506 s) | 7.07 | 5.79 | 5.74 | 6.18 | 6.34 | 6.32 | 6.32 |
| DanBarber (834 s) | 5.75 | 5.46 | 4.90 | 6.47 | 6.79 | 6.59 | 6.83 |
| DanBarber_2010_S103 (5.5 s) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 8.33 | 8.33 |
| DanielKahneman (1096 s) | 3.29 | 3.23 | 3.20 | 4.11 | 3.98 | 4.11 | 4.14 |
| EricMead (459 s) | 5.61 | 4.91 | 4.84 | 5.35 | 5.16 | 5.23 | 5.16 |
| GaryFlake (345 s) | 2.90 | 3.16 | 3.08 | 3.52 | 3.60 | 3.60 | 3.69 |
| JamesCameron (982 s) | 6.34 | 5.52 | 5.42 | 5.81 | 5.62 | 5.62 | 5.55 |
| JaneMcGonigal (1168 s) | 4.21 | 3.60 | 3.68 | 4.06 | 4.09 | 4.06 | 4.16 |
| MichaelSpecter (921 s) | 3.57 | 3.25 | 3.16 | 3.63 | 3.79 | 3.63 | 3.86 |
| RobertGupta (340 s) | 3.04 | 2.71 | 2.59 | 4.17 | 4.51 | 4.17 | 4.62 |
| Mean, all 11 | 4.40 | 3.73 | 3.64 | 4.32 | 4.38 | 5.07 | 5.17 |
| Mean, without the 5.5 s clip | 4.84 | 4.11 | 4.00 | 4.75 | 4.82 | 4.74 | 4.85 |

Reading:

- Ultra beats v3 on 9 of the 11 talks in a single pass (one tie on the 5.5 s clip) and on the mean (3.73 against 4.40; 4.11 against 4.84 without the
  short clip). It loses only on GaryFlake (3.16 against 2.90). Direction agrees with upstream (2.71 against 1.94), but our
  absolute values are about twice upstream's. Part of that gap is the normalizer: the references spell numbers out while
  the model writes digits, and in the Ultra plain hypotheses 218 of about 26800 tokens contain digits (none in the
  references), which alone accounts for at least 0.8 points. I did not run a number-aware normalizer, so the rest of the gap is
  not explained here.
- Redux packed with `--vad`: 4.38 on all 11 (4.82 without the short clip), against Ultra VAD 3.64 (4.00) and v3 plain
  4.40 (4.84). So Redux is about where v3 is on real talks and about 0.8 points behind Ultra, the same order as upstream
  (2.51 against 1.94 against 2.71).
- Packed against dequantized Redux, both with `--vad`: 4.38 against 5.17 on all 11, but 4.82 against 4.85 without the
  5.5 s clip. The all-11 gap is the 24-word clip (0 against 2 errors); on the ten real talks packed and dequantized are the same.
- VAD against plain. Ultra: VAD is better on 9 of 11 talks (one tie), mean 3.64 against 3.73 (4.00 against 4.11), gains of 0.03
  to 0.56 points (the 0.56 is DanBarber), and worse on JaneMcGonigal by 0.08. Redux dequantized: VAD is slightly worse, 5.17 against 5.07
  (4.85 against 4.74). Redux packed: plain 4.32 against VAD 4.38 (4.75 against 4.82 without the short clip), so plain is slightly better, by
  0.07 points. The honest reading is that on real talks VAD segmentation is about neutral: a small gain for Ultra, a small loss for Redux,
  all under about 0.15 points on the mean and inside per-talk noise. Peak RSS (from `/usr/bin/time -v`) with `--vad` is 7 to 11 GB on the talks over 800 s, about the same as
  the single pass, and drops to about 4 GB only on the three shortest talks (340 to 459 s). So with the first segmenter VAD did not buy a memory win (it does now, see the re-measurement below).
- Packed against dequantized Redux, plain single pass, after the long-audio fix (the seven talks that used to crash; packed run
  at `--threads 8` with `/usr/bin/time -v`): WER 4.92 packed against 4.91 dequantized on the mean of the seven (per talk within
  0.2 points, packed lower on 2, higher on 2, equal on 3). Scored with the dequantized transcript as the reference, packed
  differs by 64 of 22922 words (0.28 percent; per talk 0.05 to 0.50 percent). So on real 14 to 25 minute talks the packed
  path matches the dequantized one to a few words per talk, and there is no measurable cost from int8 activations or the
  packed kernel. The packed plain runs all exited 0. Longest talk (BillGates, 1506 s): 82.8 s wall, 11.67 GB peak RSS, load average 20 at the
  start; the other six took 64 to 85 s except JamesCameron (318 s) and MichaelSpecter (294 s), which started at load 26 to 48;
  peak RSS 6.6 to 11.7 GB, similar to the F16 models. Load averages at start were 3.5 to 48, so wall times are indicative only.
- Single-pass memory and time. Every single pass over a 5 to 25 minute talk finished on the F16 models (Ultra, dequantized Redux,
  v3), peak RSS 7 to 13 GB, so the O(T^2) attention above the 8192-frame local-attention threshold did not fail for any of these
  talks. Wall times were recorded but are not reported: the shared machine was heavily loaded during the runs (load average 40 to
  90 from other jobs), so times vary by more than 10x between identical runs.

#### Finding (fixed): packed Redux crashed on single-pass audio longer than about 11 minutes

With `redux-keep.gguf` (packed ternary), plain single pass segfaulted after about 3 s on every talk longer than about
655 s (8192 encoder frames at 80 ms): AimeeMullins, BillGates, DanBarber, DanielKahneman, JamesCameron, JaneMcGonigal and
MichaelSpecter, exit code 139. The backtrace ended in `RelPosAttention::build_graph_local_chunked`.

Cause: above `kLocalThreshold = 8192` frames the encoder switches to the local attention paths. Three of the five
attention `linear` lambdas (`build_graph_batched_local`, `build_graph_local`, `build_graph_local_chunked`) read
only `<base>.weight`, which a packed GGUF does not have, so `ggml_mul_mat` got a null tensor. The earlier synthetic clips
(218 to 354 s) never reach these paths. All five now go through one helper (`attn_linear` in
`src/relpos_attention.cpp`), which uses the packed kernel when `<base>.qweight` exists and throws
`missing encoder weight <name>` when neither form exists.

Verified after the fix on a 714 s clip (`speech.wav` repeated, 2112 words): packed and dequantized Redux both finish and
print byte-identical transcripts. `tests/test_ternary_long.cpp` forces the same paths on short audio
(`PARAKEET_ATT_CONTEXT=64`), for a single item and for a batch. The seven talks that crashed were then re-run
with the fixed build; their packed plain numbers are in the TED-LIUM table above.

### Re-measurement with the Moondream-style segmenter

The same 11 talks and scoring as above, run again after the segmenter and head changes ("Segmenter rules"). "Before" is
the PR head (`cb6208d`, first segmenter, ReLU head) and "now" is this branch, both built from source with the same
flags and run with `parakeet-cli transcribe --decoder tdt --threads 8 --vad`, one after the other for each talk. Models:
Ultra converted to Q8_0 (`--dtype q8_0`) and Redux packed (`--ternary keep`). Audio is streamed from the Hub and deleted
after the runs.

Load. The machine was shared and busy: the load average was 23 to 69 at the start of the runs. These runs were not
measured on a quiet machine, so wall times are not reported; WER and peak memory do not depend on load.

WER, percent, `--vad`:

| Talk | Ultra before | Ultra now | Redux before | Redux now |
|---|---:|---:|---:|---:|
| AimeeMullins | 3.41 | 3.74 | 4.34 | 4.38 |
| BillGates | 5.83 | 5.70 | 6.34 | 6.18 |
| DanBarber | 4.90 | 5.18 | 6.79 | 6.27 |
| DanBarber_2010_S103 | 0.00 | 0.00 | 0.00 | 0.00 |
| DanielKahneman | 3.13 | 2.94 | 3.98 | 4.04 |
| EricMead | 4.84 | 4.84 | 5.16 | 5.35 |
| GaryFlake | 3.16 | 3.16 | 3.60 | 3.60 |
| JamesCameron | 5.32 | 5.19 | 5.62 | 5.58 |
| JaneMcGonigal | 3.68 | 3.70 | 4.09 | 4.14 |
| MichaelSpecter | 3.16 | 2.94 | 3.79 | 3.35 |
| RobertGupta | 2.59 | 2.37 | 4.51 | 5.07 |
| Mean, all 11 | 3.64 | 3.61 | 4.38 | 4.36 |
| Mean, without the 5.5 s clip | 4.00 | 3.98 | 4.82 | 4.80 |

Plain single pass for reference. The plain path did not change (transcripts are byte-identical between the two builds
on the fixtures and on 8 clips of up to 180 s for three models), so plain numbers do not depend on the build. The earlier
table has Ultra F16 plain at 3.73 (all 11) and 4.11 (without the short clip), and Redux packed plain at 4.32 and 4.75.
The Redux packed plain runs repeated here on 9 of the 11 talks and gave the same WER as the earlier table on every one.
Ultra Q8_0 plain was run on two talks only (AimeeMullins 3.54, RobertGupta 2.71; Ultra F16 earlier: 3.44, 2.71); the
rest was stopped because the machine was overloaded.

Peak resident memory with `--vad` (GB, before / now, from `/usr/bin/time -v`):

| Talk | Ultra Q8_0 | Redux packed |
|---|---|---|
| AimeeMullins | 9.53 / 2.01 | 9.05 / 1.47 |
| BillGates | 11.30 / 2.05 | 10.83 / 1.51 |
| DanBarber | 6.67 / 1.93 | 6.17 / 1.39 |
| DanBarber_2010_S103 | 0.99 / 0.99 | 0.44 / 0.44 |
| DanielKahneman | 8.48 / 1.97 | 7.98 / 1.44 |
| EricMead | 4.10 / 1.86 | 3.58 / 1.32 |
| GaryFlake | 3.33 / 1.83 | 2.80 / 1.30 |
| JamesCameron | 7.69 / 1.95 | 7.20 / 1.42 |
| JaneMcGonigal | 8.97 / 1.99 | 8.48 / 1.45 |
| MichaelSpecter | 7.27 / 1.94 | 6.77 / 1.40 |
| RobertGupta | 3.29 / 1.83 | 2.76 / 1.29 |

Reading.
- WER. The mean moves by 0.02 to 0.03 points (Ultra 3.64 to 3.61, Redux 4.38 to 4.36 on all 11; 4.00 to 3.98 and 4.82 to
  4.80 without the short clip). Per talk the change is between -0.52 and +0.56 points, in both directions (Redux RobertGupta +0.56, DanBarber
  -0.52, MichaelSpecter -0.44; Ultra AimeeMullins +0.33, DanBarber +0.28), the same size as the noise between runs that
  differ only in where the cuts fall. Against plain, `--vad` now is better than plain for Ultra by 0.12 points (3.61
  against 3.73 from the earlier F16 table) and worse for Redux by 0.04 (4.36 against 4.32). The conclusion of the
  earlier sections stands: segmentation does not change accuracy in any direction we can resolve with 11 talks.
- Memory. This changes a lot. With `--vad` the peak RSS is now 1.3 to 2.1 GB on every talk of 340 s or more, against 2.8
  to 11.3 GB before. The old VAD ran the subsampler over the whole clip in one pass; the 120 s blocks bound that, and
  segments are at most 30 s. The earlier statement that `--vad` buys no memory win applied to the first version only.
  The 5.5 s clip is unchanged (it takes the plain path).
- The cuts differ from the first version (earlier threshold of 0.32 s and 8 s minimum, now 0.2 s and 1 s), so more and
  shorter segments are expected; the segment count was not recorded.

Pause detection with `vad-probe`, on the same 5 clips as the wiring probe (three 12-utterance LibriSpeech clips, `jfk.wav`,
first 120 s of `i_have_a_dream.wav`), same labels, F16-equivalent models Ultra Q8_0 and Redux packed. Frames with
speech labels: 4289 (Ultra) and 4065 (Redux); pause labels: 477 and 564 (labels come from each model's own TDT
timestamps, so the counts differ a little between models).

| Model | Head | Speech frames p >= 0.5 | Pause frames p < 0.5 | Median p, speech | Median p, pause | AUC |
|---|---|---:|---:|---:|---:|---:|
| Ultra Q8_0 | ReLU (before) | 92.8% | 58.1% | 0.999 | 0.436 | 0.931 |
| Ultra Q8_0 | SiLU (now) | 92.8% | 58.1% | 0.999 | 0.434 | 0.931 |
| Redux packed | ReLU (before) | 94.6% | 92.9% | 1.000 | 0.187 | 0.959 |
| Redux packed | SiLU (now) | 94.6% | 92.9% | 1.000 | 0.187 | 0.959 |

The head change does not move pause detection (as expected from the saturated pre-activations), and the old default is
still available as `vad-probe --variant 5`. Ultra still separates pauses weakly (58% of pause frames below 0.5).

Timing check, not a result. Five interleaved runs of `--vad` on a 117 s clip (Ultra Q8_0, 4 threads), before / now, wall
seconds with the load average at the start: 19.35 (64.5) / 17.51 (61.2), 18.48 (53.9) / 13.28 (51.0), 16.78 (50.1) / 10.78
(49.5), 13.61 (48.0) / 11.04 (52.1), 11.03 (51.8) / 11.02 (47.0). This was not measured on a quiet machine (load above 45
throughout), so only the direction is usable: the new build is not slower.

### Parity with the transformers reference (Ultra)

An independent check that the Ultra conversion and engine reproduce the HF implementation. `ParakeetForTDT` is not in the
installed transformers 5.3.0, so this used a source checkout that has it (version string 5.10.0.dev0) through
`PYTHONPATH`. The HF checkpoint has no preprocessor or tokenizer files, so `scripts/hf_reference_transcribe.py` builds a
`ParakeetFeatureExtractor(feature_size=128, sampling_rate=16000)` (per-feature normalization is built in) and decodes
token ids with the piece table from our Ultra GGUF. The run is fp32 on CPU with greedy TDT decoding; ours is the F16 GGUF.
Because the token table comes from our GGUF, this checks the encoder, decoder and search, not the tokenizer.

Same first 20 test utterances of en_us, de_de and fr_fr (60 utterances, 1509 words). Ours scored against the HF
transcript as reference, after `normalize`, and after dropping the `<unk>` piece, which the HF side prints and we omit:

| Language | Identical after normalize | WER of ours vs HF |
|---|---:|---:|
| en_us | 20 of 20 | 0.00 |
| de_de | 18 of 20 | 0.45 |
| fr_fr | 18 of 20 | 0.48 |
| total | 56 of 60 | 0.33 |

The four differing utterances are single-word choices near a tie (for example `Hirnschadens` against `Höhenschadens`,
`Laka` against `Lakas`, `vient` against `viant`, and `dix-sept` against `17`). The F16 weights in the GGUF against the fp32 weights in HF are a plausible cause, but I did not test that, for example by converting an
F32 GGUF.

Both implementations scored against the FLEURS references (same scorer as above, `<unk>` left in the HF text, where the
`unk` word counts as an error):

| Language | Ours (F16 GGUF) | HF transformers (fp32) |
|---|---:|---:|
| en_us | 3.64 | 3.64 |
| de_de | 3.56 | 3.78 |
| fr_fr | 5.13 | 6.09 |
| mean of the three | 4.11 | 4.50 |

The fr_fr gap on the HF side comes from `<unk>` pieces that the HF text prints (four in the 20 utterances) and we drop; it is a
decoding detail, not a model difference.

### Commands

```
# subsets (stream from the Hub, no token; only 16 kHz wavs, references and manifests are written, under 1 GB in total)
python3 scripts/fetch_fleurs_subset.py --out /tmp/val/fleurs --n 50
python3 scripts/fetch_tedlium_longform.py --out /tmp/val/ted

# FLEURS WER for the four models (bench --json keeps the hypotheses per language)
python3 scripts/eval_manifest_wer.py --out /tmp/val/fleurs_res --threads 8 \
    --model v3=<tdt-0.6b-v3-f16.gguf> --model ultra=<ultra-f16.gguf> \
    --model redux=<redux-keep.gguf> --model reduxdeq=<redux-deq.gguf> /tmp/val/fleurs/*/manifest.tsv

# TED-LIUM plain and --vad, one command per model (records WER, wall time, peak RSS; a crash is recorded as FAIL)
python3 scripts/eval_longform_talks.py --model <gguf> --dir /tmp/val/ted --save /tmp/val/ted_res/<name>

# HF transformers reference for Ultra, then compare
PYTHONPATH=<transformers checkout>/src python3 scripts/hf_reference_transcribe.py \
    --hf-dir <hf ultra dir> --gguf <ultra-f16.gguf> --out /tmp/val/spike_hf /tmp/val/spike/{en_us,de_de,fr_fr}/manifest.tsv
python3 scripts/compare_hyps.py /tmp/val/spike_res/ultra /tmp/val/spike_hf
```

The FLEURS run used `--threads 2` for most of the sweep instead of 8, because the machine was heavily loaded and 8 spinning
threads were much slower than 2 there; greedy decoding does not depend on the thread count except for float summation
order, which was not checked for any change in a transcript.

Limits of this section:
- 50 utterances per language is a sample, not the full FLEURS test split; per-language numbers carry about a 1 point noise band.
- The normalizer is a plain one, so numbers, hyphenation and non-Latin scripts add errors that the leaderboard normalizer
  would remove; compare models against each other, not against upstream.
- The TED-LIUM set has 11 talks, one of them 5.5 s long; a per-talk difference under 0.2 points is not meaningful.
- Timing is not reported (loaded machine). The speed numbers are in the earlier speed section.

## Tests

```
ctest --test-dir build -R ternary --output-on-failure        # test_ternary (kernels vs scalar), no model needed

PARAKEET_TEST_GGUF_REDUX_KEEP=<redux --ternary keep gguf> \
PARAKEET_TEST_GGUF_REDUX_DEQ=<redux --ternary dequant gguf> \
    ctest --test-dir build -R test_ternary_model --output-on-failure

build/tests/bench_ternary [N K T reps]                        # single thread: each kernel, the quantizer, ggml Q8_0 and F16 mul_mat

ctest --test-dir build -R test_fivevalue$ --output-on-failure  # five-value split, no model needed

PARAKEET_TEST_GGUF_PHONON_KEEP=<phonon --five-value keep gguf> \
PARAKEET_TEST_GGUF_PHONON_DEQ=<phonon dequantized f16 gguf> \
    ctest --test-dir build -R 'test_fivevalue_(model|gpu)' --output-on-failure   # _gpu skips on a CPU build
```
