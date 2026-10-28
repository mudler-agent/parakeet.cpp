# Speaker diarization

parakeet.cpp runs [nvidia/Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization),
a Sortformer model that answers "who spoke when" for up to 8 speakers, and
combines it with any Parakeet ASR model for speaker-attributed transcripts
("who said what").

## Model

- Encoder: FeatureStacking (8 mel frames stacked, 80 ms per step) and a
  31-layer pre-norm Transformer with RoPE attention (d_model 512, 8 heads).
- Head: projection to 192, a subpixel Conv1d that upsamples 8x back to 10 ms
  frames, two linear layers and a sigmoid per speaker.
- Output: per-speaker activity probabilities every 10 ms. Segments come from
  thresholding at 0.5.
- Speakers are numbered in order of first appearance.

## Converting

```
.venv/bin/python scripts/convert_parakeet_to_gguf.py \
    --model nvidia/Nemotron-3-Diarization --dtype q8_0 \
    --output models/nemotron-3-diarization.q8_0.gguf
```

`--model` also takes a local `.nemo`. The converter reads the checkpoint
directly, so it works with a NeMo that cannot instantiate the model (NeMo 3.0
has no RoPE Transformer encoder). Sizes: F32 397 MB, F16 201 MB, Q8_0 109 MB.

## Offline and streaming inference

NeMo's `diarize()` for this checkpoint runs cache-aware streaming inference
(`streaming_mode: true` in the model config), and `parakeet_capi_diarize_*`
does the same: audio is processed in 264-step chunks (21.12 s) with a
speaker cache of 264 steps that keeps speaker identities stable across chunks.
This is also what keeps long recordings correct: the model is trained on
sessions of about 105 s, and attending over a whole long clip at once is
outside that range. On a 12 minute, 3 speaker recording the offline path
(which matches NeMo offline on short clips) agrees with NeMo's `diarize()` on
22% of speech frames and puts almost everything on one speaker; the streaming
path agrees on 100%.

`DiarizationModel::run_offline` / `speaker_probs` still implement the offline
path (NeMo peak-normalizes the waveform there) for short clips and for parity
checks.

## Live streaming and latency modes

The streaming mel is frame-local and does not normalize per feature, so the live
API needs a diarization GGUF with `parakeet.preprocessor.normalize` set to `NA`.
The shipped checkpoint (`nvidia/Nemotron-3-Diarization`) has `normalize: NA` in its
`model_config.yaml`. A GGUF that says `per_feature` is refused by
`parakeet_capi_diarize_stream_begin*` with an error, not run with the wrong features.

The live streaming API (`parakeet_capi_diarize_stream_*`) takes 16 kHz PCM in
pieces of any size and computes the log-mel incrementally (bit-identical to
the whole-clip mel). Segments that continue past a chunk boundary are not
split.

The checkpoint's own configuration processes 21.12 s chunks, too slow for live
labels. The same checkpoint also supports the model card's low-latency
configurations, selected with `parakeet_capi_diarize_stream_begin_latency`:

| Mode | Input latency | Chunk | Look-ahead | FIFO | Cache | Update period |
|---|---|---|---|---|---|---|
| `PARAKEET_DIAR_LATENCY_MODEL` | 21.12 s | 264 | 0 | 0 | 264 | 264 |
| `PARAKEET_DIAR_LATENCY_LOW` | 1.04 s | 9 | 4 | 264 | 264 | 222 |
| `PARAKEET_DIAR_LATENCY_VERY_LOW` | 0.64 s | 6 | 2 | 264 | 264 | 222 |
| `PARAKEET_DIAR_LATENCY_ULTRA_LOW` | 0.32 s | 3 | 1 | 264 | 264 | 222 |

Sizes are 80 ms encoder frames; input latency is (chunk + look-ahead) x 80 ms.
Each chunk is encoded together with its look-ahead, as NeMo's
`streaming_feat_loader` does, and only the chunk itself is output and enters
the FIFO and speaker cache. `parakeet_capi_diarize_stream_active` returns the
segments still open ("who is speaking now") and
`parakeet_capi_diarize_stream_time` how much audio has been diarized.

## Parity with NeMo

Measured against NeMo main (the reference must support the RoPE encoder), with
`scripts/gen_diar_baseline.py`:

| Clip | Speakers | Offline prob max diff | Offline segments | Streaming segments |
|---|---|---|---|---|
| `tests/fixtures/two_speakers.wav`, 23.6 s | 2 | 0.004 | 5 / 5 identical | 5 / 5 identical |
| synthetic two-voice dialogue (VibeVoice sample), 68.5 s | 2 | 0.004 | 26 / 26 identical | 26 / 26 identical |
| synthetic three-voice dialogue (VibeVoice sample), 12.3 min | 3 | | | 100% frame agreement |

Every latency mode matches NeMo in the same mode: on the 23.6 s fixture all
four modes give NeMo's segments exactly (5, 5, 6 and 7 segments), and on the
68.5 s clip the 1.04 s mode, which fills the FIFO and compresses the speaker
cache, gives all 26 segments; probability max diff 0.01 or less throughout.

Segment boundaries match to the 10 ms frame. F16 gives the same results. Q8_0
keeps the same segments on the fixture and the 68.5 s clip; its probabilities
move up to about 0.06 in the checkpoint mode and 0.12 in the low-latency modes
there, where quantization noise changes which frames the cache keeps. Where a
probability sits right at the 0.5 threshold, Q8_0 can also flip a frame or two:
on a 31.5 s two-speaker clip that splits one segment into three short pieces
(99.5% frame agreement). Use F16 when segment-exact output matters.

After the speaker cache compresses, NeMo picks cache frames with
`torch.topk`, whose order for tied scores is arbitrary. parakeet.cpp breaks ties
toward the earlier frame, so streaming probabilities can drift by up to about
0.02 on long clips while the segments stay the same.

## Speaker-attributed ASR

`parakeet_capi_transcribe_and_diarize(_json)` runs an ASR context and a
diarization context on the same PCM and assigns every ASR word to the speaker
whose segments overlap it most. A word that overlaps no segment takes the
nearest segment's speaker if that segment is within 0.5 s, otherwise -1.
Consecutive words from one speaker (gaps up to 0.5 s) form an utterance.

`parakeet_capi_sas_stream_*` does the same live (`_begin_latency` takes a
latency mode): once at least 4 s of diarized audio is uncommitted it is
transcribed, and words that end at least 1 s before the diarized edge are
committed with their speakers. Transcription resumes right after the last
committed word, so no word is cut at the edge and audio the ASR skipped is
heard again with more context. Offline ASR on shorter windows loses words, so
committed text lags by about 3 to 5 s whatever the diarization latency.

## Speed

Streaming modes on the 12.3 min, 3 speaker clip (model load included; agreement
is with NeMo's default `diarize()`, so it shows what the lower latency costs):

| Mode | F16 | x real time | Agreement | Q8_0 | Agreement |
|---|---|---|---|---|---|
| checkpoint (21.12 s) | 5.7 s | 130x | 100% | 6.0 s | 99.9% |
| low (1.04 s) | 107 s | 6.9x | 99.4% | 115 s | 99.4% |
| very low (0.64 s) | 166 s | 4.4x | 99.1% | 172 s | 99.2% |
| ultra low (0.32 s) | 295 s | 2.5x | 98.3% | 332 s | 98.4% |

A low-latency chunk costs about 0.1 s on this CPU, so labels arrive about
0.1 s after the input latency.

Whole-file diarization:

End to end with the `diarize` example (model load included), AMD Ryzen 9
9950X3D, CPU:

| Audio | F32 | Q8_0 |
|---|---|---|
| 23.6 s | 0.43 s | 0.25 s |
| 68.5 s | 0.80 s | 0.60 s |
| 12.3 min | 6.8 s | 6.0 s |

Streaming cost grows linearly with length (each chunk attends over at most
528 steps), so long recordings run at about 110x real time.

## Tests

```
PARAKEET_TEST_DIAR_GGUF=diar.gguf PARAKEET_TEST_BASELINE_DIAR=diar_baseline.gguf \
PARAKEET_TEST_GGUF=asr.gguf ctest --test-dir build -R "diar|sas|combined"
```

- `test_diarization_accuracy`: offline probabilities and segments, and the
  default `diarize_pcm` segments, against NeMo.
- `test_streaming_diarization`: streaming probabilities and segments against
  NeMo streaming, from a whole-clip mel and from live 100 ms PCM pieces.
- `test_combined_offline`: speaker-attributed ASR and the streaming C-API on
  the two-speaker fixture.
- `test_sas_merge`: word to speaker assignment (no model needed).

Set `PARAKEET_TEST_DIAR_PROB_TOL=0.15` for Q8_0 (the segment checks stay
exact). The baseline includes every latency mode by default; NeMo runs the
low-latency modes slowly on CPU, so `--modes low` limits a long clip to one.

### Opt-in speaker profiles (C API)

Use the additive entry point below to offer **preview, then Name and remember**
without recording another voice sample. Existing plain/named diarization APIs
and their JSON defaults do not export profiles or embeddings. Feature-detect
these symbols when loading an older library; no existing signature changes.

```c
char* parakeet_capi_diarize_profiles_pcm_json(
    parakeet_ctx* diar, parakeet_ctx* speaker,
    parakeet_speaker_registry* reg,
    const float* samples, int n_samples, int sample_rate,
    float accept_threshold, float margin);
const char* parakeet_capi_speaker_identity(const parakeet_ctx* speaker);
```

The input is mono float PCM; positive rates are resampled to 16 kHz. Threshold
and margin semantics are the same as `parakeet_capi_diarize_named_pcm_json`
(zero selects defaults). Both a loaded diarization context and a loaded speaker
encoder are required. A NULL registry means an empty registry **only for the
new profile API**. An empty registry still produces profiles. The registry is
read-only and must not be mutated concurrently. This call never enrolls voices.
Free the returned JSON with `parakeet_capi_free_string`. On failure it returns
NULL; inspect `parakeet_capi_last_error` on both contexts, as with the named API.

The existing `speakers`, `segments`, and `names` fields are retained. An
additional top-level object has this schema (dimension 2 is illustrative):

```json
{
  "speaker_profiles": {
    "version": 1,
    "encoder": {"identity": "sha256:<64 lowercase hex digits>", "dimension": 2},
    "speakers": [
      {"speaker": 0, "clean_duration": 3.0,
       "intervals": [{"start": 0.0, "end": 3.0}],
       "unavailable_reason": null, "embedding": [0.6, 0.8]},
      {"speaker": 1, "clean_duration": 1.0,
       "intervals": [{"start": 4.0, "end": 5.0}],
       "unavailable_reason": "insufficient_clean_speech"}
    ]
  }
}
```

There is one profile for each discovered speaker slot, sorted by integer slot,
including known speakers. There are no per-segment embeddings. No discovered
speakers yields an empty profile array. Intervals are in seconds in the original
recording, clipped to available audio and excluding overlap with other speakers.
Clean pieces shorter than 0.2 seconds are discarded. The newest 30 seconds of
clean audio per speaker are retained; `clean_duration` and `intervals` describe
exactly that retained audio, not the total diarized speech. Preview these spans
from the original recording, not a separated or synthesized voice.

At least 2 seconds of retained clean speech is required. Each usable speaker
gets one encoder call and one finite, nonzero, L2-normalized embedding of the
loaded encoder's dimension. Unusable profiles omit `embedding` and report one
of `insufficient_clean_speech`, `embedding_failed`, or `invalid_embedding`.
A whole-call/model error remains a NULL result rather than an unavailable profile.
Clean duration is not a calibrated confidence or a guarantee of voice quality.

#### Trusted compatibility and enrollment

`parakeet_capi_speaker_identity` returns a borrowed string valid until context
free, or NULL for NULL/non-speaker contexts. It is SHA-256 of the **entire GGUF
file**, including metadata and quantized weights, independent of filename. Get
the dimension using `parakeet_capi_speaker_dim`. Compare profile identity and
dimension against these values from the server's configured, loaded encoder;
never use a client-selected model tag or dimension as the authority. Renaming a
file preserves identity; converting, quantizing, or editing it changes identity.

Loading hashes the file before and after encoder loading and rejects changed
bytes or read errors. This adds two sequential file reads per speaker load, not
per request. Deploy model files read-only and replace them only between context
lifetimes. The encoder loads by pathname: the checks detect ordinary updates,
not an adversary who replaces and restores a file during loading. They cannot
protect a memory-mapped model from later writes either.

An application must gate profile export and enrollment with its recognition
permissions. Only after explicit user confirmation should it validate version,
unavailable status, finite/nonzero vector, trusted identity and dimension, then
call the registration path. Pass the trusted encoder family and identity with
the vector (`parakeet_capi_speaker_registry_add_embedding_fp`, see "Encoder
fingerprint" below), so the registry records which encoder made it and every
named call checks it. The plain `add_embedding` still works and records
nothing. Profiles are not signed and are not proof of identity. Treat exported
voice vectors as sensitive data.

The native registry is **name-keyed and aggregating**:
`parakeet_capi_speaker_registry_add_embedding` calls `SpeakerRegistry::enroll`,
which combines repeated enrollments under the same name into one centroid.
Passing the same display name twice therefore merges the native entries; profile
export does not change this behavior.

An application such as LocalAI must implement duplicate-display-name registration
separately: use distinct registration IDs, keep display names separate, and store
one embedding per registration without merging or updating existing samples.
If replaying entries into the native registry, use unique registration IDs as
native keys, not display names, and map matches back to display names in the
application. This is a downstream application responsibility, not functionality
implemented by this backend's profile export. Relabel only after registration
succeeds. Profile export adds no persistence or automatic enrollment; it does not
change an application's global, in-memory registry lifecycle.

### Encoder fingerprint

Equal embedding sizes do not mean the same embedding space: ECAPA and CAM++
both give 192 values, and a registry enrolled with one names the wrong people
when the other is used. A registry therefore records which encoder made its
voices, and the encoder in use is checked against it before any name is
assigned. Two strings make the fingerprint:

- **Family**: `voicedetect:<voicedetect.arch>:<general.name>:<voicedetect.embedding_dim>`,
  read from the encoder GGUF metadata, for example
  `voicedetect:ecapa_tdnn:speechbrain/spkrec-ecapa-voxceleb:192`. It names the
  embedding space. `parakeet_capi_speaker_encoder_family` returns it.
- **Weights**: `sha256:<64 hex>` of the exact bytes of the encoder GGUF file,
  the same string as `parakeet_capi_speaker_identity`. A GGUF has no recorded
  source hash, so the file bytes are the definition. Another quantization of the
  same encoder has the same family and another weights hash. (A `voice`
  component of a bundle reports `sha256:` plus the `source_sha256` of its header,
  the hash of the single-model GGUF it came from, so the two agree when the
  component is an unchanged copy of that file.)

What the check does, the same everywhere (`parakeet-cli scene`,
`parakeet_capi_speaker_identify_pcm_json`, `parakeet_capi_diarize_named_pcm_json`,
`parakeet_capi_diarize_profiles_pcm_json`,
`parakeet_capi_transcribe_and_diarize_named_json`,
`parakeet_capi_scene_stream_begin_speaker`), with the same message text:

| Registry vs encoder | Result |
|---|---|
| Other embedding size | Error: `registry holds N-value embeddings, this model produces M`. |
| Other family | Error naming both families. No name is assigned. |
| Same family, other weights | Warning only: logged to stderr, kept in `parakeet_capi_speaker_last_warning`. Names are assigned. |
| No fingerprint (a version 1 file, or embeddings added without one) | Accepted with a warning: the encoder is unverified. |
| No fingerprint and strict mode (`parakeet_capi_speaker_registry_set_strict`, `parakeet-cli scene --strict-registry`) | Error. |
| Empty registry | Nothing to check. |

An error is reported like any other failure of that call: NULL (or nonzero)
with the message on the speaker context, and a non-zero exit in the CLI.

Enrolment records the fingerprint of the encoder that computed the voice:
`parakeet-cli enroll`, `parakeet_capi_speaker_enroll` and
`SpeakerIdentifier` enrolment do it themselves. An empty registry takes the
fingerprint of its first voice. Enrolling with another family is refused. A
registry that has voices but no fingerprint refuses a fingerprinted voice, and a
fingerprinted registry refuses a voice with none: it is never stamped
silently, because nothing can verify what made the old voices. A caller that
builds registries from stored embeddings passes the family and identity it
stored with them to `parakeet_capi_speaker_registry_add_embedding_fp`.

To stamp a registry that has no fingerprint, say which encoder made it:

```
parakeet-cli registry reg.bin                                   # show the file
parakeet-cli registry reg.bin --restamp --encoder speaker.gguf  # stamp a version 1 file
```

`registry` prints the format version, the embedding size, the family, the
weights hash and the speaker names. `--restamp` only works on a registry with no
fingerprint, checks the embedding size against the encoder, and trusts you for
the rest: it cannot tell ECAPA from CAM++ when both give 192 values.

#### Registry file format

Little-endian. `PKSR` magic, then:

```
version 1 (no fingerprint; also what a registry without one is saved as)
  offset  size  field
  0       4     "PKSR"
  4       4     u32 version = 1
  8       4     i32 dim
  12      4     u32 n, the number of speakers
  16      ...   n speaker records

version 2 (with a fingerprint)
  0       4     "PKSR"
  4       4     u32 version = 2
  8       4     i32 dim
  12      4     u32 n
  16      4     u32 family length (at most 4096)
  20      ...   family bytes (UTF-8, not terminated)
  ...     4     u32 weights length (at most 4096)
  ...     ...   weights bytes
  ...     ...   n speaker records

speaker record (both versions)
  4     u32 name length (1 to 4096)
  ...   name bytes
  4     i32 count of enrolled clips (at least 1)
  4*dim f32 sum of the L2-normalized embeddings
```

A version 2 file has at least one non-empty fingerprint string. Readers refuse
an unknown version, a truncated file, trailing bytes and implausible lengths. A
reader from before this change only knows version 1, so it refuses a version 2
file with "unsupported version" instead of reading it wrong. A version 1 file
loads unchanged.

The C-API additions are `parakeet_capi_speaker_registry_add_embedding_fp`,
`parakeet_capi_speaker_registry_encoder_family`, `..._encoder_weights`,
`parakeet_capi_speaker_registry_set_strict`, `parakeet_capi_speaker_encoder_family`
and `parakeet_capi_speaker_last_warning`. They are additive; the ABI version
stays 10 and no signature changed.
