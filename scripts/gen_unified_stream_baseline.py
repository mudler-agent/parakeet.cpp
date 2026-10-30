#!/usr/bin/env python3
"""NeMo buffered-streaming reference for a unified checkpoint (nemotron-labs-unified).

Runs the loop of NeMo's example `speech_to_text_streaming_infer_rnnt.py`
(examples/asr/asr_chunked_inference/rnnt) for ONE utterance, batch 1, CPU,
float32, and writes what the C++ BufferedStreamingSession must reproduce:

  * per step: the buffer's (left, chunk, right) sample counts after the add,
    the window length, the encoder length, the number of frames decoded, and
    the tokens that step emitted;
  * the final token ids and text (NeMo's own BatchedHyps merge);
  * optionally the encoder slice that goes into the decoder (after the prompt
    kernel), per step, as time-major f32 [n_frames, d_model].

It imports NeMo's ContextSize / StreamingBatchedAudioBuffer and the model's
decoding_computer rather than copying them, so the arithmetic is NeMo's.

The presets are given as frames of 80 ms (left, chunk, right), as in the
checkpoint's model card; NeMo's example turns seconds into frames with
int(seconds * 100 / 8). Pass --secs to use seconds instead, which is what the
example does.

Usage (dedicated NeMo-main venv, pinned commit cd5e4fbd36ba):
  python scripts/gen_unified_stream_baseline.py \
      --model /path/nemotron-labs-unified-asr-0.6b.nemo \
      --audio tests/fixtures/speech.wav --lang en \
      --preset 70,7,7 --preset 70,2,5 --preset 70,1,1 \
      --output /path/out/unified-stream-en-speech.gguf
"""
import argparse
import sys

import numpy as np
import torch


def parse_preset(s):
    parts = [int(x) for x in s.split(",")]
    if len(parts) != 3 or parts[1] < 1 or min(parts) < 0:
        raise argparse.ArgumentTypeError(f"preset must be L,C,R frames with C >= 1: {s}")
    return tuple(parts)


def load_wav_16k_mono(path):
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        raise SystemExit(f"{path}: expected 16 kHz, got {sr}")
    return np.ascontiguousarray(audio, dtype=np.float32)


@torch.inference_mode()
def run_preset(model, audio, lang_index, preset, dump_enc):
    """NeMo's streaming loop for one preset. Returns a dict of arrays and strings."""
    from nemo.collections.asr.parts.utils.rnnt_utils import batched_hyps_to_hypotheses
    from nemo.collections.asr.parts.utils.streaming_utils import ContextSize, StreamingBatchedAudioBuffer

    L, C, R = preset
    subsampling = model.encoder.subsampling_factor          # 8
    frame_samples = 160 * subsampling                       # 1280: one 80 ms encoder frame
    ctx_frames = ContextSize(left=L, chunk=C, right=R)
    ctx_samples = ContextSize(
        left=L * frame_samples, chunk=C * frame_samples, right=R * frame_samples
    )
    # The example sets the default att context to the frame triple (chunked_limited_with_rc).
    model.encoder.set_default_att_context_size(att_context_size=[L, C, R])
    decoding_computer = model.decoding.decoding.decoding_computer

    S = int(audio.shape[0])
    audio_t = torch.from_numpy(audio).unsqueeze(0)          # [1, S]
    rest = torch.tensor([S], dtype=torch.long)
    prompt = torch.tensor([lang_index], dtype=torch.long)

    left_sample = 0
    right_sample = min(ctx_samples.chunk + ctx_samples.right, S)
    buffer = StreamingBatchedAudioBuffer(
        batch_size=1, context_samples=ctx_samples, dtype=audio_t.dtype, device="cpu"
    )
    state = None
    total_hyps = None
    steps = []
    enc_slices = []
    final_text = None
    while left_sample < S:
        chunk_length = min(right_sample, S) - left_sample
        is_last_chunk_batch = torch.tensor([chunk_length >= int(rest[0])])
        is_last_chunk = right_sample >= S
        chunk_lengths_batch = torch.where(
            is_last_chunk_batch, rest, torch.full_like(rest, fill_value=chunk_length)
        )
        buffer.add_audio_batch_(
            audio_t[:, left_sample:right_sample],
            audio_lengths=chunk_lengths_batch,
            is_last_chunk=is_last_chunk,
            is_last_chunk_batch=is_last_chunk_batch,
        )
        win_samples = int(buffer.context_size_batch.total()[0])
        encoder_output, encoder_output_len = model(
            input_signal=buffer.samples,
            input_signal_length=buffer.context_size_batch.total(),
            language_prompt_indices=prompt,
        )
        encoder_output = encoder_output.transpose(1, 2)     # [1, T, D]
        enc_ctx = buffer.context_size.subsample(factor=frame_samples)
        enc_ctx_b = buffer.context_size_batch.subsample(factor=frame_samples)
        encoder_output = encoder_output[:, enc_ctx.left:]
        dec_len = torch.where(
            is_last_chunk_batch, encoder_output_len - enc_ctx_b.left, enc_ctx_b.chunk
        )
        chunk_hyps, state = decoding_computer(
            x=encoder_output, out_len=dec_len, prev_batched_state=state, multi_biasing_ids=None
        )
        step_tokens = batched_hyps_to_hypotheses(chunk_hyps, batch_size=1)[0].y_sequence.tolist()
        if total_hyps is None:
            total_hyps = chunk_hyps
        else:
            total_hyps.merge_(chunk_hyps)
        steps.append(
            dict(
                ctx=ctx_after(buffer),
                win=win_samples,
                enc_len=int(encoder_output_len[0]),
                dec_len=int(dec_len[0]),
                tokens=[int(t) for t in step_tokens],
            )
        )
        if dump_enc:
            n = int(dec_len[0])
            enc_slices.append(encoder_output[0, :n].detach().cpu().numpy().astype(np.float32))
        rest -= chunk_lengths_batch
        left_sample = right_sample
        right_sample = min(right_sample + ctx_samples.chunk, S)

    final_hyps = batched_hyps_to_hypotheses(total_hyps, batch_size=1)[0]
    final_ids = [int(t) for t in final_hyps.y_sequence.tolist()]
    final_text = model.tokenizer.ids_to_text(final_ids)
    return dict(steps=steps, final_ids=final_ids, final_text=final_text, enc_slices=enc_slices)


def ctx_after(buffer):
    c = buffer.context_size
    return [c.left, c.chunk, c.right]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="path to the .nemo checkpoint")
    ap.add_argument("--audio", required=True, help="16 kHz mono WAV")
    ap.add_argument("--lang", required=True, help="language locale, e.g. en or de")
    ap.add_argument("--preset", action="append", type=parse_preset, required=True,
                    help="L,C,R in 80 ms frames; repeat for several presets")
    ap.add_argument("--dump-enc", action="store_true",
                    help="also store the per-step encoder slices (after the prompt kernel)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    torch.set_num_threads(max(1, torch.get_num_threads()))
    torch.set_float32_matmul_precision("highest")

    from nemo.collections.asr.models import ASRModel

    model = ASRModel.restore_from(args.model, map_location="cpu")
    model.eval()
    model.preprocessor.featurizer.dither = 0.0
    model.preprocessor.featurizer.pad_to = 0
    if getattr(model.cfg.decoding, "strategy", None) != "greedy_batch":
        raise SystemExit(f"expected greedy_batch decoding, got {model.cfg.decoding.strategy}")
    lang_index = int(model.resolve_prompt_language(args.lang))
    audio = load_wav_16k_mono(args.audio)

    import gguf

    w = gguf.GGUFWriter(args.output, "parakeet-baseline-unified-stream")
    w.add_string("baseline.kind", "unified_stream")
    w.add_string("baseline.lang", args.lang)
    w.add_uint32("baseline.prompt_index", lang_index)
    w.add_uint32("baseline.total_samples", int(audio.shape[0]))
    w.add_uint32("baseline.preset_count", len(args.preset))
    w.add_string("baseline.nemo_commit", "cd5e4fbd36ba098d0c78b4552f6f9592ad34de93")
    w.add_string("baseline.audio", args.audio.rsplit("/", 1)[-1])

    for pi, preset in enumerate(args.preset):
        out = run_preset(model, audio, lang_index, preset, args.dump_enc)
        p = f"preset{pi}"
        L, C, R = preset
        ctx = np.array([s["ctx"] for s in out["steps"]], dtype=np.int32)
        w.add_uint32(f"{p}.left", L)
        w.add_uint32(f"{p}.chunk", C)
        w.add_uint32(f"{p}.right", R)
        w.add_uint32(f"{p}.n_steps", len(out["steps"]))
        w.add_string(f"{p}.text", out["final_text"])
        w.add_uint32(f"{p}.token_count", len(out["final_ids"]))
        w.add_tensor(f"{p}.step_ctx", ctx)  # [n_steps, 3] samples (left, chunk, right) after the add
        w.add_tensor(f"{p}.step_win", np.array([s["win"] for s in out["steps"]], dtype=np.int32))
        w.add_tensor(f"{p}.step_enc_len", np.array([s["enc_len"] for s in out["steps"]], dtype=np.int32))
        w.add_tensor(f"{p}.step_dec_len", np.array([s["dec_len"] for s in out["steps"]], dtype=np.int32))
        w.add_tensor(f"{p}.step_tok_count", np.array([len(s["tokens"]) for s in out["steps"]], dtype=np.int32))
        flat = [t for s in out["steps"] for t in s["tokens"]]
        w.add_tensor(f"{p}.step_tokens", np.array(flat, dtype=np.int32) if flat else np.zeros(0, dtype=np.int32))
        w.add_tensor(f"{p}.final_tokens", np.array(out["final_ids"], dtype=np.int32)
                     if out["final_ids"] else np.zeros(0, dtype=np.int32))
        for k, enc in enumerate(out["enc_slices"]):
            w.add_tensor(f"{p}.enc.{k}", np.ascontiguousarray(enc))  # [n_frames, d_model] time-major
        print(f"preset {preset}: steps={len(out['steps'])} tokens={len(out['final_ids'])} "
              f"text={out['final_text']!r}", file=sys.stderr)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    main()
