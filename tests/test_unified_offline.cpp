// Offline parity for a unified checkpoint (nvidia/nemotron-labs-unified-asr-0.6b,
// att_context_style chunked_limited_with_rc). Offline inference is full context,
// so the stages must match NeMo's offline path. Skips (77) unless the converted
// GGUF and a prompt baseline are provided:
//   PARAKEET_TEST_GGUF_UNIFIED      converted unified gguf
//   PARAKEET_TEST_BASELINE_UNIFIED  prompt baseline from gen_nemo_baseline.py --lang X
//                                   (mel [n_mels,T], encoder_out [D,T],
//                                   prompt_kernel_out [T,D], rnnt_token_ids,
//                                   baseline.rnnt_text, baseline.target_lang)
//   PARAKEET_TEST_UNIFIED_WAV       clip used for the baseline (default tests/fixtures/speech.wav)
// Also checks the loader's unified fields, and that the stream_* C-API refuses
// the model instead of running it as a cache-aware stream.
#include "model.hpp"
#include "model_loader.hpp"
#include "backend.hpp"
#include "audio_io.hpp"
#include "encoder.hpp"
#include "mel.hpp"
#include "prompt_kernel.hpp"
#include "parity.hpp"
#include "parakeet_capi.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL: %s (line %d)\n", #c, __LINE__); ++failures; } } while (0)

// Largest |a-b| over two equal-length vectors.
float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

}  // namespace

int main() {
    const char* gguf = std::getenv("PARAKEET_TEST_GGUF_UNIFIED");
    const char* base = std::getenv("PARAKEET_TEST_BASELINE_UNIFIED");
    if (!gguf || !base) {
        std::fprintf(stderr, "test_unified_offline: fixtures not set; skip\n");
        return 77;
    }
    const char* wav = std::getenv("PARAKEET_TEST_UNIFIED_WAV");
    const std::string wav_path = wav ? wav : "tests/fixtures/speech.wav";

    std::string lang, ref_text;
    if (!pktest::load_kv_str(base, "baseline.target_lang", lang)) return 1;
    if (!pktest::load_kv_str(base, "baseline.rnnt_text", ref_text)) return 1;
    pk::Audio a;
    if (!pk::load_audio_16k_mono(wav_path, a)) { std::fprintf(stderr, "audio load failed\n"); return 1; }

    // Each model copy is released before the next one loads (the f32 file is 2.5 GB).
    {
        // ---- loader: unified fields, streaming off, prompt on ----
        pk::ModelLoader ml;
        if (!ml.load(gguf)) { std::fprintf(stderr, "loader failed\n"); return 1; }
        const pk::ParakeetConfig& c = ml.config();
        CHECK(c.unified.present);
        CHECK(c.unified.num_presets() == 6);
        CHECK(c.unified.default_preset == 1);
        CHECK(c.unified.presets.size() == 18 && c.unified.presets[3] == 70 && c.unified.presets[4] == 7);
        CHECK(!c.streaming.present);  // unified models must never enable cache-aware streaming
        CHECK(c.prompt.present);
        CHECK(c.use_bias);  // this checkpoint keeps the linear biases (nemotron-3.5 does not)

        // ---- stage 1: mel from the production frontend (per-feature normalized) ----
        // With parakeet.unified.valid_length the frontend trims to NeMo's `length`
        // (floor(samples/hop)); the reference mel holds one more frame, so only the
        // first T frames are compared.
        std::vector<float> ref_mel; std::vector<int64_t> mshape;
        if (!pktest::load_baseline(base, "mel", ref_mel, mshape)) return 1;
        const int Tref = (int)mshape[1];
        pk::MelFrontend mel(ml);
        std::vector<float> feats; int n_mels = 0, T = 0;
        mel.compute(a.samples, feats, n_mels, T);
        CHECK(c.unified.valid_length);
        CHECK(T == (int)(a.samples.size() / (size_t)c.hop_length));
        CHECK(T <= Tref && (int)mshape[0] == n_mels);
        std::vector<float> ref_head((size_t)n_mels * T);
        for (int m = 0; m < n_mels; ++m)
            for (int t = 0; t < T; ++t) ref_head[(size_t)m * T + t] = ref_mel[(size_t)m * Tref + t];
        const float mel_diff = max_abs_diff(feats, ref_head);
        std::fprintf(stderr, "[mel] frames=%d (reference holds %d) max|diff| = %.3g\n", T, Tref, (double)mel_diff);
        CHECK(mel_diff < 1e-2f);  // same tolerance as test_mel

        // ---- stage 2: encoder on the production mel (full attention, pad mask) ----
        pk::ensure_weights_realized(ml);
        std::vector<float> ref_enc; std::vector<int64_t> eshape;
        if (!pktest::load_baseline(base, "encoder_out", ref_enc, eshape)) return 1;
        pk::Encoder enc(ml);
        std::vector<float> enc_out; int dm = 0, Tout = 0;
        enc.forward(feats, n_mels, T, enc_out, dm, Tout);
        CHECK(eshape.size() == 2 && (int)eshape[0] == dm && (int)eshape[1] == Tout);
        const float enc_diff = max_abs_diff(enc_out, ref_enc);
        std::fprintf(stderr, "[encoder] max|diff| = %.3g (D=%d T=%d)\n", (double)enc_diff, dm, Tout);
        CHECK(enc_diff < 1e-5f);

        // ---- stage 3: prompt kernel with the session language ----
        const int prompt_index = (int)pktest::pktest_read_u32(base, "baseline.prompt_index");
        std::vector<float> ref_pk; std::vector<int64_t> pshape;
        if (!pktest::load_baseline(base, "prompt_kernel_out", ref_pk, pshape)) return 1;
        pk::PromptKernel pkmod(ml);
        CHECK(pkmod.present());
        std::vector<float> got;  // [D, T] channels-first
        pkmod.apply(ref_enc, dm, Tout, prompt_index, got);
        std::vector<float> got_td((size_t)Tout * dm);
        for (int t = 0; t < Tout; ++t)
            for (int d = 0; d < dm; ++d) got_td[(size_t)t * dm + d] = got[(size_t)d * Tout + t];
        const float pk_diff = max_abs_diff(got_td, ref_pk);
        std::fprintf(stderr, "[prompt] lang=%s index=%d max|diff| = %.3g\n", lang.c_str(), prompt_index,
                     (double)pk_diff);
        CHECK(pk_diff < 2e-3f);
    }
    {
        // ---- stage 4: RNNT greedy token ids and text, offline ----
        auto m = pk::Model::load(gguf);
        if (!m) { std::fprintf(stderr, "Model::load failed\n"); return 1; }
        pk::Transcription tr = m->transcribe_with_timestamps(a.samples, 16000, pk::Decoder::kDefault, lang);
        std::vector<int32_t> ref_ids;
        if (!pktest::load_baseline_i32(base, "rnnt_token_ids", ref_ids)) return 1;
        std::vector<int32_t> got_ids;
        for (const auto& t : tr.tokens) got_ids.push_back(t.id);
        std::fprintf(stderr, "[tokens] got=%zu ref=%zu\n", got_ids.size(), ref_ids.size());
        CHECK(got_ids == ref_ids);
        std::fprintf(stderr, "[text] got=%s\n[text] ref=%s\n", tr.text.c_str(), ref_text.c_str());
        CHECK(tr.text == ref_text);

        // Sub-hop and empty clips have zero mel frames (valid_length trims to
        // floor(S/hop)): no text and no crash, with and without timestamps.
        CHECK(m->transcribe_16k(std::vector<float>(100, 0.1f), pk::Decoder::kDefault, lang).empty());
        CHECK(m->transcribe_16k(std::vector<float>(), pk::Decoder::kDefault, lang).empty());
        CHECK(m->transcribe_with_timestamps(std::vector<float>(), 16000, pk::Decoder::kDefault, lang).text.empty());
    }

    // ---- stage 5: stream_* must refuse the unified model, with a reason ----
    parakeet_ctx* ctx = parakeet_capi_load(gguf);
    CHECK(ctx != nullptr);
    if (ctx) {
        parakeet_stream* s = parakeet_capi_stream_begin_lang(ctx, nullptr);
        CHECK(s == nullptr);
        const char* err = parakeet_capi_last_error(ctx);
        std::string e = err ? err : "";
        std::fprintf(stderr, "[capi] stream_begin error: %s\n", e.c_str());
        CHECK(e.find("buffered streaming") != std::string::npos);
        parakeet_capi_free(ctx);
    }

    if (failures) { std::fprintf(stderr, "test_unified_offline: %d failure(s)\n", failures); return 1; }
    std::fprintf(stderr, "test_unified_offline: OK (lang=%s)\n", lang.c_str());
    return 0;
}
