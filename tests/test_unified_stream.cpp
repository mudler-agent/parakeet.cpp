// Buffered streaming parity for a unified checkpoint (nvidia/nemotron-labs-unified-asr-0.6b).
// Skips (77) unless the converted f32 GGUF and a streaming baseline are provided:
//   PARAKEET_TEST_GGUF_UNIFIED           converted unified gguf (f32 for the 1e-6 gate)
//   PARAKEET_TEST_BASELINE_UNIFIED_STREAM baseline from scripts/gen_unified_stream_baseline.py
//   PARAKEET_TEST_UNIFIED_STREAM_WAV     the clip the baseline was made from (16 kHz mono)
//   PARAKEET_TEST_FEED                   optional: feed block size in samples (default: whole clip)
//
// For every preset in the baseline, the test runs BufferedStreamingSession with
// tracing and checks each step against NeMo's streaming loop: the buffer context
// (samples), the window length, the encoder length, the decoded frame count, the
// step's tokens, and the post-prompt encoder slice. It also checks the final
// token ids and the text.
#include "model.hpp"
#include "model_loader.hpp"
#include "audio_io.hpp"
#include "buffered_streaming.hpp"
#include "parity.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL: %s (line %d)\n", #c, __LINE__); ++failures; } } while (0)

std::string key(int p, const char* name) { return "preset" + std::to_string(p) + "." + name; }

}  // namespace

int main() {
    const char* gguf = std::getenv("PARAKEET_TEST_GGUF_UNIFIED");
    const char* base = std::getenv("PARAKEET_TEST_BASELINE_UNIFIED_STREAM");
    if (!gguf || !base) {
        std::fprintf(stderr, "test_unified_stream: fixtures not set; skip\n");
        return 77;
    }
    const char* wav = std::getenv("PARAKEET_TEST_UNIFIED_STREAM_WAV");
    const std::string wav_path = wav ? wav : "tests/fixtures/speech.wav";
    const char* feed_env = std::getenv("PARAKEET_TEST_FEED");
    const int feed = feed_env ? std::max(1, std::atoi(feed_env)) : 0;
    const char* only_env = std::getenv("PARAKEET_TEST_PRESET");   // optional: run one preset index

    pk::Audio a;
    if (!pk::load_audio_16k_mono(wav_path, a)) { std::fprintf(stderr, "audio load failed\n"); return 1; }

    std::string lang;
    if (!pktest::load_kv_str(base, "baseline.lang", lang)) return 1;
    const uint32_t n_presets = pktest::pktest_read_u32(base, "baseline.preset_count");
    if (n_presets == 0) { std::fprintf(stderr, "baseline has no presets\n"); return 1; }

    auto m = pk::Model::load(gguf);
    if (!m) { std::fprintf(stderr, "model load failed\n"); return 1; }
    CHECK(m->config().unified.present);
    CHECK(!m->config().streaming.present);

    for (uint32_t p = 0; p < n_presets; ++p) {
        if (only_env && (uint32_t)std::atoi(only_env) != p) continue;
        const int L = (int)pktest::pktest_read_u32(base, key(p, "left"));
        const int C = (int)pktest::pktest_read_u32(base, key(p, "chunk"));
        const int R = (int)pktest::pktest_read_u32(base, key(p, "right"));
        const int n_steps = (int)pktest::pktest_read_u32(base, key(p, "n_steps"));
        std::string ref_text;
        pktest::load_kv_str(base, key(p, "text"), ref_text);
        std::fprintf(stderr, "[stream] preset %d: L=%d C=%d R=%d steps=%d lang=%s feed=%s\n",
                     (int)p, L, C, R, n_steps, lang.c_str(), feed ? std::to_string(feed).c_str() : "whole");

        std::vector<int32_t> ref_ctx, ref_win, ref_enc, ref_dec, ref_cnt, ref_tok, ref_final;
        CHECK(pktest::load_baseline_i32(base, key(p, "step_ctx"), ref_ctx));
        CHECK(pktest::load_baseline_i32(base, key(p, "step_win"), ref_win));
        CHECK(pktest::load_baseline_i32(base, key(p, "step_enc_len"), ref_enc));
        CHECK(pktest::load_baseline_i32(base, key(p, "step_dec_len"), ref_dec));
        CHECK(pktest::load_baseline_i32(base, key(p, "step_tok_count"), ref_cnt));
        CHECK(pktest::load_baseline_i32(base, key(p, "step_tokens"), ref_tok));
        CHECK(pktest::load_baseline_i32(base, key(p, "final_tokens"), ref_final));
        if ((int)ref_win.size() != n_steps || (int)ref_cnt.size() != n_steps || ref_ctx.size() != 3u * n_steps) {
            std::fprintf(stderr, "FAIL: baseline step arrays do not match n_steps\n");
            return 1;
        }

        pk::BufferedStreamingSession sess(m->loader(), L, C, R, lang);
        sess.set_trace(true);
        const int64_t total = (int64_t)a.samples.size();
        if (feed <= 0) {
            sess.feed(a.samples.data(), (int)total);
        } else {
            for (int64_t off = 0; off < total; off += feed) {
                const int n = (int)std::min<int64_t>(feed, total - off);
                sess.feed(a.samples.data() + off, n);
            }
        }
        sess.finalize();

        const auto& tr = sess.trace();
        CHECK((int)tr.size() == n_steps);
        int bad_steps = 0;
        double worst_enc = 0.0;
        int worst_step = -1;
        int slice_violations = 0;
        size_t tok_off = 0;
        for (int s = 0; s < n_steps && s < (int)tr.size(); ++s) {
            const pk::BufferedStreamingSession::Step& st = tr[s];
            bool ok = st.plan.left == ref_ctx[3 * s] && st.plan.chunk == ref_ctx[3 * s + 1] &&
                      st.plan.right == ref_ctx[3 * s + 2] && (st.plan.left + st.plan.chunk + st.plan.right) == ref_win[s] &&
                      st.enc_frames == ref_enc[s] && st.dec_frames == ref_dec[s] &&
                      (int)st.tokens.size() == ref_cnt[s];
            for (int k = 0; ok && k < ref_cnt[s]; ++k) ok = st.tokens[k] == ref_tok[tok_off + k];
            tok_off += (size_t)ref_cnt[s];
            if (!ok) {
                ++bad_steps;
                if (bad_steps <= 5)
                    std::fprintf(stderr, "  step %d: ctx got (%lld,%lld,%lld) ref (%d,%d,%d); win %lld/%d; enc %d/%d; dec %d/%d; tokens %zu/%d\n",
                                 s, (long long)st.plan.left, (long long)st.plan.chunk, (long long)st.plan.right,
                                 ref_ctx[3 * s], ref_ctx[3 * s + 1], ref_ctx[3 * s + 2],
                                 (long long)(st.plan.left + st.plan.chunk + st.plan.right), ref_win[s], st.enc_frames, ref_enc[s],
                                 st.dec_frames, ref_dec[s], st.tokens.size(), ref_cnt[s]);
            }
            // Encoder slice (after the prompt kernel) for this step.
            std::vector<float> ref_slice;
            std::vector<int64_t> shape;
            if (pktest::load_baseline(base, key(p, ("enc." + std::to_string(s)).c_str()), ref_slice, shape)) {
                if (ref_slice.size() != st.enc_tm.size()) {
                    std::fprintf(stderr, "  step %d: encoder slice size %zu != %zu\n", s, st.enc_tm.size(), ref_slice.size());
                    ++bad_steps;
                } else {
                    double step_max = 0.0;
                    for (size_t i = 0; i < ref_slice.size(); ++i)
                        step_max = std::max(step_max, (double)std::fabs(ref_slice[i] - st.enc_tm[i]));
                    if (step_max > worst_enc) { worst_enc = step_max; worst_step = s; }
                    // Tolerance for the encoder slice (after the prompt). Tokens are the hard gate.
                    // Steady state (full left context): measured 7e-7 in f32 on both clips, so 1e-5.
                    // Warm-up (left context still growing): the slice gap comes from the mel front end,
                    // not the encoder. Feeding NeMo's own mel into our encoder gives <= 1.9e-5 in warm-up
                    // (out/win measurement), while our mel differs from NeMo's by 1.2e-4 to 2.3e-4 there,
                    // amplified by per-feature normalization over the few window frames. Hence 1e-3.
                    const bool warm = st.plan.left < (int64_t)L * 1280;
                    const double tol = warm ? 1e-3 : 1e-5;
                    if (step_max > tol) {
                        ++slice_violations;
                        std::fprintf(stderr, "  step %d (%s): slice max|diff| %.3e > %.1e\n", s, warm ? "warm-up" : "steady", step_max, tol);
                    }
                    if (std::getenv("PARAKEET_TEST_VERBOSE"))
                        std::fprintf(stderr, "  step %3d: ctx L=%lld C=%lld R=%lld win=%lld dec=%d slice max|diff|=%.3e\n",
                                     s, (long long)st.plan.left, (long long)st.plan.chunk, (long long)st.plan.right,
                                     (long long)(st.plan.left + st.plan.chunk + st.plan.right), st.dec_frames, step_max);
                }
            }
        }
        CHECK(bad_steps == 0);
        CHECK(slice_violations == 0);

        const std::vector<int32_t>& got = sess.tokens();
        const bool tok_ok = got == ref_final;
        const std::string got_text = sess.text();
        std::fprintf(stderr, "[stream] preset %d: %d/%d steps match, final tokens %s (%zu), text %s\n",
                     (int)p, n_steps - bad_steps, n_steps, tok_ok ? "match" : "DIFFER", got.size(),
                     got_text == ref_text ? "match" : "DIFFER");
        std::fprintf(stderr, "[stream] preset %d: encoder slice max|diff| = %.3e (worst step %d of %d, dec %d)\n",
                     (int)p, worst_enc, worst_step, n_steps, worst_step >= 0 ? (int)tr[worst_step].dec_frames : -1);
        CHECK(tok_ok);
        CHECK(got_text == ref_text);
    }

    if (failures == 0) std::printf("test_unified_stream: buffered streaming matches NeMo for %u preset(s)\n", n_presets);
    return failures ? 1 : 0;
}
