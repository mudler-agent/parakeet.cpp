// Phonon-2.5 with the five-value weights kept packed (--five-value keep) must
// decode the reference clip to the same transcript as the dequantized file, and a
// batch must equal the per-item results. It also splits every five-value linear of the packed
// file into its two planes and requires lo * A + hi * B to equal, bit for bit, the weights of the
// dequantized file (when that file holds F16 or F32 encoder weights).
// Env: PARAKEET_TEST_GGUF_PHONON_KEEP (--five-value keep), PARAKEET_TEST_GGUF_PHONON_DEQ.
#include "model.hpp"
#include "audio_io.hpp"
#include "model_loader.hpp"
#include "ternary.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "ggml.h"

using namespace pk;

static const char* kReference =
    "Well, I don't wish to see it any more, observed Phoebe, turning away her eyes. "
    "It is certainly very like the old portrait.";

// Every encoder linear of the packed file, split into planes, against the same
// weight in the dequantized file. Returns the number of mismatching weights.
static long compare_planes(const char* keep_path, const char* deq_path) {
    ModelLoader kl, dl;
    if (!kl.load(keep_path) || !dl.load(deq_path)) {
        std::fprintf(stderr, "FAIL: cannot load the two files for the plane comparison\n");
        return 1;
    }
    const ParakeetConfig& c = kl.config();
    const int64_t d = c.d_model, ff = c.ff_dim;
    struct Lin { const char* name; int64_t N, K; };
    const Lin lins[] = {
        {"feed_forward1.linear1", ff, d}, {"feed_forward1.linear2", d, ff},
        {"feed_forward2.linear1", ff, d}, {"feed_forward2.linear2", d, ff},
        {"self_attn.linear_q", d, d},     {"self_attn.linear_k", d, d},
        {"self_attn.linear_v", d, d},     {"self_attn.linear_out", d, d},
        {"self_attn.linear_pos", d, d},   {"conv.pointwise_conv1", 2 * d, d},
        {"conv.pointwise_conv2", d, d}};
    long bad = 0, checked = 0, skipped = 0;
    for (int i = 0; i < (int)c.n_layers; ++i)
        for (const Lin& l : lins) {
            const std::string base = "encoder.layers." + std::to_string(i) + "." + l.name;
            const ggml_tensor* codes = kl.tensor(base + ".fvcodes");
            const ggml_tensor* hib = kl.tensor(base + ".fvhi");
            const ggml_tensor* sc = kl.tensor(base + ".fvscale");
            const ggml_tensor* w = dl.tensor(base + ".weight");
            if (!codes || !hib || !sc || !w) { std::fprintf(stderr, "FAIL: missing tensors for %s\n", base.c_str()); return 1; }
            if (w->type != GGML_TYPE_F16 && w->type != GGML_TYPE_F32) { ++skipped; continue; }
            const int N = (int)l.N, K = (int)l.K;
            if ((int64_t)ggml_nelements(w) != (int64_t)N * K) { std::fprintf(stderr, "FAIL: size of %s\n", base.c_str()); return 1; }
            const uint16_t* lo = static_cast<const uint16_t*>(sc->data);
            TernaryWeight A, B;
            fivevalue_split(static_cast<const uint8_t*>(codes->data), static_cast<const uint8_t*>(hib->data),
                            (size_t)hib->ne[0], lo, lo + N, N, K, A, B);
            for (int n = 0; n < N; ++n)
                for (int k = 0; k < K; ++k) {
                    const int g = k / kTernaryGroup;
                    const float got = A.scale(n, g) * (float)(A.code(n, k) - 1) + B.scale(n, g) * (float)(B.code(n, k) - 1);
                    const float want = w->type == GGML_TYPE_F16
                        ? ggml_fp16_to_fp32(static_cast<const uint16_t*>(w->data)[(size_t)n * K + k])
                        : static_cast<const float*>(w->data)[(size_t)n * K + k];
                    if (got != want) ++bad;
                }
            ++checked;
        }
    std::printf("plane split: %ld linears compared, %ld skipped (not F16/F32), %ld mismatching weights\n",
                checked, skipped, bad);
    return checked == 0 ? 1 : bad;
}

int main() {
    const char* keep = std::getenv("PARAKEET_TEST_GGUF_PHONON_KEEP");
    const char* deq = std::getenv("PARAKEET_TEST_GGUF_PHONON_DEQ");
    if (!keep || !deq) {
        std::puts("skip: PARAKEET_TEST_GGUF_PHONON_KEEP / _DEQ unset");
        return 77;
    }
    int failures = 0;
    if (compare_planes(keep, deq) != 0) {
        std::fprintf(stderr, "FAIL: planes differ from the dequantized weights\n");
        ++failures;
    }
    std::string two_deq, two_keep;
    for (const char* path : {deq, keep}) {
        auto m = Model::load(path);
        if (!m) { std::fprintf(stderr, "FAIL: load %s\n", path); return 1; }
        const std::string got = m->transcribe_path("tests/fixtures/speech.wav", Decoder::kTDT);
        std::printf("%s\n  -> %s\n", path, got.c_str());
        if (got != kReference) { std::fprintf(stderr, "FAIL: transcript differs for %s\n", path); ++failures; }
        const std::string two = m->transcribe_path("tests/fixtures/two_speakers.wav", Decoder::kTDT);
        std::printf("  two_speakers -> %s\n", two.c_str());
        (path == keep ? two_keep : two_deq) = two;
        if (path == keep) {
            Audio a, b;
            if (!load_audio_16k_mono("tests/fixtures/speech.wav", a) ||
                !load_audio_16k_mono("tests/fixtures/two_speakers.wav", b)) {
                std::fprintf(stderr, "FAIL: fixture load\n");
                return 1;
            }
            const auto batch = m->transcribe_pcm_batch({a.samples, b.samples}, 16000, Decoder::kTDT);
            if (batch.size() != 2 || batch[0] != got || batch[1] != two) {
                std::fprintf(stderr, "FAIL: batched transcripts differ from per-item\n");
                for (const auto& t : batch) std::fprintf(stderr, "  batch: %s\n", t.c_str());
                ++failures;
            }
        }
    }
    if (two_deq != two_keep) {
        std::fprintf(stderr, "FAIL: two_speakers transcript differs between dequantized and packed Phonon\n"
                             "  deq : %s\n  keep: %s\n", two_deq.c_str(), two_keep.c_str());
        ++failures;
    }
    if (failures) return 1;
    std::puts("test_fivevalue_model: OK");
    return 0;
}
