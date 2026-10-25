// Phonon-2.5 with the five-value weights kept packed must load and run on a GPU
// backend. The packed linears are expanded to F16 on the device, so the transcripts
// must equal the ones of the dequantized file on the same device, and the reference
// transcript of tests/fixtures/speech.wav.
//
// Skips (exit 77) when no GPU backend is the compute device, or when the model path
// is unset, so the CPU suite is unaffected.
// Env: PARAKEET_TEST_GGUF_PHONON_KEEP (--five-value keep), optionally
// PARAKEET_TEST_GGUF_PHONON_DEQ (the F16 or Q8_0 dequantized file).
#include "backend.hpp"
#include "ggml_graph.hpp"
#include "model.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

using namespace pk;

static const char* kReference =
    "Well, I don't wish to see it any more, observed Phoebe, turning away her eyes. "
    "It is certainly very like the old portrait.";

int main() {
    const char* keep = std::getenv("PARAKEET_TEST_GGUF_PHONON_KEEP");
    const char* deq = std::getenv("PARAKEET_TEST_GGUF_PHONON_DEQ");
    if (!keep) {
        std::puts("skip: PARAKEET_TEST_GGUF_PHONON_KEEP unset");
        return 77;
    }
    const std::string dev = global_backend().device_name();
    if (dev == "cpu") {
        std::puts("skip: the compute device is the CPU (no GPU backend in this build or on this host)");
        return 77;
    }
    std::printf("device: %s\n", dev.c_str());
    int failures = 0;
    auto packed = Model::load(keep);
    if (!packed) { std::fprintf(stderr, "FAIL: packed file did not load on %s\n", dev.c_str()); return 1; }
    const std::string a = packed->transcribe_path("tests/fixtures/speech.wav", Decoder::kTDT);
    const std::string b = packed->transcribe_path("tests/fixtures/two_speakers.wav", Decoder::kTDT);
    std::printf("packed speech        -> %s\npacked two_speakers  -> %s\n", a.c_str(), b.c_str());
    if (a != kReference) { std::fprintf(stderr, "FAIL: speech.wav transcript differs from the reference\n"); ++failures; }
    if (b.empty()) { std::fprintf(stderr, "FAIL: two_speakers.wav gave an empty transcript\n"); ++failures; }
    packed.reset();
    if (deq) {
        auto dense = Model::load(deq);
        if (!dense) { std::fprintf(stderr, "FAIL: dequantized file did not load\n"); return 1; }
        const std::string da = dense->transcribe_path("tests/fixtures/speech.wav", Decoder::kTDT);
        const std::string db = dense->transcribe_path("tests/fixtures/two_speakers.wav", Decoder::kTDT);
        std::printf("dense speech         -> %s\ndense two_speakers   -> %s\n", da.c_str(), db.c_str());
        if (da != a || db != b) { std::fprintf(stderr, "FAIL: packed and dense transcripts differ on the device\n"); ++failures; }
    }
    // Free the device buffers while the driver is still alive.
    shutdown_backend();
    if (failures) return 1;
    std::puts("test_fivevalue_gpu: OK");
    return 0;
}
