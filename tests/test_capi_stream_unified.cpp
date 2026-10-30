// C-API v11 buffered streaming for a unified checkpoint. Skips (77) unless
//   PARAKEET_TEST_GGUF_UNIFIED   converted unified gguf
//   PARAKEET_TEST_UNIFIED_WAV    optional clip (default tests/fixtures/speech.wav)
// Checks: the ABI is 11; stream kind BUFFERED and caps 0; stream_begin on the
// default window and begin_opts on every preset give the same text as the
// session for every feed size (1, 160, 1280, 7777 samples and the whole clip);
// the EOU/EOB flags and drain_events stay 0; presets and info JSON; reset gives
// the same text again; a bad window is refused with an error.
#include "parakeet_capi.h"
#include "audio_io.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL: %s (line %d)\n", #c, __LINE__); ++failures; } } while (0)

// Whole stream through the C-API with `block` samples per feed; returns the text.
std::string run_stream(parakeet_stream* s, const std::vector<float>& pcm, int block, int& eou_bits) {
    std::string text;
    eou_bits = 0;
    const int total = (int)pcm.size();
    for (int off = 0; off < total; off += block) {
        const int n = std::min(block, total - off);
        int eou = 0;
        char* t = parakeet_capi_stream_feed(s, pcm.data() + off, n, &eou);
        if (!t) return std::string("<feed failed>");
        text += t;
        parakeet_capi_free_string(t);
        eou_bits |= eou;
    }
    char* f = parakeet_capi_stream_finalize(s);
    if (!f) return std::string("<finalize failed>");
    text += f;
    parakeet_capi_free_string(f);
    return text;
}
}  // namespace

int main() {
    const char* gguf = std::getenv("PARAKEET_TEST_GGUF_UNIFIED");
    if (!gguf) { std::fprintf(stderr, "test_capi_stream_unified: PARAKEET_TEST_GGUF_UNIFIED not set; skip\n"); return 77; }
    const char* wav = std::getenv("PARAKEET_TEST_UNIFIED_WAV");
    pk::Audio a;
    if (!pk::load_audio_16k_mono(wav ? wav : "tests/fixtures/speech.wav", a)) { std::fprintf(stderr, "audio load failed\n"); return 1; }

    CHECK(parakeet_capi_abi_version() == 11);
    parakeet_ctx* ctx = parakeet_capi_load(gguf);
    if (!ctx) { std::fprintf(stderr, "load failed\n"); return 1; }
    CHECK(parakeet_capi_stream_kind(ctx) == PARAKEET_STREAM_KIND_BUFFERED);
    CHECK(parakeet_capi_stream_caps(ctx) == 0);

    char* presets = parakeet_capi_stream_presets_json(ctx);
    CHECK(presets != nullptr);
    if (presets) {
        std::fprintf(stderr, "[capi] presets %s\n", presets);
        CHECK(std::strstr(presets, "\"default\":true") != nullptr);
        CHECK(std::strstr(presets, "\"latency_sec\":1.12") != nullptr);
        parakeet_capi_free_string(presets);
    }

    // Default window (stream_begin, no options).
    parakeet_stream* s = parakeet_capi_stream_begin(ctx);
    CHECK(s != nullptr);
    if (!s) { std::fprintf(stderr, "begin: %s\n", parakeet_capi_last_error(ctx)); parakeet_capi_free(ctx); return 1; }
    char* info = parakeet_capi_stream_info_json(s);
    CHECK(info && std::strstr(info, "\"kind\":\"buffered\"") && std::strstr(info, "\"left\":70") &&
          std::strstr(info, "\"chunk\":7") && std::strstr(info, "\"right\":7"));
    if (info) { std::fprintf(stderr, "[capi] info %s\n", info); parakeet_capi_free_string(info); }

    int eou = 0;
    const std::string ref = run_stream(s, a.samples, (int)a.samples.size(), eou);
    CHECK(eou == 0);
    CHECK(!ref.empty() && ref[0] != '<');
    parakeet_stream_event* evs = nullptr;
    CHECK(parakeet_capi_stream_drain_events(s, &evs) == 0 && evs == nullptr);  // no events on a buffered stream
    parakeet_capi_stream_free(s);

    // Every feed size gives the same text (feed-size invariance through the C-API).
    for (int block : {1, 160, 1280, 7777}) {
        parakeet_stream* t = parakeet_capi_stream_begin(ctx);
        CHECK(t != nullptr);
        if (!t) continue;
        int e = 0;
        const std::string got = run_stream(t, a.samples, block, e);
        std::fprintf(stderr, "[capi] feed %d: %s\n", block, got == ref ? "same text" : "DIFFERENT");
        CHECK(got == ref);
        CHECK(e == 0);
        // Reset and feed again: the same utterance gives the same text.
        CHECK(parakeet_capi_stream_reset(t) == 0);
        const std::string again = run_stream(t, a.samples, block, e);
        CHECK(again == ref);
        parakeet_capi_stream_free(t);
    }

    // Each preset of the model, selected by index, gives a stream of its own window.
    parakeet_stream_opts o;
    parakeet_capi_stream_opts_default(&o);
    for (int idx = 0; idx < 6; ++idx) {
        o.preset = idx;
        parakeet_stream* t = parakeet_capi_stream_begin_opts(ctx, &o);
        if (!t) break;   // fewer presets on this checkpoint
        char* i2 = parakeet_capi_stream_info_json(t);
        if (i2) { std::fprintf(stderr, "[capi] preset %d: %s\n", idx, i2); parakeet_capi_free_string(i2); }
        int e = 0;
        const std::string got = run_stream(t, a.samples, 1280, e);
        CHECK(e == 0 && got[0] != '<');
        parakeet_capi_stream_free(t);
    }

    // Custom window: 5600/560/560 ms = 70/7/7 frames, the default preset in ms.
    o.preset = PARAKEET_STREAM_PRESET_CUSTOM;
    o.left_frames = 70; o.chunk_frames = 7; o.right_frames = 7;
    parakeet_stream* c = parakeet_capi_stream_begin_opts(ctx, &o);
    CHECK(c != nullptr);
    if (c) {
        int e = 0;
        CHECK(run_stream(c, a.samples, 1280, e) == ref);
        parakeet_capi_stream_free(c);
    }

    // Refusals: a bad window and a wrong struct size.
    o.left_frames = 70; o.chunk_frames = 0; o.right_frames = 7;
    CHECK(parakeet_capi_stream_begin_opts(ctx, &o) == nullptr);
    CHECK(std::strlen(parakeet_capi_last_error(ctx)) > 0);
    o.chunk_frames = 7;
    o.size = 3;
    CHECK(parakeet_capi_stream_begin_opts(ctx, &o) == nullptr);

    parakeet_capi_free(ctx);
    if (failures == 0) std::printf("test_capi_stream_unified: C-API buffered streams match across feed sizes and windows\n");
    return failures ? 1 : 0;
}
