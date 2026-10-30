// Speaker identification through the flat C-API.
//   PARAKEET_TEST_DIAR_GGUF, PARAKEET_TEST_VD_GGUF (both required, else skip 77)
//   PARAKEET_TEST_GGUF       optional ASR GGUF; adds the named speaker-attributed ASR check
#include "parakeet_capi.h"

#include "audio_io.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

static int failures = 0;
#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::fprintf(stderr, "FAIL: %s (line %d)\n", #cond, __LINE__);  \
            ++failures;                                                      \
        }                                                                    \
    } while (0)

static std::vector<float> slice(const std::vector<float>& x, double a, double b) {
    return std::vector<float>(x.begin() + (long)(a * 16000), x.begin() + (long)(b * 16000));
}

int main() {
    const char* diar_path = std::getenv("PARAKEET_TEST_DIAR_GGUF");
    const char* vd_path = std::getenv("PARAKEET_TEST_VD_GGUF");
    if (!diar_path || !vd_path) return 77;

    CHECK(parakeet_capi_abi_version() == 11);

    parakeet_ctx* spk = parakeet_capi_load(vd_path);
    if (!spk) { std::fprintf(stderr, "FAIL: load speaker model (built without voice-detect?)\n"); return 77; }
    CHECK(parakeet_capi_model_kind(spk) == PARAKEET_MODEL_KIND_SPEAKER);
    CHECK(parakeet_capi_speaker_dim(spk) >= 128);
    parakeet_ctx* diar = parakeet_capi_load(diar_path);
    CHECK(diar && parakeet_capi_speaker_dim(diar) == -1);
    CHECK(parakeet_capi_model_kind(diar) == PARAKEET_MODEL_KIND_DIARIZATION);

    pk::Audio wav;
    if (!pk::load_audio_16k_mono(std::string(PK_SOURCE_DIR) + "/tests/fixtures/two_speakers.wav", wav)) return 1;
    const auto a0 = slice(wav.samples, 0.6, 4.6), b0 = slice(wav.samples, 6.9, 10.9);

    parakeet_speaker_registry* reg = parakeet_capi_speaker_registry_new();
    CHECK(parakeet_capi_speaker_enroll(reg, spk, "speaker_a", a0.data(), (int)a0.size(), 16000) == 0);
    CHECK(parakeet_capi_speaker_enroll(reg, spk, "speaker_b", b0.data(), (int)b0.size(), 16000) == 0);
    CHECK(parakeet_capi_speaker_registry_size(reg) == 2);
    // Errors: empty name, no audio, wrong ctx kind.
    CHECK(parakeet_capi_speaker_enroll(reg, spk, "", a0.data(), (int)a0.size(), 16000) != 0);
    CHECK(parakeet_capi_speaker_enroll(reg, spk, "x", nullptr, 0, 16000) != 0);
    CHECK(parakeet_capi_speaker_enroll(reg, diar, "x", a0.data(), (int)a0.size(), 16000) != 0);
    CHECK(parakeet_capi_speaker_registry_size(reg) == 2);

    // One-shot identify of clips not used for enrollment.
    {
        const auto a1 = slice(wav.samples, 14.9, 18.7);
        char* j = parakeet_capi_speaker_identify_pcm_json(reg, spk, a1.data(), (int)a1.size(), 16000);
        CHECK(j && std::strstr(j, "\"name\":\"speaker_a\"") != nullptr);
        parakeet_capi_free_string(j);
        const auto b1 = slice(wav.samples, 20.2, 23.5);
        j = parakeet_capi_speaker_identify_pcm_json(reg, spk, b1.data(), (int)b1.size(), 16000);
        CHECK(j && std::strstr(j, "\"name\":\"speaker_b\"") != nullptr);
        parakeet_capi_free_string(j);
    }

    // Save and load round trip, and a corrupt file is refused without crashing.
    const std::filesystem::path tmp_dir = std::filesystem::temp_directory_path();
    const std::string path = (tmp_dir / "pk_test_capi_speaker_registry.bin").string();
    std::filesystem::remove(path);
    CHECK(parakeet_capi_speaker_registry_save(reg, path.c_str()) == 0);
    parakeet_speaker_registry* back = parakeet_capi_speaker_registry_load(path.c_str());
    CHECK(back && parakeet_capi_speaker_registry_size(back) == 2);
    parakeet_capi_speaker_registry_free(back);
    // Saving over an existing registry replaces it, and leaves no tmp file.
    {
        parakeet_speaker_registry* one = parakeet_capi_speaker_registry_new();
        CHECK(parakeet_capi_speaker_enroll(one, spk, "only_a", a0.data(), (int)a0.size(), 16000) == 0);
        CHECK(parakeet_capi_speaker_registry_save(one, path.c_str()) == 0);
        CHECK(!std::filesystem::exists(path + ".tmp"));
        parakeet_speaker_registry* again = parakeet_capi_speaker_registry_load(path.c_str());
        CHECK(again && parakeet_capi_speaker_registry_size(again) == 1);
        parakeet_capi_speaker_registry_free(again);
        // A failed save (directory that does not exist) reports an error and
        // leaves the file that was there loadable.
        const std::string bad = (tmp_dir / "pk_test_no_such_dir" / "registry.bin").string();
        CHECK(parakeet_capi_speaker_registry_save(reg, bad.c_str()) != 0);
        CHECK(std::strlen(parakeet_capi_speaker_registry_last_error(reg)) > 0);
        parakeet_speaker_registry* still = parakeet_capi_speaker_registry_load(path.c_str());
        CHECK(still && parakeet_capi_speaker_registry_size(still) == 1);
        parakeet_capi_speaker_registry_free(still);
        parakeet_capi_speaker_registry_free(one);
    }
    { FILE* f = std::fopen(path.c_str(), "wb"); std::fputs("garbage", f); std::fclose(f); }
    CHECK(parakeet_capi_speaker_registry_load(path.c_str()) == nullptr);
    CHECK(parakeet_capi_speaker_registry_load((tmp_dir / "pk_test_no_such_dir" / "registry.bin").string().c_str()) == nullptr);
    std::filesystem::remove(path);

    // Scene stream with diarization + speaker.
    {
        parakeet_scene_opts o;
        parakeet_capi_scene_opts_default(&o);
        parakeet_scene_stream* s = parakeet_capi_scene_stream_begin_speaker(nullptr, diar, nullptr, spk, reg, &o);
        CHECK(s != nullptr);
        std::string last;
        const int chunk = 3200, n = (int)wav.samples.size();
        for (int lo = 0; lo < n && s; lo += chunk) {
            const int len = std::min(chunk, n - lo);
            char* j = parakeet_capi_scene_stream_feed_json(s, wav.samples.data() + lo, len, lo + len >= n);
            CHECK(j != nullptr);
            if (j) { last = j; parakeet_capi_free_string(j); }
        }
        CHECK(last.find("\"names\":{") != std::string::npos);
        CHECK(last.find("\"name\":\"speaker_a\"") != std::string::npos);
        CHECK(last.find("\"name\":\"speaker_b\"") != std::string::npos);
        parakeet_capi_scene_stream_free(s);
    }
    // A speaker ctx without diarization, or a registry missing, is refused with an error on the ctx.
    {
        parakeet_scene_opts o;
        parakeet_capi_scene_opts_default(&o);
        CHECK(parakeet_capi_scene_stream_begin_speaker(nullptr, nullptr, nullptr, spk, reg, &o) == nullptr);
        CHECK(std::strlen(parakeet_capi_last_error(spk)) > 0);
        CHECK(parakeet_capi_scene_stream_begin_speaker(nullptr, diar, nullptr, spk, nullptr, &o) == nullptr);
        // Old entry point still works exactly as before.
        parakeet_scene_stream* s = parakeet_capi_scene_stream_begin(nullptr, diar, nullptr, &o);
        CHECK(s != nullptr);
        parakeet_capi_scene_stream_free(s);
    }
    // An old-sized opts struct (no speaker fields) is accepted and uses defaults.
    {
        parakeet_scene_opts o;
        parakeet_capi_scene_opts_default(&o);
        o.size = (int)offsetof(parakeet_scene_opts, speaker_accept_threshold);
        parakeet_scene_stream* s = parakeet_capi_scene_stream_begin_speaker(nullptr, diar, nullptr, spk, reg, &o);
        CHECK(s != nullptr);
        parakeet_capi_scene_stream_free(s);
    }
    // Invalid speaker fields are ignored when `size` does not cover them, and
    // rejected when it does.
    {
        parakeet_scene_opts o;
        parakeet_capi_scene_opts_default(&o);
        o.speaker_refresh_sec = -1.0f;
        o.speaker_min_voice_sec = -1.0f;
        o.speaker_accept_threshold = 5.0f;
        o.size = (int)offsetof(parakeet_scene_opts, speaker_accept_threshold);
        parakeet_scene_stream* s = parakeet_capi_scene_stream_begin_speaker(nullptr, diar, nullptr, spk, reg, &o);
        CHECK(s != nullptr);
        parakeet_capi_scene_stream_free(s);
        o.size = (int)sizeof(o);
        CHECK(parakeet_capi_scene_stream_begin_speaker(nullptr, diar, nullptr, spk, reg, &o) == nullptr);
        CHECK(std::strstr(parakeet_capi_last_error(spk), "invalid speaker options") != nullptr);
    }
    // Memory safety: a caller built against the v8 header owns a buffer that ends
    // at `flags`. Nothing past it may be read (run under AddressSanitizer).
    {
        struct OldOpts {
            int size;
            int diar_latency;
            parakeet_sound_opts sound;
            int flags;
        };
        parakeet_scene_opts full;
        parakeet_capi_scene_opts_default(&full);
        OldOpts* old = (OldOpts*)std::malloc(sizeof(OldOpts));
        CHECK(old != nullptr);
        if (old) {
            old->size = (int)sizeof(OldOpts);
            old->diar_latency = full.diar_latency;
            old->sound = full.sound;
            old->flags = 0;
            parakeet_scene_stream* s = parakeet_capi_scene_stream_begin_speaker(
                nullptr, diar, nullptr, spk, reg, (const parakeet_scene_opts*)old);
            CHECK(s != nullptr);
            parakeet_capi_scene_stream_free(s);
            std::free(old);
        }
    }
    // A registry built by a model with a different embedding size is refused with a
    // clear error. Needs a second speaker GGUF of another size (for example WeSpeaker
    // 256 versus CAM++ 192): PARAKEET_TEST_VD_GGUF_ALT. Skipped when unset.
    if (const char* alt_path = std::getenv("PARAKEET_TEST_VD_GGUF_ALT")) {
        parakeet_ctx* alt = parakeet_capi_load(alt_path);
        CHECK(alt != nullptr);
        if (alt && parakeet_capi_speaker_dim(alt) != parakeet_capi_speaker_dim(spk)) {
            parakeet_speaker_registry* wrong = parakeet_capi_speaker_registry_new();
            CHECK(parakeet_capi_speaker_enroll(wrong, alt, "x", a0.data(), (int)a0.size(), 16000) == 0);
            // identify with the main model against the alt-sized registry
            CHECK(parakeet_capi_speaker_identify_pcm_json(wrong, spk, a0.data(), (int)a0.size(), 16000) == nullptr);
            CHECK(std::strstr(parakeet_capi_last_error(spk), "-value embeddings") != nullptr);
            // and a scene stream refuses it up front
            parakeet_scene_opts o;
            parakeet_capi_scene_opts_default(&o);
            CHECK(parakeet_capi_scene_stream_begin_speaker(nullptr, diar, nullptr, spk, wrong, &o) == nullptr);
            CHECK(std::strstr(parakeet_capi_last_error(spk), "embeddings") != nullptr);
            parakeet_capi_speaker_registry_free(wrong);
        }
        parakeet_capi_free(alt);
    }

    // Named speaker-attributed ASR (optional: needs an ASR model).
    if (const char* asr_path = std::getenv("PARAKEET_TEST_GGUF")) {
        parakeet_ctx* asr = parakeet_capi_load(asr_path);
        CHECK(asr != nullptr);
        char* j = parakeet_capi_transcribe_and_diarize_named_json(asr, diar, spk, reg, wav.samples.data(),
                                                                   (int)wav.samples.size(), 16000);
        CHECK(j != nullptr);
        if (j) {
            CHECK(std::strstr(j, "\"name\":\"speaker_a\"") != nullptr);
            CHECK(std::strstr(j, "\"name\":\"speaker_b\"") != nullptr);
            CHECK(std::strstr(j, "\"names\":{") != nullptr);
            parakeet_capi_free_string(j);
        }
        parakeet_capi_free(asr);
    }

    parakeet_capi_speaker_registry_free(reg);
    parakeet_capi_free(diar);
    parakeet_capi_free(spk);
    if (failures) { std::fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    std::printf("test_capi_speaker: PASS\n");
    return 0;
}
