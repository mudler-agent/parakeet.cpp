// Encoder fingerprint against real voice-detect GGUFs, through the C-API.
//
//   PARAKEET_TEST_VD_GGUF      speaker encoder GGUF (required, else skip 77)
//   PARAKEET_TEST_VD_ALT_GGUF  optional: another speaker encoder with the SAME embedding
//                              size but a different family (for example ECAPA and CAM++,
//                              both 192). Adds the real model-swap case.
//   PARAKEET_TEST_DIAR_GGUF    optional diarization GGUF; adds the diarize_named and
//                              scene paths
//   PARAKEET_TEST_VD_BUNDLE    optional: a bundle (docs/bundle.md) whose voice component was
//                              built from PARAKEET_TEST_VD_GGUF, unchanged. Checks that the
//                              same encoder gives the same family and weights identity from
//                              the standalone file and from the bundle component (name in
//                              PARAKEET_TEST_VD_BUNDLE_COMPONENT, default "voice").
//
// Without a second encoder the swap is simulated: a copy of the first GGUF with a
// changed general.name has the same size and the same weights but another family.
// A copy with one added metadata key has the same family and other file bytes: the
// "other weights" case.
#include "parakeet_capi.h"

#include "audio_io.hpp"
#include "gguf.h"

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

// Copies `src` to `dst` through the GGUF API with one change.
static bool rewrite(const std::string& src, const std::string& dst, const char* key, const char* val) {
    ggml_context* data = nullptr;
    gguf_init_params p{/*no_alloc=*/false, &data};
    gguf_context* g = gguf_init_from_file(src.c_str(), p);
    if (!g) return false;
    gguf_context* out = gguf_init_empty();
    gguf_set_kv(out, g);
    gguf_set_val_str(out, key, val);
    for (ggml_tensor* t = ggml_get_first_tensor(data); t; t = ggml_get_next_tensor(data, t))
        gguf_add_tensor(out, t);
    const bool ok = gguf_write_to_file(out, dst.c_str(), false);
    gguf_free(out);
    gguf_free(g);
    ggml_free(data);
    return ok;
}

static bool contains(const char* s, const std::string& sub) { return s && std::strstr(s, sub.c_str()); }

int main() {
    const char* vd = std::getenv("PARAKEET_TEST_VD_GGUF");
    if (!vd) return 77;
    const char* alt = std::getenv("PARAKEET_TEST_VD_ALT_GGUF");
    const char* diar_path = std::getenv("PARAKEET_TEST_DIAR_GGUF");

    CHECK(parakeet_capi_abi_version() == 11);
    parakeet_ctx* a = parakeet_capi_load(vd);
    if (!a) { std::fprintf(stderr, "FAIL: load speaker model\n"); return 77; }
    const std::string fam_a = parakeet_capi_speaker_encoder_family(a);
    std::printf("encoder A: %s %s dim %d\n", fam_a.c_str(), parakeet_capi_speaker_identity(a),
                parakeet_capi_speaker_dim(a));
    CHECK(fam_a.rfind("voicedetect:", 0) == 0);
    CHECK(fam_a.size() > 12 && fam_a.compare(fam_a.size() - std::to_string(parakeet_capi_speaker_dim(a)).size(),
                                              std::string::npos, std::to_string(parakeet_capi_speaker_dim(a))) == 0);
    CHECK(std::strncmp(parakeet_capi_speaker_identity(a), "sha256:", 7) == 0);

    pk::Audio wav;
    if (!pk::load_audio_16k_mono(std::string(PK_SOURCE_DIR) + "/tests/fixtures/two_speakers.wav", wav)) return 1;
    const auto a0 = slice(wav.samples, 0.6, 4.6), b0 = slice(wav.samples, 6.9, 10.9);
    const auto probe = slice(wav.samples, 14.9, 18.7);

    // Enrolment stamps the registry with the encoder that made the voices.
    parakeet_speaker_registry* reg = parakeet_capi_speaker_registry_new();
    CHECK(parakeet_capi_speaker_enroll(reg, a, "speaker_a", a0.data(), (int)a0.size(), 16000) == 0);
    CHECK(parakeet_capi_speaker_enroll(reg, a, "speaker_b", b0.data(), (int)b0.size(), 16000) == 0);
    CHECK(fam_a == parakeet_capi_speaker_registry_encoder_family(reg));
    CHECK(std::strcmp(parakeet_capi_speaker_identity(a), parakeet_capi_speaker_registry_encoder_weights(reg)) == 0);
    CHECK(std::strlen(parakeet_capi_speaker_last_warning(a)) == 0);

    const std::string path = (std::filesystem::temp_directory_path() / "pk_test_fp_registry_model.bin").string();
    CHECK(parakeet_capi_speaker_registry_save(reg, path.c_str()) == 0);
    parakeet_speaker_registry* loaded = parakeet_capi_speaker_registry_load(path.c_str());
    CHECK(loaded && fam_a == parakeet_capi_speaker_registry_encoder_family(loaded));

    // Same encoder: names come out, no warning.
    char* j = parakeet_capi_speaker_identify_pcm_json(loaded, a, probe.data(), (int)probe.size(), 16000);
    CHECK(j && contains(j, "\"name\":\"speaker_a\""));
    parakeet_capi_free_string(j);
    CHECK(std::strlen(parakeet_capi_speaker_last_warning(a)) == 0);

    // Swap, simulated: same weights, same size, other family. Refused before any name.
    const std::filesystem::path tmp = std::filesystem::temp_directory_path();
    const std::string renamed = (tmp / "pk_test_fp_renamed.gguf").string();
    const std::string extra = (tmp / "pk_test_fp_extra_key.gguf").string();
    CHECK(rewrite(vd, renamed, "general.name", "some-other-encoder"));
    CHECK(rewrite(vd, extra, "general.comment", "requantized elsewhere"));
    {
        parakeet_ctx* b = parakeet_capi_load(renamed.c_str());
        CHECK(b && parakeet_capi_speaker_dim(b) == parakeet_capi_speaker_dim(a));
        CHECK(b && fam_a != parakeet_capi_speaker_encoder_family(b));
        if (b) {
            j = parakeet_capi_speaker_identify_pcm_json(loaded, b, probe.data(), (int)probe.size(), 16000);
            CHECK(j == nullptr);
            CHECK(contains(parakeet_capi_last_error(b), fam_a));
            CHECK(contains(parakeet_capi_last_error(b), "some-other-encoder"));
            // Enrolling with it into the stamped registry is refused as well.
            CHECK(parakeet_capi_speaker_enroll(loaded, b, "speaker_c", b0.data(), (int)b0.size(), 16000) != 0);
            CHECK(parakeet_capi_speaker_registry_size(loaded) == 2);
            parakeet_capi_free(b);
        }
    }
    // Swap, real: another encoder with the same embedding size.
    if (alt) {
        parakeet_ctx* b = parakeet_capi_load(alt);
        CHECK(b != nullptr);
        if (b) {
            std::printf("encoder B: %s dim %d\n", parakeet_capi_speaker_encoder_family(b), parakeet_capi_speaker_dim(b));
            CHECK(parakeet_capi_speaker_dim(b) == parakeet_capi_speaker_dim(a));
            CHECK(fam_a != parakeet_capi_speaker_encoder_family(b));
            j = parakeet_capi_speaker_identify_pcm_json(loaded, b, probe.data(), (int)probe.size(), 16000);
            CHECK(j == nullptr);
            CHECK(contains(parakeet_capi_last_error(b), fam_a));
            CHECK(contains(parakeet_capi_last_error(b), parakeet_capi_speaker_encoder_family(b)));
            parakeet_capi_free(b);
        } else {
            std::printf("note: PARAKEET_TEST_VD_ALT_GGUF did not load\n");
        }
    } else {
        std::printf("note: no PARAKEET_TEST_VD_ALT_GGUF, real swap case skipped\n");
    }
    // Same family, other weights: a warning and a result.
    {
        parakeet_ctx* w = parakeet_capi_load(extra.c_str());
        CHECK(w && fam_a == parakeet_capi_speaker_encoder_family(w));
        CHECK(w && std::strcmp(parakeet_capi_speaker_identity(w), parakeet_capi_speaker_identity(a)) != 0);
        if (w) {
            j = parakeet_capi_speaker_identify_pcm_json(loaded, w, probe.data(), (int)probe.size(), 16000);
            CHECK(j && contains(j, "\"name\":\"speaker_a\""));
            parakeet_capi_free_string(j);
            CHECK(contains(parakeet_capi_speaker_last_warning(w), "other weights"));
            // Strict mode does not turn a weights difference into an error.
            parakeet_capi_speaker_registry_set_strict(loaded, 1);
            j = parakeet_capi_speaker_identify_pcm_json(loaded, w, probe.data(), (int)probe.size(), 16000);
            CHECK(j != nullptr);
            parakeet_capi_free_string(j);
            parakeet_capi_speaker_registry_set_strict(loaded, 0);
            parakeet_capi_free(w);
        }
    }

    // A registry with no fingerprint (v1 file, or add_embedding without one): accepted
    // with a soft warning; a strict registry is refused.
    {
        parakeet_speaker_registry* plain = parakeet_capi_speaker_registry_new();
        const int dim = parakeet_capi_speaker_dim(a);
        std::vector<float> e1((size_t)dim, 0.0f), e2((size_t)dim, 0.0f);
        e1[0] = 1.0f;
        e2[1] = 1.0f;
        CHECK(parakeet_capi_speaker_registry_add_embedding(plain, "x", e1.data(), dim) == 0);
        CHECK(parakeet_capi_speaker_registry_add_embedding(plain, "y", e2.data(), dim) == 0);
        const std::string v1 = (tmp / "pk_test_fp_v1.bin").string();
        CHECK(parakeet_capi_speaker_registry_save(plain, v1.c_str()) == 0);
        { std::FILE* f = std::fopen(v1.c_str(), "rb"); unsigned char h[8] = {0};
          CHECK(f && std::fread(h, 1, 8, f) == 8 && h[4] == 1);
          if (f) std::fclose(f); }
        parakeet_speaker_registry* old = parakeet_capi_speaker_registry_load(v1.c_str());
        CHECK(old && std::strlen(parakeet_capi_speaker_registry_encoder_family(old)) == 0);
        j = parakeet_capi_speaker_identify_pcm_json(old, a, probe.data(), (int)probe.size(), 16000);
        CHECK(j != nullptr);
        parakeet_capi_free_string(j);
        CHECK(contains(parakeet_capi_speaker_last_warning(a), "no encoder fingerprint"));
        parakeet_capi_speaker_registry_set_strict(old, 1);
        j = parakeet_capi_speaker_identify_pcm_json(old, a, probe.data(), (int)probe.size(), 16000);
        CHECK(j == nullptr && contains(parakeet_capi_last_error(a), "strict"));
        // Calling enroll with the encoder on a registry that already has unstamped speakers is refused.
        CHECK(parakeet_capi_speaker_enroll(old, a, "z", a0.data(), (int)a0.size(), 16000) != 0);
        // add_embedding_fp with the right fingerprint into an empty registry then works end to end.
        parakeet_speaker_registry* fresh = parakeet_capi_speaker_registry_new();
        CHECK(parakeet_capi_speaker_registry_add_embedding_fp(fresh, "x", e1.data(), dim, fam_a.c_str(),
                  parakeet_capi_speaker_identity(a)) == 0);
        CHECK(parakeet_capi_speaker_registry_add_embedding_fp(fresh, "y", e2.data(), dim, fam_a.c_str(),
                  parakeet_capi_speaker_identity(a)) == 0);
        parakeet_capi_speaker_registry_set_strict(fresh, 1);
        j = parakeet_capi_speaker_identify_pcm_json(fresh, a, probe.data(), (int)probe.size(), 16000);
        CHECK(j != nullptr);
        parakeet_capi_free_string(j);
        parakeet_capi_speaker_registry_free(fresh);
        parakeet_capi_speaker_registry_free(old);
        parakeet_capi_speaker_registry_free(plain);
        std::filesystem::remove(v1);
    }

    // The same encoder inside a bundle: same family, same weights identity (the bundle's
    // recorded source hash), so a registry enrolled with the standalone file is used with
    // the bundle component with no warning, and the other way round.
    if (const char* bundle = std::getenv("PARAKEET_TEST_VD_BUNDLE")) {
        const char* comp = std::getenv("PARAKEET_TEST_VD_BUNDLE_COMPONENT");
        parakeet_ctx* c = parakeet_capi_load_component(bundle, comp ? comp : "voice");
        CHECK(c != nullptr);
        if (c) {
            CHECK(fam_a == parakeet_capi_speaker_encoder_family(c));
            CHECK(std::strcmp(parakeet_capi_speaker_identity(a), parakeet_capi_speaker_identity(c)) == 0);
            char* jb = parakeet_capi_speaker_identify_pcm_json(loaded, c, probe.data(), (int)probe.size(), 16000);
            CHECK(jb && contains(jb, "\"name\":\"speaker_a\""));
            parakeet_capi_free_string(jb);
            CHECK(std::strlen(parakeet_capi_speaker_last_warning(c)) == 0);
            // Enrolled with the bundle component, used with the standalone file.
            parakeet_speaker_registry* rb = parakeet_capi_speaker_registry_new();
            CHECK(parakeet_capi_speaker_enroll(rb, c, "speaker_a", a0.data(), (int)a0.size(), 16000) == 0);
            CHECK(std::strcmp(parakeet_capi_speaker_registry_encoder_weights(rb),
                              parakeet_capi_speaker_registry_encoder_weights(loaded)) == 0);
            jb = parakeet_capi_speaker_identify_pcm_json(rb, a, probe.data(), (int)probe.size(), 16000);
            CHECK(jb != nullptr);
            parakeet_capi_free_string(jb);
            CHECK(std::strlen(parakeet_capi_speaker_last_warning(a)) == 0);
            parakeet_capi_speaker_registry_free(rb);
            parakeet_capi_free(c);
        }
    } else {
        std::printf("note: no PARAKEET_TEST_VD_BUNDLE, bundle identity case skipped\n");
    }

    // The same check guards the diarization and scene entry points.
    if (diar_path) {
        parakeet_ctx* diar = parakeet_capi_load(diar_path);
        parakeet_ctx* b = parakeet_capi_load(renamed.c_str());
        CHECK(diar && b);
        if (diar && b) {
            char* d = parakeet_capi_diarize_named_pcm_json(diar, b, loaded, wav.samples.data(),
                                                           (int)wav.samples.size(), 16000, 0, 0);
            CHECK(d == nullptr && contains(parakeet_capi_last_error(b), fam_a));
            parakeet_capi_free_string(d);
            parakeet_scene_opts o;
            parakeet_capi_scene_opts_default(&o);
            parakeet_scene_stream* s = parakeet_capi_scene_stream_begin_speaker(nullptr, diar, nullptr, b, loaded, &o);
            CHECK(s == nullptr && contains(parakeet_capi_last_error(b), fam_a));
            parakeet_capi_scene_stream_free(s);
            // And the right encoder still names the speakers.
            d = parakeet_capi_diarize_named_pcm_json(diar, a, loaded, wav.samples.data(),
                                                     (int)wav.samples.size(), 16000, 0, 0);
            CHECK(d && contains(d, "\"name\":\"speaker_a\""));
            parakeet_capi_free_string(d);
        }
        parakeet_capi_free(b);
        parakeet_capi_free(diar);
    }

    parakeet_capi_speaker_registry_free(loaded);
    parakeet_capi_speaker_registry_free(reg);
    parakeet_capi_free(a);
    std::filesystem::remove(path);
    std::filesystem::remove(renamed);
    std::filesystem::remove(extra);
    if (failures) { std::fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    std::printf("test_speaker_fingerprint: PASS\n");
    return 0;
}
