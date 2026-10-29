// Registry entry points that need no model: add_embedding, size, save, load,
// errors. Model independent.
#include "parakeet_capi.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>

static int failures = 0;
#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::fprintf(stderr, "FAIL: %s (line %d)\n", #cond, __LINE__);  \
            ++failures;                                                      \
        }                                                                    \
    } while (0)

int main() {
    CHECK(parakeet_capi_abi_version() == 11);

    parakeet_speaker_registry* reg = parakeet_capi_speaker_registry_new();
    CHECK(reg != nullptr);
    const float a[4] = {1, 0, 0, 0};
    const float b[4] = {0, 1, 0, 0};
    CHECK(parakeet_capi_speaker_registry_add_embedding(reg, "ada", a, 4) == 0);
    CHECK(parakeet_capi_speaker_registry_add_embedding(reg, "ben", b, 4) == 0);
    CHECK(parakeet_capi_speaker_registry_size(reg) == 2);
    // Same name again: averaged into the same speaker, not a third one.
    const float a2[4] = {0.9f, 0.1f, 0, 0};
    CHECK(parakeet_capi_speaker_registry_add_embedding(reg, "ada", a2, 4) == 0);
    CHECK(parakeet_capi_speaker_registry_size(reg) == 2);

    // Errors, each leaves the registry unchanged and says why on the registry.
    const float c3[3] = {1, 0, 0};
    CHECK(parakeet_capi_speaker_registry_add_embedding(reg, "cy", c3, 3) != 0);
    CHECK(std::strstr(parakeet_capi_speaker_registry_last_error(reg), "expects") != nullptr);
    const float nan4[4] = {std::numeric_limits<float>::quiet_NaN(), 0, 0, 0};
    CHECK(parakeet_capi_speaker_registry_add_embedding(reg, "dee", nan4, 4) != 0);
    const float inf4[4] = {std::numeric_limits<float>::infinity(), 0, 0, 0};
    CHECK(parakeet_capi_speaker_registry_add_embedding(reg, "dee", inf4, 4) != 0);
    const float zero4[4] = {0, 0, 0, 0};
    CHECK(parakeet_capi_speaker_registry_add_embedding(reg, "dee", zero4, 4) != 0);
    CHECK(parakeet_capi_speaker_registry_add_embedding(reg, "", a, 4) != 0);
    CHECK(parakeet_capi_speaker_registry_add_embedding(reg, nullptr, a, 4) != 0);
    CHECK(parakeet_capi_speaker_registry_add_embedding(reg, "ee", nullptr, 4) != 0);
    CHECK(parakeet_capi_speaker_registry_add_embedding(reg, "ee", a, 0) != 0);
    CHECK(parakeet_capi_speaker_registry_add_embedding(reg, "ee", a, -1) != 0);
    CHECK(parakeet_capi_speaker_registry_add_embedding(nullptr, "ee", a, 4) != 0);
    CHECK(parakeet_capi_speaker_registry_size(reg) == 2);
    // A success clears the message.
    CHECK(parakeet_capi_speaker_registry_add_embedding(reg, "ben", b, 4) == 0);
    CHECK(std::strlen(parakeet_capi_speaker_registry_last_error(reg)) == 0);

    // A registry filled this way saves and loads like any other.
    const std::filesystem::path p = std::filesystem::temp_directory_path() / "pk_test_add_embedding.bin";
    std::filesystem::remove(p);
    CHECK(parakeet_capi_speaker_registry_save(reg, p.string().c_str()) == 0);
    parakeet_speaker_registry* back = parakeet_capi_speaker_registry_load(p.string().c_str());
    CHECK(back != nullptr && parakeet_capi_speaker_registry_size(back) == 2);
    // And a loaded registry keeps its dimension: a wrong-sized vector is refused.
    if (back) CHECK(parakeet_capi_speaker_registry_add_embedding(back, "cy", c3, 3) != 0);
    parakeet_capi_speaker_registry_free(back);
    std::filesystem::remove(p);

    parakeet_capi_speaker_registry_free(reg);

    // Encoder fingerprint (additive, ABI stays 10).
    {
        const char* fam = "voicedetect:ecapa_tdnn:ecapa:4";
        const char* w1 = "sha256:1111";
        parakeet_speaker_registry* f = parakeet_capi_speaker_registry_new();
        CHECK(std::strlen(parakeet_capi_speaker_registry_encoder_family(f)) == 0);
        CHECK(std::strlen(parakeet_capi_speaker_registry_encoder_weights(f)) == 0);
        CHECK(parakeet_capi_speaker_registry_add_embedding_fp(f, "ada", a, 4, fam, w1) == 0);
        CHECK(std::strcmp(parakeet_capi_speaker_registry_encoder_family(f), fam) == 0);
        CHECK(std::strcmp(parakeet_capi_speaker_registry_encoder_weights(f), w1) == 0);
        // Same family again, other weights: accepted.
        CHECK(parakeet_capi_speaker_registry_add_embedding_fp(f, "ben", b, 4, fam, "sha256:2222") == 0);
        CHECK(std::strcmp(parakeet_capi_speaker_registry_encoder_weights(f), w1) == 0);
        // Other family, and no fingerprint at all, are refused with a message.
        CHECK(parakeet_capi_speaker_registry_add_embedding_fp(f, "cy", a, 4, "voicedetect:campplus:c:4", w1) != 0);
        CHECK(std::strstr(parakeet_capi_speaker_registry_last_error(f), "campplus") != nullptr);
        CHECK(parakeet_capi_speaker_registry_add_embedding(f, "cy", a, 4) != 0);
        CHECK(parakeet_capi_speaker_registry_size(f) == 2);
        // NULL fingerprint arguments on the _fp call mean "none".
        parakeet_speaker_registry* n = parakeet_capi_speaker_registry_new();
        CHECK(parakeet_capi_speaker_registry_add_embedding_fp(n, "ada", a, 4, nullptr, nullptr) == 0);
        CHECK(std::strlen(parakeet_capi_speaker_registry_encoder_family(n)) == 0);
        // A fingerprint into a non-empty unfingerprinted registry is refused, not stamped.
        CHECK(parakeet_capi_speaker_registry_add_embedding_fp(n, "ben", b, 4, fam, w1) != 0);
        CHECK(parakeet_capi_speaker_registry_encoder_family(nullptr) != nullptr);
        // The fingerprint survives save and load.
        const std::filesystem::path q = std::filesystem::temp_directory_path() / "pk_test_fp_registry.bin";
        CHECK(parakeet_capi_speaker_registry_save(f, q.string().c_str()) == 0);
        parakeet_speaker_registry* back2 = parakeet_capi_speaker_registry_load(q.string().c_str());
        CHECK(back2 && std::strcmp(parakeet_capi_speaker_registry_encoder_family(back2), fam) == 0);
        CHECK(back2 && std::strcmp(parakeet_capi_speaker_registry_encoder_weights(back2), w1) == 0);
        // An unfingerprinted registry is still a v1 file.
        CHECK(parakeet_capi_speaker_registry_save(n, q.string().c_str()) == 0);
        { std::FILE* fh = std::fopen(q.string().c_str(), "rb"); unsigned char h[8] = {0};
          CHECK(fh && std::fread(h, 1, 8, fh) == 8 && h[4] == 1);
          if (fh) std::fclose(fh); }
        std::filesystem::remove(q);
        parakeet_capi_speaker_registry_set_strict(n, 1);   // no crash on NULL either
        parakeet_capi_speaker_registry_set_strict(nullptr, 1);
        CHECK(parakeet_capi_speaker_encoder_family(nullptr) == nullptr);
        CHECK(parakeet_capi_speaker_last_warning(nullptr) != nullptr);
        parakeet_capi_speaker_registry_free(back2);
        parakeet_capi_speaker_registry_free(n);
        parakeet_capi_speaker_registry_free(f);
    }
    if (failures) { std::fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    std::printf("test_capi_speaker_registry: PASS\n");
    return 0;
}
