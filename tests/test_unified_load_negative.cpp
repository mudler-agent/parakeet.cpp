// A unified GGUF (att_context_style chunked_limited_with_rc) whose preset block
// is malformed must be refused by the loader with a message, not run. Doctored
// copies keep the source's key-value section, without tensors (ModelLoader::load
// only parses the config), and are written next to the source file. Skips (77)
// when the env var is unset.
#include "model_loader.hpp"

#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <string>

#include "ggml.h"
#include "gguf.h"

using namespace pk;

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL: %s (line %d)\n", #c, __LINE__); ++failures; } } while (0)

// Removes the file on every exit path. The file sits next to the source GGUF.
struct TempFile {
    std::string path;
    TempFile(const std::string& src, const std::string& tag) {
        path = (std::filesystem::path(src).parent_path() /
                ("pk_unified_neg_" + std::to_string((long)getpid()) + "_" + tag + ".gguf")).string();
    }
    ~TempFile() { std::remove(path.c_str()); }
};

// Writes a KV-only copy of src (no tensors), then applies edit to it.
static bool doctor(const char* src, const TempFile& tf,
                   const std::function<void(gguf_context*)>& edit) {
    gguf_init_params p{/*no_alloc*/ true, nullptr};
    gguf_context* g = gguf_init_from_file(src, p);
    if (!g) return false;
    gguf_context* w = gguf_init_empty();
    gguf_set_kv(w, g);
    edit(w);
    const bool ok = gguf_write_to_file(w, tf.path.c_str(), /*only_meta*/ true);
    gguf_free(w);
    gguf_free(g);
    return ok;
}

// Replaces a key: gguf_set_val_* does not overwrite a copied key, so remove it first.
static void drop_key(gguf_context* w, const char* key) {
    gguf_remove_key(w, key);
}

static void expect_refused(const char* src, const std::string& tag,
                           const std::function<void(gguf_context*)>& edit) {
    TempFile tf(src, tag);
    CHECK(doctor(src, tf, edit));
    ModelLoader ml;
    const bool loaded = ml.load(tf.path);
    std::printf("%s: %s\n", tag.c_str(), loaded ? "LOADED (unexpected)" : "refused");
    CHECK(!loaded);
}

int main() {
    const char* src = std::getenv("PARAKEET_TEST_GGUF_UNIFIED");
    if (!src) {
        std::puts("skip: PARAKEET_TEST_GGUF_UNIFIED unset");
        return 77;
    }
    {
        // Control: an unedited copy through the same rebuild path must load.
        TempFile tf(src, "control");
        CHECK(doctor(src, tf, [](gguf_context*) {}));
        ModelLoader ml;
        CHECK(ml.load(tf.path));
        CHECK(ml.config().unified.present);
        CHECK(!ml.config().streaming.present);
    }
    {
        // A triple is missing a value: presets must be [L,C,R] triples.
        const int32_t bad[4] = {70, 7, 7, 70};
        expect_refused(src, "presets_not_triples", [&](gguf_context* w) {
            drop_key(w, "parakeet.unified.presets");
            gguf_set_arr_data(w, "parakeet.unified.presets", GGUF_TYPE_INT32, bad, 4);
        });
    }
    {
        // Chunk 0 is not a valid streaming window.
        const int32_t bad[3] = {70, 0, 7};
        expect_refused(src, "chunk_zero", [&](gguf_context* w) {
            drop_key(w, "parakeet.unified.presets");
            gguf_set_arr_data(w, "parakeet.unified.presets", GGUF_TYPE_INT32, bad, 3);
        });
    }
    {
        // The valid-length gate defaults off: without the key the file loads and the
        // mel is not trimmed. Only the converter's unified checkpoints set it.
        TempFile tf(src, "no_valid_length");
        CHECK(doctor(src, tf, [](gguf_context* w) { drop_key(w, "parakeet.unified.valid_length"); }));
        ModelLoader ml;
        CHECK(ml.load(tf.path));
        CHECK(ml.config().unified.present);
        CHECK(!ml.config().unified.valid_length);
    }
    expect_refused(src, "default_preset_out_of_range", [](gguf_context* w) {
        drop_key(w, "parakeet.unified.default_preset");
        gguf_set_val_u32(w, "parakeet.unified.default_preset", 99);
    });
    expect_refused(src, "unified_flag_unset", [](gguf_context* w) {
        drop_key(w, "parakeet.unified.present");
        gguf_set_val_bool(w, "parakeet.unified.present", false);
    });
    expect_refused(src, "presets_missing", [](gguf_context* w) {
        drop_key(w, "parakeet.unified.presets");
    });
    if (failures) { std::fprintf(stderr, "test_unified_load_negative: %d failure(s)\n", failures); return 1; }
    std::puts("test_unified_load_negative: OK");
    return 0;
}
