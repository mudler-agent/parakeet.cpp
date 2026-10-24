// Developer tool: dump the log-mel features and the encoder output of one or
// more wav files, as raw little-endian float32, to compare with a reference
// implementation (see docs/phonon.md).
//
//   dump_encoder [--mel-dir D] [--layers 0,11,23] <model.gguf> <out_dir> <a.wav> [b.wav ...]
//
// Per input, <out_dir>/<stem>.enc.f32 holds the encoder output row-major
// [d_model, T'] (time fastest), the same orientation as Encoder::forward,
// and <out_dir>/<stem>.mel.f32 holds the log-mel features this program used,
// row-major [n_mels, T]. With --mel-dir the features are read from
// D/<stem>.mel.f32 (same layout) instead of computed from the wav, so the
// encoder can be compared with a reference on identical input. With --layers
// the outputs of those encoder layers are written too, as <stem>.layerN.f32,
// row-major [T', d_model].
#include "audio_io.hpp"
#include "encoder.hpp"
#include "mel.hpp"
#include "model_loader.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static bool write_f32(const std::string& path, const std::vector<float>& v) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const bool ok = std::fwrite(v.data(), sizeof(float), v.size(), f) == v.size();
    return std::fclose(f) == 0 && ok;
}

static bool read_f32(const std::string& path, std::vector<float>& v) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    v.resize((size_t)n / sizeof(float));
    const bool ok = std::fread(v.data(), sizeof(float), v.size(), f) == v.size();
    std::fclose(f);
    return ok;
}

int main(int argc, char** argv) {
    std::string mel_dir;
    std::vector<int> layers;
    int a0 = 1;
    while (argc > a0 + 1 && (std::string(argv[a0]) == "--mel-dir" || std::string(argv[a0]) == "--layers")) {
        if (std::string(argv[a0]) == "--mel-dir") {
            mel_dir = argv[a0 + 1];
        } else {
            for (const char* p = argv[a0 + 1]; *p;) {
                char* end = nullptr;
                layers.push_back((int)std::strtol(p, &end, 10));
                p = *end == ',' ? end + 1 : end;
            }
        }
        a0 += 2;
    }
    if (argc - a0 < 3) {
        std::fprintf(stderr, "usage: dump_encoder [--mel-dir D] <model.gguf> <out_dir> <a.wav> [...]\n");
        return 2;
    }
    pk::ModelLoader ml;
    if (!ml.load(argv[a0])) return 1;
    pk::MelFrontend mel(ml);
    pk::Encoder enc(ml);
    const std::string out_dir = argv[a0 + 1];
    for (int i = a0 + 2; i < argc; ++i) {
        pk::Audio a;
        if (!pk::load_audio_16k_mono(argv[i], a)) {
            std::fprintf(stderr, "dump_encoder: cannot read %s\n", argv[i]);
            return 1;
        }
        std::string stem = argv[i];
        stem = stem.substr(stem.find_last_of('/') + 1);
        stem = stem.substr(0, stem.find_last_of('.'));
        std::vector<float> feats;
        int n_mels = 0, T = 0;
        mel.compute(a.samples, feats, n_mels, T);
        if (!write_f32(out_dir + "/" + stem + ".mel.f32", feats)) return 1;
        if (!mel_dir.empty()) {
            if (!read_f32(mel_dir + "/" + stem + ".mel.f32", feats) || feats.size() % (size_t)n_mels) {
                std::fprintf(stderr, "dump_encoder: cannot read features for %s\n", stem.c_str());
                return 1;
            }
            T = (int)(feats.size() / (size_t)n_mels);
        }
        std::vector<float> out;
        int d = 0, To = 0;
        if (layers.empty()) {
            enc.forward(feats, n_mels, T, out, d, To);
        } else {
            std::vector<std::vector<float>> lo;
            enc.forward_capture(feats, n_mels, T, out, d, To, layers, lo);
            for (size_t j = 0; j < layers.size(); ++j)
                if (!write_f32(out_dir + "/" + stem + ".layer" + std::to_string(layers[j]) + ".f32", lo[j])) return 1;
        }
        if (!write_f32(out_dir + "/" + stem + ".enc.f32", out)) return 1;
        std::printf("%s d_model=%d frames=%d\n", stem.c_str(), d, To);
    }
    return 0;
}
