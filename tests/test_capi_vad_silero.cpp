// Silero VAD through the C-API: a context loaded from a Silero GGUF, the JSON
// document for both sample rates, options, errors, threads, and the streaming
// handle (same probabilities and regions as the one-shot call, any chunking).
//
// LABEL model; run from the project root (fixtures are relative).
// Env: PARAKEET_TEST_SILERO_GGUF  Silero VAD GGUF (skip 77 if unset)
#include "parakeet_capi.h"
#include "audio_io.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL: %s (line %d)\n", #c, __LINE__); ++failures; } } while (0)

struct Seg { double s, e; };

static std::string take(char* p) {
    if (!p) return std::string();
    std::string s(p);
    parakeet_capi_free_string(p);
    return s;
}

// Numbers after `"start":` / `"end":` in the "segments" array.
static std::vector<Seg> segments_of(const std::string& j) {
    std::vector<Seg> out;
    size_t pos = j.find("\"segments\":[");
    if (pos == std::string::npos) return out;
    const size_t end = j.find(']', pos);
    while (true) {
        pos = j.find("\"start\":", pos);
        if (pos == std::string::npos || pos > end) break;
        Seg g{};
        g.s = std::atof(j.c_str() + pos + 8);
        const size_t q = j.find("\"end\":", pos);
        g.e = std::atof(j.c_str() + q + 6);
        out.push_back(g);
        pos = q + 6;
    }
    return out;
}

// The values of the array that follows `key` (e.g. "probabilities":[...]).
static std::vector<double> array_of(const std::string& j, const char* key) {
    std::vector<double> out;
    size_t pos = j.find(std::string("\"") + key + "\":[");
    if (pos == std::string::npos) return out;
    pos += std::strlen(key) + 4;
    while (j[pos] != ']') {
        char* e = nullptr;
        out.push_back(std::strtod(j.c_str() + pos, &e));
        pos = (size_t)(e - j.c_str());
        if (j[pos] == ',') ++pos;
    }
    return out;
}

// "events":[{"type":"start","time":1.2},...] as (is_start, time).
static std::vector<std::pair<bool, double>> events_of(const std::string& j) {
    std::vector<std::pair<bool, double>> out;
    size_t pos = j.find("\"events\":[");
    if (pos == std::string::npos) return out;
    const size_t end = j.find(']', pos);
    while (true) {
        pos = j.find("\"type\":\"", pos);
        if (pos == std::string::npos || pos > end) break;
        const bool st = j.compare(pos + 8, 5, "start") == 0;
        const size_t q = j.find("\"time\":", pos);
        out.emplace_back(st, std::atof(j.c_str() + q + 7));
        pos = q + 7;
    }
    return out;
}

static double covered(const std::vector<Seg>& r, double a, double b) {
    double t = 0;
    for (const auto& g : r) t += std::max(0.0, std::min(g.e, b) - std::max(g.s, a));
    return t;
}

int main() {
    const char* silero = std::getenv("PARAKEET_TEST_SILERO_GGUF");
    if (!silero) { std::puts("skip: PARAKEET_TEST_SILERO_GGUF unset"); return 77; }

    pk::Audio sp;
    if (!pk::load_audio_16k_mono("tests/fixtures/speech.wav", sp)) { std::fprintf(stderr, "no fixture\n"); return 1; }
    const double sp_len = (double)sp.samples.size() / 16000.0;
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0.f, 0.0005f);
    auto silence = [&](double sec) { std::vector<float> v((size_t)(sec * 16000)); for (auto& x : v) x = nd(rng); return v; };
    std::vector<float> clip;
    auto add = [&](const std::vector<float>& v) { clip.insert(clip.end(), v.begin(), v.end()); };
    add(silence(2.0)); add(sp.samples); add(silence(2.5)); add(sp.samples); add(silence(2.0));
    const double a0 = 2.0, a1 = 2.0 + sp_len, b0 = a1 + 2.5, b1 = b0 + sp_len;
    const double total = (double)clip.size() / 16000.0;

    parakeet_ctx* ctx = parakeet_capi_load(silero);
    if (!ctx) { std::fprintf(stderr, "load failed\n"); return 1; }
    CHECK(parakeet_capi_model_kind(ctx) == PARAKEET_MODEL_KIND_VAD);
    CHECK(parakeet_capi_abi_version() == 11);

    // 1. Defaults: the document, regions on the original timeline.
    const std::string j = take(parakeet_capi_vad_pcm_json(ctx, clip.data(), (int)clip.size(), 16000, nullptr));
    std::printf("%s\n", j.c_str());
    CHECK(j.compare(0, 16, "{\"mode\":\"speech\"") == 0);
    CHECK(j.find("\"duration\":") != std::string::npos);
    CHECK(j.find("\"frame_sec\":0.032") != std::string::npos);
    CHECK(j.find("\"backend\":\"") != std::string::npos);
    CHECK(j.find("\"segments\":[") != std::string::npos);
    CHECK(j.find("\"probabilities\"") == std::string::npos);
    const std::vector<Seg> r = segments_of(j);
    CHECK(!r.empty());
    for (size_t i = 0; i < r.size(); ++i) {
        CHECK(r[i].e > r[i].s && r[i].s >= 0 && r[i].e <= total + 1e-6);
        if (i) CHECK(r[i].s >= r[i - 1].e);
    }
    const double tol = 0.3, tol_s = 0.7;  // speech.wav starts with about half a second of quiet
    CHECK(!r.empty() && std::fabs(r.front().s - a0) < tol_s && std::fabs(r.back().e - b1) < tol);
    CHECK(covered(r, a0, a1) > 0.7 * sp_len && covered(r, b0, b1) > 0.7 * sp_len);
    CHECK(covered(r, 0.0, a0 - tol) < 1e-9 && covered(r, a1 + tol, b0 - tol) < 1e-9 && covered(r, b1 + tol, total) < 1e-9);
    CHECK(r.size() >= 2);

    // 2. Probabilities: one per 32 ms chunk; padding changes only the edges.
    const std::string jp = take(parakeet_capi_vad_pcm_json(ctx, clip.data(), (int)clip.size(), 16000, "{\"probabilities\":true}"));
    const std::vector<double> p = array_of(jp, "probabilities");
    CHECK(p.size() == (size_t)std::ceil((double)clip.size() / 512.0));
    CHECK(segments_of(jp).size() == r.size());
    {
        double lo = 0, hi = 0; int nl = 0, nh = 0;
        for (size_t i = 0; i < p.size(); ++i) {
            const double t = (double)i * 0.032;
            if (t > 0.2 && t < a0 - 0.3) { lo += p[i]; ++nl; }
            if (t > a0 + 0.5 && t < a1 - 0.5) { hi += p[i]; ++nh; }
        }
        CHECK(nl > 0 && nh > 0 && lo / nl < 0.2 && hi / nh > 0.5);
    }
    const auto np = segments_of(take(parakeet_capi_vad_pcm_json(ctx, clip.data(), (int)clip.size(), 16000, "{\"speech_pad\":0}")));
    CHECK(np.size() == r.size());
    for (size_t i = 0; i < np.size() && i < r.size(); ++i)
        CHECK(std::fabs((np[i].s - r[i].s) - 0.03) < 0.0015 && std::fabs((r[i].e - np[i].e) - 0.03) < 0.0015);
    // A huge min_pause merges everything between the first and last speech.
    CHECK(segments_of(take(parakeet_capi_vad_pcm_json(ctx, clip.data(), (int)clip.size(), 16000, "{\"min_pause\":4}"))).size() == 1);
    // A threshold of 0.999 never gives more speech than the default.
    CHECK(covered(segments_of(take(parakeet_capi_vad_pcm_json(ctx, clip.data(), (int)clip.size(), 16000, "{\"threshold\":0.999}"))), 0, total)
          <= covered(r, 0, total) + 1e-9);

    // 3. "segments" mode: cuts of at most max_segment, whole short audio kept.
    {
        std::vector<float> longclip;
        for (int i = 0; i < 4; ++i) longclip.insert(longclip.end(), clip.begin(), clip.end());
        const std::string js = take(parakeet_capi_vad_pcm_json(ctx, longclip.data(), (int)longclip.size(), 16000,
                                                              "{\"mode\":\"segments\",\"max_segment\":15}"));
        CHECK(js.find("\"mode\":\"segments\"") != std::string::npos);
        const auto sg = segments_of(js);
        CHECK(sg.size() >= 4);
        for (const auto& g : sg) CHECK(g.e - g.s <= 15.0 + 1e-6);
        CHECK(covered(sg, 0, (double)longclip.size() / 16000.0) > 0.85 * 8 * sp_len);
        const auto w = segments_of(take(parakeet_capi_vad_pcm_json(ctx, clip.data(), (int)clip.size(), 16000, "{\"mode\":\"segments\"}")));
        CHECK(w.size() == 1 && w[0].s == 0.0 && std::fabs(w[0].e - total) < 0.002);
    }

    // 4. 8 kHz is taken as it is (frame_sec stays 0.032); other rates are resampled.
    {
        std::vector<float> c8;
        for (size_t i = 0; i + 1 < clip.size(); i += 2) c8.push_back(0.5f * (clip[i] + clip[i + 1]));
        const std::string j8 = take(parakeet_capi_vad_pcm_json(ctx, c8.data(), (int)c8.size(), 8000, "{\"probabilities\":true}"));
        CHECK(j8.find("\"frame_sec\":0.032") != std::string::npos);
        CHECK(array_of(j8, "probabilities").size() == (size_t)std::ceil((double)c8.size() / 256.0));
        const auto r8 = segments_of(j8);
        CHECK(r8.size() >= 2 && std::fabs(r8.front().s - a0) < tol_s && std::fabs(r8.back().e - b1) < tol);
        // 22.05 kHz: resampled to 16 kHz first, the timeline is kept.
        std::vector<float> c22 = pk::resample_linear(clip, 16000, 22050);
        const auto r22 = segments_of(take(parakeet_capi_vad_pcm_json(ctx, c22.data(), (int)c22.size(), 22050, nullptr)));
        CHECK(r22.size() >= 2 && std::fabs(r22.front().s - a0) < tol_s && std::fabs(r22.back().e - b1) < tol);
    }

    // 5. File variant equals PCM variant.
    {
        const std::string jf = take(parakeet_capi_vad_path_json(ctx, "tests/fixtures/speech.wav", nullptr));
        const std::string jq = take(parakeet_capi_vad_pcm_json(ctx, sp.samples.data(), (int)sp.samples.size(), 16000, nullptr));
        CHECK(!jf.empty() && jf == jq && !segments_of(jf).empty());
    }

    // 6. Edge cases and errors.
    {
        const std::string je = take(parakeet_capi_vad_pcm_json(ctx, nullptr, 0, 16000, nullptr));
        CHECK(!je.empty() && segments_of(je).empty());
        std::vector<float> tiny(100, 0.0f);   // less than one chunk
        const std::string jt = take(parakeet_capi_vad_pcm_json(ctx, tiny.data(), (int)tiny.size(), 16000, "{\"probabilities\":true}"));
        CHECK(!jt.empty() && array_of(jt, "probabilities").size() == 1 && segments_of(jt).empty());
        const char* bad_opts[] = {"{\"threshold\":0}", "{\"speech_pad\":-1}", "{\"mode\":\"x\"}", "{\"nope\":1}", "[1]", "{} x"};
        for (const char* o : bad_opts) {
            CHECK(parakeet_capi_vad_pcm_json(ctx, sp.samples.data(), (int)sp.samples.size(), 16000, o) == nullptr);
            CHECK(std::strlen(parakeet_capi_last_error(ctx)) > 0);
        }
        CHECK(parakeet_capi_vad_pcm_json(ctx, nullptr, 5, 16000, nullptr) == nullptr);
        CHECK(parakeet_capi_vad_pcm_json(ctx, sp.samples.data(), 5, 0, nullptr) == nullptr);
        CHECK(parakeet_capi_vad_path_json(ctx, nullptr, nullptr) == nullptr);
        CHECK(parakeet_capi_vad_path_json(ctx, "/nonexistent.wav", nullptr) == nullptr);
        CHECK(!take(parakeet_capi_vad_pcm_json(ctx, sp.samples.data(), (int)sp.samples.size(), 16000, "  {}  ")).empty());
        CHECK(std::strlen(parakeet_capi_last_error(ctx)) == 0);
        // Not an ASR context: the ASR entry points say so.
        CHECK(parakeet_capi_transcribe_path(ctx, "tests/fixtures/speech.wav", 0) == nullptr);
        CHECK(std::strlen(parakeet_capi_last_error(ctx)) > 0);
        // A Silero context is not an ASR context for the segmented transcription either.
        CHECK(parakeet_capi_transcribe_path_json_vad_with(ctx, ctx, "tests/fixtures/speech.wav", 0, nullptr) == nullptr);
    }

    // 7. Several threads on one context, same document as serial.
    {
        std::vector<std::string> out(4);
        std::vector<std::thread> th;
        for (int t = 0; t < 4; ++t)
            th.emplace_back([&, t] { out[t] = take(parakeet_capi_vad_pcm_json(ctx, clip.data(), (int)clip.size(), 16000, nullptr)); });
        for (auto& x : th) x.join();
        for (int t = 0; t < 4; ++t) CHECK(out[t] == j);
        std::vector<std::string> out2(3);
        th.clear();
        for (int t = 0; t < 3; ++t)
            th.emplace_back([&, t] {
                parakeet_ctx* c = parakeet_capi_load(silero);
                if (c) { out2[t] = take(parakeet_capi_vad_pcm_json(c, clip.data(), (int)clip.size(), 16000, nullptr)); parakeet_capi_free(c); }
            });
        for (auto& x : th) x.join();
        for (int t = 0; t < 3; ++t) CHECK(out2[t] == j);
    }

    // 8. Streaming.
    {
        const char* opts = "{\"probabilities\":true}";
        parakeet_vad_stream* s = parakeet_capi_vad_stream_begin(ctx, 16000, opts);
        CHECK(s != nullptr);
        CHECK(parakeet_capi_vad_stream_begin(ctx, 44100, nullptr) == nullptr);
        CHECK(std::strlen(parakeet_capi_last_error(ctx)) > 0);
        CHECK(parakeet_capi_vad_stream_begin(ctx, 16000, "{\"mode\":\"segments\"}") == nullptr);
        CHECK(parakeet_capi_vad_stream_begin(ctx, 16000, "{\"bogus\":1}") == nullptr);
        // run_gate is offline only: a stream refuses it, and run_gate 0 is fine.
        CHECK(parakeet_capi_vad_stream_begin(ctx, 16000, "{\"run_gate\":0.9}") == nullptr);
        CHECK(std::string(parakeet_capi_last_error(ctx)).find("run_gate") != std::string::npos);
        {
            parakeet_vad_stream* z = parakeet_capi_vad_stream_begin(ctx, 16000, "{\"run_gate\":0}");
            CHECK(z != nullptr);
            parakeet_capi_vad_stream_free(z);
        }

        auto run = [&](parakeet_vad_stream* st, unsigned seed, std::vector<double>* probs,
                       std::vector<std::pair<bool, double>>* ev) {
            std::mt19937 g(seed);
            size_t pos = 0;
            long long next_frame = 0;
            while (pos < clip.size()) {
                size_t n = std::min<size_t>(clip.size() - pos, g() % 4 == 0 ? 0 : 1 + g() % 5000);
                const bool last = pos + n >= clip.size();
                const std::string d = take(parakeet_capi_vad_stream_feed_json(st, clip.data() + pos, (int)n, last ? 1 : 0));
                CHECK(!d.empty());
                CHECK(d.find("\"first_frame\":" + std::to_string(next_frame) + ",") != std::string::npos);
                const auto pp = array_of(d, "probabilities");
                probs->insert(probs->end(), pp.begin(), pp.end());
                next_frame += (long long)pp.size();
                const auto e = events_of(d);
                ev->insert(ev->end(), e.begin(), e.end());
                pos += n;
            }
        };
        std::vector<double> sp1, sp2;
        std::vector<std::pair<bool, double>> ev1, ev2;
        run(s, 11, &sp1, &ev1);
        // Same probabilities as the one-shot call (4 printed decimals), whatever the chunking.
        CHECK(sp1.size() == p.size());
        bool same = sp1.size() == p.size();
        for (size_t i = 0; same && i < p.size(); ++i) same = std::fabs(sp1[i] - p[i]) < 1e-9;
        CHECK(same);
        // Events equal the regions of the one-shot "speech" mode (to the printed 3 decimals).
        CHECK(ev1.size() == 2 * r.size());
        for (size_t i = 0; i < r.size() && 2 * i + 1 < ev1.size(); ++i) {
            CHECK(ev1[2 * i].first && !ev1[2 * i + 1].first);
            CHECK(std::fabs(ev1[2 * i].second - r[i].s) < 1.5e-3 && std::fabs(ev1[2 * i + 1].second - r[i].e) < 1.5e-3);
        }
        // Finished: a further feed fails until reset(); then the same run repeats.
        CHECK(parakeet_capi_vad_stream_feed_json(s, clip.data(), 10, 0) == nullptr);
        CHECK(parakeet_capi_vad_stream_reset(s) == 0);
        run(s, 99, &sp2, &ev2);
        CHECK(sp2 == sp1 && ev2 == ev1);
        parakeet_capi_vad_stream_free(s);
        // Bad arguments.
        s = parakeet_capi_vad_stream_begin(ctx, 16000, nullptr);
        CHECK(s != nullptr);
        CHECK(parakeet_capi_vad_stream_feed_json(s, nullptr, 5, 0) == nullptr);
        CHECK(parakeet_capi_vad_stream_feed_json(s, clip.data(), -1, 0) == nullptr);
        const std::string d0 = take(parakeet_capi_vad_stream_feed_json(s, nullptr, 0, 1));   // empty stream
        CHECK(!d0.empty() && events_of(d0).empty() && d0.find("\"probabilities\"") == std::string::npos);
        parakeet_capi_vad_stream_free(s);
        // An 8 kHz stream, and two streams on two threads sharing the context.
        std::vector<float> c8;
        for (size_t i = 0; i + 1 < clip.size(); i += 2) c8.push_back(0.5f * (clip[i] + clip[i + 1]));
        std::vector<std::vector<std::pair<bool, double>>> evs(3);
        std::vector<std::thread> th;
        for (int t = 0; t < 3; ++t)
            th.emplace_back([&, t] {
                const bool r8 = t == 2;
                parakeet_vad_stream* st = parakeet_capi_vad_stream_begin(ctx, r8 ? 8000 : 16000, nullptr);
                if (!st) return;
                const std::vector<float>& src = r8 ? c8 : clip;
                for (size_t pos = 0; pos < src.size(); pos += 1777) {
                    const size_t n = std::min<size_t>(1777, src.size() - pos);
                    const std::string d = take(parakeet_capi_vad_stream_feed_json(st, src.data() + pos, (int)n, pos + n >= src.size()));
                    const auto e = events_of(d);
                    evs[t].insert(evs[t].end(), e.begin(), e.end());
                }
                parakeet_capi_vad_stream_free(st);
            });
        for (auto& x : th) x.join();
        CHECK(evs[0].size() == 2 * r.size() && evs[0] == evs[1]);
        CHECK(evs[2].size() >= 4 && evs[2].size() % 2 == 0 && std::fabs(evs[2].front().second - a0) < tol_s && std::fabs(evs[2].back().second - b1) < tol);
    }
    parakeet_capi_free(ctx);

    if (failures) return 1;
    std::puts("test_capi_vad_silero: OK");
    return 0;
}
