// Single-thread throughput of each ternary kernel, next to ggml's own mul_mat
// with Q8_0 and F16 weights of the same shape (1 thread, activations F32, so the
// ggml rows include their per-call activation quantization; the ternary rows do
// not include ternary_quant_rows, which is timed on its own). Usage: bench_ternary [N K T reps]
#include <algorithm>
#include <atomic>
#include <thread>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ternary_kernels.hpp"

using namespace pk;

// Time ggml_mul_mat(W[K,N] of type wt, X[K,T] f32) on one thread; returns seconds per run.
static double time_ggml_mul_mat(ggml_type wt, int N, int K, int T, int reps, const std::vector<float>& x,
                                std::mt19937& rng) {
    ggml_init_params ip{};
    ip.mem_size = (size_t)N * K * 4 + (size_t)T * K * 4 + (size_t)T * N * 4 + (64u << 20);
    ip.no_alloc = false;
    ggml_context* ctx = ggml_init(ip);
    ggml_tensor* w = ggml_new_tensor_2d(ctx, wt, K, N);
    std::vector<float> wf((size_t)N * K);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    for (auto& v : wf) v = d(rng);
    if (wt == GGML_TYPE_F32) std::memcpy(w->data, wf.data(), ggml_nbytes(w));
    else if (wt == GGML_TYPE_F16) ggml_fp32_to_fp16_row(wf.data(), (ggml_fp16_t*)w->data, (int64_t)N * K);
    else ggml_quantize_chunk(wt, wf.data(), w->data, 0, N, K, nullptr);
    ggml_tensor* xt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, T);
    std::memcpy(xt->data, x.data(), ggml_nbytes(xt));
    ggml_tensor* y = ggml_mul_mat(ctx, w, xt);
    ggml_cgraph* g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, y);
    ggml_cplan plan = ggml_graph_plan(g, 1, nullptr);
    std::vector<uint8_t> work(plan.work_size);
    plan.work_data = work.data();
    ggml_graph_compute(g, &plan);  // warm up
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) ggml_graph_compute(g, &plan);
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / reps;
    ggml_free(ctx);
    return sec;
}

// repack mode: bench_ternary repack [weights=264]. Times the load-time repack of
// a Redux-sized weight set (alternating 4096x1024 and 1024x4096) with the old
// scalar reference, the new serial repack, and the new repack across a thread pool.
static void repack_reference_bench(const uint8_t* q, const uint16_t* s, int N, int K, TernaryWeight& out) {
    const int G = K / kTernaryGroup, nb = (K + 4) / 5;
    out.N = N;
    out.K = K;
    const int B = out.row_blocks();
    out.planes.assign((size_t)B * G * 512, 0);
    out.scales.assign((size_t)B * G * kTernaryRowBlock, 0.0f);
    std::vector<uint8_t> row(K);
    for (int n = 0; n < N; ++n) {
        const int b = n / kTernaryRowBlock, i = n % kTernaryRowBlock;
        for (int g = 0; g < G; ++g)
            out.scales[((size_t)b * G + g) * kTernaryRowBlock + i] = ggml_fp16_to_fp32(s[(size_t)n * G + g]);
        const uint8_t* qr = q + (size_t)n * nb;
        for (int bb = 0; bb < nb; ++bb) {
            int v = qr[bb];
            for (int d = 0; d < 5; ++d) {
                const int k = bb * 5 + d;
                if (k < K) row[k] = (uint8_t)(v % 3);
                v /= 3;
            }
        }
        for (int k = 0; k < K; ++k) {
            const int g = k / kTernaryGroup, r = k % kTernaryGroup;
            const int st = r / 16, p = (r % 16) / 4, j = r % 4;
            out.planes[((size_t)b * G + g) * 512 + (size_t)st * 64 + 4 * i + j] |= (uint8_t)(row[k] << (2 * p));
        }
    }
}

static int repack_bench(int total) {
    std::mt19937 rng(3);
    const int shapes[2][2] = {{4096, 1024}, {1024, 4096}};
    std::vector<uint8_t> q[2];
    std::vector<uint16_t> s[2];
    for (int k = 0; k < 2; ++k) {
        const int N = shapes[k][0], K = shapes[k][1];
        q[k].resize((size_t)N * ((K + 4) / 5));
        for (auto& b : q[k]) b = (uint8_t)(rng() % 243);
        s[k].assign((size_t)N * (K / kTernaryGroup), ggml_fp32_to_fp16(0.05f));
    }
    using clk = std::chrono::steady_clock;
    auto ms = [](clk::time_point a) { return std::chrono::duration<double, std::milli>(clk::now() - a).count(); };
    std::vector<TernaryWeight> out(total);
    for (int k = 0; k < 2; ++k) {
        TernaryWeight w;
        auto t0 = clk::now();
        repack_reference_bench(q[k].data(), s[k].data(), shapes[k][0], shapes[k][1], w);
        const double old_ms = ms(t0);
        TernaryWeight w2;
        t0 = clk::now();
        ternary_repack(q[k].data(), s[k].data(), shapes[k][0], shapes[k][1], w2);
        std::printf("per weight %dx%d: old %.2f ms, new %.2f ms, identical=%d\n", shapes[k][0], shapes[k][1], old_ms,
                    ms(t0), (int)(w.planes == w2.planes));
    }
    auto run = [&](bool old, int nt) {
        std::atomic<int> next{0};
        auto fn = [&] {
            for (int i; (i = next.fetch_add(1)) < total;) {
                const int k = i & 1;
                if (old) repack_reference_bench(q[k].data(), s[k].data(), shapes[k][0], shapes[k][1], out[i]);
                else ternary_repack(q[k].data(), s[k].data(), shapes[k][0], shapes[k][1], out[i]);
            }
        };
        const auto t0 = clk::now();
        std::vector<std::thread> th;
        for (int t = 1; t < nt; ++t) th.emplace_back(fn);
        fn();
        for (auto& t : th) t.join();
        return ms(t0);
    };
    auto report = [&](const char* name, bool old, int nt) {
        std::vector<double> v;
        for (int r = 0; r < 5; ++r) v.push_back(run(old, nt));
        std::sort(v.begin(), v.end());
        std::printf("total %d weights, %-22s threads=%d: min %.1f ms, median %.1f ms\n", total, name, nt, v[0], v[2]);
    };
    report("old reference", true, 1);
    report("new serial", false, 1);
    report("new pool", false, 4);
    report("new pool", false, 8);
    return 0;
}

int main(int argc, char** argv) {
    if (argc > 1 && !std::strcmp(argv[1], "repack")) return repack_bench(argc > 2 ? std::atoi(argv[2]) : 264);
    const int N = argc > 1 ? std::atoi(argv[1]) : 4096;
    const int K = argc > 2 ? std::atoi(argv[2]) : 1024;
    const int T = argc > 3 ? std::atoi(argv[3]) : 200;
    const int reps = argc > 4 ? std::atoi(argv[4]) : 20;
    std::mt19937 rng(1);
    const int nb = (K + 4) / 5, G = K / kTernaryGroup;
    std::vector<uint8_t> q((size_t)N * nb);
    for (auto& b : q) b = (uint8_t)(rng() % 243);
    std::vector<uint16_t> s((size_t)N * G, ggml_fp32_to_fp16(0.05f));
    TernaryWeight w;
    ternary_repack(q.data(), s.data(), N, K, w);
    std::vector<float> x((size_t)T * K);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    for (auto& v : x) v = d(rng);
    std::vector<uint8_t> act(ternary_act_row_bytes(K) * T);
    ternary_quant_rows(x.data(), K, 0, T, act.data());
    std::vector<float> y((size_t)T * N);
    {
        // activation quantization (op 1 of ternary_linear), per call
        ternary_quant_rows(x.data(), K, 0, T, act.data());
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) ternary_quant_rows(x.data(), K, 0, T, act.data());
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / reps;
        std::printf("quant    N=-    K=%d T=%d  %8.3f ms  (ternary_quant_rows, not in the kernel rows)\n", K, T,
                    sec * 1e3);
    }
    for (const TernaryKernel* k : ternary_all_kernels()) {
        // the scalar reference is 50x slower; a few runs are enough for it
        const int kr = std::strcmp(k->name, "scalar") ? reps : std::max(1, reps / 10);
        k->fn(w, act.data(), T, y.data(), 0, N);  // warm up
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < kr; ++r) k->fn(w, act.data(), T, y.data(), 0, N);
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / kr;
        std::printf("%-8s N=%d K=%d T=%d  %8.3f ms  %7.2f GMAC/s\n", k->name, N, K, T, sec * 1e3,
                    (double)N * K * T / sec / 1e9);
    }
    {
        // five-value linear: two planes (a second random weight stands in for plane hi), run one
        // after the other plus the add, against the fused kernel
        std::vector<uint8_t> q2((size_t)N * nb);
        for (auto& b : q2) b = (uint8_t)(rng() % 243);
        TernaryWeight w2;
        ternary_repack(q2.data(), s.data(), N, K, w2);
        std::vector<float> y2((size_t)T * N);
        for (const TernaryKernel* k : ternary_all_kernels()) {
            if (!k->fn2) continue;
            auto run_two = [&] {
                k->fn(w, act.data(), T, y.data(), 0, N);
                k->fn(w2, act.data(), T, y2.data(), 0, N);
                for (size_t i = 0; i < y.size(); ++i) y[i] += y2[i];
            };
            auto run_fused = [&] { k->fn2(w, w2, act.data(), T, y.data(), 0, N); };
            for (int rep = 0; rep < 3; ++rep) {   // interleaved, best of three
                static double best_two = 1e9, best_fused = 1e9;
                run_two();
                auto t0 = std::chrono::steady_clock::now();
                for (int r = 0; r < reps; ++r) run_two();
                best_two = std::min(best_two, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / reps);
                run_fused();
                t0 = std::chrono::steady_clock::now();
                for (int r = 0; r < reps; ++r) run_fused();
                best_fused = std::min(best_fused, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / reps);
                if (rep == 2)
                    std::printf("%-8s N=%d K=%d T=%d  two planes + add %8.3f ms   fused %8.3f ms   (best of 3)\n", k->name, N,
                                K, T, best_two * 1e3, best_fused * 1e3);
            }
        }
    }
    for (ggml_type wt : {GGML_TYPE_Q8_0, GGML_TYPE_F16}) {
        const double sec = time_ggml_mul_mat(wt, N, K, T, reps, x, rng);
        std::printf("ggml_%-3s N=%d K=%d T=%d  %8.3f ms  %7.2f GMAC/s\n", ggml_type_name(wt), N, K, T, sec * 1e3,
                    (double)N * K * T / sec / 1e9);
    }
    return 0;
}
