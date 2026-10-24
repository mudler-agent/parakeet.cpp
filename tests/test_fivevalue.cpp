// Unit test for the five-value split (Phonon-2.5): the two ternary planes equal the
// dequantized weights exactly, the planes never overlap, malformed input is
// rejected, and the integer kernel run on both planes matches the float product
// of the dequantized weights with the dequantized activations. Model-independent.
#include "ternary.hpp"
#include "ternary_kernels.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <vector>

#include "ggml.h"

using namespace pk;

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL: %s (line %d)\n", #c, __LINE__); ++failures; } } while (0)

// Upstream container encoding, written independently of the loader: trits (5 per
// byte, base 3, least significant first, digit 1 = zero), then one flag bit per
// nonzero weight in row-major order, least significant bit first.
struct Encoded {
    std::vector<uint8_t> codes, hibits;
    std::vector<uint16_t> lo, hi;
    std::vector<int> sign;       // -1, 0, +1
    std::vector<char> is_hi;     // only meaningful where sign != 0
};

static Encoded make(int N, int K, unsigned seed, int pct_hi, int pct_zero) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> pct(0, 99), sg(0, 1);
    std::uniform_real_distribution<float> lo_r(0.01f, 0.2f), hi_r(0.2f, 0.9f);
    Encoded e;
    e.sign.assign((size_t)N * K, 0);
    e.is_hi.assign((size_t)N * K, 0);
    for (size_t i = 0; i < e.sign.size(); ++i) {
        if (pct(rng) < pct_zero) continue;
        e.sign[i] = sg(rng) ? 1 : -1;
        e.is_hi[i] = pct(rng) < pct_hi;
    }
    for (int n = 0; n < N; ++n) {
        e.lo.push_back(ggml_fp32_to_fp16(lo_r(rng)));
    }
    for (int n = 0; n < N; ++n) e.hi.push_back(ggml_fp32_to_fp16(hi_r(rng)));
    const int nb = (K + 4) / 5;
    e.codes.assign((size_t)N * nb, 0);
    for (int n = 0; n < N; ++n)
        for (int b = 0; b < nb; ++b) {
            int v = 0, mul = 1;
            for (int d = 0; d < 5; ++d) {
                const int k = b * 5 + d;
                const int digit = k < K ? e.sign[(size_t)n * K + k] + 1 : 1;
                v += digit * mul;
                mul *= 3;
            }
            e.codes[(size_t)n * nb + b] = (uint8_t)v;
        }
    size_t nnz = 0;
    for (size_t i = 0; i < e.sign.size(); ++i) nnz += e.sign[i] != 0;
    e.hibits.assign((nnz + 7) / 8, 0);
    size_t bit = 0;
    for (size_t i = 0; i < e.sign.size(); ++i)
        if (e.sign[i] != 0) {
            if (e.is_hi[i]) e.hibits[bit >> 3] |= (uint8_t)(1u << (bit & 7));
            ++bit;
        }
    return e;
}

static void run_case(int N, int K, int T, unsigned seed, int pct_hi, int pct_zero) {
    const Encoded e = make(N, K, seed, pct_hi, pct_zero);
    const int G = K / kTernaryGroup;
    TernaryWeight A, B;
    fivevalue_split(e.codes.data(), e.hibits.data(), e.hibits.size(), e.lo.data(), e.hi.data(), N, K, A, B);
    CHECK(A.N == N && A.K == K && B.N == N && B.K == K);

    std::vector<float> W;
    fivevalue_dequant(e.codes.data(), e.hibits.data(), e.hibits.size(), e.lo.data(), e.hi.data(), N, K, W);
    for (int n = 0; n < N; ++n) {
        const float lo = ggml_fp16_to_fp32(e.lo[n]), hi = ggml_fp16_to_fp32(e.hi[n]);
        for (int g = 0; g < G; ++g) {
            CHECK(A.scale(n, g) == lo);
            CHECK(B.scale(n, g) == hi);
        }
        for (int k = 0; k < K; ++k) {
            const size_t i = (size_t)n * K + k;
            const int ca = A.code(n, k), cb = B.code(n, k);
            CHECK(!(ca != 1 && cb != 1));                    // never both planes
            // lo * A + hi * B equals the dequantized weight, exactly
            const float w = lo * (float)(ca - 1) + hi * (float)(cb - 1);
            CHECK(w == W[i]);
            // and equals the definition: sign * (hi if flagged else lo)
            const float want = (float)e.sign[i] * (e.is_hi[i] ? hi : lo);
            CHECK(W[i] == want);
            CHECK((ca != 1) == (e.sign[i] != 0 && !e.is_hi[i]));
            CHECK((cb != 1) == (e.sign[i] != 0 && e.is_hi[i]));
        }
    }

    // kernel on both planes == float product with the dequantized activations
    std::mt19937 rng(seed + 1);
    std::uniform_real_distribution<float> xr(-3.0f, 3.0f);
    std::vector<float> x((size_t)T * K);
    for (auto& v : x) v = xr(rng);
    std::vector<uint8_t> act((size_t)T * ternary_act_row_bytes(K));
    ternary_quant_rows(x.data(), K, 0, T, act.data());
    std::vector<float> ya((size_t)T * N), yb((size_t)T * N);
    ternary_matmul_rows(A, act.data(), T, ya.data(), 0, N);
    ternary_matmul_rows(B, act.data(), T, yb.data(), 0, N);
    for (int t = 0; t < T; ++t) {
        const uint8_t* row = act.data() + (size_t)t * ternary_act_row_bytes(K);
        const int8_t* q = reinterpret_cast<const int8_t*>(row);
        const float sa = *reinterpret_cast<const float*>(row + K);
        for (int n = 0; n < N; ++n) {
            double ref = 0.0, mag = 0.0;
            for (int k = 0; k < K; ++k) {
                const double term = (double)W[(size_t)n * K + k] * (double)q[k] * (double)sa;
                ref += term;
                mag += std::fabs(term);
            }
            const double got = (double)ya[(size_t)t * N + n] + (double)yb[(size_t)t * N + n];
            CHECK(std::fabs(got - ref) <= 1e-5 * (mag + 1e-3));
        }
    }
}

template <class F> static bool throws(F f) {
    try {
        f();
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

int main() {
    run_case(32, 128, 3, 1, 30, 30);
    run_case(17, 256, 2, 2, 50, 10);      // N not a multiple of 16
    run_case(16, 384, 2, 3, 0, 50);       // no hi weights at all
    run_case(16, 128, 2, 4, 100, 50);     // every nonzero weight is hi
    run_case(5, 128, 1, 5, 30, 100);      // all zeros: empty flag stream
    run_case(48, 1024, 2, 6, 25, 35);     // a real width; 1024 is not a multiple of 5

    // malformed input is rejected
    const Encoded e = make(16, 128, 9, 30, 30);
    TernaryWeight A, B;
    CHECK(throws([&] {   // flag stream one byte short
        fivevalue_split(e.codes.data(), e.hibits.data(), e.hibits.size() - 1, e.lo.data(), e.hi.data(), 16, 128, A, B);
    }));
    std::vector<uint8_t> longer = e.hibits;
    longer.push_back(0);
    CHECK(throws([&] {   // flag stream one byte long
        fivevalue_split(e.codes.data(), longer.data(), longer.size(), e.lo.data(), e.hi.data(), 16, 128, A, B);
    }));
    std::vector<uint8_t> bad = e.codes;
    bad[7] = 243;
    CHECK(throws([&] {
        fivevalue_split(bad.data(), e.hibits.data(), e.hibits.size(), e.lo.data(), e.hi.data(), 16, 128, A, B);
    }));
    std::vector<uint16_t> nan_scale = e.lo;
    nan_scale[3] = 0x7e00;
    CHECK(throws([&] {
        fivevalue_split(e.codes.data(), e.hibits.data(), e.hibits.size(), nan_scale.data(), e.hi.data(), 16, 128, A, B);
    }));
    CHECK(!throws([&] {
        fivevalue_split(e.codes.data(), e.hibits.data(), e.hibits.size(), e.lo.data(), e.hi.data(), 16, 128, A, B);
    }));

    if (failures) {
        std::fprintf(stderr, "test_fivevalue: %d failure(s)\n", failures);
        return 1;
    }
    std::puts("test_fivevalue: OK");
    return 0;
}
