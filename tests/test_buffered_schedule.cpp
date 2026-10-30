// CPU-only test of the buffered streaming schedule (pk::BufferedSchedule), no
// checkpoint needed. The schedule is NeMo's streaming loop with the audio fed in
// blocks. For each preset and total length, and for several feed block sizes:
//   * the step edges equal NeMo's loop: e0 = min(chunk + right, S), e_{k+1} = min(e_k + chunk, S);
//   * every feed partition gives the same steps as feeding the whole clip at once;
//   * exactly one step is last, and it ends at the end of the audio (or there are no steps when S = 0);
//   * the window [win_start, edge) has the context length, and the samples added sum to S.
// The token-level check against NeMo is in test_unified_stream (checkpoint-gated).
#include "buffered_streaming.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL: %s (line %d)\n", #c, __LINE__); ++failures; } } while (0)

bool same(const std::vector<pk::BufferedStepPlan>& a, const std::vector<pk::BufferedStepPlan>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        const pk::BufferedStepPlan &x = a[i], &y = b[i];
        if (x.edge != y.edge || x.num != y.num || x.is_last != y.is_last || x.left != y.left ||
            x.chunk != y.chunk || x.right != y.right || x.win_start != y.win_start)
            return false;
    }
    return true;
}

// Steps for a whole clip of S samples, fed in blocks of `block` (S itself = all at once).
std::vector<pk::BufferedStepPlan> run_schedule(int L, int C, int R, int64_t S, int64_t block) {
    pk::BufferedSchedule sch(L, C, R);
    std::vector<pk::BufferedStepPlan> out;
    if (block <= 0) block = S;
    for (int64_t got = 0; got < S;) {
        got = std::min<int64_t>(S, got + block);
        sch.advance(got, out);
    }
    pk::BufferedStepPlan last;
    if (sch.finish(S, last)) out.push_back(last);
    return out;
}

}  // namespace

int main() {
    const int presets[][3] = {{70, 13, 13}, {70, 7, 7}, {70, 2, 5}, {70, 1, 3}, {70, 1, 1}, {25, 2, 5}};
    const int64_t lengths[] = {0, 1, 159, 160, 1279, 1280, 1281, 17920, 17921, 50000, 123457};
    const int64_t blocks[] = {1, 160, 1280, 7777, 0};   // 0 = the whole clip at once

    for (const auto& pr : presets) {
        const int L = pr[0], C = pr[1], R = pr[2];
        const int64_t F = 1280;
        for (int64_t S : lengths) {
            const std::vector<pk::BufferedStepPlan> ref = run_schedule(L, C, R, S, 0);

            // Edges as NeMo's loop computes them (clamped to the length).
            std::vector<int64_t> nemo_edges;
            if (S > 0) {
                int64_t e = std::min<int64_t>(C * F + R * F, S);
                while (true) {
                    nemo_edges.push_back(e);
                    if (e >= S) break;
                    e = std::min<int64_t>(e + C * F, S);
                }
            }
            CHECK(ref.size() == nemo_edges.size());
            if (ref.size() == nemo_edges.size())
                for (size_t k = 0; k < ref.size(); ++k) CHECK(ref[k].edge == nemo_edges[k]);

            // Only the last step is last, and it ends at the end of the audio.
            int n_last = 0;
            int64_t added = 0;
            for (size_t k = 0; k < ref.size(); ++k) {
                const pk::BufferedStepPlan& p = ref[k];
                if (p.is_last) ++n_last;
                added += p.num;
                CHECK(p.num > 0);
                CHECK(p.win_start >= 0);
                CHECK(p.edge - p.win_start == p.left + p.chunk + p.right);
                CHECK(p.left % F == 0 || p.is_last);
                CHECK(p.chunk % F == 0 || p.is_last);
                if (!p.is_last) CHECK(p.chunk == C * F);
            }
            if (S > 0) {
                CHECK(n_last == 1);
                CHECK(!ref.empty() && ref.back().is_last && ref.back().edge == S);
            } else {
                CHECK(ref.empty());
            }
            CHECK(added == S);

            // Feeding in any block size gives the same steps.
            for (int64_t b : blocks) {
                const auto got = run_schedule(L, C, R, S, b);
                if (!same(got, ref)) {
                    std::fprintf(stderr, "FAIL: preset %d/%d/%d, S=%lld, block=%lld: steps differ\n",
                                 L, C, R, (long long)S, (long long)b);
                    ++failures;
                }
            }
        }
    }

    if (failures == 0) std::printf("test_buffered_schedule: %zu presets x %zu lengths x %zu feed sizes match\n",
                                   sizeof(presets) / sizeof(presets[0]), sizeof(lengths) / sizeof(lengths[0]),
                                   sizeof(blocks) / sizeof(blocks[0]));
    return failures ? 1 : 0;
}
