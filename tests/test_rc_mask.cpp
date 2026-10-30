// Window-relative chunked-limited attention mask (NeMo att_context_style
// "chunked_limited_with_rc", unified checkpoints). CPU only, no checkpoint.
//
// tests/fixtures/rc_mask_nemo.txt holds NeMo's own ConformerEncoder._create_masks
// output for the unified presets, the misaligned 1.04 s grid (chunk 13 with a
// 70-frame left context), warm-up windows and a partial last window. See
// scripts/gen_rc_mask_fixture.py. For every valid query row, build_rc_mask must
// agree with NeMo on every key. Padded query rows are unmasked by design (their
// output is zeroed by the query mask), so they are only checked to be finite.
#include "relpos_attention.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL: %s (line %d)\n", #c, __LINE__); ++failures; } } while (0)

// Parse "case L C R T valid" blocks; each block is T rows of T '0'/'1' chars.
struct Case {
    int L, C, R, T, valid;
    std::vector<std::string> rows;   // rows[qi][kj] in {'0','1'}
};

bool load_cases(const std::string& path, std::vector<Case>& out) {
    std::ifstream in(path);
    if (!in) return false;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream hs(line);
        std::string tag;
        Case c{};
        hs >> tag >> c.L >> c.C >> c.R >> c.T >> c.valid;
        if (tag != "case") return false;
        c.rows.reserve(c.T);
        for (int qi = 0; qi < c.T; ++qi) {
            if (!std::getline(in, line) || (int)line.size() != c.T) return false;
            c.rows.push_back(line);
        }
        out.push_back(std::move(c));
    }
    return !out.empty();
}

}  // namespace

int main() {
    const char* env = std::getenv("PARAKEET_TEST_RC_MASK");
    const std::string path = env ? env : "tests/fixtures/rc_mask_nemo.txt";
    std::vector<Case> cases;
    if (!load_cases(path, cases)) {
        std::fprintf(stderr, "test_rc_mask: cannot read %s\n", path.c_str());
        return 1;
    }

    int total_rows = 0;
    for (const Case& c : cases) {
        std::vector<float> md((size_t)c.T * c.T, 0.0f);
        pk::RcMask rc{c.L, c.C, c.R};
        pk::build_rc_mask(md.data(), c.T, c.valid, rc);
        int bad = 0;
        for (int qi = 0; qi < c.T; ++qi) {
            for (int kj = 0; kj < c.T; ++kj) {
                const float v = md[(size_t)qi * c.T + kj];
                if (qi >= c.valid) {
                    // padded query row: unmasked by design
                    CHECK(v == 0.0f);
                    continue;
                }
                const bool ours = (v == 0.0f);
                const bool nemo = (c.rows[qi][kj] == '1');
                if (ours != nemo) ++bad;
            }
            if (qi < c.valid) ++total_rows;
        }
        if (bad) std::fprintf(stderr, "FAIL: L=%d C=%d R=%d T=%d valid=%d: %d key(s) disagree with NeMo\n",
                              c.L, c.C, c.R, c.T, c.valid, bad);
        failures += bad ? 1 : 0;
    }

    // Structural checks that do not need the fixture.
    {
        // Misaligned 1.04 s grid: chunk 13, left 70. Query 80 is in chunk 6 (cs = 78):
        // window [78-70, 78+12+13] = [8, 95] for T = 96, so key 7 is not visible.
        const int T = 96;
        std::vector<float> md((size_t)T * T, 0.0f);
        pk::build_rc_mask(md.data(), T, T, pk::RcMask{70, 13, 13});
        CHECK(md[(size_t)80 * T + 8] == 0.0f);
        CHECK(md[(size_t)80 * T + 95] == 0.0f);
        CHECK(md[(size_t)80 * T + 7] == -std::numeric_limits<float>::infinity());
        CHECK(md[(size_t)80 * T + 96 - 1] == 0.0f);
        // Every valid row sees itself (its own chunk lies inside its window).
        for (int qi = 0; qi < T; ++qi) CHECK(md[(size_t)qi * T + qi] == 0.0f);
        // Keys at or beyond valid_len are never visible from a valid row.
        std::vector<float> md2((size_t)T * T, 0.0f);
        pk::build_rc_mask(md2.data(), T, 40, pk::RcMask{70, 7, 7});
        for (int qi = 0; qi < 40; ++qi)
            for (int kj = 40; kj < T; ++kj) CHECK(md2[(size_t)qi * T + kj] == -std::numeric_limits<float>::infinity());
    }

    if (failures == 0)
        std::printf("test_rc_mask: %zu NeMo cases, %d valid query rows, all keys match\n",
                    cases.size(), total_rows);
    return failures ? 1 : 0;
}
