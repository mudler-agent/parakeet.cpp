// CPU-only checks of the pure pieces behind pk::BufferedStreamingSession, with no
// checkpoint. Covers the text delta that take_new_text returns (prefix comparison,
// UTF-8 hold-back), and is extended by the other error-path cases in this file.
#include "buffered_streaming.hpp"

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL: %s (line %d)\n", #c, __LINE__); ++failures; } } while (0)

// Feeds a sequence of full transcripts, one per step, and returns the concatenated output.
std::string emit_all(const std::vector<std::string>& steps, bool final_flush, std::vector<std::string>* deltas = nullptr) {
    std::string emitted, out;
    for (size_t i = 0; i < steps.size(); ++i) {
        const bool last = final_flush && i + 1 == steps.size();
        std::string d = pk::take_text_delta(emitted, steps[i], last);
        if (deltas) deltas->push_back(d);
        out += d;
    }
    return out;
}

void test_text_delta() {
    // Growing transcript: each step returns only the new bytes.
    {
        std::vector<std::string> deltas;
        const std::string out = emit_all({"hel", "hello", "hello wo", "hello world"}, false, &deltas);
        CHECK(out == "hello world");
        CHECK(deltas.size() == 4);
        if (deltas.size() == 4) {
            CHECK(deltas[0] == "hel");
            CHECK(deltas[1] == "lo");
            CHECK(deltas[2] == " wo");
            CHECK(deltas[3] == "rld");
        }
    }

    // No change: empty deltas, and repeated calls do not duplicate.
    {
        std::string emitted;
        CHECK(pk::take_text_delta(emitted, "abc", false) == "abc");
        CHECK(pk::take_text_delta(emitted, "abc", false).empty());
        CHECK(pk::take_text_delta(emitted, "", false).empty());
        CHECK(emitted == "abc");
    }

    // Two-byte character split across steps: "é" = C3 A9.
    {
        std::vector<std::string> deltas;
        const std::string out = emit_all({"caf\xC3", "caf\xC3\xA9 ok"}, false, &deltas);
        CHECK(out == "caf\xC3\xA9 ok");
        CHECK(deltas.size() == 2);
        if (deltas.size() == 2) {
            CHECK(deltas[0] == "caf");           // the lead byte is held back
            CHECK(deltas[1] == "\xC3\xA9 ok");
        }
    }

    // Four-byte character ("grinning face", F0 9F 98 80) split over three steps.
    {
        std::vector<std::string> deltas;
        const std::string out = emit_all({"\xF0", "\xF0\x9F\x98", "\xF0\x9F\x98\x80" "!"}, false, &deltas);
        CHECK(out == "\xF0\x9F\x98\x80" "!");
        CHECK(deltas.size() == 3);
        if (deltas.size() == 3) {
            CHECK(deltas[0].empty());
            CHECK(deltas[1].empty());
            CHECK(deltas[2] == "\xF0\x9F\x98\x80" "!");
        }
    }

    // End of stream flushes an incomplete trailing sequence instead of holding it forever.
    {
        std::vector<std::string> deltas;
        const std::string out = emit_all({"x\xC3"}, true, &deltas);
        CHECK(out == "x\xC3");
        CHECK(deltas.size() == 1);
    }

    // Revision under emitted bytes: nothing is emitted and nothing is duplicated.
    // Once the transcript agrees with the emitted prefix again, the rest is emitted.
    {
        std::string emitted = "abc";
        CHECK(pk::take_text_delta(emitted, "xyz", false).empty());
        CHECK(emitted == "abc");
        CHECK(pk::take_text_delta(emitted, "abc", false).empty());   // agrees, nothing new
        CHECK(pk::take_text_delta(emitted, "abcdef", false) == "def");
        CHECK(emitted == "abcdef");
    }

    // Transcript shorter than what was emitted (a truncation): hold back, no output.
    {
        std::string emitted = "abc";
        CHECK(pk::take_text_delta(emitted, "ab", false).empty());
        CHECK(pk::take_text_delta(emitted, "abcd", false) == "d");
    }

    // Revision that happens inside a held-back sequence still never splits a character.
    {
        std::string emitted;
        CHECK(pk::take_text_delta(emitted, "a\xE2\x82", false) == "a");   // euro sign, 2 of 3 bytes
        CHECK(pk::take_text_delta(emitted, "a\xE2\x82\xAC" "1", false) == "\xE2\x82\xAC" "1");
        CHECK(emitted == "a\xE2\x82\xAC" "1");
    }
}

// True when f() throws std::runtime_error.
template <class F>
bool throws_runtime_error(F&& f) {
    try {
        f();
    } catch (const std::runtime_error&) {
        return true;
    } catch (...) {
    }
    return false;
}

void test_failure_latch() {
    pk::StepFailureLatch latch;
    int ran = 0;

    // A normal call runs and leaves the latch clear.
    latch.run([&] { ++ran; });
    CHECK(ran == 1);
    CHECK(!latch.failed());

    // A throwing call rethrows the original exception and sets the latch.
    bool rethrown = false;
    try {
        latch.run([] { throw std::logic_error("step"); });
    } catch (const std::logic_error&) {
        rethrown = true;
    }
    CHECK(rethrown);
    CHECK(latch.failed());

    // Every later check and run throws std::runtime_error, and the function is not called.
    CHECK(throws_runtime_error([&] { latch.check(); }));
    CHECK(throws_runtime_error([&] { latch.run([&] { ++ran; }); }));
    CHECK(ran == 1);

    // clear() (what reset does) makes the latch usable again.
    latch.clear();
    CHECK(!latch.failed());
    latch.run([&] { ++ran; });
    CHECK(ran == 2);

    // A failure of a function that throws a runtime_error is also latched.
    CHECK(throws_runtime_error([&] { latch.run([] { throw std::runtime_error("boom"); }); }));
    CHECK(latch.failed());
    CHECK(throws_runtime_error([&] { latch.check(); }));
}

void test_window_bounds() {
    // Valid windows, including the limit itself.
    CHECK(pk::check_buffered_window(70, 13, 13).empty());
    CHECK(pk::check_buffered_window(0, 1, 0).empty());
    CHECK(pk::check_buffered_window(0, 4000, 0).empty());
    CHECK(pk::check_buffered_window(3000, 500, 500).empty());
    CHECK(pk::check_buffered_window(0, 1, 3999).empty());

    // Refused: chunk below 1, negative parts.
    CHECK(!pk::check_buffered_window(70, 0, 13).empty());
    CHECK(!pk::check_buffered_window(-1, 7, 7).empty());
    CHECK(!pk::check_buffered_window(70, 7, -1).empty());
    CHECK(!pk::check_buffered_window(0, -5, 0).empty());

    // Refused: one step over the total limit (4001 frames), and each part over it.
    CHECK(!pk::check_buffered_window(2000, 1000, 1001).empty());
    CHECK(!pk::check_buffered_window(0, 4001, 0).empty());
    CHECK(!pk::check_buffered_window(4001, 1, 0).empty());
    CHECK(!pk::check_buffered_window(0, 1, 4001).empty());

    // Overflow-sized values, as the C API (int fields) and the CLI (64-bit frames) can pass.
    const int64_t kI32 = 2147483647LL;
    CHECK(!pk::check_buffered_window(kI32, kI32, kI32).empty());
    CHECK(!pk::check_buffered_window(kI32, 1, kI32).empty());
    CHECK(!pk::check_buffered_window(0, kI32, 0).empty());
    CHECK(!pk::check_buffered_window(0, 1, kI32).empty());
    const int64_t kI64 = 9223372036854775807LL;
    CHECK(!pk::check_buffered_window(kI64, kI64, kI64).empty());
    CHECK(!pk::check_buffered_window(kI64, 1, 0).empty());
    CHECK(!pk::check_buffered_window(0, kI64, 0).empty());
    CHECK(!pk::check_buffered_window(0, 1, kI64).empty());
    CHECK(!pk::check_buffered_window(-kI64, 1, 0).empty());   // sum never formed: negative part refused
    // The limit check sums in 64 bits: two large parts that fit each on their own do not wrap.
    CHECK(!pk::check_buffered_window(kI64 / 2, kI64 / 2, kI64 / 2).empty());
}

// Schedule-level error paths, no model: zero-length input, sub-frame input, end of
// stream with nothing left, and a finish after the last step.
void test_schedule_edges() {
    // Zero-length: no step, no finish.
    {
        pk::BufferedSchedule sch(70, 7, 7);
        std::vector<pk::BufferedStepPlan> plans;
        sch.advance(0, plans);
        CHECK(plans.empty());
        pk::BufferedStepPlan last;
        CHECK(!sch.finish(0, last));
        CHECK(!sch.done());
        // A second end of stream on an empty schedule is still a no-op.
        CHECK(!sch.finish(0, last));
    }
    // Sub-frame input (less than one 80 ms frame, 1279 samples): no step, and finish gives one last step.
    {
        pk::BufferedSchedule sch(70, 7, 7);
        std::vector<pk::BufferedStepPlan> plans;
        sch.advance(1279, plans);
        CHECK(plans.empty());
        pk::BufferedStepPlan last;
        CHECK(sch.finish(1279, last));
        CHECK(last.is_last && last.edge == 1279 && last.num == 1279);
        CHECK(sch.done());
    }
    // After the last step the schedule is done: finish and advance add nothing.
    {
        pk::BufferedSchedule sch(70, 7, 7);
        std::vector<pk::BufferedStepPlan> plans;
        sch.advance(50000, plans);
        pk::BufferedStepPlan last;
        CHECK(sch.finish(50000, last));
        CHECK(sch.done());
        const size_t before = plans.size();
        sch.advance(60000, plans);
        CHECK(plans.size() == before);
        CHECK(!sch.finish(60000, last));
    }
}

}  // namespace

int main() {
    test_text_delta();
    test_failure_latch();
    test_window_bounds();
    test_schedule_edges();

    if (failures == 0) std::printf("test_buffered_session_pure: all checks passed\n");
    return failures ? 1 : 0;
}
