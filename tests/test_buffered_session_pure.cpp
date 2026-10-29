// CPU-only checks of the pure pieces behind pk::BufferedStreamingSession, with no
// checkpoint. Covers the text delta that take_new_text returns (prefix comparison,
// UTF-8 hold-back), and is extended by the other error-path cases in this file.
#include "buffered_streaming.hpp"

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

}  // namespace

int main() {
    test_text_delta();
    test_failure_latch();

    if (failures == 0) std::printf("test_buffered_session_pure: all checks passed\n");
    return failures ? 1 : 0;
}
