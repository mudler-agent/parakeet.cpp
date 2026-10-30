// CPU-only check that the buffered stream JSON (parakeet_capi_stream_info_json and
// parakeet_capi_stream_presets_json use these builders) does not depend on LC_NUMERIC.
// The numbers are written with '.' even under a comma-decimal locale. No checkpoint.
//
// The C-locale output is always checked. The comma-locale check needs a comma-decimal
// locale installed on the machine; without one the test exits 77 (skipped) after the
// C-locale checks pass.
#include "buffered_streaming.hpp"

#include <clocale>
#include <cstdio>
#include <cstddef>
#include <string>

namespace {

int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL: %s (line %d)\n", #c, __LINE__); ++failures; } } while (0)

// True when a digit, a comma and a digit follow each other (a decimal comma).
bool has_decimal_comma(const std::string& s) {
    for (size_t i = 1; i + 1 < s.size(); ++i)
        if (s[i] == ',' && s[i - 1] >= '0' && s[i - 1] <= '9' && s[i + 1] >= '0' && s[i + 1] <= '9')
            return true;
    return false;
}

// Writes the buffered JSON for a window under the current locale and checks the number format.
void check_json_numbers(const char* tag) {
    const std::string info = pk::buffered_info_json(70, 7, 7, 0.08);
    const std::string preset = pk::buffered_preset_json(0, 70, 7, 7, 0.08, true);
    std::fprintf(stderr, "[locale %s] %s\n[locale %s] %s\n", tag, info.c_str(), tag, preset.c_str());

    CHECK(info.find("\"frame_sec\":0.080000") != std::string::npos);
    CHECK(info.find("\"latency_sec\":1.120000") != std::string::npos);
    CHECK(info.find("\"step_sec\":0.560000") != std::string::npos);
    CHECK(info.find("\"first_step_samples\":17920") != std::string::npos);
    CHECK(info.find("\"step_samples\":8960") != std::string::npos);
    CHECK(preset.find("\"latency_sec\":1.12") != std::string::npos);
    CHECK(preset.find("\"default\":true") != std::string::npos);
    // Whatever the locale, no number has a decimal comma (commas only separate fields).
    CHECK(!has_decimal_comma(info));
    CHECK(!has_decimal_comma(preset));
    CHECK(pk::format_fixed(1.12, 2) == "1.12");
    CHECK(pk::format_fixed(0.08, 6) == "0.080000");
}

}  // namespace

int main() {
    std::setlocale(LC_NUMERIC, "C");
    check_json_numbers("C");

    // Comma-decimal locales to try, first one installed wins.
    const char* comma_locales[] = {"de_DE.UTF-8", "de_DE.utf8", "fr_FR.UTF-8", "fr_FR.utf8",
                                   "it_IT.UTF-8", "it_IT.utf8", "es_ES.UTF-8", "es_ES.utf8"};
    const char* chosen = nullptr;
    for (const char* name : comma_locales) {
        if (std::setlocale(LC_NUMERIC, name)) { chosen = name; break; }
    }
    if (failures) {
        std::setlocale(LC_NUMERIC, "C");
        std::fprintf(stderr, "test_buffered_locale: %d failure(s)\n", failures);
        return 1;
    }
    if (!chosen) {
        std::setlocale(LC_NUMERIC, "C");
        std::fprintf(stderr, "test_buffered_locale: no comma-decimal locale installed; C-locale checks passed, comma check skipped\n");
        return 77;
    }

    const std::lconv* lc = std::localeconv();
    CHECK(lc && lc->decimal_point && lc->decimal_point[0] == ',');
    // printf-style %f would write "0,080000" here; the builders must not.
    check_json_numbers(chosen);
    std::setlocale(LC_NUMERIC, "C");

    if (failures) {
        std::fprintf(stderr, "test_buffered_locale: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_buffered_locale: C and %s checks passed\n", chosen);
    return 0;
}
