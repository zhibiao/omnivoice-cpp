#include "duration_estimator.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#include <winnls.h>
#endif

namespace omnivoice {

namespace {

enum class ScriptWeight : uint8_t {
    Default,
    Latin,
    Greek,
    Cyrillic,
    Armenian,
    Hebrew,
    Arabic,
    Indic,
    ThaiLao,
    KhmerMyanmar,
    Hangul,
    Ethiopic,
    Georgian,
    Kana,
    Cjk,
    Yi,
};

struct ScriptRange {
    char32_t end;
    ScriptWeight weight;
};

constexpr std::array<ScriptRange, 88> SCRIPT_RANGES = {{
    {0x02AF, ScriptWeight::Latin},
    {0x03FF, ScriptWeight::Greek},
    {0x052F, ScriptWeight::Cyrillic},
    {0x058F, ScriptWeight::Armenian},
    {0x05FF, ScriptWeight::Hebrew},
    {0x077F, ScriptWeight::Arabic},
    {0x089F, ScriptWeight::Arabic},
    {0x08FF, ScriptWeight::Arabic},
    {0x097F, ScriptWeight::Indic},
    {0x09FF, ScriptWeight::Indic},
    {0x0A7F, ScriptWeight::Indic},
    {0x0AFF, ScriptWeight::Indic},
    {0x0B7F, ScriptWeight::Indic},
    {0x0BFF, ScriptWeight::Indic},
    {0x0C7F, ScriptWeight::Indic},
    {0x0CFF, ScriptWeight::Indic},
    {0x0D7F, ScriptWeight::Indic},
    {0x0DFF, ScriptWeight::Indic},
    {0x0EFF, ScriptWeight::ThaiLao},
    {0x0FFF, ScriptWeight::Indic},
    {0x109F, ScriptWeight::KhmerMyanmar},
    {0x10FF, ScriptWeight::Georgian},
    {0x11FF, ScriptWeight::Hangul},
    {0x137F, ScriptWeight::Ethiopic},
    {0x139F, ScriptWeight::Ethiopic},
    {0x13FF, ScriptWeight::Default},
    {0x167F, ScriptWeight::Default},
    {0x169F, ScriptWeight::Default},
    {0x16FF, ScriptWeight::Default},
    {0x171F, ScriptWeight::Default},
    {0x173F, ScriptWeight::Default},
    {0x175F, ScriptWeight::Default},
    {0x177F, ScriptWeight::Default},
    {0x17FF, ScriptWeight::KhmerMyanmar},
    {0x18AF, ScriptWeight::Default},
    {0x18FF, ScriptWeight::Default},
    {0x194F, ScriptWeight::Indic},
    {0x19DF, ScriptWeight::Indic},
    {0x19FF, ScriptWeight::KhmerMyanmar},
    {0x1A1F, ScriptWeight::Indic},
    {0x1AAF, ScriptWeight::Indic},
    {0x1B7F, ScriptWeight::Indic},
    {0x1BBF, ScriptWeight::Indic},
    {0x1BFF, ScriptWeight::Indic},
    {0x1C4F, ScriptWeight::Indic},
    {0x1C7F, ScriptWeight::Indic},
    {0x1C8F, ScriptWeight::Cyrillic},
    {0x1CBF, ScriptWeight::Georgian},
    {0x1CCF, ScriptWeight::Indic},
    {0x1CFF, ScriptWeight::Indic},
    {0x1D7F, ScriptWeight::Latin},
    {0x1DBF, ScriptWeight::Latin},
    {0x1DFF, ScriptWeight::Default},
    {0x1EFF, ScriptWeight::Latin},
    {0x309F, ScriptWeight::Kana},
    {0x30FF, ScriptWeight::Kana},
    {0x312F, ScriptWeight::Cjk},
    {0x318F, ScriptWeight::Hangul},
    {0x9FFF, ScriptWeight::Cjk},
    {0xA4CF, ScriptWeight::Yi},
    {0xA4FF, ScriptWeight::Default},
    {0xA63F, ScriptWeight::Default},
    {0xA69F, ScriptWeight::Cyrillic},
    {0xA6FF, ScriptWeight::Default},
    {0xA7FF, ScriptWeight::Latin},
    {0xA82F, ScriptWeight::Indic},
    {0xA87F, ScriptWeight::Default},
    {0xA8DF, ScriptWeight::Indic},
    {0xA8FF, ScriptWeight::Indic},
    {0xA92F, ScriptWeight::Indic},
    {0xA95F, ScriptWeight::Indic},
    {0xA97F, ScriptWeight::Hangul},
    {0xA9DF, ScriptWeight::Indic},
    {0xA9FF, ScriptWeight::KhmerMyanmar},
    {0xAA5F, ScriptWeight::Indic},
    {0xAA7F, ScriptWeight::KhmerMyanmar},
    {0xAADF, ScriptWeight::Indic},
    {0xAAFF, ScriptWeight::Indic},
    {0xAB2F, ScriptWeight::Ethiopic},
    {0xAB6F, ScriptWeight::Latin},
    {0xABBF, ScriptWeight::Default},
    {0xABFF, ScriptWeight::Indic},
    {0xD7AF, ScriptWeight::Hangul},
    {0xFAFF, ScriptWeight::Cjk},
    {0xFDFF, ScriptWeight::Arabic},
    {0xFE6F, ScriptWeight::Default},
    {0xFEFF, ScriptWeight::Arabic},
    {0xFFEF, ScriptWeight::Latin},
}};

struct Codepoint {
    char32_t value = 0;
    size_t begin = 0;
    size_t end = 0;
};

Status decode_utf8(
    const std::string& text,
    std::vector<Codepoint>& output) {
    output.clear();
    size_t offset = 0;
    while (offset < text.size()) {
        const size_t begin = offset;
        const uint8_t first = static_cast<uint8_t>(text[offset++]);
        char32_t value = 0;
        int continuation = 0;
        if (first < 0x80) {
            value = first;
        } else if ((first & 0xe0) == 0xc0) {
            value = first & 0x1f;
            continuation = 1;
        } else if ((first & 0xf0) == 0xe0) {
            value = first & 0x0f;
            continuation = 2;
        } else if ((first & 0xf8) == 0xf0) {
            value = first & 0x07;
            continuation = 3;
        } else {
            return Status::error("duration text has invalid UTF-8");
        }
        if (offset + static_cast<size_t>(continuation) > text.size()) {
            return Status::error("duration text has truncated UTF-8");
        }
        for (int index = 0; index < continuation; ++index) {
            const uint8_t next = static_cast<uint8_t>(text[offset++]);
            if ((next & 0xc0) != 0x80) {
                return Status::error(
                    "duration text has invalid UTF-8 continuation");
            }
            value = (value << 6) | (next & 0x3f);
        }
        if ((continuation == 1 && value < 0x80) ||
            (continuation == 2 && value < 0x800) ||
            (continuation == 3 && value < 0x10000) ||
            value > 0x10ffff ||
            (value >= 0xd800 && value <= 0xdfff)) {
            return Status::error("duration text has invalid code point");
        }
        output.push_back({value, begin, offset});
    }
    return Status::ok();
}

double script_weight(ScriptWeight weight) {
    switch (weight) {
    case ScriptWeight::Cjk:
        return 3.0;
    case ScriptWeight::Hangul:
        return 2.5;
    case ScriptWeight::Kana:
        return 2.2;
    case ScriptWeight::Ethiopic:
    case ScriptWeight::Yi:
        return 3.0;
    case ScriptWeight::Indic:
    case ScriptWeight::KhmerMyanmar:
        return 1.8;
    case ScriptWeight::ThaiLao:
        return 1.5;
    case ScriptWeight::Arabic:
    case ScriptWeight::Hebrew:
        return 1.5;
    default:
        return 1.0;
    }
}

ScriptWeight range_weight(char32_t value) {
    const auto it = std::lower_bound(
        SCRIPT_RANGES.begin(),
        SCRIPT_RANGES.end(),
        value,
        [](const ScriptRange& range, char32_t codepoint) {
            return range.end < codepoint;
        });
    return it == SCRIPT_RANGES.end()
        ? ScriptWeight::Default
        : it->weight;
}

#if defined(_WIN32)
void fill_utf16(char32_t value, WCHAR (&buffer)[2], int& length) {
    if (value <= 0xffff) {
        buffer[0] = static_cast<WCHAR>(value);
        length = 1;
        return;
    }
    value -= 0x10000;
    buffer[0] = static_cast<WCHAR>(0xd800 + (value >> 10));
    buffer[1] = static_cast<WCHAR>(0xdc00 + (value & 0x3ff));
    length = 2;
}

bool unicode_category_weight(char32_t value, double& weight) {
    WCHAR buffer[2] = {};
    int length = 0;
    fill_utf16(value, buffer, length);
    WORD type1 = 0;
    WORD type3 = 0;
    if (!GetStringTypeW(CT_CTYPE1, buffer, length, &type1) ||
        !GetStringTypeW(CT_CTYPE3, buffer, length, &type3)) {
        return false;
    }
    const bool nonspacing_mark =
        (type3 & (C3_NONSPACING | C3_DIACRITIC)) != 0 &&
        (type3 & C3_ALPHA) == 0;
    const bool spacing_mark =
        (type3 & C3_VOWELMARK) != 0 &&
        (type3 & C3_ALPHA) == 0;
    if (nonspacing_mark || spacing_mark) {
        weight = 0.0;
        return true;
    }
    if ((type1 & C1_PUNCT) != 0 ||
        (value >= 0x2000 && (type3 & C3_SYMBOL) != 0)) {
        weight = 0.5;
        return true;
    }
    if ((type1 & C1_SPACE) != 0) {
        weight = 0.2;
        return true;
    }
    if ((type1 & C1_DIGIT) != 0) {
        weight = 3.5;
        return true;
    }
    return false;
}
#endif

double character_weight(char32_t value) {
    if ((value >= U'A' && value <= U'Z') ||
        (value >= U'a' && value <= U'z')) {
        return 1.0;
    }
    if (value == U' ') {
        return 0.2;
    }
    if (value == 0x0640) {
        return 0.0;
    }
    if (value > 0xffff) {
        if (value >= 0x1f000 && value <= 0x1faff) {
            return 0.5;
        }
        if (value > 0x20000) {
            return 3.0;
        }
        return 1.0;
    }

#if defined(_WIN32)
    double category_weight = 0.0;
    if (unicode_category_weight(value, category_weight)) {
        return category_weight;
    }
#endif

    if (value > 0x20000) {
        return 3.0;
    }
    return script_weight(range_weight(value));
}

} // namespace

Status RuleDurationEstimator::calculate_total_weight(
    const std::string& text,
    double& weight) const {
    std::vector<Codepoint> codepoints;
    Status status = decode_utf8(text, codepoints);
    if (!status) {
        return status;
    }
    weight = 0.0;
    for (const Codepoint& codepoint : codepoints) {
        weight += character_weight(codepoint.value);
    }
    return Status::ok();
}

Status RuleDurationEstimator::estimate_duration(
    const std::string& target_text,
    const std::string& reference_text,
    double reference_duration,
    const DurationEstimateOptions& options,
    double& duration) const {
    duration = 0.0;
    if (reference_duration <= 0.0 || reference_text.empty()) {
        return Status::ok();
    }
    if (!std::isfinite(reference_duration) ||
        !std::isfinite(options.boost_strength) ||
        options.boost_strength <= 0.0) {
        return Status::error("invalid duration estimation parameters");
    }

    double reference_weight = 0.0;
    Status status = calculate_total_weight(
        reference_text,
        reference_weight);
    if (!status) {
        return status;
    }
    if (reference_weight == 0.0) {
        return Status::ok();
    }
    double target_weight = 0.0;
    status = calculate_total_weight(target_text, target_weight);
    if (!status) {
        return status;
    }

    const double speed_factor = reference_weight / reference_duration;
    duration = target_weight / speed_factor;
    if (options.low_threshold.has_value() &&
        *options.low_threshold > 0.0 &&
        duration < *options.low_threshold) {
        const double alpha = 1.0 / options.boost_strength;
        duration = *options.low_threshold *
            std::pow(duration / *options.low_threshold, alpha);
    }
    return Status::ok();
}

Status estimate_target_frames(
    const std::string& target_text,
    const std::string& reference_text,
    int32_t reference_frames,
    float speed,
    int32_t& target_frames,
    const RuleDurationEstimator& estimator) {
    target_frames = 0;
    if (!std::isfinite(speed) || speed <= 0.0f) {
        return Status::error("speed must be positive");
    }

    std::string effective_reference_text = reference_text;
    int32_t effective_reference_frames = reference_frames;
    if (reference_frames <= 0 || reference_text.empty()) {
        effective_reference_text = "Nice to meet you.";
        effective_reference_frames = 25;
    }

    double estimate = 0.0;
    Status status = estimator.estimate_duration(
        target_text,
        effective_reference_text,
        static_cast<double>(effective_reference_frames),
        {},
        estimate);
    if (!status) {
        return status;
    }
    if (speed != 1.0f) {
        estimate /= static_cast<double>(speed);
    }
    if (!std::isfinite(estimate)) {
        return Status::error("duration estimate is not finite");
    }
    target_frames = std::max(1, static_cast<int32_t>(estimate));
    return Status::ok();
}

} // namespace omnivoice
