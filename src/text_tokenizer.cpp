#include "text_tokenizer.h"

#include "gguf_model.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <string>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#include <winnls.h>
#endif

namespace omnivoice {

namespace {

struct Rune {
    char32_t value = 0;
    size_t begin = 0;
    size_t end = 0;
};

bool decode_utf8(const std::string& text, std::vector<Rune>& runes, std::string& error) {
    runes.clear();
    size_t i = 0;
    while (i < text.size()) {
        const size_t begin = i;
        const uint8_t first = static_cast<uint8_t>(text[i++]);
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
            error = "invalid UTF-8 leading byte at offset " + std::to_string(begin);
            return false;
        }
        if (i + static_cast<size_t>(continuation) > text.size()) {
            error = "truncated UTF-8 sequence at offset " + std::to_string(begin);
            return false;
        }
        for (int j = 0; j < continuation; ++j) {
            const uint8_t next = static_cast<uint8_t>(text[i++]);
            if ((next & 0xc0) != 0x80) {
                error = "invalid UTF-8 continuation byte at offset " + std::to_string(i - 1);
                return false;
            }
            value = (value << 6) | (next & 0x3f);
        }
        if ((continuation == 1 && value < 0x80) ||
            (continuation == 2 && value < 0x800) ||
            (continuation == 3 && value < 0x10000) ||
            value > 0x10ffff ||
            (value >= 0xd800 && value <= 0xdfff)) {
            error = "invalid UTF-8 code point at offset " + std::to_string(begin);
            return false;
        }
        runes.push_back({value, begin, i});
    }
    return true;
}

void append_utf8(std::string& output, char32_t value) {
    if (value <= 0x7f) {
        output.push_back(static_cast<char>(value));
    } else if (value <= 0x7ff) {
        output.push_back(static_cast<char>(0xc0 | (value >> 6)));
        output.push_back(static_cast<char>(0x80 | (value & 0x3f)));
    } else if (value <= 0xffff) {
        output.push_back(static_cast<char>(0xe0 | (value >> 12)));
        output.push_back(static_cast<char>(0x80 | ((value >> 6) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | (value & 0x3f)));
    } else {
        output.push_back(static_cast<char>(0xf0 | (value >> 18)));
        output.push_back(static_cast<char>(0x80 | ((value >> 12) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | ((value >> 6) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | (value & 0x3f)));
    }
}

Status normalize_nfc(const std::string& input, std::string& output) {
    output = input;
#if defined(_WIN32)
    if (input.empty()) {
        return Status::ok();
    }

    const int input_length = MultiByteToWideChar(
        CP_UTF8,
        MB_ERR_INVALID_CHARS,
        input.data(),
        static_cast<int>(input.size()),
        nullptr,
        0);
    if (input_length <= 0) {
        return Status::error("text is not valid UTF-8");
    }

    std::wstring wide(static_cast<size_t>(input_length), L'\0');
    if (MultiByteToWideChar(
            CP_UTF8,
            MB_ERR_INVALID_CHARS,
            input.data(),
            static_cast<int>(input.size()),
            wide.data(),
            input_length) != input_length) {
        return Status::error("failed to decode UTF-8 for NFC normalization");
    }

    const int normalized_length = NormalizeString(
        NormalizationC,
        wide.data(),
        input_length,
        nullptr,
        0);
    if (normalized_length <= 0) {
        return Status::error("failed to query NFC normalized text length");
    }

    std::wstring normalized(
        static_cast<size_t>(normalized_length),
        L'\0');
    const int written = NormalizeString(
        NormalizationC,
        wide.data(),
        input_length,
        normalized.data(),
        normalized_length);
    if (written <= 0) {
        return Status::error("failed to normalize text to NFC");
    }
    normalized.resize(static_cast<size_t>(written));

    const int utf8_length = WideCharToMultiByte(
        CP_UTF8,
        WC_ERR_INVALID_CHARS,
        normalized.data(),
        static_cast<int>(normalized.size()),
        nullptr,
        0,
        nullptr,
        nullptr);
    if (utf8_length <= 0) {
        return Status::error("failed to encode NFC normalized text as UTF-8");
    }

    output.assign(static_cast<size_t>(utf8_length), '\0');
    if (WideCharToMultiByte(
            CP_UTF8,
            WC_ERR_INVALID_CHARS,
            normalized.data(),
            static_cast<int>(normalized.size()),
            output.data(),
            utf8_length,
            nullptr,
            nullptr) != utf8_length) {
        return Status::error("failed to encode NFC normalized text as UTF-8");
    }
#endif
    return Status::ok();
}

const std::array<char32_t, 256>& byte_encoder() {
    static const std::array<char32_t, 256> table = [] {
        std::array<char32_t, 256> result{};
        std::array<bool, 256> assigned{};
        uint32_t next = 0;
        auto assign_range = [&](int begin, int end) {
            for (int byte = begin; byte <= end; ++byte) {
                result[static_cast<size_t>(byte)] = static_cast<char32_t>(byte);
                assigned[static_cast<size_t>(byte)] = true;
            }
        };
        assign_range(0x21, 0x7e);
        assign_range(0xa1, 0xac);
        assign_range(0xae, 0xff);
        for (size_t byte = 0; byte < result.size(); ++byte) {
            if (!assigned[byte]) {
                result[byte] = static_cast<char32_t>(256 + next++);
            }
        }
        return result;
    }();
    return table;
}

bool in_range(char32_t value, char32_t begin, char32_t end) {
    return value >= begin && value <= end;
}

bool is_combining_mark(char32_t value) {
    return in_range(value, 0x0300, 0x036f) ||
           in_range(value, 0x0483, 0x0489) ||
           in_range(value, 0x0591, 0x05bd) ||
           in_range(value, 0x0610, 0x061a) ||
           in_range(value, 0x064b, 0x065f) ||
           in_range(value, 0x1ab0, 0x1aff) ||
           in_range(value, 0x1dc0, 0x1dff) ||
           in_range(value, 0x20d0, 0x20ff) ||
           in_range(value, 0xfe20, 0xfe2f);
}

bool is_unicode_letter(char32_t value) {
    if (is_combining_mark(value)) {
        return false;
    }
    return in_range(value, 0x0041, 0x005a) ||
           in_range(value, 0x0061, 0x007a) ||
           value == 0x00aa || value == 0x00b5 || value == 0x00ba ||
           in_range(value, 0x00c0, 0x00d6) ||
           in_range(value, 0x00d8, 0x00f6) ||
           in_range(value, 0x00f8, 0x02c1) ||
           in_range(value, 0x02c6, 0x02d1) ||
           in_range(value, 0x02e0, 0x02e4) ||
           in_range(value, 0x0370, 0x0374) ||
           in_range(value, 0x0376, 0x0377) ||
           in_range(value, 0x037a, 0x037d) ||
           in_range(value, 0x037f, 0x03f5) ||
           in_range(value, 0x03f7, 0x0481) ||
           in_range(value, 0x048a, 0x052f) ||
           in_range(value, 0x0531, 0x0556) ||
           value == 0x0559 ||
           in_range(value, 0x0560, 0x0588) ||
           in_range(value, 0x05d0, 0x05ea) ||
           in_range(value, 0x05ef, 0x05f2) ||
           in_range(value, 0x0620, 0x064a) ||
           in_range(value, 0x066e, 0x066f) ||
           in_range(value, 0x0671, 0x06d3) ||
           value == 0x06d5 ||
           in_range(value, 0x06e5, 0x06e6) ||
           in_range(value, 0x06ee, 0x06ef) ||
           in_range(value, 0x06fa, 0x06fc) ||
           value == 0x06ff ||
           in_range(value, 0x0710, 0x072f) ||
           in_range(value, 0x074d, 0x07a5) ||
           value == 0x07b1 ||
           in_range(value, 0x07ca, 0x07ea) ||
           in_range(value, 0x0800, 0x0815) ||
           in_range(value, 0x0840, 0x0858) ||
           in_range(value, 0x0860, 0x086a) ||
           in_range(value, 0x0904, 0x0939) ||
           in_range(value, 0x0958, 0x0961) ||
           in_range(value, 0x0971, 0x0980) ||
           in_range(value, 0x0985, 0x09a8) ||
           in_range(value, 0x09aa, 0x09b0) ||
           in_range(value, 0x09b6, 0x09b9) ||
           in_range(value, 0x0a05, 0x0a0a) ||
           in_range(value, 0x0a0f, 0x0a10) ||
           in_range(value, 0x0a13, 0x0a28) ||
           in_range(value, 0x0a2a, 0x0a30) ||
           in_range(value, 0x0a32, 0x0a33) ||
           in_range(value, 0x0a35, 0x0a36) ||
           in_range(value, 0x0a38, 0x0a39) ||
           in_range(value, 0x0b05, 0x0b0c) ||
           in_range(value, 0x0b0f, 0x0b10) ||
           in_range(value, 0x0b13, 0x0b28) ||
           in_range(value, 0x0b2a, 0x0b30) ||
           in_range(value, 0x0b32, 0x0b33) ||
           in_range(value, 0x0b35, 0x0b39) ||
           in_range(value, 0x0b85, 0x0b8a) ||
           in_range(value, 0x0b8e, 0x0b90) ||
           in_range(value, 0x0b92, 0x0b95) ||
           in_range(value, 0x0ba8, 0x0baa) ||
           in_range(value, 0x0bae, 0x0bb9) ||
           in_range(value, 0x0c05, 0x0c0c) ||
           in_range(value, 0x0c0e, 0x0c10) ||
           in_range(value, 0x0c12, 0x0c28) ||
           in_range(value, 0x0c2a, 0x0c39) ||
           in_range(value, 0x0c85, 0x0c8c) ||
           in_range(value, 0x0c8e, 0x0c90) ||
           in_range(value, 0x0c92, 0x0ca8) ||
           in_range(value, 0x0caa, 0x0cb3) ||
           in_range(value, 0x0cb5, 0x0cb9) ||
           in_range(value, 0x0d05, 0x0d0c) ||
           in_range(value, 0x0d0e, 0x0d10) ||
           in_range(value, 0x0d12, 0x0d3a) ||
           in_range(value, 0x0d85, 0x0d96) ||
           in_range(value, 0x0d9a, 0x0db1) ||
           in_range(value, 0x0db3, 0x0dbb) ||
           in_range(value, 0x0dc0, 0x0dc6) ||
           in_range(value, 0x0e01, 0x0e30) ||
           in_range(value, 0x0e32, 0x0e33) ||
           in_range(value, 0x0e40, 0x0e46) ||
           in_range(value, 0x0e81, 0x0e82) ||
           in_range(value, 0x0e84, 0x0e84) ||
           in_range(value, 0x0e86, 0x0e8a) ||
           in_range(value, 0x0e8c, 0x0ea3) ||
           in_range(value, 0x0ea5, 0x0ea5) ||
           in_range(value, 0x0ea7, 0x0eb0) ||
           in_range(value, 0x0eb2, 0x0eb3) ||
           in_range(value, 0x0ebd, 0x0ebd) ||
           in_range(value, 0x0ec0, 0x0ec4) ||
           in_range(value, 0x0ec6, 0x0ec6) ||
           in_range(value, 0x0f40, 0x0f47) ||
           in_range(value, 0x0f49, 0x0f6c) ||
           in_range(value, 0x0f88, 0x0f8c) ||
           in_range(value, 0x1000, 0x102a) ||
           in_range(value, 0x1050, 0x1055) ||
           in_range(value, 0x10a0, 0x10c5) ||
           in_range(value, 0x10d0, 0x10fa) ||
           in_range(value, 0x1100, 0x11ff) ||
           in_range(value, 0x1200, 0x1248) ||
           in_range(value, 0x1250, 0x1256) ||
           in_range(value, 0x1258, 0x1258) ||
           in_range(value, 0x125a, 0x125d) ||
           in_range(value, 0x1260, 0x1288) ||
           in_range(value, 0x128a, 0x128d) ||
           in_range(value, 0x1290, 0x12b0) ||
           in_range(value, 0x12b2, 0x12b5) ||
           in_range(value, 0x12b8, 0x12be) ||
           in_range(value, 0x12c0, 0x12c0) ||
           in_range(value, 0x12c2, 0x12c5) ||
           in_range(value, 0x12c8, 0x12d6) ||
           in_range(value, 0x12d8, 0x1310) ||
           in_range(value, 0x1312, 0x1315) ||
           in_range(value, 0x1318, 0x135a) ||
           in_range(value, 0x1380, 0x138f) ||
           in_range(value, 0x13a0, 0x13f5) ||
           in_range(value, 0x13f8, 0x13fd) ||
           in_range(value, 0x1401, 0x166c) ||
           in_range(value, 0x166f, 0x167f) ||
           in_range(value, 0x1681, 0x169a) ||
           in_range(value, 0x16a0, 0x16ea) ||
           in_range(value, 0x16f1, 0x16f8) ||
           in_range(value, 0x1700, 0x170c) ||
           in_range(value, 0x170e, 0x1711) ||
           in_range(value, 0x1720, 0x1731) ||
           in_range(value, 0x1740, 0x1751) ||
           in_range(value, 0x1760, 0x176c) ||
           in_range(value, 0x176e, 0x1770) ||
           in_range(value, 0x1780, 0x17b3) ||
           in_range(value, 0x1820, 0x1878) ||
           in_range(value, 0x1880, 0x1884) ||
           in_range(value, 0x1887, 0x18a8) ||
           in_range(value, 0x18b0, 0x18f5) ||
           in_range(value, 0x1900, 0x191e) ||
           in_range(value, 0x1950, 0x196d) ||
           in_range(value, 0x1970, 0x1974) ||
           in_range(value, 0x1980, 0x19ab) ||
           in_range(value, 0x19b0, 0x19c9) ||
           in_range(value, 0x1a00, 0x1a16) ||
           in_range(value, 0x1a20, 0x1a54) ||
           in_range(value, 0x1b05, 0x1b33) ||
           in_range(value, 0x1b45, 0x1b4b) ||
           in_range(value, 0x1b83, 0x1ba0) ||
           in_range(value, 0x1bae, 0x1baf) ||
           in_range(value, 0x1bba, 0x1be5) ||
           in_range(value, 0x1c00, 0x1c23) ||
           in_range(value, 0x1c4d, 0x1c4f) ||
           in_range(value, 0x1c5a, 0x1c7d) ||
           in_range(value, 0x1c80, 0x1c88) ||
           in_range(value, 0x1c90, 0x1cba) ||
           in_range(value, 0x1cbd, 0x1cbf) ||
           in_range(value, 0x1ce9, 0x1cec) ||
           in_range(value, 0x1cee, 0x1cf3) ||
           in_range(value, 0x1cf5, 0x1cf6) ||
           in_range(value, 0x1d00, 0x1dbf) ||
           in_range(value, 0x1e00, 0x1f15) ||
           in_range(value, 0x1f18, 0x1f1d) ||
           in_range(value, 0x1f20, 0x1f45) ||
           in_range(value, 0x1f48, 0x1f4d) ||
           in_range(value, 0x1f50, 0x1f57) ||
           in_range(value, 0x1f59, 0x1f59) ||
           in_range(value, 0x1f5b, 0x1f5b) ||
           in_range(value, 0x1f5d, 0x1f5d) ||
           in_range(value, 0x1f5f, 0x1f7d) ||
           in_range(value, 0x1f80, 0x1fb4) ||
           in_range(value, 0x1fb6, 0x1fbc) ||
           in_range(value, 0x1fbe, 0x1fbe) ||
           in_range(value, 0x1fc2, 0x1fc4) ||
           in_range(value, 0x1fc6, 0x1fcc) ||
           in_range(value, 0x1fd0, 0x1fd3) ||
           in_range(value, 0x1fd6, 0x1fdb) ||
           in_range(value, 0x1fe0, 0x1fec) ||
           in_range(value, 0x1ff2, 0x1ff4) ||
           in_range(value, 0x1ff6, 0x1ffc) ||
           in_range(value, 0x2071, 0x2071) ||
           in_range(value, 0x207f, 0x207f) ||
           in_range(value, 0x2090, 0x209c) ||
           in_range(value, 0x2102, 0x2102) ||
           in_range(value, 0x2107, 0x2107) ||
           in_range(value, 0x210a, 0x2113) ||
           in_range(value, 0x2115, 0x2115) ||
           in_range(value, 0x2119, 0x211d) ||
           in_range(value, 0x2124, 0x2124) ||
           in_range(value, 0x2126, 0x2126) ||
           in_range(value, 0x2128, 0x2128) ||
           in_range(value, 0x212a, 0x212d) ||
           in_range(value, 0x212f, 0x2139) ||
           in_range(value, 0x213c, 0x213f) ||
           in_range(value, 0x2145, 0x2149) ||
           in_range(value, 0x214e, 0x214e) ||
           in_range(value, 0x2160, 0x2188) ||
           in_range(value, 0x2c00, 0x2c2f) ||
           in_range(value, 0x2c30, 0x2c5f) ||
           in_range(value, 0x2c60, 0x2ce4) ||
           in_range(value, 0x2ceb, 0x2cee) ||
           in_range(value, 0x2cf2, 0x2cf3) ||
           in_range(value, 0x2d00, 0x2d25) ||
           in_range(value, 0x2d27, 0x2d27) ||
           in_range(value, 0x2d2d, 0x2d2d) ||
           in_range(value, 0x2d30, 0x2d67) ||
           in_range(value, 0x2d6f, 0x2d6f) ||
           in_range(value, 0x2d80, 0x2d96) ||
           in_range(value, 0x2da0, 0x2da6) ||
           in_range(value, 0x2da8, 0x2dae) ||
           in_range(value, 0x2db0, 0x2db6) ||
           in_range(value, 0x2db8, 0x2dbe) ||
           in_range(value, 0x2dc0, 0x2dc6) ||
           in_range(value, 0x2dc8, 0x2dce) ||
           in_range(value, 0x2dd0, 0x2dd6) ||
           in_range(value, 0x2dd8, 0x2dde) ||
           in_range(value, 0x3005, 0x3006) ||
           in_range(value, 0x3031, 0x3035) ||
           in_range(value, 0x303b, 0x303c) ||
           in_range(value, 0x3041, 0x3096) ||
           in_range(value, 0x309d, 0x309f) ||
           in_range(value, 0x30a1, 0x30fa) ||
           in_range(value, 0x30fc, 0x30ff) ||
           in_range(value, 0x3105, 0x312f) ||
           in_range(value, 0x3131, 0x318e) ||
           in_range(value, 0x31a0, 0x31bf) ||
           in_range(value, 0x31f0, 0x31ff) ||
           in_range(value, 0x3400, 0x4dbf) ||
           in_range(value, 0x4e00, 0x9fff) ||
           in_range(value, 0xa000, 0xa48c) ||
           in_range(value, 0xa4d0, 0xa4fd) ||
           in_range(value, 0xa500, 0xa60c) ||
           in_range(value, 0xa610, 0xa61f) ||
           in_range(value, 0xa62a, 0xa62b) ||
           in_range(value, 0xa640, 0xa66e) ||
           in_range(value, 0xa67f, 0xa69d) ||
           in_range(value, 0xa6a0, 0xa6e5) ||
           in_range(value, 0xa717, 0xa71f) ||
           in_range(value, 0xa722, 0xa788) ||
           in_range(value, 0xa78b, 0xa7ff) ||
           in_range(value, 0xa800, 0xa801) ||
           in_range(value, 0xa803, 0xa805) ||
           in_range(value, 0xa807, 0xa80a) ||
           in_range(value, 0xa80c, 0xa822) ||
           in_range(value, 0xa840, 0xa873) ||
           in_range(value, 0xa882, 0xa8b3) ||
           in_range(value, 0xa8f2, 0xa8f7) ||
           in_range(value, 0xa8fb, 0xa8fb) ||
           in_range(value, 0xa8fd, 0xa8fe) ||
           in_range(value, 0xa90a, 0xa925) ||
           in_range(value, 0xa930, 0xa946) ||
           in_range(value, 0xa960, 0xa97c) ||
           in_range(value, 0xa984, 0xa9b2) ||
           in_range(value, 0xa9cf, 0xa9cf) ||
           in_range(value, 0xa9e0, 0xa9e4) ||
           in_range(value, 0xa9e6, 0xa9ef) ||
           in_range(value, 0xa9fa, 0xa9fe) ||
           in_range(value, 0xaa00, 0xaa28) ||
           in_range(value, 0xaa40, 0xaa42) ||
           in_range(value, 0xaa44, 0xaa4b) ||
           in_range(value, 0xaa60, 0xaa76) ||
           in_range(value, 0xaa7a, 0xaa7a) ||
           in_range(value, 0xaa7e, 0xaaaf) ||
           in_range(value, 0xaab1, 0xaab1) ||
           in_range(value, 0xaab5, 0xaab6) ||
           in_range(value, 0xaab9, 0xaabd) ||
           in_range(value, 0xaac0, 0xaac0) ||
           in_range(value, 0xaac2, 0xaac2) ||
           in_range(value, 0xaadb, 0xaadd) ||
           in_range(value, 0xaae0, 0xaaea) ||
           in_range(value, 0xaaf2, 0xaaf4) ||
           in_range(value, 0xab01, 0xab06) ||
           in_range(value, 0xab09, 0xab0e) ||
           in_range(value, 0xab11, 0xab16) ||
           in_range(value, 0xab20, 0xab26) ||
           in_range(value, 0xab28, 0xab2e) ||
           in_range(value, 0xab30, 0xab5a) ||
           in_range(value, 0xab5c, 0xab69) ||
           in_range(value, 0xab70, 0xabe2) ||
           in_range(value, 0xac00, 0xd7a3) ||
           in_range(value, 0xd7b0, 0xd7c6) ||
           in_range(value, 0xd7cb, 0xd7fb) ||
           in_range(value, 0xf900, 0xfa6d) ||
           in_range(value, 0xfa70, 0xfad9) ||
           in_range(value, 0xfb00, 0xfb06) ||
           in_range(value, 0xfb13, 0xfb17) ||
           in_range(value, 0xfb1d, 0xfb28) ||
           in_range(value, 0xfb2a, 0xfb36) ||
           in_range(value, 0xfb38, 0xfb3c) ||
           in_range(value, 0xfb3e, 0xfb3e) ||
           in_range(value, 0xfb40, 0xfb41) ||
           in_range(value, 0xfb43, 0xfb44) ||
           in_range(value, 0xfb46, 0xfbb1) ||
           in_range(value, 0xfbd3, 0xfd3d) ||
           in_range(value, 0xfd50, 0xfd8f) ||
           in_range(value, 0xfd92, 0xfdc7) ||
           in_range(value, 0xfdf0, 0xfdfb) ||
           in_range(value, 0xfe70, 0xfe74) ||
           in_range(value, 0xfe76, 0xfefc) ||
           in_range(value, 0xff21, 0xff3a) ||
           in_range(value, 0xff41, 0xff5a) ||
           in_range(value, 0xff66, 0xffbe) ||
           in_range(value, 0xffc2, 0xffc7) ||
           in_range(value, 0xffca, 0xffcf) ||
           in_range(value, 0xffd2, 0xffd7) ||
           in_range(value, 0xffda, 0xffdc) ||
           in_range(value, 0x10000, 0x1000b) ||
           in_range(value, 0x1000d, 0x10026) ||
           in_range(value, 0x10028, 0x1003a) ||
           in_range(value, 0x1003c, 0x1003d) ||
           in_range(value, 0x1003f, 0x1004d) ||
           in_range(value, 0x10050, 0x1005d) ||
           in_range(value, 0x10080, 0x100fa) ||
           in_range(value, 0x10280, 0x1029c) ||
           in_range(value, 0x102a0, 0x102d0) ||
           in_range(value, 0x10300, 0x1031f) ||
           in_range(value, 0x1032d, 0x10340) ||
           in_range(value, 0x10342, 0x10349) ||
           in_range(value, 0x10350, 0x10375) ||
           in_range(value, 0x10380, 0x1039d) ||
           in_range(value, 0x103a0, 0x103c3) ||
           in_range(value, 0x103c8, 0x103cf) ||
           in_range(value, 0x10400, 0x1049d) ||
           in_range(value, 0x104b0, 0x104d3) ||
           in_range(value, 0x104d8, 0x104fb) ||
           in_range(value, 0x10500, 0x10527) ||
           in_range(value, 0x10530, 0x10563) ||
           in_range(value, 0x10600, 0x10736) ||
           in_range(value, 0x10740, 0x10755) ||
           in_range(value, 0x10760, 0x10767) ||
           in_range(value, 0x10800, 0x10805) ||
           in_range(value, 0x10808, 0x10808) ||
           in_range(value, 0x1080a, 0x10835) ||
           in_range(value, 0x10837, 0x10838) ||
           in_range(value, 0x1083c, 0x1083c) ||
           in_range(value, 0x1083f, 0x10855) ||
           in_range(value, 0x10860, 0x10876) ||
           in_range(value, 0x10880, 0x1089e) ||
           in_range(value, 0x108e0, 0x108f2) ||
           in_range(value, 0x108f4, 0x108f5) ||
           in_range(value, 0x10900, 0x10915) ||
           in_range(value, 0x10920, 0x10939) ||
           in_range(value, 0x10980, 0x109b7) ||
           in_range(value, 0x109be, 0x109bf) ||
           in_range(value, 0x10a00, 0x10a00) ||
           in_range(value, 0x10a10, 0x10a13) ||
           in_range(value, 0x10a15, 0x10a17) ||
           in_range(value, 0x10a19, 0x10a35) ||
           in_range(value, 0x10a60, 0x10a7c) ||
           in_range(value, 0x10a80, 0x10a9c) ||
           in_range(value, 0x10ac0, 0x10ac7) ||
           in_range(value, 0x10ac9, 0x10ae4) ||
           in_range(value, 0x10b00, 0x10b35) ||
           in_range(value, 0x10b40, 0x10b55) ||
           in_range(value, 0x10b60, 0x10b72) ||
           in_range(value, 0x10b80, 0x10b91) ||
           in_range(value, 0x10c00, 0x10c48) ||
           in_range(value, 0x10c80, 0x10cb2) ||
           in_range(value, 0x10cc0, 0x10cf2) ||
           in_range(value, 0x10d00, 0x10d23) ||
           in_range(value, 0x10e80, 0x10ea9) ||
           in_range(value, 0x10eb0, 0x10eb1) ||
           in_range(value, 0x10f00, 0x10f1c) ||
           in_range(value, 0x10f27, 0x10f27) ||
           in_range(value, 0x10f30, 0x10f45) ||
           in_range(value, 0x10f70, 0x10f81) ||
           in_range(value, 0x10fb0, 0x10fc4) ||
           in_range(value, 0x10fe0, 0x10ff6) ||
           in_range(value, 0x11003, 0x11037) ||
           in_range(value, 0x11083, 0x110af) ||
           in_range(value, 0x110d0, 0x110e8) ||
           in_range(value, 0x11103, 0x11126) ||
           in_range(value, 0x11144, 0x11144) ||
           in_range(value, 0x11147, 0x11147) ||
           in_range(value, 0x11150, 0x11172) ||
           in_range(value, 0x11176, 0x11176) ||
           in_range(value, 0x11183, 0x111b2) ||
           in_range(value, 0x111c1, 0x111c4) ||
           in_range(value, 0x111da, 0x111da) ||
           in_range(value, 0x111dc, 0x111dc) ||
           in_range(value, 0x11200, 0x11211) ||
           in_range(value, 0x11213, 0x1122b) ||
           in_range(value, 0x1123f, 0x11240) ||
           in_range(value, 0x11280, 0x11286) ||
           in_range(value, 0x11288, 0x11288) ||
           in_range(value, 0x1128a, 0x1128d) ||
           in_range(value, 0x1128f, 0x1129d) ||
           in_range(value, 0x1129f, 0x112a8) ||
           in_range(value, 0x112b0, 0x112de) ||
           in_range(value, 0x11305, 0x1130c) ||
           in_range(value, 0x1130f, 0x11310) ||
           in_range(value, 0x11313, 0x11328) ||
           in_range(value, 0x1132a, 0x11330) ||
           in_range(value, 0x11332, 0x11333) ||
           in_range(value, 0x11335, 0x11339) ||
           in_range(value, 0x11350, 0x11350) ||
           in_range(value, 0x1135d, 0x11361) ||
           in_range(value, 0x11400, 0x11434) ||
           in_range(value, 0x11447, 0x1144a) ||
           in_range(value, 0x1145f, 0x11461) ||
           in_range(value, 0x11480, 0x114af) ||
           in_range(value, 0x114c4, 0x114c5) ||
           in_range(value, 0x114c7, 0x114c7) ||
           in_range(value, 0x11580, 0x115ae) ||
           in_range(value, 0x115d8, 0x115db) ||
           in_range(value, 0x11600, 0x1162f) ||
           in_range(value, 0x11644, 0x11644) ||
           in_range(value, 0x11680, 0x116aa) ||
           in_range(value, 0x116b8, 0x116b8) ||
           in_range(value, 0x11700, 0x1171a) ||
           in_range(value, 0x11740, 0x11746) ||
           in_range(value, 0x11800, 0x1182b) ||
           in_range(value, 0x118a0, 0x118df) ||
           in_range(value, 0x118ff, 0x11906) ||
           in_range(value, 0x11909, 0x11909) ||
           in_range(value, 0x1190c, 0x11913) ||
           in_range(value, 0x11915, 0x11916) ||
           in_range(value, 0x11918, 0x1192f) ||
           in_range(value, 0x1193f, 0x1193f) ||
           in_range(value, 0x11941, 0x11941) ||
           in_range(value, 0x119a0, 0x119a7) ||
           in_range(value, 0x119aa, 0x119d0) ||
           in_range(value, 0x119e1, 0x119e1) ||
           in_range(value, 0x11a00, 0x11a00) ||
           in_range(value, 0x11a0b, 0x11a32) ||
           in_range(value, 0x11a50, 0x11a50) ||
           in_range(value, 0x11a5c, 0x11a89) ||
           in_range(value, 0x11a9d, 0x11a9d) ||
           in_range(value, 0x11ab0, 0x11af8) ||
           in_range(value, 0x11c00, 0x11c08) ||
           in_range(value, 0x11c0a, 0x11c2e) ||
           in_range(value, 0x11c40, 0x11c40) ||
           in_range(value, 0x11c72, 0x11c8f) ||
           in_range(value, 0x11d00, 0x11d06) ||
           in_range(value, 0x11d0b, 0x11d30) ||
           in_range(value, 0x11d46, 0x11d46) ||
           in_range(value, 0x11d60, 0x11d65) ||
           in_range(value, 0x11d6a, 0x11d89) ||
           in_range(value, 0x11ee0, 0x11ef2) ||
           in_range(value, 0x11f02, 0x11f02) ||
           in_range(value, 0x11f04, 0x11f10) ||
           in_range(value, 0x11f12, 0x11f33) ||
           in_range(value, 0x12000, 0x12399) ||
           in_range(value, 0x12480, 0x12543) ||
           in_range(value, 0x12f90, 0x12ff0) ||
           in_range(value, 0x13000, 0x1342f) ||
           in_range(value, 0x13460, 0x143fa) ||
           in_range(value, 0x14400, 0x14646) ||
           in_range(value, 0x16800, 0x16a38) ||
           in_range(value, 0x16a40, 0x16a5e) ||
           in_range(value, 0x16a70, 0x16abe) ||
           in_range(value, 0x16ad0, 0x16aed) ||
           in_range(value, 0x16b00, 0x16b2f) ||
           in_range(value, 0x16b40, 0x16b43) ||
           in_range(value, 0x16b63, 0x16b77) ||
           in_range(value, 0x16b7d, 0x16b8f) ||
           in_range(value, 0x16e40, 0x16e7f) ||
           in_range(value, 0x16f00, 0x16f4a) ||
           in_range(value, 0x16f50, 0x16f50) ||
           in_range(value, 0x16f93, 0x16f9f) ||
           in_range(value, 0x16fe0, 0x16fe1) ||
           in_range(value, 0x16fe3, 0x16fe3) ||
           in_range(value, 0x17000, 0x187f7) ||
           in_range(value, 0x18800, 0x18cd5) ||
           in_range(value, 0x18d00, 0x18d08) ||
           in_range(value, 0x1aff0, 0x1aff3) ||
           in_range(value, 0x1aff5, 0x1affb) ||
           in_range(value, 0x1affd, 0x1affe) ||
           in_range(value, 0x1b000, 0x1b122) ||
           in_range(value, 0x1b132, 0x1b132) ||
           in_range(value, 0x1b150, 0x1b152) ||
           in_range(value, 0x1b155, 0x1b155) ||
           in_range(value, 0x1b164, 0x1b167) ||
           in_range(value, 0x1b170, 0x1b2fb) ||
           in_range(value, 0x1d400, 0x1d454) ||
           in_range(value, 0x1d456, 0x1d49c) ||
           in_range(value, 0x1d49e, 0x1d49f) ||
           in_range(value, 0x1d4a2, 0x1d4a2) ||
           in_range(value, 0x1d4a5, 0x1d4a6) ||
           in_range(value, 0x1d4a9, 0x1d4ac) ||
           in_range(value, 0x1d4ae, 0x1d4b9) ||
           in_range(value, 0x1d4bb, 0x1d4bb) ||
           in_range(value, 0x1d4bd, 0x1d4c3) ||
           in_range(value, 0x1d4c5, 0x1d505) ||
           in_range(value, 0x1d507, 0x1d50a) ||
           in_range(value, 0x1d50d, 0x1d514) ||
           in_range(value, 0x1d516, 0x1d51c) ||
           in_range(value, 0x1d51e, 0x1d539) ||
           in_range(value, 0x1d53b, 0x1d53e) ||
           in_range(value, 0x1d540, 0x1d544) ||
           in_range(value, 0x1d546, 0x1d546) ||
           in_range(value, 0x1d54a, 0x1d550) ||
           in_range(value, 0x1d552, 0x1d6a5) ||
           in_range(value, 0x1d6a8, 0x1d6c0) ||
           in_range(value, 0x1d6c2, 0x1d6da) ||
           in_range(value, 0x1d6dc, 0x1d6fa) ||
           in_range(value, 0x1d6fc, 0x1d714) ||
           in_range(value, 0x1d716, 0x1d734) ||
           in_range(value, 0x1d736, 0x1d74e) ||
           in_range(value, 0x1d750, 0x1d76e) ||
           in_range(value, 0x1d770, 0x1d788) ||
           in_range(value, 0x1d78a, 0x1d7a8) ||
           in_range(value, 0x1d7aa, 0x1d7c2) ||
           in_range(value, 0x1d7c4, 0x1d7cb) ||
           in_range(value, 0x1df00, 0x1df1e) ||
           in_range(value, 0x1df25, 0x1df2a) ||
           in_range(value, 0x1e030, 0x1e06d) ||
           in_range(value, 0x1e100, 0x1e12c) ||
           in_range(value, 0x1e137, 0x1e13d) ||
           in_range(value, 0x1e14e, 0x1e14e) ||
           in_range(value, 0x1e290, 0x1e2ad) ||
           in_range(value, 0x1e2c0, 0x1e2eb) ||
           in_range(value, 0x1e4d0, 0x1e4eb) ||
           in_range(value, 0x1e7e0, 0x1e7e6) ||
           in_range(value, 0x1e7e8, 0x1e7eb) ||
           in_range(value, 0x1e7ed, 0x1e7ee) ||
           in_range(value, 0x1e7f0, 0x1e7fe) ||
           in_range(value, 0x1e800, 0x1e8c4) ||
           in_range(value, 0x1e900, 0x1e943) ||
           in_range(value, 0x1e94b, 0x1e94b) ||
           in_range(value, 0x1ee00, 0x1ee03) ||
           in_range(value, 0x1ee05, 0x1ee1f) ||
           in_range(value, 0x1ee21, 0x1ee22) ||
           in_range(value, 0x1ee24, 0x1ee24) ||
           in_range(value, 0x1ee27, 0x1ee27) ||
           in_range(value, 0x1ee29, 0x1ee32) ||
           in_range(value, 0x1ee34, 0x1ee37) ||
           in_range(value, 0x1ee39, 0x1ee39) ||
           in_range(value, 0x1ee3b, 0x1ee3b) ||
           in_range(value, 0x1ee42, 0x1ee42) ||
           in_range(value, 0x1ee47, 0x1ee47) ||
           in_range(value, 0x1ee49, 0x1ee49) ||
           in_range(value, 0x1ee4b, 0x1ee4b) ||
           in_range(value, 0x1ee4d, 0x1ee4f) ||
           in_range(value, 0x1ee51, 0x1ee52) ||
           in_range(value, 0x1ee54, 0x1ee54) ||
           in_range(value, 0x1ee57, 0x1ee57) ||
           in_range(value, 0x1ee59, 0x1ee59) ||
           in_range(value, 0x1ee5b, 0x1ee5b) ||
           in_range(value, 0x1ee5d, 0x1ee5d) ||
           in_range(value, 0x1ee5f, 0x1ee5f) ||
           in_range(value, 0x1ee61, 0x1ee62) ||
           in_range(value, 0x1ee64, 0x1ee64) ||
           in_range(value, 0x1ee67, 0x1ee6a) ||
           in_range(value, 0x1ee6c, 0x1ee72) ||
           in_range(value, 0x1ee74, 0x1ee77) ||
           in_range(value, 0x1ee79, 0x1ee7c) ||
           in_range(value, 0x1ee7e, 0x1ee7e) ||
           in_range(value, 0x1ee80, 0x1ee89) ||
           in_range(value, 0x1ee8b, 0x1ee9b) ||
           in_range(value, 0x1eea1, 0x1eea3) ||
           in_range(value, 0x1eea5, 0x1eea9) ||
           in_range(value, 0x1eeab, 0x1eebb) ||
           in_range(value, 0x20000, 0x2a6df) ||
           in_range(value, 0x2a700, 0x2b739) ||
           in_range(value, 0x2b740, 0x2b81d) ||
           in_range(value, 0x2b820, 0x2cea1) ||
           in_range(value, 0x2ceb0, 0x2ebe0) ||
           in_range(value, 0x2f800, 0x2fa1d) ||
           in_range(value, 0x30000, 0x3134a) ||
           in_range(value, 0x31350, 0x323af);
}

bool is_unicode_number(char32_t value) {
    return in_range(value, 0x0030, 0x0039) ||
           in_range(value, 0x00b2, 0x00b3) ||
           value == 0x00b9 ||
           in_range(value, 0x00bc, 0x00be) ||
           in_range(value, 0x0660, 0x0669) ||
           in_range(value, 0x06f0, 0x06f9) ||
           in_range(value, 0x07c0, 0x07c9) ||
           in_range(value, 0x0966, 0x096f) ||
           in_range(value, 0x09e6, 0x09ef) ||
           in_range(value, 0x0a66, 0x0a6f) ||
           in_range(value, 0x0ae6, 0x0aef) ||
           in_range(value, 0x0b66, 0x0b6f) ||
           in_range(value, 0x0be6, 0x0bef) ||
           in_range(value, 0x0c66, 0x0c6f) ||
           in_range(value, 0x0ce6, 0x0cef) ||
           in_range(value, 0x0d66, 0x0d6f) ||
           in_range(value, 0x0de6, 0x0def) ||
           in_range(value, 0x0e50, 0x0e59) ||
           in_range(value, 0x0ed0, 0x0ed9) ||
           in_range(value, 0x0f20, 0x0f29) ||
           in_range(value, 0x1040, 0x1049) ||
           in_range(value, 0x1090, 0x1099) ||
           in_range(value, 0x17e0, 0x17e9) ||
           in_range(value, 0x1810, 0x1819) ||
           in_range(value, 0x1946, 0x194f) ||
           in_range(value, 0x19d0, 0x19d9) ||
           in_range(value, 0x1a80, 0x1a89) ||
           in_range(value, 0x1a90, 0x1a99) ||
           in_range(value, 0x1b50, 0x1b59) ||
           in_range(value, 0x1bb0, 0x1bb9) ||
           in_range(value, 0x1c40, 0x1c49) ||
           in_range(value, 0x1c50, 0x1c59) ||
           in_range(value, 0xa620, 0xa629) ||
           in_range(value, 0xa8d0, 0xa8d9) ||
           in_range(value, 0xa900, 0xa909) ||
           in_range(value, 0xa9d0, 0xa9d9) ||
           in_range(value, 0xa9f0, 0xa9f9) ||
           in_range(value, 0xaa50, 0xaa59) ||
           in_range(value, 0xabf0, 0xabf9) ||
           in_range(value, 0xff10, 0xff19) ||
           in_range(value, 0x104a0, 0x104a9) ||
           in_range(value, 0x10d30, 0x10d39) ||
           in_range(value, 0x11066, 0x1106f) ||
           in_range(value, 0x110f0, 0x110f9) ||
           in_range(value, 0x11136, 0x1113f) ||
           in_range(value, 0x111d0, 0x111d9) ||
           in_range(value, 0x112f0, 0x112f9) ||
           in_range(value, 0x11450, 0x11459) ||
           in_range(value, 0x114d0, 0x114d9) ||
           in_range(value, 0x11650, 0x11659) ||
           in_range(value, 0x116c0, 0x116c9) ||
           in_range(value, 0x11730, 0x11739) ||
           in_range(value, 0x118e0, 0x118e9) ||
           in_range(value, 0x11950, 0x11959) ||
           in_range(value, 0x11c50, 0x11c59) ||
           in_range(value, 0x11d50, 0x11d59) ||
           in_range(value, 0x11da0, 0x11da9) ||
           in_range(value, 0x16a60, 0x16a69) ||
           in_range(value, 0x16ac0, 0x16ac9) ||
           in_range(value, 0x16b50, 0x16b59) ||
           in_range(value, 0x1d7ce, 0x1d7ff) ||
           in_range(value, 0x1e140, 0x1e149) ||
           in_range(value, 0x1e2f0, 0x1e2f9) ||
           in_range(value, 0x1e4f0, 0x1e4f9) ||
           in_range(value, 0x1e950, 0x1e959);
}

bool is_whitespace(char32_t value) {
    return in_range(value, 0x0009, 0x000d) ||
           value == 0x0020 ||
           value == 0x0085 ||
           value == 0x00a0 ||
           value == 0x1680 ||
           in_range(value, 0x2000, 0x200a) ||
           value == 0x2028 ||
           value == 0x2029 ||
           value == 0x202f ||
           value == 0x205f ||
           value == 0x3000;
}

bool is_contraction_at(
    const std::vector<Rune>& runes,
    size_t position,
    std::initializer_list<char32_t> suffix) {
    if (position >= runes.size() || runes[position].value != '\'') {
        return false;
    }
    if (position + suffix.size() >= runes.size()) {
        return false;
    }
    size_t offset = 1;
    for (char32_t expected : suffix) {
        char32_t actual = runes[position + offset].value;
        if (actual >= U'A' && actual <= U'Z') {
            actual = actual - U'A' + U'a';
        }
        if (actual != expected) {
            return false;
        }
        ++offset;
    }
    return true;
}

size_t consume_letters(const std::vector<Rune>& runes, size_t begin) {
    size_t end = begin;
    while (end < runes.size() && is_unicode_letter(runes[end].value)) {
        ++end;
    }
    return end;
}

size_t consume_symbols(const std::vector<Rune>& runes, size_t begin) {
    size_t end = begin;
    while (end < runes.size()) {
        const char32_t value = runes[end].value;
        if (is_whitespace(value) ||
            is_unicode_letter(value) ||
            is_unicode_number(value)) {
            break;
        }
        ++end;
    }
    return end;
}

size_t consume_newlines(const std::vector<Rune>& runes, size_t begin) {
    size_t end = begin;
    while (end < runes.size()) {
        const char32_t value = runes[end].value;
        if (value != '\r' && value != '\n') {
            break;
        }
        ++end;
    }
    return end;
}

// Mirrors the first matching alternative of the Qwen ByteLevel pre-tokenizer:
// (?i:'s|'t|'re|'ve|'m|'ll|'d)
// |[^\r\n\p{L}\p{N}]?\p{L}+
// |\p{N}
// | ?[^\s\p{L}\p{N}]+[\r\n]*
// |\s*[\r\n]+
// |\s+(?!\S)
// |\s+
std::vector<std::string> pretokenize(const std::string& text) {
    std::vector<Rune> runes;
    std::string error;
    if (!decode_utf8(text, runes, error)) {
        return {};
    }

    std::vector<std::string> pieces;
    size_t position = 0;
    while (position < runes.size()) {
        const size_t begin = runes[position].begin;

        if (is_contraction_at(runes, position, {U's'}) ||
            is_contraction_at(runes, position, {U't'}) ||
            is_contraction_at(runes, position, {U'm'}) ||
            is_contraction_at(runes, position, {U'd'})) {
            pieces.emplace_back(text.substr(begin, runes[position + 1].end - begin));
            position += 2;
            continue;
        }
        if (is_contraction_at(runes, position, {U'r', U'e'}) ||
            is_contraction_at(runes, position, {U'v', U'e'}) ||
            is_contraction_at(runes, position, {U'l', U'l'})) {
            pieces.emplace_back(text.substr(begin, runes[position + 2].end - begin));
            position += 3;
            continue;
        }

        const char32_t current = runes[position].value;
        if (is_unicode_letter(current)) {
            const size_t end = consume_letters(runes, position);
            pieces.emplace_back(text.substr(begin, runes[end - 1].end - begin));
            position = end;
            continue;
        }
        if (position + 1 < runes.size() &&
            current != '\r' &&
            current != '\n' &&
            !is_unicode_number(current) &&
            is_unicode_letter(runes[position + 1].value)) {
            const size_t end = consume_letters(runes, position + 1);
            pieces.emplace_back(text.substr(begin, runes[end - 1].end - begin));
            position = end;
            continue;
        }

        if (is_unicode_number(current)) {
            pieces.emplace_back(text.substr(begin, runes[position].end - begin));
            ++position;
            continue;
        }

        size_t symbol_begin = std::string::npos;
        if (current == ' ' &&
            position + 1 < runes.size() &&
            !is_whitespace(runes[position + 1].value) &&
            !is_unicode_letter(runes[position + 1].value) &&
            !is_unicode_number(runes[position + 1].value)) {
            symbol_begin = position + 1;
        } else if (!is_whitespace(current) &&
                   !is_unicode_letter(current) &&
                   !is_unicode_number(current)) {
            symbol_begin = position;
        }
        if (symbol_begin != std::string::npos) {
            size_t end = consume_symbols(runes, symbol_begin);
            end = consume_newlines(runes, end);
            pieces.emplace_back(text.substr(begin, runes[end - 1].end - begin));
            position = end;
            continue;
        }

        if (is_whitespace(current)) {
            size_t whitespace_end = position;
            while (whitespace_end < runes.size() &&
                   is_whitespace(runes[whitespace_end].value)) {
                ++whitespace_end;
            }

            size_t newline_begin = position;
            while (newline_begin < whitespace_end &&
                   runes[newline_begin].value != '\r' &&
                   runes[newline_begin].value != '\n') {
                ++newline_begin;
            }
            if (newline_begin < whitespace_end) {
                const size_t end = consume_newlines(runes, newline_begin);
                pieces.emplace_back(text.substr(begin, runes[end - 1].end - begin));
                position = end;
                continue;
            }

            size_t end = whitespace_end;
            if (end < runes.size() && end - position > 1) {
                --end;
            }
            pieces.emplace_back(text.substr(begin, runes[end - 1].end - begin));
            position = end;
            continue;
        }

        pieces.emplace_back(text.substr(begin, runes[position].end - begin));
        ++position;
    }
    return pieces;
}

Status required_key(
    const GGUFModel& model,
    const std::string& key,
    gguf_type type) {
    if (!model.has_key(key)) {
        return Status::error("GGUF is missing required key: " + key);
    }
    const gguf_type actual = model.key_type(key);
    if (actual != type) {
        return Status::error("GGUF key has unexpected type: " + key);
    }
    return Status::ok();
}

Status required_array(
    const GGUFModel& model,
    const std::string& key,
    gguf_type element_type) {
    Status status = required_key(model, key, GGUF_TYPE_ARRAY);
    if (!status) {
        return status;
    }
    const gguf_context* raw = model.raw_gguf();
    const int64_t id = gguf_find_key(raw, key.c_str());
    if (id < 0 || gguf_get_arr_type(raw, id) != element_type) {
        return Status::error(
            "GGUF array has unexpected element type: " + key);
    }
    return Status::ok();
}

} // namespace

Status load_text_tokenizer_config_from_gguf(
    const GGUFModel& model,
    TextTokenizerConfig& config) {
    config = {};

    Status status = required_key(
        model,
        "omnivoice.schema_version",
        GGUF_TYPE_UINT32);
    if (!status) {
        return status;
    }
    if (model.get_u32("omnivoice.schema_version") != 1) {
        return Status::error(
            "unsupported OmniVoice GGUF schema version");
    }

    status = required_key(
        model,
        "tokenizer.ggml.model",
        GGUF_TYPE_STRING);
    if (!status) {
        return status;
    }
    if (model.get_string("tokenizer.ggml.model") != "qwen2") {
        return Status::error("GGUF text tokenizer is not Qwen2 BPE");
    }

    status = required_key(
        model,
        "omnivoice.text.vocab_size",
        GGUF_TYPE_UINT32);
    if (!status) {
        return status;
    }
    const uint32_t vocab_size =
        model.get_u32("omnivoice.text.vocab_size");
    if (vocab_size == 0 ||
        vocab_size > static_cast<uint32_t>(
            std::numeric_limits<int32_t>::max())) {
        return Status::error("GGUF tokenizer vocab_size is out of range");
    }
    config.vocab_size = static_cast<int32_t>(vocab_size);

    status = required_key(
        model,
        "tokenizer.ggml.padding_token_id",
        GGUF_TYPE_INT32);
    if (!status) {
        return status;
    }
    config.pad_token_id =
        model.get_i32("tokenizer.ggml.padding_token_id");

    status = required_key(
        model,
        "tokenizer.ggml.eos_token_id",
        GGUF_TYPE_INT32);
    if (!status) {
        return status;
    }
    config.eos_token_id =
        model.get_i32("tokenizer.ggml.eos_token_id");

    status = required_key(
        model,
        "omnivoice.text.nfc_normalization",
        GGUF_TYPE_BOOL);
    if (!status) {
        return status;
    }
    config.nfc_normalization =
        model.get_bool("omnivoice.text.nfc_normalization");

    status = required_array(
        model,
        "tokenizer.ggml.tokens",
        GGUF_TYPE_STRING);
    if (!status) {
        return status;
    }
    config.tokens = model.get_string_array("tokenizer.ggml.tokens");

    status = required_array(
        model,
        "tokenizer.ggml.merges",
        GGUF_TYPE_STRING);
    if (!status) {
        return status;
    }
    config.merges = model.get_string_array("tokenizer.ggml.merges");

    status = required_array(
        model,
        "tokenizer.ggml.added_token_names",
        GGUF_TYPE_STRING);
    if (!status) {
        return status;
    }
    config.added_token_names =
        model.get_string_array("tokenizer.ggml.added_token_names");

    status = required_array(
        model,
        "tokenizer.ggml.added_token_ids",
        GGUF_TYPE_INT32);
    if (!status) {
        return status;
    }
    config.added_token_ids =
        model.get_i32_array("tokenizer.ggml.added_token_ids");

    if (config.tokens.size() != static_cast<size_t>(config.vocab_size)) {
        return Status::error(
            "GGUF tokenizer token count does not match vocab_size");
    }
    return Status::ok();
}

Status TextTokenizer::load(
    const TextTokenizerConfig& config) {
    token_to_id_.clear();
    merge_ranks_.clear();
    added_tokens_.clear();
    tokens_.clear();
    nfc_normalization_ = config.nfc_normalization;
    pad_token_id_ = config.pad_token_id;
    eos_token_id_ = config.eos_token_id;

    const std::vector<std::string>& tokens = config.tokens;
    if (tokens.empty()) {
        return Status::error("text tokenizer vocabulary is empty");
    }
    if (config.vocab_size != 0 &&
        config.vocab_size != static_cast<int32_t>(tokens.size())) {
        return Status::error("text tokenizer vocab_size does not match tokens");
    }
    if (config.pad_token_id < -1 ||
        config.pad_token_id >= static_cast<int32_t>(tokens.size())) {
        return Status::error("text tokenizer pad_token_id is out of range");
    }
    if (config.eos_token_id < -1 ||
        config.eos_token_id >= static_cast<int32_t>(tokens.size())) {
        return Status::error("text tokenizer eos_token_id is out of range");
    }
    for (size_t id = 0; id < tokens.size(); ++id) {
        if (tokens[id].empty()) {
            return Status::error("empty token at ID " + std::to_string(id));
        }
        if (!token_to_id_.emplace(tokens[id], static_cast<int32_t>(id)).second) {
            return Status::error("duplicate tokenizer token: " + tokens[id]);
        }
    }

    const std::vector<std::string>& merges = config.merges;
    if (merges.empty()) {
        return Status::error("text tokenizer merges are empty");
    }
    for (size_t rank = 0; rank < merges.size(); ++rank) {
        const std::string& merge = merges[rank];
        if (merge.find(' ') == std::string::npos) {
            return Status::error("invalid BPE merge: " + merge);
        }
        if (!merge_ranks_.emplace(merge, static_cast<uint32_t>(rank)).second) {
            return Status::error("duplicate BPE merge: " + merge);
        }
    }

    if (config.added_token_names.size() != config.added_token_ids.size()) {
        return Status::error("added token name/id arrays have different sizes");
    }
    for (size_t i = 0; i < config.added_token_names.size(); ++i) {
        const std::string& name = config.added_token_names[i];
        const int32_t id = config.added_token_ids[i];
        if (id < 0 || static_cast<size_t>(id) >= tokens.size()) {
            return Status::error("added token ID out of range: " + name);
        }
        if (token_to_id_.find(name) == token_to_id_.end()) {
            return Status::error("added token is missing from token table: " + name);
        }
        added_tokens_.push_back({name, id});
    }
    std::sort(added_tokens_.begin(), added_tokens_.end(), [](const AddedToken& lhs, const AddedToken& rhs) {
        if (lhs.text.size() != rhs.text.size()) {
            return lhs.text.size() > rhs.text.size();
        }
        return lhs.id < rhs.id;
    });
    tokens_ = config.tokens;
    return Status::ok();
}

int32_t TextTokenizer::special_token_id(const std::string& name) const {
    const auto it = token_to_id_.find(name);
    return it == token_to_id_.end() ? -1 : it->second;
}

bool TextTokenizer::has_token(const std::string& token) const {
    return token_to_id_.find(token) != token_to_id_.end();
}

size_t TextTokenizer::vocab_size() const noexcept {
    return tokens_.size();
}

Status TextTokenizer::encode(
    const std::string& text,
    std::vector<int32_t>& token_ids) const {
    token_ids.clear();
    size_t position = 0;
    while (position < text.size()) {
        size_t next_position = std::string::npos;
        int32_t next_id = -1;
        size_t next_length = 0;
        for (const AddedToken& added : added_tokens_) {
            if (added.text.empty() || position + added.text.size() > text.size()) {
                continue;
            }
            if (text.compare(position, added.text.size(), added.text) == 0) {
                next_position = position;
                next_id = added.id;
                next_length = added.text.size();
                break;
            }
        }
        if (next_position != std::string::npos) {
            token_ids.push_back(next_id);
            position += next_length;
            continue;
        }

        size_t next_added = text.size();
        for (const AddedToken& added : added_tokens_) {
            const size_t found = text.find(added.text, position);
            if (found != std::string::npos && found < next_added) {
                next_added = found;
            }
        }
        Status status = encode_regular(text.substr(position, next_added - position), token_ids);
        if (!status) {
            return status;
        }
        position = next_added;
    }
    return Status::ok();
}

Status TextTokenizer::encode_regular(
    const std::string& text,
    std::vector<int32_t>& token_ids) const {
    if (text.empty()) {
        return Status::ok();
    }

    std::string normalized;
    if (nfc_normalization_) {
        Status status = normalize_nfc(text, normalized);
        if (!status) {
            return status;
        }
    } else {
        normalized = text;
    }

    std::string error;
    std::vector<Rune> runes;
    if (!decode_utf8(normalized, runes, error)) {
        return Status::error(error);
    }
    for (const std::string& piece : pretokenize(normalized)) {
        Status status = encode_piece(piece, token_ids);
        if (!status) {
            return status;
        }
    }
    return Status::ok();
}

Status TextTokenizer::encode_piece(
    const std::string& piece,
    std::vector<int32_t>& token_ids) const {
    const auto& encoder = byte_encoder();
    std::vector<std::string> symbols;
    symbols.reserve(piece.size());
    for (const unsigned char byte : piece) {
        std::string symbol;
        append_utf8(symbol, encoder[byte]);
        symbols.push_back(std::move(symbol));
    }

    while (symbols.size() > 1) {
        uint32_t best_rank = std::numeric_limits<uint32_t>::max();
        std::string best_pair;
        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
            const std::string pair = symbols[i] + " " + symbols[i + 1];
            const auto it = merge_ranks_.find(pair);
            if (it != merge_ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pair = pair;
            }
        }
        if (best_rank == std::numeric_limits<uint32_t>::max()) {
            break;
        }

        // Hugging Face BPE merges every non-overlapping occurrence of the
        // selected highest-priority pair before selecting the next pair.
        std::vector<std::string> merged;
        merged.reserve(symbols.size());
        for (size_t i = 0; i < symbols.size();) {
            if (i + 1 < symbols.size() &&
                symbols[i] + " " + symbols[i + 1] == best_pair) {
                merged.push_back(symbols[i] + symbols[i + 1]);
                i += 2;
            } else {
                merged.push_back(std::move(symbols[i]));
                ++i;
            }
        }
        symbols = std::move(merged);
    }

    for (const std::string& symbol : symbols) {
        const auto it = token_to_id_.find(symbol);
        if (it == token_to_id_.end()) {
            return Status::error("BPE symbol is missing from vocabulary");
        }
        token_ids.push_back(it->second);
    }
    return Status::ok();
}

} // namespace omnivoice
