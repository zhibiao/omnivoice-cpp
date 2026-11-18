#include "gguf_model.h"
#include "text_tokenizer.h"
#include "text_tokenizer_internal.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

bool expect_equal(
    const std::vector<int32_t>& actual,
    const std::vector<int32_t>& expected,
    const char* label) {
    if (actual == expected) {
        return true;
    }
    std::cerr << label << " token mismatch\nexpected:";
    for (int32_t id : expected) {
        std::cerr << ' ' << id;
    }
    std::cerr << "\nactual:";
    for (int32_t id : actual) {
        std::cerr << ' ' << id;
    }
    std::cerr << '\n';
    return false;
}

bool expect_pieces_equal(
    const std::vector<std::string>& actual,
    const std::vector<std::string>& expected,
    const char* label) {
    if (actual == expected) {
        return true;
    }
    std::cerr << label << " piece mismatch\nexpected:";
    for (const std::string& piece : expected) {
        std::cerr << " [" << piece << ']';
    }
    std::cerr << "\nactual:";
    for (const std::string& piece : actual) {
        std::cerr << " [" << piece << ']';
    }
    std::cerr << '\n';
    return false;
}

bool test_unicode_helpers() {
    namespace unicode = omnivoice::text_tokenizer_internal;

    if (!unicode::is_unicode_letter(U'A') ||
        !unicode::is_unicode_letter(
            static_cast<char32_t>(0x00e9)) ||
        !unicode::is_unicode_letter(U'\u4e2d') ||
        !unicode::is_unicode_letter(
            static_cast<char32_t>(0x10400)) ||
        unicode::is_unicode_letter(U'1') ||
        unicode::is_unicode_letter(
            static_cast<char32_t>(0x0301)) ||
        unicode::is_unicode_letter(
            static_cast<char32_t>(0x2160))) {
        std::cerr << "Unicode letter classification mismatch\n";
        return false;
    }

    if (!unicode::is_unicode_number(U'1') ||
        !unicode::is_unicode_number(
            static_cast<char32_t>(0x0660)) ||
        !unicode::is_unicode_number(
            static_cast<char32_t>(0xff11)) ||
        !unicode::is_unicode_number(
            static_cast<char32_t>(0x2160)) ||
        !unicode::is_unicode_number(
            static_cast<char32_t>(0x00b2)) ||
        unicode::is_unicode_number(U'A') ||
        unicode::is_unicode_number(U'\u4e2d') ||
        unicode::is_unicode_number(
            static_cast<char32_t>(0x0301))) {
        std::cerr << "Unicode number classification mismatch\n";
        return false;
    }

    if (!unicode::is_whitespace(U'\t') ||
        !unicode::is_whitespace(U'\n') ||
        !unicode::is_whitespace(U' ') ||
        !unicode::is_whitespace(
            static_cast<char32_t>(0x00a0)) ||
        !unicode::is_whitespace(
            static_cast<char32_t>(0x3000)) ||
        unicode::is_whitespace(
            static_cast<char32_t>(0x200b)) ||
        unicode::is_whitespace(U'A') ||
        unicode::is_unicode_letter(
            static_cast<char32_t>(0x1f600)) ||
        unicode::is_unicode_number(
            static_cast<char32_t>(0x1f600))) {
        std::cerr << "Unicode classification mismatch\n";
        return false;
    }

    std::u32string codepoints;
    std::string error;
    const std::string mixed =
        std::string(u8"A") + u8"\u00e9" + u8"\u4e2d" + u8"\U0001f600";
    if (!unicode::decode_utf8(mixed, codepoints, error) ||
        codepoints.size() != 4 ||
        codepoints[0] != U'A' ||
        codepoints[1] != static_cast<char32_t>(0x00e9) ||
        codepoints[2] != U'\u4e2d' ||
        codepoints[3] != static_cast<char32_t>(0x1f600) ||
        unicode::utf32_slice_to_utf8(codepoints, 0, codepoints.size()) !=
            mixed) {
        std::cerr << "Unicode UTF-8/UTF-32 round trip mismatch\n";
        return false;
    }

    const std::vector<std::string> invalid_inputs = {
        std::string(1, static_cast<char>(0xff)),
        std::string("\xc0\xaf", 2),
        std::string("\xed\xa0\x80", 3),
        std::string("\xe2\x82", 2),
        std::string("\xe2\x41\xa1", 3),
    };
    for (const std::string& invalid : invalid_inputs) {
        if (unicode::decode_utf8(invalid, codepoints, error) ||
            error.find("byte offset") == std::string::npos) {
            std::cerr << "invalid UTF-8 was not reported correctly\n";
            return false;
        }
    }

    const std::u32string all_scalars = [] {
        std::u32string result;
        result.reserve(0x110000 - 0x800);
        for (char32_t value = 0; value <= 0x10ffff; ++value) {
            if (value < 0xd800 || value > 0xdfff) {
                result.push_back(value);
            }
        }
        return result;
    }();
    const std::string all_scalars_utf8 =
        unicode::utf32_slice_to_utf8(
            all_scalars,
            0,
            all_scalars.size());
    if (!unicode::decode_utf8(
            all_scalars_utf8,
            codepoints,
            error) ||
        codepoints != all_scalars) {
        std::cerr << "all Unicode scalar values failed UTF round trip\n";
        return false;
    }

    const std::vector<std::pair<std::string, std::string>>
        normalization_cases = {
            {u8"\u00e9", u8"e\u0301"},
            {u8"\u00c5", u8"A\u030a"},
            {u8"\u00c5", u8"\u212b"},
            {u8"\uac00", u8"\u1100\u1161"},
            {u8"\u304c", u8"\u304b\u3099"},
            {u8"\u1e0a", u8"D\u0307"},
        };
    for (const auto& item : normalization_cases) {
        std::string normalized;
        if (!unicode::normalize_nfc(
                item.second,
                normalized,
                codepoints) ||
            normalized != item.first) {
            std::cerr << "direct NFC normalization mismatch\n";
            return false;
        }
        std::string normalized_again;
        if (!unicode::normalize_nfc(
                normalized,
                normalized_again,
                codepoints) ||
            normalized_again != normalized) {
            std::cerr << "NFC normalization is not idempotent\n";
            return false;
        }
    }
    const std::string invalid_utf8(1, static_cast<char>(0xff));
    std::string normalized;
    if (unicode::normalize_nfc(
            invalid_utf8,
            normalized,
            codepoints)) {
        std::cerr << "NFC normalization accepted invalid UTF-8\n";
        return false;
    }

    struct PretokenizeCase {
        const char* text;
        std::vector<std::string> expected;
    };
    const std::vector<PretokenizeCase> pretokenize_cases = {
        {"Hello world", {"Hello", " world"}},
        {"abc  def", {"abc", " ", " def"}},
        {"123", {"1", "2", "3"}},
        {" !!!", {" !!!"}},
        {"abc\n\ndef", {"abc", "\n\n", "def"}},
        {"hello's", {"hello", "'s"}},
        {"it's we're I'VE", {"it", "'s", " we", "'re", " I", "'VE"}},
        {"a\r\n\r\nb", {"a", "\r\n\r\n", "b"}},
        {"a   ", {"a", "   "}},
        {"a  b", {"a", " ", " b"}},
        {"1A", {"1", "A"}},
        {"!!!\n", {"!!!\n"}},
        {"\t\n", {"\t\n"}},
        {u8"e\u0301", {"e", u8"\u0301"}},
        {u8"\uff11", {u8"\uff11"}},
        {u8"\u4e2dA", {u8"\u4e2dA"}},
    };
    for (const PretokenizeCase& item : pretokenize_cases) {
        if (!unicode::decode_utf8(item.text, codepoints, error)) {
            std::cerr << "failed to decode pretokenize test input\n";
            return false;
        }
        if (!expect_pieces_equal(
                unicode::pretokenize(codepoints),
                item.expected,
                item.text)) {
            return false;
        }
    }
    return true;
}

} // namespace

int main() {
    if (!test_unicode_helpers()) {
        return 1;
    }

    std::unique_ptr<omnivoice::GGUFModel> gguf;
    omnivoice::Status status = omnivoice::GGUFModel::open(
        std::string(OMNIVOICE_TEST_DATA_DIR) +
            "/omnivoice-tokenizer-m1.gguf",
        gguf);
    if (!status) {
        std::cerr << "tokenizer GGUF load failed: "
                  << status.message() << '\n';
        return 1;
    }

    omnivoice::TextTokenizerConfig config;
    status = omnivoice::load_text_tokenizer_config_from_gguf(
        *gguf,
        config);
    if (!status) {
        std::cerr << "tokenizer config load failed: "
                  << status.message() << '\n';
        return 1;
    }
    omnivoice::TextTokenizer tokenizer;
    status = tokenizer.load(config);
    if (!status) {
        std::cerr << "tokenizer load failed: " << status.message() << '\n';
        return 1;
    }
    if (tokenizer.vocab_size() != 151676) {
        std::cerr << "unexpected vocabulary size\n";
        return 1;
    }
    if (tokenizer.special_token_id("<|endoftext|>") != 151643 ||
        tokenizer.special_token_id("<|im_end|>") != 151645) {
        std::cerr << "unexpected tokenizer special IDs\n";
        return 1;
    }
    if (tokenizer.special_token_id("<|denoise|>") != 151669 ||
        tokenizer.special_token_id("<|lang_start|>") != 151670 ||
        tokenizer.special_token_id("<|lang_end|>") != 151671 ||
        tokenizer.special_token_id("<|instruct_start|>") != 151672 ||
        tokenizer.special_token_id("<|instruct_end|>") != 151673 ||
        tokenizer.special_token_id("<|text_start|>") != 151674 ||
        tokenizer.special_token_id("<|text_end|>") != 151675) {
        std::cerr << "unexpected OmniVoice special IDs\n";
        return 1;
    }

    std::vector<int32_t> ids;
    status = tokenizer.encode(
        "<|text_start|>Hello<|text_end|>",
        ids);
    if (!status) {
        std::cerr << "special token encode failed: " << status.message() << '\n';
        return 1;
    }
    if (ids.size() < 3 ||
        ids.front() != 151674 ||
        ids.back() != 151675) {
        std::cerr << "special token split failed\n";
        return 1;
    }
    for (int32_t id : ids) {
        if (id < 0 || id >= 151676) {
            std::cerr << "token ID out of range\n";
            return 1;
        }
    }

    std::vector<int32_t> composed;
    std::vector<int32_t> decomposed;
    status = tokenizer.encode("\xC3\xA9", composed);
    if (!status) {
        return 1;
    }
    status = tokenizer.encode("e\xCC\x81", decomposed);
    if (!status) {
        return 1;
    }
    if (!expect_equal(composed, decomposed, "NFC")) {
        return 1;
    }

    const std::vector<std::pair<std::string, std::string>> nfc_cases = {
        {u8"\u00c5", u8"A\u030a"},
        {u8"\u212b", u8"\u00c5"},
        {u8"\uac00", u8"\u1100\u1161"},
        {u8"\u304c", u8"\u304b\u3099"},
        {u8"\u1e0a", u8"D\u0307"},
    };
    for (const auto& item : nfc_cases) {
        status = tokenizer.encode(item.first, composed);
        if (!status) {
            std::cerr << "NFC composed input failed: "
                      << status.message() << '\n';
            return 1;
        }
        status = tokenizer.encode(item.second, decomposed);
        if (!status) {
            std::cerr << "NFC decomposed input failed: "
                      << status.message() << '\n';
            return 1;
        }
        if (!expect_equal(composed, decomposed, "extended NFC")) {
            return 1;
        }
    }

    config.nfc_normalization = false;
    omnivoice::TextTokenizer raw_tokenizer;
    status = raw_tokenizer.load(config);
    if (!status) {
        std::cerr << "tokenizer without NFC failed to load: "
                  << status.message() << '\n';
        return 1;
    }
    status = raw_tokenizer.encode("\xC3\xA9", composed);
    if (!status) {
        std::cerr << "non-NFC composed input failed: "
                  << status.message() << '\n';
        return 1;
    }
    status = raw_tokenizer.encode("e\xCC\x81", decomposed);
    if (!status) {
        std::cerr << "non-NFC decomposed input failed: "
                  << status.message() << '\n';
        return 1;
    }
    if (composed == decomposed) {
        std::cerr << "nfc_normalization=false was not respected\n";
        return 1;
    }
    const std::string invalid_raw_utf8(1, static_cast<char>(0xff));
    status = raw_tokenizer.encode(invalid_raw_utf8, ids);
    if (status) {
        std::cerr << "non-NFC tokenizer accepted invalid UTF-8\n";
        return 1;
    }

    std::ifstream golden_file(
        std::string(OMNIVOICE_TEST_DATA_DIR) + "/tokenizer_cases.json");
    if (!golden_file) {
        std::cerr << "failed to open tokenizer golden data\n";
        return 1;
    }
    nlohmann::json golden;
    golden_file >> golden;
    for (const nlohmann::json& item : golden.at("cases")) {
        const std::string text = item.at("text").get<std::string>();
        const std::vector<int32_t> expected =
            item.at("ids").get<std::vector<int32_t>>();
        status = tokenizer.encode(text, ids);
        if (!status) {
            std::cerr << "golden encode failed for: " << text
                      << " (" << status.message() << ")\n";
            return 1;
        }
        if (!expect_equal(ids, expected, "Python golden")) {
            std::cerr << "input: " << text << '\n';
            return 1;
        }
    }

    const std::string invalid_utf8(1, static_cast<char>(0xff));
    status = tokenizer.encode(invalid_utf8, ids);
    if (status) {
        std::cerr << "invalid UTF-8 must be rejected\n";
        return 1;
    }

    return 0;
}
