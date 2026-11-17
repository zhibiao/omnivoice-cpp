#include "gguf_model.h"
#include "text_tokenizer.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
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

} // namespace

int main() {
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
