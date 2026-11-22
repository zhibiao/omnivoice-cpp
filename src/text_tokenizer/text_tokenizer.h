#pragma once

#include "status.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace omnivoice {

class GGUFModel;

struct TextTokenizerConfig {
    int32_t vocab_size = 0;
    int32_t pad_token_id = -1;
    int32_t eos_token_id = -1;
    bool nfc_normalization = true;

    std::vector<std::string> tokens;
    std::vector<std::string> merges;
    std::vector<std::string> added_token_names;
    std::vector<int32_t> added_token_ids;
};

// Reads tokenizer metadata from the converted GGUF. This adapter is separate
// from TextTokenizer so the tokenizer implementation remains backend-free.
Status load_text_tokenizer_config_from_gguf(
    const GGUFModel& model,
    TextTokenizerConfig& config);

class TextTokenizer {
public:
    Status load(const TextTokenizerConfig& config);

    Status encode(
        const std::string& text,
        std::vector<int32_t>& token_ids) const;

    int32_t special_token_id(const std::string& name) const;
    bool has_token(const std::string& token) const;
    size_t vocab_size() const noexcept;

private:
    struct AddedToken {
        std::string text;
        int32_t id = 0;
    };

    Status encode_regular(
        const std::string& text,
        std::vector<int32_t>& token_ids) const;

    Status encode_piece(
        const std::string& piece,
        std::vector<int32_t>& token_ids) const;

    std::unordered_map<std::string, int32_t> token_to_id_;
    std::unordered_map<std::string, uint32_t> merge_ranks_;
    std::vector<AddedToken> added_tokens_;
    std::vector<std::string> tokens_;
    bool nfc_normalization_ = true;
    int32_t pad_token_id_ = -1;
    int32_t eos_token_id_ = -1;
};

} // namespace omnivoice
