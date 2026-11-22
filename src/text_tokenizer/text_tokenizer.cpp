#include "text_tokenizer/text_tokenizer.h"

#include "gguf_model.h"
#include "text_tokenizer/text_tokenizer_internal.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <string>
#include <string_view>

#include <uni-algo-1.2.0/conv.h>
#include <uni-algo-1.2.0/norm.h>
#include <uni-algo-1.2.0/prop.h>

namespace omnivoice {

namespace text_tokenizer_internal {

bool decode_utf8(
    std::string_view text,
    std::u32string& codepoints,
    std::string& error) {
    una::error decode_error;
    codepoints = una::strict::utf8to32u(text, decode_error);
    if (decode_error.get_code() == una::error::code::success) {
        error.clear();
        return true;
    }

    error = "invalid UTF-8";
    if (decode_error.has_pos()) {
        error += " at byte offset " + std::to_string(decode_error.pos());
    }
    return false;
}

std::string utf32_slice_to_utf8(
    std::u32string_view text,
    size_t begin,
    size_t end) {
    if (begin > end || end > text.size()) {
        return {};
    }
    return una::utf32to8(text.substr(begin, end - begin));
}

bool is_unicode_letter(char32_t value) {
    return una::codepoint::prop{value}.General_Category_L();
}

bool is_unicode_number(char32_t value) {
    return una::codepoint::prop{value}.General_Category_N();
}

bool is_whitespace(char32_t value) {
    return una::codepoint::is_whitespace(value);
}

Status normalize_nfc(
    std::string_view input,
    std::string& output,
    std::u32string& codepoints) {
    std::string error;
    if (!decode_utf8(input, codepoints, error)) {
        return Status::error(error);
    }
    output = una::norm::to_nfc_utf8(input);
    if (output != input && !decode_utf8(output, codepoints, error)) {
        return Status::error(error);
    }
    return Status::ok();
}

} // namespace text_tokenizer_internal

namespace {

using text_tokenizer_internal::decode_utf8;
using text_tokenizer_internal::is_unicode_letter;
using text_tokenizer_internal::is_unicode_number;
using text_tokenizer_internal::is_whitespace;
using text_tokenizer_internal::normalize_nfc;
using text_tokenizer_internal::utf32_slice_to_utf8;

// GPT-2 ByteLevel maps each raw byte to a Unicode code point. Keep this tiny
// encoder local to the ByteLevel implementation.
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

bool is_contraction_at(
    std::u32string_view codepoints,
    size_t position,
    std::initializer_list<char32_t> suffix) {
    if (position >= codepoints.size() || codepoints[position] != '\'') {
        return false;
    }
    if (position + suffix.size() >= codepoints.size()) {
        return false;
    }
    size_t offset = 1;
    for (char32_t expected : suffix) {
        char32_t actual = codepoints[position + offset];
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

size_t consume_letters(std::u32string_view codepoints, size_t begin) {
    size_t end = begin;
    while (end < codepoints.size() &&
           is_unicode_letter(codepoints[end])) {
        ++end;
    }
    return end;
}

size_t consume_symbols(std::u32string_view codepoints, size_t begin) {
    size_t end = begin;
    while (end < codepoints.size()) {
        const char32_t value = codepoints[end];
        if (is_whitespace(value) ||
            is_unicode_letter(value) ||
            is_unicode_number(value)) {
            break;
        }
        ++end;
    }
    return end;
}

size_t consume_newlines(std::u32string_view codepoints, size_t begin) {
    size_t end = begin;
    while (end < codepoints.size()) {
        const char32_t value = codepoints[end];
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
std::vector<std::string> pretokenize_impl(std::u32string_view codepoints) {
    std::vector<std::string> pieces;
    size_t position = 0;
    while (position < codepoints.size()) {
        if (is_contraction_at(codepoints, position, {U's'}) ||
            is_contraction_at(codepoints, position, {U't'}) ||
            is_contraction_at(codepoints, position, {U'm'}) ||
            is_contraction_at(codepoints, position, {U'd'})) {
            pieces.push_back(utf32_slice_to_utf8(
                codepoints, position, position + 2));
            position += 2;
            continue;
        }
        if (is_contraction_at(codepoints, position, {U'r', U'e'}) ||
            is_contraction_at(codepoints, position, {U'v', U'e'}) ||
            is_contraction_at(codepoints, position, {U'l', U'l'})) {
            pieces.push_back(utf32_slice_to_utf8(
                codepoints, position, position + 3));
            position += 3;
            continue;
        }

        const char32_t current = codepoints[position];
        if (is_unicode_letter(current)) {
            const size_t end = consume_letters(codepoints, position);
            pieces.push_back(
                utf32_slice_to_utf8(codepoints, position, end));
            position = end;
            continue;
        }
        if (position + 1 < codepoints.size() &&
            current != '\r' &&
            current != '\n' &&
            !is_unicode_number(current) &&
            is_unicode_letter(codepoints[position + 1])) {
            const size_t end = consume_letters(codepoints, position + 1);
            pieces.push_back(
                utf32_slice_to_utf8(codepoints, position, end));
            position = end;
            continue;
        }

        if (is_unicode_number(current)) {
            pieces.push_back(utf32_slice_to_utf8(
                codepoints, position, position + 1));
            ++position;
            continue;
        }

        size_t symbol_begin = std::string::npos;
        if (current == ' ' &&
            position + 1 < codepoints.size() &&
            !is_whitespace(codepoints[position + 1]) &&
            !is_unicode_letter(codepoints[position + 1]) &&
            !is_unicode_number(codepoints[position + 1])) {
            symbol_begin = position + 1;
        } else if (!is_whitespace(current) &&
                   !is_unicode_letter(current) &&
                   !is_unicode_number(current)) {
            symbol_begin = position;
        }
        if (symbol_begin != std::string::npos) {
            size_t end = consume_symbols(codepoints, symbol_begin);
            end = consume_newlines(codepoints, end);
            pieces.push_back(
                utf32_slice_to_utf8(codepoints, position, end));
            position = end;
            continue;
        }

        if (is_whitespace(current)) {
            size_t whitespace_end = position;
            while (whitespace_end < codepoints.size() &&
                   is_whitespace(codepoints[whitespace_end])) {
                ++whitespace_end;
            }

            size_t newline_begin = position;
            while (newline_begin < whitespace_end &&
                   codepoints[newline_begin] != '\r' &&
                   codepoints[newline_begin] != '\n') {
                ++newline_begin;
            }
            if (newline_begin < whitespace_end) {
                const size_t end =
                    consume_newlines(codepoints, newline_begin);
                pieces.push_back(
                    utf32_slice_to_utf8(codepoints, position, end));
                position = end;
                continue;
            }

            size_t end = whitespace_end;
            if (end < codepoints.size() && end - position > 1) {
                --end;
            }
            pieces.push_back(
                utf32_slice_to_utf8(codepoints, position, end));
            position = end;
            continue;
        }

        pieces.push_back(utf32_slice_to_utf8(
            codepoints, position, position + 1));
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

namespace text_tokenizer_internal {

std::vector<std::string> pretokenize(std::u32string_view text) {
    return pretokenize_impl(text);
}

} // namespace text_tokenizer_internal

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
    std::u32string codepoints;
    if (nfc_normalization_) {
        Status status = normalize_nfc(text, normalized, codepoints);
        if (!status) {
            return status;
        }
    } else {
        normalized = text;
        std::string error;
        if (!decode_utf8(normalized, codepoints, error)) {
            return Status::error(error);
        }
    }
    for (const std::string& piece :
         text_tokenizer_internal::pretokenize(codepoints)) {
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
