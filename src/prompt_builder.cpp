#include "prompt_builder.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace omnivoice {

namespace {

struct Codepoint {
    char32_t value = 0;
    size_t begin = 0;
    size_t end = 0;
};

bool decode_utf8(
    const std::string& text,
    std::vector<Codepoint>& output,
    std::string& error) {
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
            error = "invalid UTF-8 leading byte";
            return false;
        }
        if (offset + static_cast<size_t>(continuation) > text.size()) {
            error = "truncated UTF-8 sequence";
            return false;
        }
        for (int index = 0; index < continuation; ++index) {
            const uint8_t next = static_cast<uint8_t>(text[offset++]);
            if ((next & 0xc0) != 0x80) {
                error = "invalid UTF-8 continuation byte";
                return false;
            }
            value = (value << 6) | (next & 0x3f);
        }
        if ((continuation == 1 && value < 0x80) ||
            (continuation == 2 && value < 0x800) ||
            (continuation == 3 && value < 0x10000) ||
            value > 0x10ffff ||
            (value >= 0xd800 && value <= 0xdfff)) {
            error = "invalid UTF-8 code point";
            return false;
        }
        output.push_back({value, begin, offset});
    }
    return true;
}

bool is_unicode_whitespace(char32_t value) {
    return (value >= 0x0009 && value <= 0x000d) ||
        value == 0x0020 ||
        value == 0x0085 ||
        value == 0x00a0 ||
        value == 0x1680 ||
        (value >= 0x2000 && value <= 0x200a) ||
        value == 0x2028 ||
        value == 0x2029 ||
        value == 0x202f ||
        value == 0x205f ||
        value == 0x3000;
}

bool is_cjk(char32_t value) {
    return value >= 0x4e00 && value <= 0x9fff;
}

Status trim_unicode_whitespace(
    const std::string& text,
    std::string& output) {
    std::vector<Codepoint> codepoints;
    std::string error;
    if (!decode_utf8(text, codepoints, error)) {
        return Status::error(error);
    }
    size_t begin = 0;
    size_t end = codepoints.size();
    while (begin < end &&
           is_unicode_whitespace(codepoints[begin].value)) {
        ++begin;
    }
    while (end > begin &&
           is_unicode_whitespace(codepoints[end - 1].value)) {
        --end;
    }
    if (begin == end) {
        output.clear();
    } else {
        output = text.substr(
            codepoints[begin].begin,
            codepoints[end - 1].end - codepoints[begin].begin);
    }
    return Status::ok();
}

void replace_all(
    std::string& text,
    const std::string& from,
    const std::string& to) {
    size_t offset = 0;
    while ((offset = text.find(from, offset)) != std::string::npos) {
        text.replace(offset, from.size(), to);
        offset += to.size();
    }
}

Status remove_spaces_around_cjk(std::string& text) {
    std::vector<Codepoint> codepoints;
    std::string error;
    if (!decode_utf8(text, codepoints, error)) {
        return Status::error(error);
    }
    std::string result;
    result.reserve(text.size());
    for (size_t index = 0; index < codepoints.size(); ++index) {
        const Codepoint& current = codepoints[index];
        if (current.value == ' ' &&
            index > 0 &&
            is_cjk(codepoints[index - 1].value)) {
            continue;
        }
        if (current.value == ' ' &&
            index + 1 < codepoints.size() &&
            is_cjk(codepoints[index + 1].value)) {
            continue;
        }
        result.append(
            text,
            current.begin,
            current.end - current.begin);
    }
    text = std::move(result);
    return Status::ok();
}

constexpr std::string_view NONVERBAL_TAGS[] = {
    "[laughter]",
    "[sigh]",
    "[confirmation-en]",
    "[question-en]",
    "[question-ah]",
    "[question-oh]",
    "[question-ei]",
    "[question-yi]",
    "[surprise-ah]",
    "[surprise-oh]",
    "[surprise-wa]",
    "[surprise-yo]",
    "[dissatisfaction-hnn]",
};

bool is_nonverbal_tag_at(
    const std::string& text,
    size_t position,
    size_t& length) {
    for (std::string_view tag : NONVERBAL_TAGS) {
        if (text.compare(position, tag.size(), tag) == 0) {
            length = tag.size();
            return true;
        }
    }
    return false;
}

Status encode_with_nonverbal_tags(
    const TextTokenizer& tokenizer,
    const std::string& text,
    std::vector<int32_t>& token_ids) {
    size_t position = 0;
    while (position < text.size()) {
        size_t next_tag = std::string::npos;
        for (size_t offset = position; offset < text.size(); ++offset) {
            size_t ignored = 0;
            if (is_nonverbal_tag_at(text, offset, ignored)) {
                next_tag = offset;
                break;
            }
        }

        const size_t segment_end =
            next_tag == std::string::npos ? text.size() : next_tag;
        if (segment_end > position) {
            std::vector<int32_t> segment_ids;
            Status status = tokenizer.encode(
                text.substr(position, segment_end - position),
                segment_ids);
            if (!status) {
                return status;
            }
            token_ids.insert(
                token_ids.end(),
                segment_ids.begin(),
                segment_ids.end());
        }
        if (next_tag == std::string::npos) {
            break;
        }

        size_t tag_length = 0;
        is_nonverbal_tag_at(text, next_tag, tag_length);
        std::vector<int32_t> tag_ids;
        Status status = tokenizer.encode(
            text.substr(next_tag, tag_length),
            tag_ids);
        if (!status) {
            return status;
        }
        token_ids.insert(token_ids.end(), tag_ids.begin(), tag_ids.end());
        position = next_tag + tag_length;
    }
    return Status::ok();
}

Status append_tokens(
    const TextTokenizer& tokenizer,
    const std::string& text,
    std::vector<int32_t>& output) {
    std::vector<int32_t> ids;
    Status status = tokenizer.encode(text, ids);
    if (!status) {
        return status;
    }
    output.insert(output.end(), ids.begin(), ids.end());
    return Status::ok();
}

bool equals_none(const std::string& value) {
    if (value.empty()) {
        return true;
    }
    if (value.size() != 4) {
        return false;
    }
    const char lower[] = {'n', 'o', 'n', 'e'};
    for (size_t index = 0; index < 4; ++index) {
        char actual = value[index];
        if (actual >= 'A' && actual <= 'Z') {
            actual = static_cast<char>(actual - 'A' + 'a');
        }
        if (actual != lower[index]) {
            return false;
        }
    }
    return true;
}

} // namespace

Status combine_reference_target_text(
    const std::string& reference_text,
    const std::string& target_text,
    std::string& combined_text) {
    std::string reference;
    std::string target;
    Status status = trim_unicode_whitespace(reference_text, reference);
    if (!status) {
        return status;
    }
    status = trim_unicode_whitespace(target_text, target);
    if (!status) {
        return status;
    }
    if (reference.empty()) {
        return Status::error("reference text is empty");
    }
    if (target.empty()) {
        return Status::error("target text is empty");
    }

    combined_text = reference + " " + target;
    combined_text.erase(
        std::remove(combined_text.begin(), combined_text.end(), '\r'),
        combined_text.end());
    combined_text.erase(
        std::remove(combined_text.begin(), combined_text.end(), '\n'),
        combined_text.end());
    replace_all(combined_text, "\xEF\xBC\x88", "(");
    replace_all(combined_text, "\xEF\xBC\x89", ")");

    std::string collapsed;
    collapsed.reserve(combined_text.size());
    bool previous_space = false;
    for (char value : combined_text) {
        const bool space = value == ' ' || value == '\t';
        if (space) {
            if (!previous_space) {
                collapsed.push_back(' ');
            }
        } else {
            collapsed.push_back(value);
        }
        previous_space = space;
    }
    combined_text = std::move(collapsed);
    return remove_spaces_around_cjk(combined_text);
}

Status build_voice_clone_prompt(
    const TextTokenizer& tokenizer,
    const OmniVoiceConfig& config,
    const VoiceClonePromptInput& input,
    PromptPlan& plan) {
    plan = {};
    if (config.lm.num_audio_codebooks <= 0) {
        return Status::error("configuration has no audio codebooks");
    }
    if (input.target_frames <= 0) {
        return Status::error("target frame count must be positive");
    }
    if (input.reference_codes.codebooks != config.lm.num_audio_codebooks ||
        input.reference_codes.frames <= 0 ||
        input.reference_codes.data.size() !=
            input.reference_codes.size()) {
        return Status::error("reference audio codes have an invalid shape");
    }
    for (int32_t code : input.reference_codes.data) {
        if (code < 0 ||
            code >= config.lm.audio_vocab_size ||
            code == config.lm.audio_mask_id) {
            return Status::error("reference audio code is out of range");
        }
    }
    if (!tokenizer.has_token("<|denoise|>") ||
        !tokenizer.has_token("<|lang_start|>") ||
        !tokenizer.has_token("<|lang_end|>") ||
        !tokenizer.has_token("<|instruct_start|>") ||
        !tokenizer.has_token("<|instruct_end|>") ||
        !tokenizer.has_token("<|text_start|>") ||
        !tokenizer.has_token("<|text_end|>")) {
        return Status::error("tokenizer is missing OmniVoice special tokens");
    }

    const std::string language =
        equals_none(input.language) ? "None" : input.language;
    std::string style_text;
    if (input.denoise) {
        style_text += "<|denoise|>";
    }
    style_text += "<|lang_start|>";
    style_text += language;
    style_text += "<|lang_end|><|instruct_start|>None<|instruct_end|>";

    Status status = append_tokens(tokenizer, style_text, plan.style_tokens);
    if (!status) {
        return status;
    }

    std::string combined_text;
    status = combine_reference_target_text(
        input.reference_text,
        input.target_text,
        combined_text);
    if (!status) {
        return status;
    }
    status = append_tokens(
        tokenizer,
        "<|text_start|>",
        plan.text_tokens);
    if (!status) {
        return status;
    }
    status = encode_with_nonverbal_tags(
        tokenizer,
        combined_text,
        plan.text_tokens);
    if (!status) {
        return status;
    }
    status = append_tokens(
        tokenizer,
        "<|text_end|>",
        plan.text_tokens);
    if (!status) {
        return status;
    }

    plan.codebooks = config.lm.num_audio_codebooks;
    plan.layout.style_frames =
        static_cast<int32_t>(plan.style_tokens.size());
    plan.layout.text_frames =
        static_cast<int32_t>(plan.text_tokens.size());
    plan.layout.reference_start =
        plan.layout.style_frames + plan.layout.text_frames;
    plan.layout.target_start =
        plan.layout.reference_start + input.reference_codes.frames;
    plan.layout.target_frames = input.target_frames;
    plan.layout.sequence_length =
        plan.layout.target_start + input.target_frames;

    const size_t sequence_length =
        static_cast<size_t>(plan.layout.sequence_length);
    plan.input_ids.assign(
        static_cast<size_t>(plan.codebooks) * sequence_length,
        config.lm.audio_mask_id);
    plan.audio_mask.assign(sequence_length, 0);

    for (int32_t codebook = 0; codebook < plan.codebooks; ++codebook) {
        const size_t row =
            static_cast<size_t>(codebook) * sequence_length;
        std::copy(
            plan.style_tokens.begin(),
            plan.style_tokens.end(),
            plan.input_ids.begin() + static_cast<std::ptrdiff_t>(row));
        std::copy(
            plan.text_tokens.begin(),
            plan.text_tokens.end(),
            plan.input_ids.begin() + static_cast<std::ptrdiff_t>(
                row + static_cast<size_t>(plan.layout.style_frames)));
        const size_t reference_input_offset =
            row + static_cast<size_t>(plan.layout.reference_start);
        const size_t reference_code_offset =
            static_cast<size_t>(codebook) *
                static_cast<size_t>(input.reference_codes.frames);
        std::copy(
            input.reference_codes.data.begin() +
                static_cast<std::ptrdiff_t>(reference_code_offset),
            input.reference_codes.data.begin() +
                static_cast<std::ptrdiff_t>(
                    reference_code_offset +
                    static_cast<size_t>(input.reference_codes.frames)),
            plan.input_ids.begin() +
                static_cast<std::ptrdiff_t>(reference_input_offset));
    }
    std::fill(
        plan.audio_mask.begin() +
            static_cast<std::ptrdiff_t>(plan.layout.reference_start),
        plan.audio_mask.end(),
        1);
    return Status::ok();
}

} // namespace omnivoice
