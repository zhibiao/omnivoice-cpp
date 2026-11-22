#include "text_tokenizer/text_chunker.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
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
            return false;
        }
        if (offset + static_cast<size_t>(continuation) > text.size()) {
            return false;
        }
        for (int index = 0; index < continuation; ++index) {
            const uint8_t next = static_cast<uint8_t>(text[offset++]);
            if ((next & 0xc0) != 0x80) {
                return false;
            }
            value = (value << 6) | (next & 0x3f);
        }
        if ((continuation == 1 && value < 0x80) ||
            (continuation == 2 && value < 0x800) ||
            (continuation == 3 && value < 0x10000) ||
            value > 0x10ffff ||
            (value >= 0xd800 && value <= 0xdfff)) {
            return false;
        }
        output.push_back({value, begin, offset});
    }
    return true;
}

bool is_split_punctuation(char32_t value) {
    switch (value) {
    case '.':
    case ',':
    case ';':
    case ':':
    case '!':
    case '?':
    case 0x3002:
    case 0xff0c:
    case 0xff1b:
    case 0xff1a:
    case 0xff01:
    case 0xff1f:
        return true;
    default:
        return false;
    }
}

bool is_closing_mark(char32_t value) {
    switch (value) {
    case '"':
    case '\'':
    case 0x201c:
    case 0x201d:
    case 0x2018:
    case 0x2019:
    case 0xff09:
    case ']':
    case 0x300b:
    case '>':
    case 0x300d:
    case 0x3011:
        return true;
    default:
        return false;
    }
}

bool is_abbreviation(const std::string& sentence) {
    static constexpr std::string_view ABBREVIATIONS[] = {
        "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "Rev.",
        "Fr.", "Hon.", "Pres.", "Gov.", "Capt.", "Gen.", "Sen.",
        "Rep.", "Col.", "Maj.", "Lt.", "Cmdr.", "Sgt.", "Cpl.",
        "Co.", "Corp.", "Inc.", "Ltd.", "Est.", "Dept.", "St.",
        "Ave.", "Blvd.", "Rd.", "Mt.", "Ft.", "No.", "Jan.",
        "Feb.", "Mar.", "Apr.", "Aug.", "Sep.", "Sept.", "Oct.",
        "Nov.", "Dec.", "i.e.", "e.g.", "vs.", "Vs.", "Etc.",
        "approx.", "fig.", "def.",
    };
    size_t end = sentence.find_last_not_of(" \t\r\n");
    if (end == std::string::npos) {
        return false;
    }
    const size_t begin = sentence.find_last_of(" \t\r\n", end);
    const size_t word_begin =
        begin == std::string::npos ? 0 : begin + 1;
    const std::string_view word(
        sentence.data() + word_begin,
        end - word_begin + 1);
    return std::find(
        std::begin(ABBREVIATIONS),
        std::end(ABBREVIATIONS),
        word) != std::end(ABBREVIATIONS);
}

int32_t length(const std::vector<Codepoint>& codepoints,
               size_t begin,
               size_t end) {
    return static_cast<int32_t>(end - begin);
}

std::string trim_ascii(const char* data, size_t size) {
    size_t begin = 0;
    size_t end = size;
    while (begin < end &&
           (data[begin] == ' ' || data[begin] == '\t' ||
            data[begin] == '\r' || data[begin] == '\n')) {
        ++begin;
    }
    while (end > begin &&
           (data[end - 1] == ' ' || data[end - 1] == '\t' ||
            data[end - 1] == '\r' || data[end - 1] == '\n')) {
        --end;
    }
    return std::string(data + begin, end - begin);
}

} // namespace

int32_t utf8_codepoint_count(const std::string& text) {
    std::vector<Codepoint> codepoints;
    return decode_utf8(text, codepoints)
        ? static_cast<int32_t>(codepoints.size())
        : -1;
}

Status plan_text_chunks(
    const std::string& text,
    int32_t estimated_frames,
    int32_t frame_rate,
    float chunk_duration,
    float chunk_threshold,
    std::vector<std::string>& chunks) {
    chunks.clear();
    if (text.empty() ||
        estimated_frames <= 0 ||
        frame_rate <= 0 ||
        chunk_duration <= 0.0f ||
        chunk_threshold <= 0.0f) {
        return Status::error("invalid long-text chunk parameters");
    }
    const int32_t threshold_frames = std::max(
        1,
        static_cast<int32_t>(
            chunk_threshold * frame_rate));
    if (estimated_frames <= threshold_frames) {
        chunks.push_back(text);
        return Status::ok();
    }
    const int32_t characters = utf8_codepoint_count(text);
    if (characters <= 0) {
        return Status::error("target text is not valid UTF-8");
    }
    const double tokens_per_character =
        static_cast<double>(estimated_frames) /
        static_cast<double>(characters);
    const int32_t chunk_characters = std::max(
        1,
        static_cast<int32_t>(
            chunk_duration * frame_rate /
            tokens_per_character));
    Status status = chunk_text_punctuation(
        text,
        chunk_characters,
        3,
        chunks);
    if (!status) {
        return status;
    }
    if (chunks.empty()) {
        return Status::error("long text produced no chunks");
    }
    return Status::ok();
}

Status chunk_text_punctuation(
    const std::string& text,
    int32_t chunk_length,
    int32_t minimum_chunk_length,
    std::vector<std::string>& chunks) {
    chunks.clear();
    if (text.empty()) {
        return Status::ok();
    }
    if (chunk_length <= 0 || minimum_chunk_length < 0) {
        return Status::error("invalid text chunk length");
    }
    std::vector<Codepoint> codepoints;
    if (!decode_utf8(text, codepoints)) {
        return Status::error("invalid UTF-8 in text chunker");
    }

    struct Sentence {
        size_t begin = 0;
        size_t end = 0;
    };
    std::vector<Sentence> sentences;
    size_t current_begin = 0;
    size_t current_end = 0;
    for (size_t index = 0; index < codepoints.size(); ++index) {
        const Codepoint& codepoint = codepoints[index];
        if (current_end == current_begin &&
            !sentences.empty() &&
            (is_split_punctuation(codepoint.value) ||
             is_closing_mark(codepoint.value))) {
            sentences.back().end = index + 1;
            continue;
        }
        if (current_end == current_begin) {
            current_begin = index;
        }
        current_end = index + 1;
        if (is_split_punctuation(codepoint.value)) {
            bool abbreviation = false;
            if (codepoint.value == '.') {
                const std::string sentence = text.substr(
                    codepoints[current_begin].begin,
                    codepoint.end -
                        codepoints[current_begin].begin);
                abbreviation = is_abbreviation(sentence);
            }
            if (!abbreviation) {
                sentences.push_back({current_begin, current_end});
                current_begin = current_end;
            }
        }
    }
    if (current_end > current_begin) {
        sentences.push_back({current_begin, current_end});
    }

    std::vector<Sentence> merged;
    size_t merged_begin = 0;
    size_t merged_end = 0;
    bool merged_active = false;
    for (const Sentence& sentence : sentences) {
        const int32_t sentence_length =
            length(codepoints, sentence.begin, sentence.end);
        const int32_t current_length = merged_active
            ? length(codepoints, merged_begin, merged_end)
            : 0;
        if (!merged_active ||
            current_length + sentence_length <= chunk_length) {
            if (!merged_active) {
                merged_begin = sentence.begin;
                merged_active = true;
            }
            merged_end = sentence.end;
        } else {
            merged.push_back({merged_begin, merged_end});
            merged_begin = sentence.begin;
            merged_end = sentence.end;
        }
    }
    if (merged_active) {
        merged.push_back({merged_begin, merged_end});
    }

    std::vector<Sentence> final_chunks;
    const bool first_short =
        !merged.empty() &&
        length(codepoints, merged[0].begin, merged[0].end) <
            minimum_chunk_length;
    for (size_t index = 0; index < merged.size(); ++index) {
        const Sentence& chunk = merged[index];
        if (index == 1 && first_short && !final_chunks.empty()) {
            final_chunks.back().end = chunk.end;
            continue;
        }
        const int32_t chunk_size =
            length(codepoints, chunk.begin, chunk.end);
        if (chunk_size >= minimum_chunk_length ||
            final_chunks.empty()) {
            final_chunks.push_back(chunk);
        } else {
            final_chunks.back().end = chunk.end;
        }
    }
    for (const Sentence& chunk : final_chunks) {
        const size_t byte_begin = codepoints[chunk.begin].begin;
        const size_t byte_end = codepoints[chunk.end - 1].end;
        std::string value = trim_ascii(
            text.data() + byte_begin,
            byte_end - byte_begin);
        if (!value.empty()) {
            chunks.push_back(std::move(value));
        }
    }
    return Status::ok();
}

} // namespace omnivoice
