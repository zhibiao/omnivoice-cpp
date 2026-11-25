#include "voice_clone_prompt.h"

#include "audio_toolkit/audio_processing.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace omnivoice {

namespace {

constexpr std::array<char, 8> MAGIC = {
    'O', 'V', 'P', 'R', 'O', 'M', 'P', 'T'
};
constexpr uint32_t FORMAT_VERSION = 1;
constexpr uint32_t MAX_TEXT_BYTES = 4 * 1024 * 1024;
constexpr uint32_t MAX_LANGUAGE_BYTES = 1024;
constexpr uint32_t MAX_CODES = 64 * 1024 * 1024;
constexpr int32_t HIGGS_CODEBOOK_SIZE = 1024;

void write_u32(std::ostream& output, uint32_t value) {
    const std::array<uint8_t, 4> bytes = {
        static_cast<uint8_t>(value),
        static_cast<uint8_t>(value >> 8),
        static_cast<uint8_t>(value >> 16),
        static_cast<uint8_t>(value >> 24),
    };
    output.write(
        reinterpret_cast<const char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
}

void write_i32(std::ostream& output, int32_t value) {
    write_u32(output, static_cast<uint32_t>(value));
}

void write_f32(std::ostream& output, float value) {
    uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    write_u32(output, bits);
}

bool read_u32(std::istream& input, uint32_t& value) {
    std::array<uint8_t, 4> bytes{};
    input.read(
        reinterpret_cast<char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
    if (!input) {
        return false;
    }
    value =
        static_cast<uint32_t>(bytes[0]) |
        (static_cast<uint32_t>(bytes[1]) << 8) |
        (static_cast<uint32_t>(bytes[2]) << 16) |
        (static_cast<uint32_t>(bytes[3]) << 24);
    return true;
}

bool read_i32(std::istream& input, int32_t& value) {
    uint32_t bits = 0;
    if (!read_u32(input, bits)) {
        return false;
    }
    value = static_cast<int32_t>(bits);
    return true;
}

bool read_f32(std::istream& input, float& value) {
    uint32_t bits = 0;
    if (!read_u32(input, bits)) {
        return false;
    }
    std::memcpy(&value, &bits, sizeof(value));
    return true;
}

Status read_string(
    std::istream& input,
    uint32_t maximum_size,
    const char* label,
    std::string& value) {
    uint32_t size = 0;
    if (!read_u32(input, size)) {
        return Status::error(
            std::string("truncated prompt ") + label);
    }
    if (size == 0) {
        return Status::error(
            std::string("prompt ") + label + " is empty");
    }
    if (size > maximum_size) {
        return Status::error(
            std::string("prompt ") + label + " is too large");
    }
    value.resize(size);
    input.read(value.data(), static_cast<std::streamsize>(size));
    if (!input) {
        return Status::error(
            std::string("truncated prompt ") + label);
    }
    return Status::ok();
}

} // namespace

Status validate_voice_clone_prompt(
    const VoiceClonePrompt& prompt) {
    if (prompt.reference_text.empty()) {
        return Status::error(
            "voice prompt reference text must not be empty");
    }
    if (prompt.reference_text.size() > MAX_TEXT_BYTES ||
        prompt.language.empty() ||
        prompt.language.size() > MAX_LANGUAGE_BYTES) {
        return Status::error("voice prompt text is too large");
    }
    if (prompt.sample_rate != 24000 ||
        prompt.reference_codes.codebooks != 8 ||
        prompt.reference_codes.frames <= 0 ||
        prompt.reference_codes.data.size() !=
            prompt.reference_codes.size() ||
        prompt.reference_codes.size() > MAX_CODES) {
        return Status::error("voice prompt audio code shape is invalid");
    }
    if (!std::isfinite(prompt.reference_rms)) {
        return Status::error("voice prompt RMS is not finite");
    }
    for (int32_t code : prompt.reference_codes.data) {
        if (code < 0 || code >= HIGGS_CODEBOOK_SIZE) {
            return Status::error("voice prompt code is out of range");
        }
    }
    return Status::ok();
}

Status create_voice_clone_prompt(
    HiggsAudioEncoder& encoder,
    const AudioBuffer& reference_audio,
    const std::string& reference_text,
    const std::string& language,
    bool denoise,
    bool preprocess_reference,
    VoiceClonePrompt& prompt) {
    prompt = {};
    if (!encoder.is_loaded()) {
        return Status::error(
            "Higgs encoder must be loaded before creating a prompt");
    }
    Status status = validate_audio_24k_mono(reference_audio);
    if (!status) {
        return status;
    }
    if (reference_text.empty()) {
        return Status::error(
            "voice prompt reference text must not be empty");
    }
    const float original_rms = compute_rms(reference_audio);
    AudioBuffer encoded_audio = reference_audio;
    if (original_rms > 0.0f && original_rms < 0.1f) {
        const float gain = 0.1f / original_rms;
        for (float& sample : encoded_audio.samples) {
            sample *= gain;
        }
    }
    if (preprocess_reference) {
        SilenceTrimOptions options;
        options.threshold_dbfs = -50.0f;
        options.min_silence_ms = 200;
        options.keep_silence_ms = 100;
        status = remove_silence(encoded_audio, options);
        if (!status) {
            return status;
        }
    }
    prompt.reference_text = reference_text;
    prompt.language = language.empty() ? "None" : language;
    prompt.reference_rms = original_rms;
    prompt.sample_rate = reference_audio.sample_rate;
    prompt.denoise = denoise;
    status = encoder.encode(encoded_audio, prompt.reference_codes);
    if (!status) {
        prompt = {};
        return status;
    }
    status = validate_voice_clone_prompt(prompt);
    if (!status) {
        prompt = {};
    }
    return status;
}

Status save_voice_clone_prompt(
    const std::string& path,
    const VoiceClonePrompt& prompt) {
    Status status = validate_voice_clone_prompt(prompt);
    if (!status) {
        return status;
    }
    std::ofstream output(
        std::filesystem::u8path(path),
        std::ios::binary | std::ios::trunc);
    if (!output) {
        return Status::error(
            "failed to open voice prompt for writing: " + path);
    }
    output.write(
        MAGIC.data(),
        static_cast<std::streamsize>(MAGIC.size()));
    write_u32(output, FORMAT_VERSION);
    write_u32(output, static_cast<uint32_t>(prompt.sample_rate));
    write_u32(
        output,
        static_cast<uint32_t>(prompt.reference_codes.codebooks));
    write_u32(
        output,
        static_cast<uint32_t>(prompt.reference_codes.frames));
    write_f32(output, prompt.reference_rms);
    const std::array<uint8_t, 4> flags = {
        static_cast<uint8_t>(prompt.denoise ? 1 : 0),
        0,
        0,
        0,
    };
    output.write(
        reinterpret_cast<const char*>(flags.data()),
        static_cast<std::streamsize>(flags.size()));
    write_u32(
        output,
        static_cast<uint32_t>(prompt.language.size()));
    output.write(
        prompt.language.data(),
        static_cast<std::streamsize>(prompt.language.size()));
    write_u32(
        output,
        static_cast<uint32_t>(prompt.reference_text.size()));
    output.write(
        prompt.reference_text.data(),
        static_cast<std::streamsize>(prompt.reference_text.size()));
    for (int32_t code : prompt.reference_codes.data) {
        write_i32(output, code);
    }
    output.flush();
    if (!output) {
        return Status::error(
            "failed to write voice prompt: " + path);
    }
    return Status::ok();
}

Status load_voice_clone_prompt(
    const std::string& path,
    VoiceClonePrompt& prompt) {
    prompt = {};
    std::ifstream input(
        std::filesystem::u8path(path),
        std::ios::binary);
    if (!input) {
        return Status::error(
            "failed to open voice prompt: " + path);
    }
    std::array<char, MAGIC.size()> magic{};
    input.read(
        magic.data(),
        static_cast<std::streamsize>(magic.size()));
    if (!input || magic != MAGIC) {
        return Status::error("invalid voice prompt magic");
    }

    uint32_t version = 0;
    uint32_t sample_rate = 0;
    uint32_t codebooks = 0;
    uint32_t frames = 0;
    if (!read_u32(input, version) ||
        !read_u32(input, sample_rate) ||
        !read_u32(input, codebooks) ||
        !read_u32(input, frames) ||
        !read_f32(input, prompt.reference_rms)) {
        return Status::error("truncated voice prompt header");
    }
    if (version != FORMAT_VERSION) {
        return Status::error("unsupported voice prompt version");
    }
    std::array<uint8_t, 4> flags{};
    input.read(
        reinterpret_cast<char*>(flags.data()),
        static_cast<std::streamsize>(flags.size()));
    if (!input || (flags[0] & ~uint8_t{1}) != 0) {
        return Status::error("invalid voice prompt flags");
    }
    prompt.denoise = (flags[0] & 1) != 0;
    prompt.sample_rate = static_cast<int32_t>(sample_rate);
    prompt.reference_codes.codebooks =
        static_cast<int32_t>(codebooks);
    prompt.reference_codes.frames =
        static_cast<int32_t>(frames);

    Status status = read_string(
        input,
        MAX_LANGUAGE_BYTES,
        "language",
        prompt.language);
    if (!status) {
        prompt = {};
        return status;
    }
    status = read_string(
        input,
        MAX_TEXT_BYTES,
        "reference text",
        prompt.reference_text);
    if (!status) {
        prompt = {};
        return status;
    }
    if (codebooks > 1000 ||
        frames > 10000000 ||
        static_cast<uint64_t>(codebooks) * frames > MAX_CODES) {
        prompt = {};
        return Status::error("voice prompt code dimensions are too large");
    }
    prompt.reference_codes.data.resize(
        static_cast<size_t>(codebooks) *
        static_cast<size_t>(frames));
    for (int32_t& code : prompt.reference_codes.data) {
        if (!read_i32(input, code)) {
            prompt = {};
            return Status::error("truncated voice prompt codes");
        }
    }
    char extra = 0;
    if (input.read(&extra, 1)) {
        prompt = {};
        return Status::error("voice prompt has trailing data");
    }
    status = validate_voice_clone_prompt(prompt);
    if (!status) {
        prompt = {};
    }
    return status;
}

} // namespace omnivoice
