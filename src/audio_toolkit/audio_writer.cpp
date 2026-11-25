#include "audio_toolkit/audio_writer.h"

#include "audio_toolkit/audio_processing.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <vector>

namespace omnivoice {

namespace {

void append_u16(std::vector<uint8_t>& output, uint16_t value) {
    output.push_back(static_cast<uint8_t>(value));
    output.push_back(static_cast<uint8_t>(value >> 8));
}

void append_u32(std::vector<uint8_t>& output, uint32_t value) {
    output.push_back(static_cast<uint8_t>(value));
    output.push_back(static_cast<uint8_t>(value >> 8));
    output.push_back(static_cast<uint8_t>(value >> 16));
    output.push_back(static_cast<uint8_t>(value >> 24));
}

Status append_pcm_s16le(
    const AudioBuffer& audio,
    std::vector<uint8_t>& output) {
    output.reserve(output.size() + audio.samples.size() * 2);
    for (float sample : audio.samples) {
        if (!std::isfinite(sample)) {
            return Status::error("audio samples contain non-finite values");
        }
        sample = std::clamp(sample, -1.0f, 1.0f);
        const int16_t pcm = sample >= 0.0f
            ? static_cast<int16_t>(sample * 32767.0f)
            : static_cast<int16_t>(sample * 32768.0f);
        append_u16(output, static_cast<uint16_t>(pcm));
    }
    return Status::ok();
}

Status build_wav(
    const AudioBuffer& audio,
    std::vector<uint8_t>& output) {
    if (audio.samples.empty()) {
        return Status::error("WAV output has no samples");
    }
    if (audio.sample_rate <= 0) {
        return Status::error("WAV sample rate must be positive");
    }
    if (audio.channels <= 0 || audio.channels > 2) {
        return Status::error("WAV writer supports mono or stereo");
    }
    if (audio.samples.size() >
        (std::numeric_limits<uint32_t>::max() - 36) / 2) {
        return Status::error("WAV output is too large");
    }

    const uint16_t channels = static_cast<uint16_t>(audio.channels);
    const uint32_t data_size =
        static_cast<uint32_t>(audio.samples.size() * sizeof(int16_t));
    const uint16_t block_align = static_cast<uint16_t>(channels * 2);
    const uint32_t byte_rate =
        static_cast<uint32_t>(audio.sample_rate) * block_align;

    output.clear();
    output.reserve(44 + data_size);
    output.insert(output.end(), {'R', 'I', 'F', 'F'});
    append_u32(output, 36 + data_size);
    output.insert(output.end(), {'W', 'A', 'V', 'E'});
    output.insert(output.end(), {'f', 'm', 't', ' '});
    append_u32(output, 16);
    append_u16(output, 1);
    append_u16(output, channels);
    append_u32(output, static_cast<uint32_t>(audio.sample_rate));
    append_u32(output, byte_rate);
    append_u16(output, block_align);
    append_u16(output, 16);
    output.insert(output.end(), {'d', 'a', 't', 'a'});
    append_u32(output, data_size);
    return append_pcm_s16le(audio, output);
}

} // namespace

Status save_wav_24k_mono(
    const std::string& path,
    const AudioBuffer& audio) {
    Status status = validate_audio_24k_mono(audio);
    if (!status) {
        return status;
    }
    return write_wav(path, audio);
}

Status encode_wav_24k_mono(
    const AudioBuffer& audio,
    std::vector<uint8_t>& bytes) {
    bytes.clear();
    Status status = validate_audio_24k_mono(audio);
    if (!status) {
        return status;
    }
    status = build_wav(audio, bytes);
    if (!status) {
        bytes.clear();
    }
    return status;
}

Status encode_pcm_s16le(
    const AudioBuffer& audio,
    std::vector<uint8_t>& bytes) {
    bytes.clear();
    if (audio.samples.empty()) {
        return Status::error("audio buffer is empty");
    }
    if (audio.sample_rate != OMNIVOICE_SAMPLE_RATE) {
        return Status::error("audio buffer must be 24000 Hz");
    }
    if (audio.channels != 1) {
        return Status::error("audio buffer must be mono");
    }
    const Status status = append_pcm_s16le(audio, bytes);
    if (!status) {
        bytes.clear();
    }
    return status;
}

Status write_wav(
    const std::string& path,
    const AudioBuffer& audio) {
    if (path.empty()) {
        return Status::error("WAV output path is empty");
    }
    std::vector<uint8_t> bytes;
    Status status = build_wav(audio, bytes);
    if (!status) {
        return status;
    }

    std::ofstream file(
        std::filesystem::u8path(path),
        std::ios::binary | std::ios::trunc);
    if (!file) {
        return Status::error("failed to open WAV output: " + path);
    }
    file.write(
        reinterpret_cast<const char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
    if (!file) {
        return Status::error("failed to write WAV output: " + path);
    }
    return Status::ok();
}

} // namespace omnivoice
