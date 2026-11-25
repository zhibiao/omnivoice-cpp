#include "audio_toolkit/audio_loader.h"

#include "audio_toolkit/audio_processing.h"
#include "audio_toolkit/audio_resampler.h"

#include "miniaudio-0.11.26/miniaudio.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <utility>
#include <vector>

namespace omnivoice {

namespace {

constexpr uint16_t WAVE_FORMAT_PCM = 1;
constexpr uint16_t WAVE_FORMAT_IEEE_FLOAT = 3;
constexpr uint16_t WAVE_FORMAT_EXTENSIBLE = 0xfffe;
constexpr size_t MAX_WAV_FILE_BYTES = 1ull << 30;
constexpr size_t MAX_AUDIO_FILE_BYTES = 512ull << 20;
constexpr int32_t MAX_AUDIO_CHANNELS = 8;
constexpr ma_uint64 DECODE_CHUNK_FRAMES = 4096;

enum class AudioContainer {
    Wave,
    Mp3,
    Unsupported,
};

AudioContainer sniff_container(const std::vector<uint8_t>& data) {
    if (data.size() >= 12 &&
        std::memcmp(data.data(), "RIFF", 4) == 0 &&
        std::memcmp(data.data() + 8, "WAVE", 4) == 0) {
        return AudioContainer::Wave;
    }
    if (data.size() >= 3 &&
        std::memcmp(data.data(), "ID3", 3) == 0) {
        return AudioContainer::Mp3;
    }
    // MPEG audio frame sync with Layer III (AAC ADTS uses layer 00).
    if (data.size() >= 2 &&
        data[0] == 0xff &&
        (data[1] & 0xe0) == 0xe0 &&
        ((data[1] >> 1) & 0x03) == 0x01) {
        return AudioContainer::Mp3;
    }
    return AudioContainer::Unsupported;
}

uint16_t read_u16(const std::vector<uint8_t>& data, size_t offset) {
    return static_cast<uint16_t>(data[offset]) |
        (static_cast<uint16_t>(data[offset + 1]) << 8);
}

uint32_t read_u32(const std::vector<uint8_t>& data, size_t offset) {
    return static_cast<uint32_t>(data[offset]) |
        (static_cast<uint32_t>(data[offset + 1]) << 8) |
        (static_cast<uint32_t>(data[offset + 2]) << 16) |
        (static_cast<uint32_t>(data[offset + 3]) << 24);
}

bool range_is_valid(size_t offset, uint64_t size, size_t total) {
    return offset <= total && size <= static_cast<uint64_t>(total - offset);
}

Status read_file_common(
    const std::string& path,
    size_t maximum_bytes,
    const char* kind,
    std::vector<uint8_t>& data) {
    std::ifstream file(
        std::filesystem::u8path(path),
        std::ios::binary | std::ios::ate);
    if (!file) {
        return Status::error("failed to open " + std::string(kind) +
            " file: " + path);
    }
    const std::streamoff size = file.tellg();
    if (size <= 0) {
        return Status::error(std::string(kind) +
            " file is empty: " + path);
    }
    if (static_cast<uint64_t>(size) > maximum_bytes) {
        return Status::error(std::string(kind) +
            " file is too large: " + path);
    }
    data.resize(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);
    file.read(
        reinterpret_cast<char*>(data.data()),
        static_cast<std::streamsize>(data.size()));
    if (!file) {
        return Status::error("failed to read " + std::string(kind) +
            " file: " + path);
    }
    return Status::ok();
}

Status read_wav_file(
    const std::string& path,
    std::vector<uint8_t>& data) {
    return read_file_common(
        path,
        MAX_WAV_FILE_BYTES,
        "WAV",
        data);
}

Status read_audio_file(
    const std::string& path,
    std::vector<uint8_t>& data) {
    return read_file_common(
        path,
        MAX_AUDIO_FILE_BYTES,
        "audio",
        data);
}

Status parse_fmt_chunk(
    const std::vector<uint8_t>& data,
    size_t offset,
    uint32_t size,
    WavInfo& info) {
    if (size < 16 || !range_is_valid(offset, size, data.size())) {
        return Status::error("invalid WAV fmt chunk");
    }

    uint16_t format_tag = read_u16(data, offset);
    const uint16_t channels = read_u16(data, offset + 2);
    const uint32_t sample_rate = read_u32(data, offset + 4);
    const uint32_t byte_rate = read_u32(data, offset + 8);
    const uint16_t block_align = read_u16(data, offset + 12);
    info.bits_per_sample = read_u16(data, offset + 14);

    if (channels == 0) {
        return Status::error("WAV has no channels");
    }
    if (channels > MAX_AUDIO_CHANNELS) {
        return Status::error("WAV has too many channels");
    }
    info.channels = static_cast<int32_t>(channels);

    if (sample_rate == 0) {
        return Status::error("WAV sample rate must be positive");
    }
    if (sample_rate > static_cast<uint32_t>(
            std::numeric_limits<int32_t>::max())) {
        return Status::error("WAV sample rate is out of range");
    }
    info.sample_rate = static_cast<int32_t>(sample_rate);

    if (format_tag == WAVE_FORMAT_EXTENSIBLE) {
        if (size < 40) {
            return Status::error("WAV extensible fmt chunk is truncated");
        }
        const uint16_t extension_size = read_u16(data, offset + 16);
        if (extension_size < 22) {
            return Status::error("WAV extensible fmt extension is invalid");
        }
        format_tag = read_u16(data, offset + 24);
    }
    info.format_tag = format_tag;

    if (format_tag != WAVE_FORMAT_PCM &&
        format_tag != WAVE_FORMAT_IEEE_FLOAT) {
        return Status::error("unsupported WAV sample format");
    }
    if (format_tag == WAVE_FORMAT_PCM &&
        info.bits_per_sample != 16 &&
        info.bits_per_sample != 24 &&
        info.bits_per_sample != 32) {
        return Status::error("unsupported PCM bit depth");
    }
    if (format_tag == WAVE_FORMAT_IEEE_FLOAT &&
        info.bits_per_sample != 32) {
        return Status::error("only 32-bit float WAV is supported");
    }

    const uint32_t bytes_per_sample = info.bits_per_sample / 8;
    const uint32_t expected_block_align =
        static_cast<uint32_t>(info.channels) * bytes_per_sample;
    if (block_align != expected_block_align) {
        return Status::error("WAV block alignment is inconsistent");
    }
    const uint64_t expected_byte_rate =
        static_cast<uint64_t>(sample_rate) * block_align;
    if (byte_rate != expected_byte_rate) {
        return Status::error("WAV byte rate is inconsistent");
    }
    return Status::ok();
}

int32_t read_i24(const std::vector<uint8_t>& data, size_t offset) {
    int32_t value =
        static_cast<int32_t>(data[offset]) |
        (static_cast<int32_t>(data[offset + 1]) << 8) |
        (static_cast<int32_t>(data[offset + 2]) << 16);
    if ((value & 0x00800000) != 0) {
        value |= static_cast<int32_t>(0xff000000);
    }
    return value;
}

Status decode_samples(
    const std::vector<uint8_t>& data,
    size_t offset,
    uint32_t size,
    const WavInfo& info,
    AudioBuffer& audio) {
    if (!range_is_valid(offset, size, data.size())) {
        return Status::error("WAV data chunk is truncated");
    }
    const uint32_t bytes_per_sample = info.bits_per_sample / 8;
    if (bytes_per_sample == 0 || size % bytes_per_sample != 0) {
        return Status::error("WAV data size is not sample aligned");
    }

    const size_t count = size / bytes_per_sample;
    audio.samples.resize(count);
    for (size_t index = 0; index < count; ++index) {
        const size_t sample_offset = offset + index * bytes_per_sample;
        float value = 0.0f;
        if (info.format_tag == WAVE_FORMAT_IEEE_FLOAT) {
            const uint32_t raw = read_u32(data, sample_offset);
            std::memcpy(&value, &raw, sizeof(value));
            if (!std::isfinite(value)) {
                return Status::error("WAV contains non-finite float samples");
            }
            if (value < -1.000001f || value > 1.000001f) {
                return Status::error(
                    "float WAV samples must be normalized to [-1, 1]");
            }
            value = std::clamp(value, -1.0f, 1.0f);
        } else if (info.bits_per_sample == 16) {
            const int16_t raw =
                static_cast<int16_t>(read_u16(data, sample_offset));
            value = static_cast<float>(raw) / 32768.0f;
        } else if (info.bits_per_sample == 24) {
            value = static_cast<float>(read_i24(data, sample_offset)) /
                8388608.0f;
        } else {
            const int32_t raw =
                static_cast<int32_t>(read_u32(data, sample_offset));
            value = static_cast<float>(
                static_cast<double>(raw) / 2147483648.0);
        }
        audio.samples[index] = value;
    }
    return Status::ok();
}

struct WavChunks {
    WavInfo info;
    size_t data_offset = 0;
    uint32_t data_size = 0;
};

Status parse_wav_chunks(
    const std::vector<uint8_t>& data,
    WavChunks& chunks) {
    const uint32_t riff_size = read_u32(data, 4);
    if (riff_size > data.size() - 8) {
        return Status::error("WAV RIFF size exceeds file size");
    }

    bool have_fmt = false;
    bool have_data = false;

    size_t offset = 12;
    while (offset + 8 <= data.size()) {
        const uint32_t chunk_size = read_u32(data, offset + 4);
        const size_t payload = offset + 8;
        if (!range_is_valid(payload, chunk_size, data.size())) {
            return Status::error("WAV chunk exceeds file size");
        }

        if (std::memcmp(data.data() + offset, "fmt ", 4) == 0) {
            if (have_fmt) {
                return Status::error("WAV contains multiple fmt chunks");
            }
            Status status =
                parse_fmt_chunk(data, payload, chunk_size, chunks.info);
            if (!status) {
                return status;
            }
            have_fmt = true;
        } else if (std::memcmp(data.data() + offset, "data", 4) == 0) {
            if (have_data) {
                return Status::error("WAV contains multiple data chunks");
            }
            chunks.data_offset = payload;
            chunks.data_size = chunk_size;
            have_data = true;
        }

        offset = payload + chunk_size + (chunk_size & 1u);
    }
    if (!have_fmt) {
        return Status::error("WAV has no fmt chunk");
    }
    if (!have_data) {
        return Status::error("WAV has no data chunk");
    }

    const uint32_t block_align =
        static_cast<uint32_t>(chunks.info.channels) *
        chunks.info.bits_per_sample / 8;
    if (chunks.data_size == 0 || chunks.data_size % block_align != 0) {
        return Status::error("WAV data chunk has an invalid size");
    }
    chunks.info.frames = chunks.data_size / block_align;
    return Status::ok();
}

Status parse_wav_file(
    const std::string& path,
    std::vector<uint8_t>& data,
    WavChunks& chunks) {
    Status status = read_wav_file(path, data);
    if (!status) {
        return status;
    }
    if (data.size() < 12 ||
        std::memcmp(data.data(), "RIFF", 4) != 0 ||
        std::memcmp(data.data() + 8, "WAVE", 4) != 0) {
        return Status::error("file is not a RIFF/WAVE file");
    }
    return parse_wav_chunks(data, chunks);
}

Status decode_wav_audio(
    const std::vector<uint8_t>& data,
    const WavChunks& chunks,
    AudioBuffer& audio,
    WavInfo* info) {
    audio = {};
    Status status = decode_samples(
        data,
        chunks.data_offset,
        chunks.data_size,
        chunks.info,
        audio);
    if (!status) {
        return status;
    }
    audio.sample_rate = chunks.info.sample_rate;
    audio.channels = chunks.info.channels;
    if (info != nullptr) {
        *info = chunks.info;
    }
    return Status::ok();
}

Status decode_wav(
    const std::vector<uint8_t>& data,
    AudioBuffer& audio) {
    WavChunks chunks;
    Status status = parse_wav_chunks(data, chunks);
    if (!status) {
        return status;
    }
    return decode_wav_audio(data, chunks, audio, nullptr);
}

Status decode_with_miniaudio(
    const std::string& path,
    const std::vector<uint8_t>& data,
    AudioBuffer& audio) {
    // Channels and sample rate of 0 keep the decoder's native values, so the
    // samples are returned as stored in the container.
    const ma_decoder_config config = ma_decoder_config_init(
        ma_format_f32,
        0,
        0);
    ma_decoder decoder;
    ma_result result = ma_decoder_init_memory(
        data.data(),
        data.size(),
        &config,
        &decoder);
    if (result != MA_SUCCESS) {
        return Status::error(
            "failed to decode audio: " + path +
            " (miniaudio " +
            std::to_string(static_cast<int>(result)) + ")");
    }

    ma_uint32 channels = 0;
    ma_uint32 sample_rate = 0;
    result = ma_decoder_get_data_format(
        &decoder,
        nullptr,
        &channels,
        &sample_rate,
        nullptr,
        0);
    if (result != MA_SUCCESS ||
        channels == 0 ||
        channels > static_cast<ma_uint32>(MAX_AUDIO_CHANNELS) ||
        sample_rate == 0) {
        ma_decoder_uninit(&decoder);
        return Status::error("unsupported audio layout: " + path);
    }

    std::vector<float> samples;
    std::vector<float> chunk(
        static_cast<size_t>(DECODE_CHUNK_FRAMES) * channels);
    for (;;) {
        ma_uint64 frames_read = 0;
        result = ma_decoder_read_pcm_frames(
            &decoder,
            chunk.data(),
            DECODE_CHUNK_FRAMES,
            &frames_read);
        if (result != MA_SUCCESS && result != MA_AT_END) {
            ma_decoder_uninit(&decoder);
            return Status::error(
                "failed to decode audio: " + path +
                " (miniaudio " +
                std::to_string(static_cast<int>(result)) + ")");
        }
        if (frames_read > 0) {
            samples.insert(
                samples.end(),
                chunk.begin(),
                chunk.begin() +
                    static_cast<std::ptrdiff_t>(frames_read * channels));
        }
        if (frames_read == 0 || result == MA_AT_END) {
            break;
        }
    }
    ma_decoder_uninit(&decoder);

    if (samples.empty()) {
        return Status::error("audio decoded to no samples: " + path);
    }
    for (float sample : samples) {
        if (!std::isfinite(sample)) {
            return Status::error(
                "audio contains non-finite samples: " + path);
        }
    }
    audio = {};
    audio.samples = std::move(samples);
    audio.sample_rate = static_cast<int32_t>(sample_rate);
    audio.channels = static_cast<int32_t>(channels);
    return Status::ok();
}

} // namespace

Status load_audio(
    const std::string& path,
    AudioBuffer& audio) {
    audio = {};
    std::vector<uint8_t> data;
    Status status = read_audio_file(path, data);
    if (!status) {
        return status;
    }

    switch (sniff_container(data)) {
    case AudioContainer::Wave:
        return decode_wav(data, audio);
    case AudioContainer::Mp3:
        return decode_with_miniaudio(path, data, audio);
    case AudioContainer::Unsupported:
        break;
    }
    return Status::error(
        "unsupported audio format (expected WAV or MP3): " + path);
}

Status load_wav_24k_mono(
    const std::string& path,
    AudioBuffer& audio,
    WavInfo* info) {
    audio = {};
    std::vector<uint8_t> data;
    WavChunks chunks;
    Status status = parse_wav_file(path, data, chunks);
    if (!status) {
        return status;
    }
    if (chunks.info.channels != 1) {
        return Status::error("WAV input must be mono");
    }
    if (chunks.info.sample_rate != OMNIVOICE_SAMPLE_RATE) {
        return Status::error("WAV input must be 24000 Hz");
    }
    return decode_wav_audio(data, chunks, audio, info);
}

Status load_audio_24k_mono(
    const std::string& path,
    AudioBuffer& audio) {
    audio = {};

    AudioBuffer decoded;
    Status status = load_audio(path, decoded);
    if (!status) {
        return status;
    }

    AudioBuffer mono;
    status = downmix_to_mono(decoded, mono);
    if (!status) {
        return status;
    }

    AudioBuffer converted;
    if (mono.sample_rate != OMNIVOICE_SAMPLE_RATE) {
        status = resample_audio(
            mono,
            OMNIVOICE_SAMPLE_RATE,
            converted);
        if (!status) {
            return status;
        }
    } else {
        converted = std::move(mono);
    }

    // The band-limited kernel can overshoot slightly on full-scale input.
    for (float& sample : converted.samples) {
        sample = std::clamp(sample, -1.0f, 1.0f);
    }
    audio = std::move(converted);
    return validate_audio_24k_mono(audio);
}

} // namespace omnivoice
