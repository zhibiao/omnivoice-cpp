#include "audio_kit/audio_io.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <vector>

namespace omnivoice {

namespace {

constexpr uint16_t WAVE_FORMAT_PCM = 1;
constexpr uint16_t WAVE_FORMAT_IEEE_FLOAT = 3;
constexpr uint16_t WAVE_FORMAT_EXTENSIBLE = 0xfffe;
constexpr size_t MAX_WAV_FILE_BYTES = 1ull << 30;

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

bool range_is_valid(size_t offset, uint64_t size, size_t total) {
    return offset <= total && size <= static_cast<uint64_t>(total - offset);
}

Status read_file(const std::string& path, std::vector<uint8_t>& data) {
    std::ifstream file(
        std::filesystem::u8path(path),
        std::ios::binary | std::ios::ate);
    if (!file) {
        return Status::error("failed to open WAV file: " + path);
    }
    const std::streamoff size = file.tellg();
    if (size <= 0) {
        return Status::error("WAV file is empty: " + path);
    }
    if (static_cast<uint64_t>(size) > MAX_WAV_FILE_BYTES) {
        return Status::error("WAV file is too large: " + path);
    }
    data.resize(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);
    file.read(
        reinterpret_cast<char*>(data.data()),
        static_cast<std::streamsize>(data.size()));
    if (!file) {
        return Status::error("failed to read WAV file: " + path);
    }
    return Status::ok();
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
    info.channels = read_u16(data, offset + 2);
    const uint32_t sample_rate = read_u32(data, offset + 4);
    const uint32_t byte_rate = read_u32(data, offset + 8);
    const uint16_t block_align = read_u16(data, offset + 12);
    info.bits_per_sample = read_u16(data, offset + 14);

    if (sample_rate > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
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
    if (info.channels != 1) {
        return Status::error("WAV input must be mono");
    }
    if (sample_rate != static_cast<uint32_t>(OMNIVOICE_SAMPLE_RATE)) {
        return Status::error("WAV input must be 24000 Hz");
    }
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

float window_db(
    const std::vector<float>& samples,
    size_t begin,
    size_t end) {
    double energy = 0.0;
    for (size_t index = begin; index < end; ++index) {
        energy += static_cast<double>(samples[index]) * samples[index];
    }
    const double rms = std::sqrt(energy / static_cast<double>(end - begin));
    return static_cast<float>(20.0 * std::log10(rms + 1e-12));
}

} // namespace

Status validate_audio_24k_mono(const AudioBuffer& audio) {
    if (audio.samples.empty()) {
        return Status::error("audio buffer is empty");
    }
    if (audio.sample_rate != OMNIVOICE_SAMPLE_RATE) {
        return Status::error("audio buffer must be 24000 Hz");
    }
    if (audio.channels != 1) {
        return Status::error("audio buffer must be mono");
    }
    for (float sample : audio.samples) {
        if (!std::isfinite(sample)) {
            return Status::error("audio buffer contains non-finite samples");
        }
        if (sample < -1.0f || sample > 1.0f) {
            return Status::error("audio buffer samples must be in [-1, 1]");
        }
    }
    return Status::ok();
}

Status load_wav_24k_mono(
    const std::string& path,
    AudioBuffer& audio,
    WavInfo* info) {
    audio = {};
    std::vector<uint8_t> data;
    Status status = read_file(path, data);
    if (!status) {
        return status;
    }
    if (data.size() < 12 ||
        std::memcmp(data.data(), "RIFF", 4) != 0 ||
        std::memcmp(data.data() + 8, "WAVE", 4) != 0) {
        return Status::error("file is not a RIFF/WAVE file");
    }

    const uint32_t riff_size = read_u32(data, 4);
    if (riff_size > data.size() - 8) {
        return Status::error("WAV RIFF size exceeds file size");
    }

    WavInfo parsed;
    bool have_fmt = false;
    bool have_data = false;
    size_t data_offset = 0;
    uint32_t data_size = 0;

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
            status = parse_fmt_chunk(data, payload, chunk_size, parsed);
            if (!status) {
                return status;
            }
            have_fmt = true;
        } else if (std::memcmp(data.data() + offset, "data", 4) == 0) {
            if (have_data) {
                return Status::error("WAV contains multiple data chunks");
            }
            data_offset = payload;
            data_size = chunk_size;
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

    const uint32_t block_align = parsed.channels * parsed.bits_per_sample / 8;
    if (data_size == 0 || data_size % block_align != 0) {
        return Status::error("WAV data chunk has an invalid size");
    }
    parsed.frames = data_size / block_align;
    status = decode_samples(
        data,
        data_offset,
        data_size,
        parsed,
        audio);
    if (!status) {
        return status;
    }
    audio.sample_rate = parsed.sample_rate;
    audio.channels = parsed.channels;
    if (info != nullptr) {
        *info = parsed;
    }
    return Status::ok();
}

Status save_wav_24k_mono(
    const std::string& path,
    const AudioBuffer& audio) {
    Status status = validate_audio_24k_mono(audio);
    if (!status) {
        return status;
    }
    return write_wav(path, audio);
}

Status write_wav(
    const std::string& path,
    const AudioBuffer& audio) {
    if (path.empty()) {
        return Status::error("WAV output path is empty");
    }
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

    std::vector<uint8_t> header;
    header.reserve(44);
    header.insert(header.end(), {'R', 'I', 'F', 'F'});
    append_u32(header, 36 + data_size);
    header.insert(header.end(), {'W', 'A', 'V', 'E'});
    header.insert(header.end(), {'f', 'm', 't', ' '});
    append_u32(header, 16);
    append_u16(header, 1);
    append_u16(header, channels);
    append_u32(header, static_cast<uint32_t>(audio.sample_rate));
    append_u32(header, byte_rate);
    append_u16(header, block_align);
    append_u16(header, 16);
    header.insert(header.end(), {'d', 'a', 't', 'a'});
    append_u32(header, data_size);

    std::ofstream file(
        std::filesystem::u8path(path),
        std::ios::binary | std::ios::trunc);
    if (!file) {
        return Status::error("failed to open WAV output: " + path);
    }
    file.write(
        reinterpret_cast<const char*>(header.data()),
        static_cast<std::streamsize>(header.size()));

    for (float sample : audio.samples) {
        if (!std::isfinite(sample)) {
            return Status::error("WAV samples contain non-finite values");
        }
        sample = std::clamp(sample, -1.0f, 1.0f);
        const int16_t pcm = sample >= 0.0f
            ? static_cast<int16_t>(sample * 32767.0f)
            : static_cast<int16_t>(sample * 32768.0f);
        const uint16_t raw = static_cast<uint16_t>(pcm);
        const char bytes[2] = {
            static_cast<char>(raw),
            static_cast<char>(raw >> 8),
        };
        file.write(bytes, 2);
    }
    if (!file) {
        return Status::error("failed to write WAV output: " + path);
    }
    return Status::ok();
}

float compute_rms(const AudioBuffer& audio) {
    if (audio.samples.empty()) {
        return 0.0f;
    }
    double energy = 0.0;
    for (float sample : audio.samples) {
        energy += static_cast<double>(sample) * sample;
    }
    return static_cast<float>(
        std::sqrt(energy / static_cast<double>(audio.samples.size())));
}

Status trim_to_hop_multiple(
    AudioBuffer& audio,
    int32_t hop_length) {
    Status status = validate_audio_24k_mono(audio);
    if (!status) {
        return status;
    }
    if (hop_length <= 0) {
        return Status::error("hop length must be positive");
    }
    const size_t remainder = audio.samples.size() %
        static_cast<size_t>(hop_length);
    if (remainder != 0) {
        audio.samples.resize(audio.samples.size() - remainder);
    }
    if (audio.samples.empty()) {
        return Status::error("audio has no complete hop frame");
    }
    return Status::ok();
}

Status pad_to_hop_multiple(
    AudioBuffer& audio,
    int32_t hop_length) {
    Status status = validate_audio_24k_mono(audio);
    if (!status) {
        return status;
    }
    if (hop_length <= 0) {
        return Status::error("hop length must be positive");
    }
    const size_t remainder = audio.samples.size() %
        static_cast<size_t>(hop_length);
    if (remainder != 0) {
        audio.samples.resize(
            audio.samples.size() +
            static_cast<size_t>(hop_length) - remainder,
            0.0f);
    }
    return Status::ok();
}

Status remove_silence(
    AudioBuffer& audio,
    const SilenceTrimOptions& options) {
    Status status = validate_audio_24k_mono(audio);
    if (!status) {
        return status;
    }
    if (options.threshold_dbfs >= 0.0f) {
        return Status::error("silence threshold must be below 0 dBFS");
    }
    if (options.min_silence_ms < 10 ||
        options.keep_silence_ms < 0) {
        return Status::error("invalid silence trim duration");
    }

    const size_t window = static_cast<size_t>(
        audio.sample_rate * 10 / 1000);
    const size_t window_count =
        (audio.samples.size() + window - 1) / window;

    size_t first_signal_window = window_count;
    size_t last_signal_window = 0;
    std::vector<uint8_t> silent(window_count, 0);
    for (size_t index = 0; index < window_count; ++index) {
        const size_t begin = index * window;
        const size_t end = std::min(begin + window, audio.samples.size());
        const float db = window_db(audio.samples, begin, end);
        silent[index] = db <= options.threshold_dbfs ? 1 : 0;
        if (silent[index] == 0) {
            if (first_signal_window == window_count) {
                first_signal_window = index;
            }
            last_signal_window = index;
        }
    }
    if (first_signal_window == window_count) {
        return Status::error("audio contains no non-silent samples");
    }

    const size_t keep = static_cast<size_t>(
        audio.sample_rate) *
        static_cast<size_t>(options.keep_silence_ms) / 1000;
    const size_t signal_begin = first_signal_window * window;
    const size_t signal_end = std::min(
        (last_signal_window + 1) * window,
        audio.samples.size());
    const size_t begin = signal_begin > keep ? signal_begin - keep : 0;
    const size_t end = std::min(
        audio.samples.size(),
        signal_end + keep);

    std::vector<float> trimmed(
        audio.samples.begin() + static_cast<std::ptrdiff_t>(begin),
        audio.samples.begin() + static_cast<std::ptrdiff_t>(end));
    if (trimmed.empty()) {
        return Status::error("silence trimming produced empty audio");
    }

    const size_t min_silence_samples = static_cast<size_t>(
        audio.sample_rate) *
        static_cast<size_t>(options.min_silence_ms) / 1000;
    if (min_silence_samples > 0) {
        std::vector<float> compacted;
        compacted.reserve(trimmed.size());
        size_t position = 0;
        while (position < trimmed.size()) {
            const size_t run_begin = position;
            const bool is_silent =
                window_db(
                    trimmed,
                    position,
                    std::min(position + window, trimmed.size()))
                <= options.threshold_dbfs;
            size_t run_end = std::min(
                position + window,
                trimmed.size());
            while (is_silent && run_end < trimmed.size()) {
                const size_t next_end = std::min(
                    run_end + window,
                    trimmed.size());
                if (window_db(trimmed, run_end, next_end) >
                    options.threshold_dbfs) {
                    break;
                }
                run_end = next_end;
            }

            if (is_silent && run_end - run_begin >= min_silence_samples) {
                const size_t left_keep = std::min(
                    keep,
                    run_end - run_begin);
                const size_t right_keep = std::min(
                    keep,
                    run_end - run_begin - left_keep);
                compacted.insert(
                    compacted.end(),
                    trimmed.begin() + static_cast<std::ptrdiff_t>(run_begin),
                    trimmed.begin() + static_cast<std::ptrdiff_t>(
                        run_begin + left_keep));
                compacted.insert(
                    compacted.end(),
                    trimmed.begin() + static_cast<std::ptrdiff_t>(
                        run_end - right_keep),
                    trimmed.begin() + static_cast<std::ptrdiff_t>(run_end));
            } else {
                compacted.insert(
                    compacted.end(),
                    trimmed.begin() + static_cast<std::ptrdiff_t>(run_begin),
                    trimmed.begin() + static_cast<std::ptrdiff_t>(run_end));
            }
            position = run_end;
        }
        trimmed = std::move(compacted);
    }

    if (trimmed.empty()) {
        return Status::error("silence trimming produced empty audio");
    }
    audio.samples = std::move(trimmed);
    return Status::ok();
}

} // namespace omnivoice
