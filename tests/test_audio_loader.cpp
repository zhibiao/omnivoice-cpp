#include "audio_toolkit/audio_processing.h"
#include "audio_toolkit/audio_loader.h"
#include "audio_toolkit/audio_writer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

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

void append_tag(std::vector<uint8_t>& output, const char* tag) {
    output.insert(output.end(), tag, tag + 4);
}

std::filesystem::path write_test_wav(
    const std::string& name,
    uint16_t format_tag,
    uint16_t bits_per_sample,
    uint32_t sample_rate,
    uint16_t channels,
    const std::vector<uint8_t>& sample_bytes) {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / name;
    const uint16_t block_align =
        static_cast<uint16_t>(channels * bits_per_sample / 8);
    const uint32_t byte_rate = sample_rate * block_align;

    std::vector<uint8_t> file;
    append_tag(file, "RIFF");
    append_u32(file, static_cast<uint32_t>(36 + sample_bytes.size()));
    append_tag(file, "WAVE");
    append_tag(file, "fmt ");
    append_u32(file, 16);
    append_u16(file, format_tag);
    append_u16(file, channels);
    append_u32(file, sample_rate);
    append_u32(file, byte_rate);
    append_u16(file, block_align);
    append_u16(file, bits_per_sample);
    append_tag(file, "data");
    append_u32(file, static_cast<uint32_t>(sample_bytes.size()));
    file.insert(file.end(), sample_bytes.begin(), sample_bytes.end());

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(
        reinterpret_cast<const char*>(file.data()),
        static_cast<std::streamsize>(file.size()));
    return path;
}

bool close_to(float actual, float expected, float tolerance = 1e-5f) {
    return std::abs(actual - expected) <= tolerance;
}

std::vector<uint8_t> pcm16_bytes(
    const std::vector<int16_t>& samples) {
    std::vector<uint8_t> bytes;
    for (int16_t sample : samples) {
        append_u16(bytes, static_cast<uint16_t>(sample));
    }
    return bytes;
}

std::vector<uint8_t> pcm24_bytes(
    const std::vector<int32_t>& samples) {
    std::vector<uint8_t> bytes;
    for (int32_t sample : samples) {
        bytes.push_back(static_cast<uint8_t>(sample));
        bytes.push_back(static_cast<uint8_t>(sample >> 8));
        bytes.push_back(static_cast<uint8_t>(sample >> 16));
    }
    return bytes;
}

std::vector<uint8_t> pcm32_bytes(
    const std::vector<int32_t>& samples) {
    std::vector<uint8_t> bytes;
    for (int32_t sample : samples) {
        append_u32(bytes, static_cast<uint32_t>(sample));
    }
    return bytes;
}

std::vector<uint8_t> float32_bytes(
    const std::vector<float>& samples) {
    std::vector<uint8_t> bytes;
    for (float sample : samples) {
        uint32_t raw = 0;
        std::memcpy(&raw, &sample, sizeof(raw));
        append_u32(bytes, raw);
    }
    return bytes;
}

std::string utf8_string(const std::filesystem::path& path) {
    return path.u8string();
}

std::filesystem::path write_raw_file(
    const std::string& name,
    const std::vector<uint8_t>& bytes) {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / name;
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(
        reinterpret_cast<const char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
    return path;
}

std::filesystem::path write_extensible_wav(
    const std::string& name,
    uint32_t sample_rate,
    uint16_t channels,
    uint16_t bits_per_sample,
    const std::vector<uint8_t>& sample_bytes) {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / name;
    const uint16_t block_align =
        static_cast<uint16_t>(channels * bits_per_sample / 8);
    const uint32_t byte_rate = sample_rate * block_align;

    std::vector<uint8_t> file;
    append_tag(file, "RIFF");
    append_u32(
        file,
        static_cast<uint32_t>(60 + sample_bytes.size()));
    append_tag(file, "WAVE");
    append_tag(file, "fmt ");
    append_u32(file, 40);
    append_u16(file, 0xfffe);  // WAVE_FORMAT_EXTENSIBLE
    append_u16(file, channels);
    append_u32(file, sample_rate);
    append_u32(file, byte_rate);
    append_u16(file, block_align);
    append_u16(file, bits_per_sample);
    append_u16(file, 22);  // cbSize
    append_u16(file, bits_per_sample);  // wValidBitsPerSample
    append_u32(file, 0);  // dwChannelMask
    // SubFormat KSDATAFORMAT_SUBTYPE_PCM.
    append_u16(file, 1);
    const uint8_t guid_tail[] = {
        0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80,
        0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71,
    };
    file.insert(file.end(), std::begin(guid_tail), std::end(guid_tail));
    append_tag(file, "data");
    append_u32(file, static_cast<uint32_t>(sample_bytes.size()));
    file.insert(file.end(), sample_bytes.begin(), sample_bytes.end());

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(
        reinterpret_cast<const char*>(file.data()),
        static_cast<std::streamsize>(file.size()));
    return path;
}

} // namespace

int main() {
    std::vector<std::filesystem::path> paths;
    auto cleanup = [&]() {
        for (const auto& path : paths) {
            std::error_code error;
            std::filesystem::remove(path, error);
        }
    };

    const auto pcm16_path = write_test_wav(
        "omnivoice_audio_loader_pcm16.wav",
        1,
        16,
        24000,
        1,
        pcm16_bytes({-32768, -16384, 0, 16384, 32767}));
    paths.push_back(pcm16_path);

    omnivoice::AudioBuffer pcm16;
    omnivoice::WavInfo pcm16_info;
    omnivoice::Status status = omnivoice::load_wav_24k_mono(
        pcm16_path.string(),
        pcm16,
        &pcm16_info);
    if (!status) {
        std::cerr << "PCM16 load failed: " << status.message() << '\n';
        cleanup();
        return 1;
    }
    if (pcm16_info.format_tag != 1 ||
        pcm16_info.bits_per_sample != 16 ||
        pcm16_info.frames != 5 ||
        !close_to(pcm16.samples[0], -1.0f) ||
        !close_to(pcm16.samples[1], -0.5f)) {
        std::cerr << "PCM16 conversion mismatch\n";
        cleanup();
        return 1;
    }

    const auto pcm24_path = write_test_wav(
        "omnivoice_audio_loader_pcm24.wav",
        1,
        24,
        24000,
        1,
        pcm24_bytes({0x400000, -0x400000}));
    paths.push_back(pcm24_path);
    omnivoice::AudioBuffer pcm24;
    status = omnivoice::load_wav_24k_mono(pcm24_path.string(), pcm24);
    if (!status ||
        pcm24.samples.size() != 2 ||
        !close_to(pcm24.samples[0], 0.5f) ||
        !close_to(pcm24.samples[1], -0.5f)) {
        std::cerr << "PCM24 conversion mismatch\n";
        cleanup();
        return 1;
    }

    const auto pcm32_path = write_test_wav(
        "omnivoice_audio_loader_pcm32.wav",
        1,
        32,
        24000,
        1,
        pcm32_bytes({1073741824, -1073741824}));
    paths.push_back(pcm32_path);
    omnivoice::AudioBuffer pcm32;
    status = omnivoice::load_wav_24k_mono(pcm32_path.string(), pcm32);
    if (!status ||
        pcm32.samples.size() != 2 ||
        !close_to(pcm32.samples[0], 0.5f) ||
        !close_to(pcm32.samples[1], -0.5f)) {
        std::cerr << "PCM32 conversion mismatch\n";
        cleanup();
        return 1;
    }

    const auto float_path = write_test_wav(
        "omnivoice_audio_loader_float.wav",
        3,
        32,
        24000,
        1,
        float32_bytes({-0.25f, 0.25f, 0.75f}));
    paths.push_back(float_path);
    omnivoice::AudioBuffer floats;
    status = omnivoice::load_wav_24k_mono(float_path.string(), floats);
    if (!status ||
        floats.samples.size() != 3 ||
        !close_to(floats.samples[0], -0.25f) ||
        !close_to(floats.samples[2], 0.75f)) {
        std::cerr << "float32 conversion mismatch\n";
        cleanup();
        return 1;
    }

    const auto wrong_rate_path = write_test_wav(
        "omnivoice_audio_loader_48k.wav",
        1,
        16,
        48000,
        1,
        pcm16_bytes({0, 100}));
    paths.push_back(wrong_rate_path);
    if (omnivoice::load_wav_24k_mono(wrong_rate_path.string(), pcm16)) {
        std::cerr << "48 kHz input must be rejected\n";
        cleanup();
        return 1;
    }

    const auto stereo_path = write_test_wav(
        "omnivoice_audio_loader_stereo.wav",
        1,
        16,
        24000,
        2,
        pcm16_bytes({0, 100, 200, 300}));
    paths.push_back(stereo_path);
    if (omnivoice::load_wav_24k_mono(stereo_path.string(), pcm16)) {
        std::cerr << "stereo input must be rejected\n";
        cleanup();
        return 1;
    }

    // load_audio keeps native sample rate, channel count and interleaving.
    const auto native_path = write_test_wav(
        "omnivoice_audio_loader_native_44k_stereo.wav",
        1,
        16,
        44100,
        2,
        pcm16_bytes({16384, -16384, 8192, 4096, 0, -32768}));
    paths.push_back(native_path);
    omnivoice::AudioBuffer native;
    status = omnivoice::load_audio(native_path.string(), native);
    if (!status ||
        native.sample_rate != 44100 ||
        native.channels != 2 ||
        native.samples.size() != 6 ||
        !close_to(native.samples[0], 0.5f) ||
        !close_to(native.samples[1], -0.5f) ||
        !close_to(native.samples[2], 0.25f) ||
        !close_to(native.samples[3], 0.125f) ||
        !close_to(native.samples[4], 0.0f) ||
        !close_to(native.samples[5], -1.0f)) {
        std::cerr << "load_audio native WAV mismatch\n";
        cleanup();
        return 1;
    }

    omnivoice::AudioBuffer native_48k;
    status = omnivoice::load_audio(wrong_rate_path.string(), native_48k);
    if (!status ||
        native_48k.sample_rate != 48000 ||
        native_48k.channels != 1 ||
        native_48k.samples.size() != 2 ||
        !close_to(native_48k.samples[1], 100.0f / 32768.0f)) {
        std::cerr << "load_audio must not resample WAV input\n";
        cleanup();
        return 1;
    }

    // The general loader and the strict loader agree on 24 kHz mono files.
    omnivoice::AudioBuffer loose;
    omnivoice::AudioBuffer strict_reloaded;
    status = omnivoice::load_audio(pcm16_path.string(), loose);
    if (status) {
        status = omnivoice::load_wav_24k_mono(
            pcm16_path.string(),
            strict_reloaded);
    }
    if (!status ||
        loose.sample_rate != strict_reloaded.sample_rate ||
        loose.channels != strict_reloaded.channels ||
        loose.samples != strict_reloaded.samples) {
        std::cerr << "load_audio differs from load_wav_24k_mono\n";
        cleanup();
        return 1;
    }

    // WAVE_FORMAT_EXTENSIBLE headers are accepted.
    const auto extensible_path = write_extensible_wav(
        "omnivoice_audio_loader_extensible.wav",
        32000,
        2,
        16,
        pcm16_bytes({1000, -1000, 2000, -2000}));
    paths.push_back(extensible_path);
    omnivoice::AudioBuffer extensible;
    status = omnivoice::load_audio(extensible_path.string(), extensible);
    if (!status ||
        extensible.sample_rate != 32000 ||
        extensible.channels != 2 ||
        extensible.samples.size() != 4 ||
        !close_to(extensible.samples[2], 2000.0f / 32768.0f)) {
        std::cerr << "extensible WAV load mismatch\n";
        cleanup();
        return 1;
    }
    if (omnivoice::load_wav_24k_mono(extensible_path.string(), pcm16)) {
        std::cerr << "strict loader accepted a 32 kHz stereo file\n";
        cleanup();
        return 1;
    }

    // Unsupported PCM bit depths are rejected by both loaders.
    const auto pcm8_path = write_test_wav(
        "omnivoice_audio_loader_pcm8.wav",
        1,
        8,
        24000,
        1,
        {0, 128});
    paths.push_back(pcm8_path);
    if (omnivoice::load_audio(pcm8_path.string(), pcm16) ||
        omnivoice::load_wav_24k_mono(pcm8_path.string(), pcm16)) {
        std::cerr << "8-bit PCM must be rejected\n";
        cleanup();
        return 1;
    }

    // Truncated and non-audio files are rejected.
    std::vector<uint8_t> truncated;
    append_tag(truncated, "RIFF");
    append_u32(truncated, 4);
    append_tag(truncated, "WAVE");
    const auto truncated_path = write_raw_file(
        "omnivoice_audio_loader_truncated.wav",
        truncated);
    paths.push_back(truncated_path);
    if (omnivoice::load_audio(truncated_path.string(), pcm16) ||
        omnivoice::load_wav_24k_mono(truncated_path.string(), pcm16)) {
        std::cerr << "truncated WAV must be rejected\n";
        cleanup();
        return 1;
    }
    const auto garbage_path = write_raw_file(
        "omnivoice_audio_loader_garbage.bin",
        std::vector<uint8_t>(32, 0x41));
    paths.push_back(garbage_path);
    if (omnivoice::load_audio(garbage_path.string(), pcm16)) {
        std::cerr << "non-audio data must be rejected\n";
        cleanup();
        return 1;
    }
    if (omnivoice::load_audio(
            (std::filesystem::temp_directory_path() /
             "omnivoice_missing_audio.wav").string(),
            pcm16)) {
        std::cerr << "missing audio must be rejected\n";
        cleanup();
        return 1;
    }

    // MP3 is decoded at its native rate and channel count.
    const std::filesystem::path mp3_path =
        std::filesystem::path(OMNIVOICE_TEST_DATA_DIR) /
        "audio_tone_44k_stereo.mp3";
    if (std::filesystem::is_regular_file(mp3_path)) {
        omnivoice::AudioBuffer mp3;
        status = omnivoice::load_audio(mp3_path.u8string(), mp3);
        float mp3_peak = 0.0f;
        for (float sample : mp3.samples) {
            mp3_peak = std::max(mp3_peak, std::abs(sample));
        }
        if (!status ||
            mp3.sample_rate != 44100 ||
            mp3.channels != 2 ||
            mp3.samples.empty() ||
            mp3.samples.size() % 2 != 0 ||
            mp3.duration_seconds() < 0.4 ||
            mp3.duration_seconds() > 0.6 ||
            mp3_peak < 0.4f ||
            mp3_peak > 0.6f) {
            std::cerr << "native MP3 decode mismatch\n";
            cleanup();
            return 1;
        }
    } else {
        std::cout << "native MP3 decode skipped: test data unavailable\n";
    }

    // 24 kHz stereo is downmixed without changing the frame count.
    const auto stereo_24_path = write_test_wav(
        "omnivoice_audio_loader_stereo_24k.wav",
        1,
        16,
        24000,
        2,
        pcm16_bytes({16384, 8192, 16384, 8192}));
    paths.push_back(stereo_24_path);
    omnivoice::AudioBuffer downmixed;
    status = omnivoice::load_audio_24k_mono(
        stereo_24_path.string(),
        downmixed);
    if (!status ||
        downmixed.sample_rate != 24000 ||
        downmixed.channels != 1 ||
        downmixed.samples.size() != 2 ||
        !close_to(downmixed.samples[0], 0.375f) ||
        !close_to(downmixed.samples[1], 0.375f)) {
        std::cerr << "24 kHz stereo downmix mismatch\n";
        cleanup();
        return 1;
    }

    omnivoice::AudioBuffer rms_audio;
    rms_audio.samples = {0.5f, -0.5f};
    if (!close_to(omnivoice::compute_rms(rms_audio), 0.5f)) {
        std::cerr << "RMS mismatch\n";
        cleanup();
        return 1;
    }

    omnivoice::AudioBuffer hop_audio;
    hop_audio.samples.assign(1000, 0.0f);
    status = omnivoice::trim_to_hop_multiple(hop_audio, 960);
    if (!status || hop_audio.samples.size() != 960) {
        std::cerr << "trim_to_hop_multiple failed\n";
        cleanup();
        return 1;
    }
    hop_audio.samples.assign(1000, 0.0f);
    status = omnivoice::pad_to_hop_multiple(hop_audio, 960);
    if (!status || hop_audio.samples.size() != 1920) {
        std::cerr << "pad_to_hop_multiple failed\n";
        cleanup();
        return 1;
    }

    omnivoice::AudioBuffer silent_audio;
    silent_audio.samples.resize(4800, 0.0f);
    if (omnivoice::remove_silence(silent_audio)) {
        std::cerr << "all-silent audio must be rejected\n";
        cleanup();
        return 1;
    }

    omnivoice::AudioBuffer padded_audio;
    padded_audio.samples.resize(2400, 0.0f);
    padded_audio.samples.insert(
        padded_audio.samples.end(),
        2400,
        0.0f);
    for (size_t index = 4800; index < 9600; ++index) {
        padded_audio.samples.push_back(
            0.2f * std::sin(
                2.0f * 3.14159265358979323846f *
                440.0f * static_cast<float>(index) / 24000.0f));
    }
    padded_audio.samples.insert(
        padded_audio.samples.end(),
        2400,
        0.0f);
    const size_t before_trim = padded_audio.samples.size();
    status = omnivoice::remove_silence(padded_audio);
    if (!status ||
        padded_audio.samples.empty() ||
        padded_audio.samples.size() >= before_trim) {
        std::cerr << "silence trimming failed\n";
        cleanup();
        return 1;
    }

    const auto roundtrip_path =
        std::filesystem::temp_directory_path() /
        std::filesystem::u8path(
            "omnivoice 跨平台路径 roundtrip.wav");
    paths.push_back(roundtrip_path);
    omnivoice::AudioBuffer source;
    source.samples = {0.0f, 0.25f, -0.25f, 0.5f};
    status = omnivoice::save_wav_24k_mono(
        utf8_string(roundtrip_path),
        source);
    if (!status) {
        std::cerr << "WAV save failed: " << status.message() << '\n';
        cleanup();
        return 1;
    }
    omnivoice::AudioBuffer loaded;
    status = omnivoice::load_wav_24k_mono(
        utf8_string(roundtrip_path),
        loaded);
    if (!status || loaded.samples.size() != source.samples.size()) {
        std::cerr << "WAV round-trip failed\n";
        cleanup();
        return 1;
    }
    for (size_t index = 0; index < source.samples.size(); ++index) {
        if (!close_to(loaded.samples[index], source.samples[index], 4e-5f)) {
            std::cerr << "WAV round-trip sample mismatch\n";
            cleanup();
            return 1;
        }
    }

    // Direct validation coverage for the 24 kHz mono contract.
    {
        omnivoice::AudioBuffer empty;
        if (omnivoice::validate_audio_24k_mono(empty)) {
            std::cerr << "empty audio should fail validation\n";
            cleanup();
            return 1;
        }
        omnivoice::AudioBuffer wrong_rate;
        wrong_rate.samples = {0.0f};
        wrong_rate.sample_rate = 16000;
        if (omnivoice::validate_audio_24k_mono(wrong_rate)) {
            std::cerr << "wrong sample rate should fail validation\n";
            cleanup();
            return 1;
        }
        omnivoice::AudioBuffer stereo;
        stereo.samples = {0.0f, 0.0f};
        stereo.channels = 2;
        if (omnivoice::validate_audio_24k_mono(stereo)) {
            std::cerr << "stereo audio should fail validation\n";
            cleanup();
            return 1;
        }
        omnivoice::AudioBuffer non_finite;
        non_finite.samples = {0.0f, std::nanf("")};
        if (omnivoice::validate_audio_24k_mono(non_finite)) {
            std::cerr << "non-finite audio should fail validation\n";
            cleanup();
            return 1;
        }
        omnivoice::AudioBuffer out_of_range;
        out_of_range.samples = {1.5f};
        if (omnivoice::validate_audio_24k_mono(out_of_range)) {
            std::cerr << "out-of-range audio should fail validation\n";
            cleanup();
            return 1;
        }
        omnivoice::AudioBuffer valid;
        valid.samples = {0.0f, 0.5f, -0.5f};
        if (!omnivoice::validate_audio_24k_mono(valid)) {
            std::cerr << "valid audio should pass validation\n";
            cleanup();
            return 1;
        }
    }

    cleanup();
    return 0;
}
