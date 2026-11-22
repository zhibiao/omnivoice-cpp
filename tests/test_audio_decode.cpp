#include "audio_kit.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double PI = 3.14159265358979323846;

int32_t count_zero_crossings(const std::vector<float>& samples) {
    int32_t crossings = 0;
    float previous = 0.0f;
    for (float sample : samples) {
        const float sign = sample > 0.0f
            ? 1.0f
            : (sample < 0.0f ? -1.0f : 0.0f);
        if (sign == 0.0f) {
            continue;
        }
        if (previous != 0.0f && sign != previous) {
            ++crossings;
        }
        previous = sign;
    }
    return crossings;
}

} // namespace

int main() {
    const std::filesystem::path data_dir = OMNIVOICE_TEST_DATA_DIR;
    const std::filesystem::path mp3_path =
        data_dir / "audio_tone_44k_stereo.mp3";
    const std::filesystem::path unsupported_path =
        data_dir / "text_chunker_cases.json";
    if (!std::filesystem::is_regular_file(mp3_path) ||
        !std::filesystem::is_regular_file(unsupported_path)) {
        std::cerr << "audio decode test data is unavailable; skipping\n";
        return 77;
    }

    std::vector<std::filesystem::path> temporaries;
    const auto cleanup = [&]() {
        std::error_code ignored;
        for (const auto& path : temporaries) {
            std::filesystem::remove(path, ignored);
        }
    };

    try {
        // MP3 at 44.1 kHz stereo is decoded, downmixed and resampled.
        omnivoice::AudioBuffer mp3;
        omnivoice::Status status =
            omnivoice::load_audio_24k_mono(mp3_path.u8string(), mp3);
        if (!status) {
            std::cerr << "MP3 decode failed: " << status.message() << '\n';
            cleanup();
            return 1;
        }
        float peak = 0.0f;
        for (float sample : mp3.samples) {
            peak = std::max(peak, std::abs(sample));
        }
        const int32_t crossings = count_zero_crossings(mp3.samples);
        const bool mp3_valid =
            mp3.sample_rate == 24000 &&
            mp3.channels == 1 &&
            mp3.samples.size() >= 11000 &&
            mp3.samples.size() <= 13000 &&
            peak > 0.4f &&
            peak < 0.6f &&
            std::abs(crossings - 440) <= 12;

        // Unsupported containers are rejected.
        omnivoice::AudioBuffer unsupported;
        status = omnivoice::load_audio_24k_mono(
            unsupported_path.u8string(),
            unsupported);
        const bool unsupported_rejected = !status;

        // Strict 24 kHz mono WAV keeps working.
        omnivoice::AudioBuffer wav_source;
        for (int32_t index = 0; index < 2400; ++index) {
            wav_source.samples.push_back(static_cast<float>(
                0.25 * std::sin(
                    2.0 * PI * 440.0 * index / 24000.0)));
        }
        const std::filesystem::path wav_path =
            std::filesystem::temp_directory_path() /
            "omnivoice_decode_strict.wav";
        temporaries.push_back(wav_path);
        status = omnivoice::save_wav_24k_mono(
            wav_path.u8string(),
            wav_source);
        if (!status) {
            std::cerr << "WAV fixture write failed: "
                      << status.message() << '\n';
            cleanup();
            return 1;
        }
        omnivoice::AudioBuffer wav_loaded;
        status = omnivoice::load_audio_24k_mono(
            wav_path.u8string(),
            wav_loaded);
        const bool wav_valid =
            status &&
            wav_loaded.sample_rate == 24000 &&
            wav_loaded.channels == 1 &&
            wav_loaded.samples.size() == wav_source.samples.size();

        // WAV that needs resampling/downmixing goes through the fallback.
        omnivoice::AudioBuffer stereo;
        stereo.sample_rate = 44100;
        stereo.channels = 2;
        for (int32_t frame = 0; frame < 4410; ++frame) {
            const float value = static_cast<float>(
                0.25 * std::sin(2.0 * PI * 440.0 * frame / 44100.0));
            stereo.samples.push_back(value);
            stereo.samples.push_back(value);
        }
        const std::filesystem::path stereo_path =
            std::filesystem::temp_directory_path() /
            "omnivoice_decode_44k_stereo.wav";
        temporaries.push_back(stereo_path);
        status = omnivoice::write_wav(
            stereo_path.u8string(),
            stereo);
        if (!status) {
            std::cerr << "44.1 kHz WAV fixture write failed: "
                      << status.message() << '\n';
            cleanup();
            return 1;
        }
        omnivoice::AudioBuffer converted;
        status = omnivoice::load_audio_24k_mono(
            stereo_path.u8string(),
            converted);
        const bool converted_valid =
            status &&
            converted.sample_rate == 24000 &&
            converted.channels == 1 &&
            converted.samples.size() >= 2390 &&
            converted.samples.size() <= 2410;

        // Missing files are rejected.
        omnivoice::AudioBuffer missing;
        status = omnivoice::load_audio_24k_mono(
            (data_dir / "missing_audio.wav").u8string(),
            missing);
        const bool missing_rejected = !status;

        std::cout << "decode_mp3_valid: "
                  << (mp3_valid ? "true" : "false") << '\n'
                  << "decode_mp3_sample_rate: " << mp3.sample_rate << '\n'
                  << "decode_mp3_channels: " << mp3.channels << '\n'
                  << "decode_mp3_samples: " << mp3.samples.size() << '\n'
                  << "decode_mp3_peak: " << peak << '\n'
                  << "decode_mp3_zero_crossings: " << crossings << '\n'
                  << "decode_unsupported_rejected: "
                  << (unsupported_rejected ? "true" : "false") << '\n'
                  << "decode_wav_valid: "
                  << (wav_valid ? "true" : "false") << '\n'
                  << "decode_wav_converted: "
                  << (converted_valid ? "true" : "false") << '\n'
                  << "decode_wav_converted_samples: "
                  << converted.samples.size() << '\n'
                  << "decode_missing_rejected: "
                  << (missing_rejected ? "true" : "false") << '\n';

        cleanup();
        if (!mp3_valid || !unsupported_rejected || !wav_valid ||
            !converted_valid || !missing_rejected) {
            return 1;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "audio decode test failed: "
                  << error.what() << '\n';
        cleanup();
        return 1;
    }
}
