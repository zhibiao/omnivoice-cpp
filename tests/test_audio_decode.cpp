#include "audio_toolkit/audio_loader.h"
#include "audio_toolkit/audio_writer.h"

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

        // load_audio keeps the MP3's native format.
        omnivoice::AudioBuffer mp3_native;
        status = omnivoice::load_audio(mp3_path.u8string(), mp3_native);
        const bool mp3_native_valid =
            status &&
            mp3_native.sample_rate == 44100 &&
            mp3_native.channels == 2 &&
            !mp3_native.samples.empty() &&
            std::abs(
                mp3_native.duration_seconds() - mp3.duration_seconds()) <
                0.01;

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
        const int32_t converted_crossings = status
            ? count_zero_crossings(converted.samples)
            : -1;
        const bool converted_valid =
            status &&
            converted.sample_rate == 24000 &&
            converted.channels == 1 &&
            converted.samples.size() >= 2390 &&
            converted.samples.size() <= 2410 &&
            std::abs(converted_crossings - 88) <= 4;

        // Asymmetric stereo (L=0.4, R=0) proves channels are really mixed.
        omnivoice::AudioBuffer asymmetric;
        asymmetric.sample_rate = 44100;
        asymmetric.channels = 2;
        for (int32_t frame = 0; frame < 480; ++frame) {
            asymmetric.samples.push_back(0.4f);
            asymmetric.samples.push_back(0.0f);
        }
        const std::filesystem::path asymmetric_path =
            std::filesystem::temp_directory_path() /
            "omnivoice_decode_asymmetric.wav";
        temporaries.push_back(asymmetric_path);
        status = omnivoice::write_wav(
            asymmetric_path.u8string(),
            asymmetric);
        omnivoice::AudioBuffer mixed;
        if (status) {
            status = omnivoice::load_audio_24k_mono(
                asymmetric_path.u8string(),
                mixed);
        }
        const float dc_mix = status && !mixed.samples.empty()
            ? mixed.samples[mixed.samples.size() / 2]
            : -1.0f;
        const bool downmix_valid =
            status &&
            mixed.sample_rate == 24000 &&
            mixed.channels == 1 &&
            dc_mix > 0.0f &&
            dc_mix < 0.39f;

        // Uncommon rates are brought to 24 kHz with the same duration.
        bool low_rate_valid = false;
        bool high_rate_valid = false;
        for (int32_t rate : {8000, 96000}) {
            const int32_t frames = rate / 10;
            omnivoice::AudioBuffer source;
            source.sample_rate = rate;
            source.channels = 1;
            for (int32_t index = 0; index < frames; ++index) {
                source.samples.push_back(static_cast<float>(
                    0.25 * std::sin(
                        2.0 * PI * 440.0 * index / rate)));
            }
            const std::filesystem::path path =
                std::filesystem::temp_directory_path() /
                (rate == 8000
                     ? "omnivoice_decode_8k.wav"
                     : "omnivoice_decode_96k.wav");
            temporaries.push_back(path);
            status = omnivoice::write_wav(path.u8string(), source);
            omnivoice::AudioBuffer converted_rate;
            if (status) {
                status = omnivoice::load_audio_24k_mono(
                    path.u8string(),
                    converted_rate);
            }
            const int64_t expected_frames =
                (static_cast<int64_t>(24000) * frames + rate - 1) / rate;
            const bool valid =
                status &&
                converted_rate.sample_rate == 24000 &&
                converted_rate.channels == 1 &&
                converted_rate.samples.size() ==
                    static_cast<size_t>(expected_frames) &&
                std::abs(
                    count_zero_crossings(converted_rate.samples) - 88) <= 5;
            if (rate == 8000) {
                low_rate_valid = valid;
            } else {
                high_rate_valid = valid;
            }
        }

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
                  << "decode_wav_converted_zero_crossings: "
                  << converted_crossings << '\n'
                  << "decode_wav_downmix_valid: "
                  << (downmix_valid ? "true" : "false") << '\n'
                  << "decode_wav_downmix_value: " << dc_mix << '\n'
                  << "decode_mp3_native: "
                  << (mp3_native_valid ? "true" : "false") << '\n'
                  << "decode_wav_8k: "
                  << (low_rate_valid ? "true" : "false") << '\n'
                  << "decode_wav_96k: "
                  << (high_rate_valid ? "true" : "false") << '\n'
                  << "decode_missing_rejected: "
                  << (missing_rejected ? "true" : "false") << '\n';

        cleanup();
        if (!mp3_valid || !mp3_native_valid || !unsupported_rejected ||
            !wav_valid || !converted_valid || !low_rate_valid ||
            !high_rate_valid || !downmix_valid || !missing_rejected) {
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
