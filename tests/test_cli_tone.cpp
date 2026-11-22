#include "audio_kit.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

int main() {
    const std::filesystem::path output =
        std::filesystem::temp_directory_path() /
        "omnivoice_cli_tone.wav";
    std::error_code ignored;
    std::filesystem::remove(output, ignored);

#if defined(_WIN32)
    // cmd.exe strips the first and last quote of the line, so the whole
    // command needs an extra outer pair to keep the executable quoted.
    const std::string command =
        "\"\"" OMNIVOICE_CLI_PATH "\" --write-test-tone \"" +
        output.u8string() + "\"\"";
#else
    const std::string command =
        "\"" OMNIVOICE_CLI_PATH "\" --write-test-tone \"" +
        output.u8string() + "\"";
#endif
    const int exit_code = std::system(command.c_str());
    if (exit_code != 0) {
        std::cerr << "CLI --write-test-tone exited with "
                  << exit_code << '\n'
                  << "command: " << command << '\n';
        return 1;
    }
    if (!std::filesystem::is_regular_file(output)) {
        std::cerr << "CLI did not create the output WAV\n";
        return 1;
    }

    omnivoice::AudioBuffer audio;
    omnivoice::Status status =
        omnivoice::load_wav_24k_mono(output.u8string(), audio);
    std::filesystem::remove(output, ignored);
    if (!status) {
        std::cerr << "CLI output is not a valid 24 kHz mono WAV: "
                  << status.message() << '\n';
        return 1;
    }

    // write_test_tone emits 0.5 s at 24 kHz: 12000 mono samples.
    constexpr size_t EXPECTED_SAMPLES = 12000;
    float peak = 0.0f;
    int32_t zero_crossings = 0;
    float previous_sign = 0.0f;
    for (float sample : audio.samples) {
        peak = std::max(peak, std::abs(sample));
        const float sign = sample > 0.0f
            ? 1.0f
            : (sample < 0.0f ? -1.0f : 0.0f);
        if (sign != 0.0f) {
            if (previous_sign != 0.0f && sign != previous_sign) {
                ++zero_crossings;
            }
            previous_sign = sign;
        }
    }

    // 440 Hz over 0.5 s spans 220 cycles, i.e. ~440 zero crossings.
    const bool shape_valid =
        audio.sample_rate == 24000 &&
        audio.channels == 1 &&
        audio.samples.size() == EXPECTED_SAMPLES &&
        std::abs(peak - 0.2f) < 1e-3f &&
        std::abs(zero_crossings - 440) <= 4;

    std::cout << "cli_tone_sample_rate: " << audio.sample_rate << '\n'
              << "cli_tone_channels: " << audio.channels << '\n'
              << "cli_tone_samples: " << audio.samples.size() << '\n'
              << "cli_tone_peak: " << peak << '\n'
              << "cli_tone_zero_crossings: " << zero_crossings << '\n'
              << "cli_tone_shape_valid: "
              << (shape_valid ? "true" : "false") << '\n';
    return shape_valid ? 0 : 1;
}
