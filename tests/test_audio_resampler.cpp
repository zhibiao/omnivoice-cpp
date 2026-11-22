#include "audio_kit/audio_resampler.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

constexpr double PI = 3.14159265358979323846;
constexpr int32_t INPUT_RATE = 24000;
constexpr int32_t OUTPUT_RATE = 16000;
// The sinc kernel spans a few input samples on each side; skip the
// transient at both ends before comparing against the analytic signal.
constexpr int32_t GUARD_SAMPLES = 16;

std::vector<float> make_sine(double frequency, size_t sample_count) {
    std::vector<float> samples(sample_count);
    for (size_t index = 0; index < sample_count; ++index) {
        samples[index] = static_cast<float>(std::sin(
            2.0 * PI * frequency *
            static_cast<double>(index) /
            static_cast<double>(INPUT_RATE)));
    }
    return samples;
}

float maximum_interior_error(
    const std::vector<float>& actual,
    const std::vector<float>& expected) {
    if (actual.size() != expected.size()) {
        return INFINITY;
    }
    if (actual.size() <= static_cast<size_t>(2 * GUARD_SAMPLES)) {
        return INFINITY;
    }
    float result = 0.0f;
    for (size_t index = GUARD_SAMPLES;
         index < actual.size() - GUARD_SAMPLES;
         ++index) {
        result = std::max(
            result,
            std::abs(actual[index] - expected[index]));
    }
    return result;
}

} // namespace

int main() {
    // Empty input stays empty.
    if (!omnivoice::resample_24k_to_16k({}).empty()) {
        std::cerr << "empty input should produce empty output\n";
        return 1;
    }

    // Output length follows the documented 3:2 decimation.
    const std::vector<float> impulse(2400, 0.0f);
    const std::vector<float> resampled_impulse =
        omnivoice::resample_24k_to_16k(impulse);
    if (resampled_impulse.size() != 1600) {
        std::cerr << "unexpected resampled length: "
                  << resampled_impulse.size() << '\n';
        return 1;
    }

    // DC gain must stay unity away from the transient.
    std::vector<float> dc(2400, 0.5f);
    const std::vector<float> resampled_dc =
        omnivoice::resample_24k_to_16k(dc);
    const std::vector<float> expected_dc(resampled_dc.size(), 0.5f);
    const float dc_error =
        maximum_interior_error(resampled_dc, expected_dc);

    // A 1 kHz tone is well inside the passband, so the resampled signal
    // should match the analytic 16 kHz tone.
    const std::vector<float> tone = make_sine(1000.0, 2400);
    const std::vector<float> resampled_tone =
        omnivoice::resample_24k_to_16k(tone);
    std::vector<float> expected_tone(resampled_tone.size());
    for (size_t index = 0; index < expected_tone.size(); ++index) {
        expected_tone[index] = static_cast<float>(std::sin(
            2.0 * PI * 1000.0 *
            static_cast<double>(index) /
            static_cast<double>(OUTPUT_RATE)));
    }
    const float tone_error =
        maximum_interior_error(resampled_tone, expected_tone);

    // Linearity: resample(a + b) == resample(a) + resample(b).
    const std::vector<float> secondary =
        make_sine(2500.0, tone.size());
    std::vector<float> combined(tone.size());
    for (size_t index = 0; index < tone.size(); ++index) {
        combined[index] = tone[index] + secondary[index];
    }
    const std::vector<float> resampled_secondary =
        omnivoice::resample_24k_to_16k(secondary);
    const std::vector<float> resampled_combined =
        omnivoice::resample_24k_to_16k(combined);
    float linearity_error = 0.0f;
    for (size_t index = GUARD_SAMPLES;
         index < resampled_combined.size() - GUARD_SAMPLES;
         ++index) {
        const float expected =
            resampled_tone[index] + resampled_secondary[index];
        linearity_error = std::max(
            linearity_error,
            std::abs(resampled_combined[index] - expected));
    }

    // A silent input must stay silent.
    std::vector<float> silent(2400, 0.0f);
    const std::vector<float> resampled_silent =
        omnivoice::resample_24k_to_16k(silent);
    float silent_peak = 0.0f;
    for (float sample : resampled_silent) {
        silent_peak = std::max(silent_peak, std::abs(sample));
    }

    std::cout << "resampler_dc_max_error: " << dc_error << '\n'
              << "resampler_tone_max_error: " << tone_error << '\n'
              << "resampler_linearity_max_error: "
              << linearity_error << '\n'
              << "resampler_silent_peak: " << silent_peak << '\n';

    if (!std::isfinite(dc_error) || dc_error > 1e-3f ||
        !std::isfinite(tone_error) || tone_error > 2e-3f ||
        !std::isfinite(linearity_error) || linearity_error > 1e-5f ||
        silent_peak > 1e-6f) {
        return 1;
    }
    return 0;
}
