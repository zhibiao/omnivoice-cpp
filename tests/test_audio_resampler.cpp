#include "audio_toolkit/audio_resampler.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr double PI = 3.14159265358979323846;
constexpr int32_t INPUT_RATE = 24000;
constexpr int32_t OUTPUT_RATE = 16000;
// The sinc kernel spans a few input samples on each side; skip the
// transient at both ends before comparing against the analytic signal.
constexpr int32_t GUARD_SAMPLES = 16;
// Generic conversions use wider kernels and larger ratios.
constexpr int32_t GENERIC_GUARD_SAMPLES = 128;

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

std::vector<float> make_sine(
    double frequency,
    int32_t sample_rate,
    size_t sample_count,
    float amplitude = 1.0f) {
    std::vector<float> samples(sample_count);
    for (size_t index = 0; index < sample_count; ++index) {
        samples[index] = amplitude * static_cast<float>(std::sin(
            2.0 * PI * frequency *
            static_cast<double>(index) /
            static_cast<double>(sample_rate)));
    }
    return samples;
}

float maximum_interior_error(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    int32_t guard = GUARD_SAMPLES) {
    if (actual.size() != expected.size()) {
        return INFINITY;
    }
    if (actual.size() <= static_cast<size_t>(2 * guard)) {
        return INFINITY;
    }
    float result = 0.0f;
    for (size_t index = static_cast<size_t>(guard);
         index < actual.size() - static_cast<size_t>(guard);
         ++index) {
        result = std::max(
            result,
            std::abs(actual[index] - expected[index]));
    }
    return result;
}

float peak_interior(
    const std::vector<float>& samples,
    int32_t guard = GENERIC_GUARD_SAMPLES) {
    if (samples.size() <= static_cast<size_t>(2 * guard)) {
        return INFINITY;
    }
    float peak = 0.0f;
    for (size_t index = static_cast<size_t>(guard);
         index < samples.size() - static_cast<size_t>(guard);
         ++index) {
        peak = std::max(peak, std::abs(samples[index]));
    }
    return peak;
}

std::vector<float> channel_samples(
    const omnivoice::AudioBuffer& audio,
    int32_t channel) {
    if (audio.channels <= 0 ||
        audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        return {};
    }
    const size_t channels = static_cast<size_t>(audio.channels);
    std::vector<float> result(audio.samples.size() / channels);
    for (size_t frame = 0; frame < result.size(); ++frame) {
        result[frame] = audio.samples[frame * channels +
            static_cast<size_t>(channel)];
    }
    return result;
}

void test_24k_to_16k() {
    // Empty input stays empty.
    check(
        omnivoice::resample_24k_to_16k({}).empty(),
        "24k->16k: empty input should produce empty output");

    // Output length follows the documented 3:2 decimation.
    const std::vector<float> impulse(2400, 0.0f);
    const std::vector<float> resampled_impulse =
        omnivoice::resample_24k_to_16k(impulse);
    check(
        resampled_impulse.size() == 1600,
        "24k->16k: unexpected resampled length");

    // DC gain must stay unity away from the transient.
    std::vector<float> dc(2400, 0.5f);
    const std::vector<float> resampled_dc =
        omnivoice::resample_24k_to_16k(dc);
    const std::vector<float> expected_dc(resampled_dc.size(), 0.5f);
    const float dc_error =
        maximum_interior_error(resampled_dc, expected_dc);
    check(
        std::isfinite(dc_error) && dc_error <= 1e-3f,
        "24k->16k: DC gain deviates from unity");

    // A 1 kHz tone is well inside the passband, so the resampled signal
    // should match the analytic 16 kHz tone.
    const std::vector<float> tone =
        make_sine(1000.0, INPUT_RATE, 2400);
    const std::vector<float> resampled_tone =
        omnivoice::resample_24k_to_16k(tone);
    const std::vector<float> expected_tone =
        make_sine(1000.0, OUTPUT_RATE, resampled_tone.size());
    const float tone_error =
        maximum_interior_error(resampled_tone, expected_tone);
    check(
        std::isfinite(tone_error) && tone_error <= 2e-3f,
        "24k->16k: 1 kHz tone deviates from the analytic signal");

    // Linearity: resample(a + b) == resample(a) + resample(b).
    const std::vector<float> secondary =
        make_sine(2500.0, INPUT_RATE, tone.size());
    std::vector<float> combined(tone.size());
    for (size_t index = 0; index < tone.size(); ++index) {
        combined[index] = tone[index] + secondary[index];
    }
    const std::vector<float> resampled_secondary =
        omnivoice::resample_24k_to_16k(secondary);
    const std::vector<float> resampled_combined =
        omnivoice::resample_24k_to_16k(combined);
    float linearity_error = 0.0f;
    if (resampled_combined.size() == resampled_tone.size() &&
        resampled_secondary.size() == resampled_tone.size()) {
        for (size_t index = GUARD_SAMPLES;
             index < resampled_combined.size() - GUARD_SAMPLES;
             ++index) {
            const float expected =
                resampled_tone[index] + resampled_secondary[index];
            linearity_error = std::max(
                linearity_error,
                std::abs(resampled_combined[index] - expected));
        }
    } else {
        linearity_error = INFINITY;
    }
    check(
        std::isfinite(linearity_error) && linearity_error <= 1e-5f,
        "24k->16k: resampler is not linear");

    // A silent input must stay silent.
    std::vector<float> silent(2400, 0.0f);
    const std::vector<float> resampled_silent =
        omnivoice::resample_24k_to_16k(silent);
    check(
        peak_interior(resampled_silent) == 0.0f,
        "24k->16k: silent input produced non-zero samples");
}

void test_common_rates_to_24k() {
    struct RateCase {
        int32_t rate;
        size_t frames;
    };
    const RateCase cases[] = {
        {8000, 800},
        {11025, 1102},
        {16000, 1600},
        {22050, 2205},
        {32000, 3200},
        {44100, 4410},
        {48000, 4800},
        {88200, 8820},
        {96000, 9600},
        {192000, 19200},
    };

    for (const RateCase& test : cases) {
        const std::string label = std::to_string(test.rate) + "->24000";

        omnivoice::AudioBuffer input;
        input.sample_rate = test.rate;
        input.channels = 1;
        input.samples = make_sine(1000.0, test.rate, test.frames, 0.4f);

        omnivoice::AudioBuffer output;
        const omnivoice::Status status =
            omnivoice::resample_audio(input, 24000, output);
        check(status.is_ok(), label + ": resample_audio failed: " +
            status.message());
        if (!status) {
            continue;
        }

        const int64_t expected_frames =
            (static_cast<int64_t>(24000) *
                 static_cast<int64_t>(test.frames) +
             test.rate - 1) / test.rate;
        check(
            output.sample_rate == 24000,
            label + ": wrong output sample rate");
        check(output.channels == 1, label + ": wrong output channels");
        check(
            output.samples.size() == static_cast<size_t>(expected_frames),
            label + ": wrong output length");
        if (output.samples.size() !=
            static_cast<size_t>(expected_frames)) {
            continue;
        }

        const std::vector<float> expected =
            make_sine(1000.0, 24000, output.samples.size(), 0.4f);
        const float tone_error = maximum_interior_error(
            output.samples,
            expected,
            GENERIC_GUARD_SAMPLES);
        check(
            std::isfinite(tone_error) && tone_error <= 2e-3f,
            label + ": 1 kHz tone error " + std::to_string(tone_error));

        omnivoice::AudioBuffer dc;
        dc.sample_rate = test.rate;
        dc.channels = 1;
        dc.samples.assign(test.frames, 0.25f);
        omnivoice::AudioBuffer dc_output;
        const omnivoice::Status dc_status =
            omnivoice::resample_audio(dc, 24000, dc_output);
        check(
            dc_status.is_ok(),
            label + ": DC resample failed: " + dc_status.message());
        if (dc_status.is_ok()) {
            const float dc_error = maximum_interior_error(
                dc_output.samples,
                std::vector<float>(dc_output.samples.size(), 0.25f),
                GENERIC_GUARD_SAMPLES);
            check(
                std::isfinite(dc_error) && dc_error <= 1e-3f,
                label + ": DC gain error " + std::to_string(dc_error));
        }
    }
}

void test_passband_and_alias_rejection() {
    // 48 kHz -> 24 kHz, a 1 kHz and an 8 kHz tone stay intact.
    const struct {
        double frequency;
        float tolerance;
    } passband[] = {
        {1000.0, 2e-3f},
        {4000.0, 2e-3f},
        {8000.0, 5e-3f},
    };
    for (const auto& test : passband) {
        omnivoice::AudioBuffer input;
        input.sample_rate = 48000;
        input.channels = 1;
        input.samples =
            make_sine(test.frequency, 48000, 4800, 0.5f);
        omnivoice::AudioBuffer output;
        const omnivoice::Status status =
            omnivoice::resample_audio(input, 24000, output);
        const std::string label = "48k->24k " +
            std::to_string(static_cast<int>(test.frequency)) + " Hz";
        check(status.is_ok(), label + ": resample failed");
        if (!status) {
            continue;
        }
        const std::vector<float> expected = make_sine(
            test.frequency,
            24000,
            output.samples.size(),
            0.5f);
        const float error = maximum_interior_error(
            output.samples,
            expected,
            GENERIC_GUARD_SAMPLES);
        check(
            std::isfinite(error) && error <= test.tolerance,
            label + ": passband error " + std::to_string(error));
    }

    // Content above the 24 kHz Nyquist rate must be filtered out, not
    // folded back into the passband.
    for (double frequency : {16000.0, 20000.0}) {
        omnivoice::AudioBuffer input;
        input.sample_rate = 48000;
        input.channels = 1;
        input.samples = make_sine(frequency, 48000, 4800, 0.5f);
        omnivoice::AudioBuffer output;
        const omnivoice::Status status =
            omnivoice::resample_audio(input, 24000, output);
        const std::string label = "48k->24k alias " +
            std::to_string(static_cast<int>(frequency)) + " Hz";
        check(status.is_ok(), label + ": resample failed");
        if (!status) {
            continue;
        }
        const float peak = peak_interior(output.samples);
        check(
            peak < 5e-3f,
            label + ": aliased peak " + std::to_string(peak));
    }

    // Upsampling an 8 kHz signal must not create images in 0..12 kHz.
    omnivoice::AudioBuffer input;
    input.sample_rate = 8000;
    input.channels = 1;
    input.samples = make_sine(1000.0, 8000, 800, 0.5f);
    omnivoice::AudioBuffer output;
    const omnivoice::Status status =
        omnivoice::resample_audio(input, 24000, output);
    check(status.is_ok(), "8k->24k: resample failed");
    if (status) {
        const float error = maximum_interior_error(
            output.samples,
            make_sine(1000.0, 24000, output.samples.size(), 0.5f),
            GENERIC_GUARD_SAMPLES);
        check(
            std::isfinite(error) && error <= 2e-3f,
            "8k->24k: anti-imaging error " + std::to_string(error));
    }
}

void test_stereo_independence() {
    omnivoice::AudioBuffer input;
    input.sample_rate = 44100;
    input.channels = 2;
    const std::vector<float> left =
        make_sine(1000.0, 44100, 4410, 0.4f);
    const std::vector<float> right =
        make_sine(2000.0, 44100, 4410, 0.25f);
    input.samples.reserve(left.size() * 2);
    for (size_t frame = 0; frame < left.size(); ++frame) {
        input.samples.push_back(left[frame]);
        input.samples.push_back(right[frame]);
    }

    omnivoice::AudioBuffer output;
    const omnivoice::Status status =
        omnivoice::resample_audio(input, 24000, output);
    check(status.is_ok(), "stereo: resample failed");
    if (!status) {
        return;
    }
    check(output.channels == 2, "stereo: channel count changed");
    check(
        output.samples.size() == 2400u * 2u,
        "stereo: wrong interleaved length");

    const std::vector<float> left_out = channel_samples(output, 0);
    const std::vector<float> right_out = channel_samples(output, 1);
    const float left_error = maximum_interior_error(
        left_out,
        make_sine(1000.0, 24000, left_out.size(), 0.4f),
        GENERIC_GUARD_SAMPLES);
    const float right_error = maximum_interior_error(
        right_out,
        make_sine(2000.0, 24000, right_out.size(), 0.25f),
        GENERIC_GUARD_SAMPLES);
    check(
        std::isfinite(left_error) && left_error <= 2e-3f,
        "stereo: left channel error " + std::to_string(left_error));
    check(
        std::isfinite(right_error) && right_error <= 2e-3f,
        "stereo: right channel error " + std::to_string(right_error));
}

void test_identity_and_wrapper() {
    omnivoice::AudioBuffer input;
    input.sample_rate = 48000;
    input.channels = 2;
    input.samples = {0.0f, 0.1f, -0.2f, 0.3f, 0.4f, -0.5f};

    omnivoice::AudioBuffer identity;
    const omnivoice::Status identity_status =
        omnivoice::resample_audio(input, 48000, identity);
    check(identity_status.is_ok(), "identity: resample failed");
    if (identity_status.is_ok()) {
        check(
            identity.sample_rate == 48000 &&
                identity.channels == 2,
            "identity: metadata changed");
        check(
            identity.samples == input.samples,
            "identity: samples changed");
    }

    omnivoice::AudioBuffer direct;
    omnivoice::AudioBuffer wrapped;
    const omnivoice::Status direct_status =
        omnivoice::resample_audio(input, 24000, direct);
    const omnivoice::Status wrapped_status =
        omnivoice::resample_audio_to_24k(input, wrapped);
    check(direct_status.is_ok(), "wrapper: direct resample failed");
    check(wrapped_status.is_ok(), "wrapper: wrapper resample failed");
    if (direct_status.is_ok() && wrapped_status.is_ok()) {
        check(
            wrapped.sample_rate == 24000 &&
                wrapped.channels == direct.channels &&
                wrapped.samples == direct.samples,
            "wrapper: resample_audio_to_24k differs from resample_audio");
    }
}

void test_errors() {
    omnivoice::AudioBuffer empty;
    omnivoice::AudioBuffer output;
    check(
        !omnivoice::resample_audio(empty, 24000, output),
        "errors: empty input must be rejected");

    omnivoice::AudioBuffer input;
    input.sample_rate = 48000;
    input.samples = {0.1f, 0.2f};
    check(
        !omnivoice::resample_audio(input, 0, output),
        "errors: zero target rate must be rejected");

    input.sample_rate = 0;
    check(
        !omnivoice::resample_audio(input, 24000, output),
        "errors: zero source rate must be rejected");

    input.sample_rate = 48000;
    input.channels = 0;
    check(
        !omnivoice::resample_audio(input, 24000, output),
        "errors: zero channel count must be rejected");

    input.channels = 2;
    input.samples = {0.1f, 0.2f, 0.3f};
    check(
        !omnivoice::resample_audio(input, 24000, output),
        "errors: non frame-aligned samples must be rejected");

    input.sample_rate = 44101;
    input.channels = 1;
    input.samples = {0.1f, 0.2f, 0.3f, 0.4f};
    check(
        !omnivoice::resample_audio(input, 24000, output),
        "errors: oversized kernel ratio must be rejected");
}

} // namespace

int main() {
    test_24k_to_16k();
    test_common_rates_to_24k();
    test_passband_and_alias_rejection();
    test_stereo_independence();
    test_identity_and_wrapper();
    test_errors();

    if (failures != 0) {
        std::cerr << failures << " resampler check(s) failed\n";
        return 1;
    }
    std::cout << "resampler: all checks passed\n";
    return 0;
}
