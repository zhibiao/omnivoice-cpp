#include "audio_toolkit/audio_resampler.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace omnivoice {

namespace {

constexpr int32_t LOWPASS_FILTER_WIDTH = 6;
constexpr double ROLLOFF = 0.99;
// Upper bound for the precomputed polyphase table. Common rates stay far
// below this; pathological ratios are rejected instead of exhausting memory.
constexpr int64_t MAX_KERNEL_ENTRIES = 4'000'000;

struct ResampleKernel {
    int32_t width = 0;
    int32_t phases = 0;
    int32_t taps = 0;
    std::vector<float> values;  // [phases, taps]
};

struct ResamplePlan {
    int32_t reduced_source = 0;
    int32_t reduced_target = 0;
    ResampleKernel kernel;
};

int32_t greatest_common_divisor(int32_t left, int32_t right) {
    while (right != 0) {
        const int32_t remainder = left % right;
        left = right;
        right = remainder;
    }
    return left;
}

Status build_plan(
    int32_t source_rate,
    int32_t target_rate,
    ResamplePlan& plan) {
    if (source_rate <= 0) {
        return Status::error("source sample rate must be positive");
    }
    if (target_rate <= 0) {
        return Status::error("target sample rate must be positive");
    }

    const int32_t divisor =
        greatest_common_divisor(source_rate, target_rate);
    const int64_t reduced_source = source_rate / divisor;
    const int64_t reduced_target = target_rate / divisor;

    const double base_freq =
        static_cast<double>(std::min(reduced_source, reduced_target)) *
        ROLLOFF;
    const double width_value = std::ceil(
        static_cast<double>(LOWPASS_FILTER_WIDTH) *
        static_cast<double>(reduced_source) /
        base_freq);
    const int64_t taps_value =
        2 * static_cast<int64_t>(width_value) + reduced_source;
    if (width_value < 0.0 ||
        width_value > 1'000'000.0 ||
        taps_value > std::numeric_limits<int32_t>::max() ||
        reduced_target > std::numeric_limits<int32_t>::max()) {
        return Status::error("resample ratio is not supported");
    }
    if (reduced_target * taps_value > MAX_KERNEL_ENTRIES) {
        return Status::error(
            "resample ratio needs a kernel table that is too large; "
            "use a common sample rate");
    }

    plan.reduced_source = static_cast<int32_t>(reduced_source);
    plan.reduced_target = static_cast<int32_t>(reduced_target);

    ResampleKernel& kernel = plan.kernel;
    kernel.width = static_cast<int32_t>(width_value);
    kernel.phases = plan.reduced_target;
    kernel.taps = static_cast<int32_t>(taps_value);
    const double scale =
        base_freq / static_cast<double>(reduced_source);

    kernel.values.resize(
        static_cast<size_t>(kernel.phases) *
        static_cast<size_t>(kernel.taps));
    constexpr double PI = 3.14159265358979323846264338327950288;

    for (int32_t phase = 0; phase < kernel.phases; ++phase) {
        for (int32_t index = 0; index < kernel.taps; ++index) {
            double t =
                static_cast<double>(-phase) /
                    static_cast<double>(kernel.phases) +
                static_cast<double>(index - kernel.width) /
                    static_cast<double>(reduced_source);
            t *= base_freq;
            t = std::clamp(
                t,
                -static_cast<double>(LOWPASS_FILTER_WIDTH),
                static_cast<double>(LOWPASS_FILTER_WIDTH));
            const double window = std::pow(
                std::cos(
                    t * PI /
                    static_cast<double>(LOWPASS_FILTER_WIDTH) /
                    2.0),
                2.0);
            t *= PI;
            const double sinc =
                std::abs(t) < 1e-20 ? 1.0 : std::sin(t) / t;
            kernel.values[
                static_cast<size_t>(phase) *
                    static_cast<size_t>(kernel.taps) +
                static_cast<size_t>(index)] =
                static_cast<float>(sinc * window * scale);
        }
    }
    return Status::ok();
}

int64_t output_frame_count(
    int32_t reduced_source,
    int32_t reduced_target,
    int64_t input_frames) {
    return static_cast<int64_t>(std::ceil(
        static_cast<double>(reduced_target) *
        static_cast<double>(input_frames) /
        static_cast<double>(reduced_source)));
}

void resample_channel(
    const float* input,
    int64_t input_frames,
    int32_t channels,
    int32_t channel,
    const ResamplePlan& plan,
    float* output,
    int64_t output_frames) {
    const int32_t reduced_source = plan.reduced_source;
    const int32_t reduced_target = plan.reduced_target;
    const int32_t width = plan.kernel.width;
    const int32_t taps = plan.kernel.taps;

    for (int64_t output_index = 0;
         output_index < output_frames;
         ++output_index) {
        const int64_t convolution_index =
            output_index / reduced_target;
        const int32_t phase = static_cast<int32_t>(
            output_index % reduced_target);
        const float* filter =
            plan.kernel.values.data() +
            static_cast<ptrdiff_t>(phase) * taps;

        float sum = 0.0f;
        for (int64_t tap = 0; tap < taps; ++tap) {
            const int64_t source_index =
                convolution_index * reduced_source +
                tap - width;
            if (source_index >= 0 &&
                source_index < input_frames) {
                sum +=
                    input[source_index * channels + channel] *
                    filter[static_cast<size_t>(tap)];
            }
        }
        output[output_index * channels + channel] = sum;
    }
}

std::vector<float> resample_mono(
    const std::vector<float>& input,
    const ResamplePlan& plan) {
    const int64_t input_frames =
        static_cast<int64_t>(input.size());
    const int64_t output_frames = output_frame_count(
        plan.reduced_source,
        plan.reduced_target,
        input_frames);

    std::vector<float> output(static_cast<size_t>(output_frames));
    resample_channel(
        input.data(),
        input_frames,
        1,
        0,
        plan,
        output.data(),
        output_frames);
    return output;
}

} // namespace

Status resample_audio(
    const AudioBuffer& input,
    int32_t target_rate,
    AudioBuffer& output) {
    output = {};
    if (input.samples.empty()) {
        return Status::error("audio buffer is empty");
    }
    if (input.sample_rate <= 0) {
        return Status::error("audio sample rate must be positive");
    }
    if (input.channels <= 0) {
        return Status::error("audio channels must be positive");
    }
    if (target_rate <= 0) {
        return Status::error("target sample rate must be positive");
    }

    const size_t channels = static_cast<size_t>(input.channels);
    if (input.samples.size() % channels != 0) {
        return Status::error("audio samples are not frame aligned");
    }

    if (input.sample_rate == target_rate) {
        output = input;
        return Status::ok();
    }

    ResamplePlan plan;
    Status status = build_plan(input.sample_rate, target_rate, plan);
    if (!status) {
        return status;
    }

    const int64_t input_frames =
        static_cast<int64_t>(input.samples.size() / channels);
    const int64_t output_frames = output_frame_count(
        plan.reduced_source,
        plan.reduced_target,
        input_frames);
    if (output_frames <= 0) {
        return Status::error("resampling produced no samples");
    }
    if (static_cast<uint64_t>(output_frames) >
        static_cast<uint64_t>(std::numeric_limits<size_t>::max()) /
            channels) {
        return Status::error("resampled audio is too large");
    }

    output.sample_rate = target_rate;
    output.channels = input.channels;
    output.samples.resize(
        static_cast<size_t>(output_frames) * channels);
    for (int32_t channel = 0; channel < input.channels; ++channel) {
        resample_channel(
            input.samples.data(),
            input_frames,
            input.channels,
            channel,
            plan,
            output.samples.data(),
            output_frames);
    }
    for (float sample : output.samples) {
        if (!std::isfinite(sample)) {
            output = {};
            return Status::error(
                "resampling produced non-finite samples");
        }
    }
    return Status::ok();
}

Status resample_audio_to_24k(
    const AudioBuffer& input,
    AudioBuffer& output) {
    return resample_audio(input, OMNIVOICE_SAMPLE_RATE, output);
}

std::vector<float> resample_24k_to_16k(
    const std::vector<float>& input) {
    if (input.empty()) {
        return {};
    }

    ResamplePlan plan;
    const Status status = build_plan(24000, 16000, plan);
    if (!status) {
        return {};
    }
    return resample_mono(input, plan);
}

} // namespace omnivoice
