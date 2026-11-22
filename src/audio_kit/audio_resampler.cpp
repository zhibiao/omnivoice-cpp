#include "audio_kit/audio_resampler.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace omnivoice {

namespace {

constexpr int32_t ORIGINAL_RATE = 24000;
constexpr int32_t TARGET_RATE = 16000;
constexpr int32_t GCD = 8000;
constexpr int32_t REDUCED_ORIGINAL = ORIGINAL_RATE / GCD;
constexpr int32_t REDUCED_TARGET = TARGET_RATE / GCD;
constexpr int32_t LOWPASS_FILTER_WIDTH = 6;
constexpr double ROLLOFF = 0.99;

struct ResampleKernel {
    int32_t width = 0;
    std::vector<float> values;  // [REDUCED_TARGET, 2*width+reduced_original]
};

ResampleKernel make_kernel() {
    const double base_freq =
        static_cast<double>(
            std::min(REDUCED_ORIGINAL, REDUCED_TARGET)) *
        ROLLOFF;
    const int32_t width = static_cast<int32_t>(std::ceil(
        static_cast<double>(LOWPASS_FILTER_WIDTH) *
        static_cast<double>(REDUCED_ORIGINAL) / base_freq));
    const int32_t kernel_size =
        2 * width + REDUCED_ORIGINAL;
    const double scale =
        base_freq / static_cast<double>(REDUCED_ORIGINAL);

    ResampleKernel kernel;
    kernel.width = width;
    kernel.values.resize(
        static_cast<size_t>(REDUCED_TARGET) *
        static_cast<size_t>(kernel_size));
    constexpr double PI = 3.14159265358979323846264338327950288;

    for (int32_t phase = 0; phase < REDUCED_TARGET; ++phase) {
        for (int32_t index = 0; index < kernel_size; ++index) {
            double t =
                static_cast<double>(-phase) /
                    static_cast<double>(REDUCED_TARGET) +
                static_cast<double>(index - width) /
                    static_cast<double>(REDUCED_ORIGINAL);
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
                    static_cast<size_t>(kernel_size) +
                static_cast<size_t>(index)] =
                static_cast<float>(sinc * window * scale);
        }
    }
    return kernel;
}

} // namespace

std::vector<float> resample_24k_to_16k(
    const std::vector<float>& input) {
    if (input.empty()) {
        return {};
    }

    const ResampleKernel kernel = make_kernel();
    const int64_t input_length =
        static_cast<int64_t>(input.size());
    const int64_t output_length = static_cast<int64_t>(std::ceil(
        static_cast<double>(REDUCED_TARGET) *
        static_cast<double>(input_length) /
        static_cast<double>(REDUCED_ORIGINAL)));
    const int64_t kernel_size =
        static_cast<int64_t>(kernel.values.size()) /
        REDUCED_TARGET;

    std::vector<float> output(static_cast<size_t>(output_length));
    for (int64_t output_index = 0;
         output_index < output_length;
         ++output_index) {
        const int64_t convolution_index =
            output_index / REDUCED_TARGET;
        const int32_t phase = static_cast<int32_t>(
            output_index % REDUCED_TARGET);
        const float* filter =
            kernel.values.data() +
            static_cast<ptrdiff_t>(phase) * kernel_size;

        float sum = 0.0f;
        for (int64_t tap = 0; tap < kernel_size; ++tap) {
            const int64_t source_index =
                convolution_index * REDUCED_ORIGINAL +
                tap - kernel.width;
            if (source_index >= 0 &&
                source_index < input_length) {
                sum +=
                    input[static_cast<size_t>(source_index)] *
                    filter[static_cast<size_t>(tap)];
            }
        }
        output[static_cast<size_t>(output_index)] = sum;
    }
    return output;
}

} // namespace omnivoice
