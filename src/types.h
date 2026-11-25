#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace omnivoice {

// Native sample rate of the OmniVoice codec.
constexpr int32_t OMNIVOICE_SAMPLE_RATE = 24000;

// Number of samples covered by one codec frame.
constexpr int32_t OMNIVOICE_HOP_LENGTH = 960;

using LogCallback = std::function<void(const std::string&)>;

inline void log_message(
    const LogCallback& log,
    const std::string& message) {
    if (log != nullptr) {
        log(message);
    }
}

struct AudioBuffer {
    std::vector<float> samples;
    int32_t sample_rate = OMNIVOICE_SAMPLE_RATE;
    int32_t channels = 1;

    // Samples are frame-major. For stereo, the layout is L,R,L,R...
    double duration_seconds() const {
        if (samples.empty() || sample_rate <= 0 || channels <= 0) {
            return 0.0;
        }
        return static_cast<double>(samples.size()) /
            (static_cast<double>(sample_rate) *
             static_cast<double>(channels));
    }
};

// OmniVoice audio codes are codebook-major: [codebooks, frames].
// data[codebook * frames + frame] stores one quantized token.
struct AudioCodes {
    int32_t frames = 0;
    int32_t codebooks = 8;
    std::vector<int32_t> data;

    size_t size() const {
        return static_cast<size_t>(frames) *
            static_cast<size_t>(codebooks);
    }
};

} // namespace omnivoice
