#pragma once

#include <vector>

namespace omnivoice {

// Matches torchaudio.functional.resample with sinc_interp_hann,
// lowpass_filter_width=6 and rolloff=0.99.
std::vector<float> resample_24k_to_16k(
    const std::vector<float>& input);

} // namespace omnivoice
