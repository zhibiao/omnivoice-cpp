#pragma once

#include "status.h"
#include "types.h"

#include <vector>

namespace omnivoice {

// High-quality band-limited resampler: Hann-windowed sinc with
// lowpass_filter_width=6 and rolloff=0.99, matching
// torchaudio.functional.resample(sinc_interp_hann).
//
// The channel layout is preserved and every channel is filtered
// independently. The output frame count is
// ceil(target_rate * input_frames / input.sample_rate).
//
// Typical rates (8, 11.025, 16, 22.05, 32, 44.1, 48, 88.2, 96 kHz and integer
// multiples of 24 kHz) reduce to small kernels. Ratios whose kernel table
// would be unreasonably large are rejected instead of allocating gigabytes.
Status resample_audio(
    const AudioBuffer& input,
    int32_t target_rate,
    AudioBuffer& output);

// Convenience wrapper that resamples to OMNIVOICE_SAMPLE_RATE (24 kHz).
Status resample_audio_to_24k(
    const AudioBuffer& input,
    AudioBuffer& output);

// Mono 24 kHz -> 16 kHz, used by the Higgs audio encoder. Kept bit-exact with
// the original fixed-ratio implementation so encoder outputs do not drift.
std::vector<float> resample_24k_to_16k(
    const std::vector<float>& input);

} // namespace omnivoice
