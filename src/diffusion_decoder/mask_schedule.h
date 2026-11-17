#pragma once

#include "status.h"

#include <cstdint>
#include <vector>

namespace omnivoice {

struct MaskSchedule {
    int32_t target_frames = 0;
    int32_t codebooks = 0;
    int32_t total_positions = 0;
    std::vector<int32_t> counts;
};

// Matches OmniVoice _get_time_steps:
// t' = t_shift * t / (1 + (t_shift - 1) * t).
Status get_mask_time_steps(
    int32_t num_steps,
    float t_shift,
    std::vector<float>& timesteps);

// Each count is the number of (codebook, frame) cells unmasked in one step.
Status build_mask_schedule(
    int32_t target_frames,
    int32_t codebooks,
    int32_t num_steps,
    float t_shift,
    MaskSchedule& schedule);

} // namespace omnivoice
