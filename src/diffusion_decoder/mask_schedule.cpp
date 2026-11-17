#include "diffusion_decoder/mask_schedule.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace omnivoice {

Status get_mask_time_steps(
    int32_t num_steps,
    float t_shift,
    std::vector<float>& timesteps) {
    timesteps.clear();
    if (num_steps <= 0) {
        return Status::error("mask schedule num_steps must be positive");
    }
    if (!std::isfinite(t_shift) || t_shift <= 0.0f) {
        return Status::error("mask schedule t_shift must be positive");
    }

    timesteps.resize(static_cast<size_t>(num_steps) + 1);
    for (int32_t index = 0; index <= num_steps; ++index) {
        const float step = static_cast<float>(index) /
            static_cast<float>(num_steps);
        const float denominator =
            1.0f + (t_shift - 1.0f) * step;
        if (denominator <= 0.0f || !std::isfinite(denominator)) {
            return Status::error("mask schedule generated an invalid timestep");
        }
        timesteps[static_cast<size_t>(index)] =
            t_shift * step / denominator;
    }
    return Status::ok();
}

Status build_mask_schedule(
    int32_t target_frames,
    int32_t codebooks,
    int32_t num_steps,
    float t_shift,
    MaskSchedule& schedule) {
    schedule = {};
    if (target_frames <= 0 || codebooks <= 0) {
        return Status::error(
            "mask schedule dimensions must be positive");
    }
    const int64_t total =
        static_cast<int64_t>(target_frames) * codebooks;
    if (total > std::numeric_limits<int32_t>::max()) {
        return Status::error("mask schedule position count is too large");
    }

    std::vector<float> timesteps;
    Status status = get_mask_time_steps(
        num_steps,
        t_shift,
        timesteps);
    if (!status) {
        return status;
    }

    schedule.target_frames = target_frames;
    schedule.codebooks = codebooks;
    schedule.total_positions = static_cast<int32_t>(total);
    schedule.counts.reserve(static_cast<size_t>(num_steps));

    int64_t remaining = total;
    for (int32_t step = 0; step < num_steps; ++step) {
        int64_t count = remaining;
        if (step != num_steps - 1) {
            const double delta =
                static_cast<double>(timesteps[static_cast<size_t>(step + 1)]) -
                static_cast<double>(timesteps[static_cast<size_t>(step)]);
            const double fraction =
                static_cast<double>(total) * delta;
            if (fraction < 0.0 ||
                !std::isfinite(fraction)) {
                return Status::error("mask schedule produced an invalid count");
            }
            count = static_cast<int64_t>(std::ceil(fraction));
            count = std::min(count, remaining);
        }
        schedule.counts.push_back(static_cast<int32_t>(count));
        remaining -= count;
    }
    if (remaining != 0) {
        return Status::error("mask schedule did not consume all positions");
    }
    return Status::ok();
}

} // namespace omnivoice
