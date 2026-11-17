#pragma once

#include "status.h"

#include <cstdint>
#include <optional>
#include <string>

namespace omnivoice {

struct DurationEstimateOptions {
    std::optional<double> low_threshold = 50.0;
    double boost_strength = 3.0;
};

class RuleDurationEstimator {
public:
    Status calculate_total_weight(
        const std::string& text,
        double& weight) const;

    Status estimate_duration(
        const std::string& target_text,
        const std::string& reference_text,
        double reference_duration,
        const DurationEstimateOptions& options,
        double& duration) const;
};

Status estimate_target_frames(
    const std::string& target_text,
    const std::string& reference_text,
    int32_t reference_frames,
    float speed,
    int32_t& target_frames,
    const RuleDurationEstimator& estimator = {});

} // namespace omnivoice
