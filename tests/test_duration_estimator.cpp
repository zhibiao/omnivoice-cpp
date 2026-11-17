#include "duration_estimator.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

namespace {

bool close_to(double actual, double expected, double tolerance = 1e-9) {
    return std::abs(actual - expected) <= tolerance;
}

} // namespace

int main() {
    std::ifstream file(
        std::string(OMNIVOICE_TEST_DATA_DIR) + "/duration_cases.json");
    if (!file) {
        std::cerr << "failed to open duration golden data\n";
        return 1;
    }
    nlohmann::json golden;
    file >> golden;
    const std::string reference_text =
        golden.at("reference_text").get<std::string>();

    omnivoice::RuleDurationEstimator estimator;
    for (const nlohmann::json& item : golden.at("cases")) {
        const std::string text = item.at("text").get<std::string>();
        double weight = 0.0;
        omnivoice::Status status =
            estimator.calculate_total_weight(text, weight);
        if (!status ||
            !close_to(weight, item.at("weight").get<double>())) {
            std::cerr << "duration weight mismatch for " << text
                      << " actual=" << weight << '\n';
            return 1;
        }

        double duration5 = 0.0;
        status = estimator.estimate_duration(
            text,
            reference_text,
            5.0,
            {},
            duration5);
        if (!status ||
            !close_to(
                duration5,
                item.at("duration_ref5").get<double>(),
                1e-9)) {
            std::cerr << "duration_ref5 mismatch for " << text
                      << " actual=" << duration5 << '\n';
            return 1;
        }

        double duration50 = 0.0;
        status = estimator.estimate_duration(
            text,
            reference_text,
            50.0,
            {},
            duration50);
        if (!status ||
            !close_to(
                duration50,
                item.at("duration_ref50").get<double>(),
                1e-9)) {
            std::cerr << "duration_ref50 mismatch for " << text
                      << " actual=" << duration50 << '\n';
            return 1;
        }
    }

    int32_t frames = 0;
    omnivoice::Status status = omnivoice::estimate_target_frames(
        "Hello, world.",
        reference_text,
        5,
        1.0f,
        frames,
        estimator);
    if (!status || frames != 23) {
        std::cerr << "target frame estimate mismatch: " << frames << '\n';
        return 1;
    }
    status = omnivoice::estimate_target_frames(
        "Hello, world.",
        reference_text,
        5,
        2.0f,
        frames,
        estimator);
    if (!status || frames != 11) {
        std::cerr << "speed-adjusted frame estimate mismatch: "
                  << frames << '\n';
        return 1;
    }

    double fallback_duration = 0.0;
    status = estimator.estimate_duration(
        "Hello, world.",
        "Nice to meet you.",
        25.0,
        {},
        fallback_duration);
    if (!status) {
        return 1;
    }
    status = omnivoice::estimate_target_frames(
        "Hello, world.",
        "",
        0,
        1.0f,
        frames,
        estimator);
    if (!status ||
        frames != static_cast<int32_t>(fallback_duration)) {
        std::cerr << "duration fallback mismatch\n";
        return 1;
    }

    status = omnivoice::estimate_target_frames(
        "Hello",
        reference_text,
        5,
        0.0f,
        frames,
        estimator);
    if (status) {
        std::cerr << "zero speed must be rejected\n";
        return 1;
    }
    return 0;
}
