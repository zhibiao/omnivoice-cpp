#include "diffusion_decoder/mask_schedule.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

int main() {
    std::ifstream file(
        std::string(OMNIVOICE_TEST_DATA_DIR) +
        "/mask_schedule_cases.json");
    if (!file) {
        std::cerr << "failed to open mask schedule golden data\n";
        return 1;
    }
    nlohmann::json cases;
    file >> cases;

    for (const nlohmann::json& item : cases) {
        omnivoice::MaskSchedule schedule;
        const omnivoice::Status status =
            omnivoice::build_mask_schedule(
                item.at("target_frames").get<int32_t>(),
                item.at("codebooks").get<int32_t>(),
                item.at("num_steps").get<int32_t>(),
                item.at("t_shift").get<float>(),
                schedule);
        if (!status) {
            std::cerr << "mask schedule failed: "
                      << status.message() << '\n';
            return 1;
        }
        const std::vector<int32_t> expected =
            item.at("counts").get<std::vector<int32_t>>();
        if (schedule.counts != expected) {
            std::cerr << "mask schedule count mismatch for frames="
                      << schedule.target_frames << '\n';
            return 1;
        }
        const int32_t sum = std::accumulate(
            schedule.counts.begin(),
            schedule.counts.end(),
            int32_t{0});
        if (sum != schedule.total_positions) {
            std::cerr << "mask schedule did not cover all positions\n";
            return 1;
        }
    }

    std::vector<float> timesteps;
    omnivoice::Status status = omnivoice::get_mask_time_steps(
        4,
        0.1f,
        timesteps);
    const std::vector<float> expected_steps = {
        0.0f,
        0.03225806728005409f,
        0.09090909361839294f,
        0.23076920211315155f,
        0.9999997615814209f,
    };
    if (!status || timesteps.size() != expected_steps.size()) {
        std::cerr << "mask timestep shape mismatch\n";
        return 1;
    }
    for (size_t index = 0; index < timesteps.size(); ++index) {
        if (std::abs(timesteps[index] - expected_steps[index]) > 1e-6f) {
            std::cerr << "mask timestep value mismatch\n";
            return 1;
        }
    }

    omnivoice::MaskSchedule invalid;
    if (omnivoice::build_mask_schedule(
            0, 8, 32, 0.1f, invalid) ||
        omnivoice::build_mask_schedule(
            10, 8, 0, 0.1f, invalid) ||
        omnivoice::build_mask_schedule(
            10, 8, 32, 0.0f, invalid)) {
        std::cerr << "invalid mask schedule parameters must fail\n";
        return 1;
    }
    return 0;
}
