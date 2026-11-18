#include "diffusion_decoder/token_selector.h"

#include "ggml-backend.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

bool close_to(float actual, float expected, float tolerance = 2e-5f) {
    return std::abs(actual - expected) <= tolerance;
}

std::vector<float> flatten_floats(const nlohmann::json& value) {
    std::vector<float> result;
    const auto append = [&](const auto& self,
                            const nlohmann::json& node) -> void {
        if (node.is_number()) {
            result.push_back(node.get<float>());
            return;
        }
        for (const nlohmann::json& child : node) {
            self(self, child);
        }
    };
    append(append, value);
    return result;
}

std::vector<int32_t> flatten_i32(const nlohmann::json& value) {
    std::vector<int32_t> result;
    const auto append = [&](const auto& self,
                            const nlohmann::json& node) -> void {
        if (node.is_number_integer()) {
            result.push_back(node.get<int32_t>());
            return;
        }
        for (const nlohmann::json& child : node) {
            self(self, child);
        }
    };
    append(append, value);
    return result;
}

std::vector<float> to_frame_major(
    const std::vector<float>& codebook_major,
    int32_t codebooks,
    int32_t frames,
    int32_t vocab) {
    std::vector<float> result(codebook_major.size());
    for (int32_t frame = 0; frame < frames; ++frame) {
        for (int32_t codebook = 0; codebook < codebooks; ++codebook) {
            const size_t source =
                static_cast<size_t>(codebook) *
                    static_cast<size_t>(frames) *
                    static_cast<size_t>(vocab) +
                static_cast<size_t>(frame) *
                    static_cast<size_t>(vocab);
            const size_t destination =
                static_cast<size_t>(frame) *
                    static_cast<size_t>(codebooks) *
                    static_cast<size_t>(vocab) +
                static_cast<size_t>(codebook) *
                    static_cast<size_t>(vocab);
            std::copy(
                codebook_major.begin() +
                    static_cast<std::ptrdiff_t>(source),
                codebook_major.begin() +
                    static_cast<std::ptrdiff_t>(source + vocab),
                result.begin() +
                    static_cast<std::ptrdiff_t>(destination));
        }
    }
    return result;
}

} // namespace

int main() {
    std::ifstream file(
        std::string(OMNIVOICE_TEST_DATA_DIR) +
        "/token_selector_cases.json");
    if (!file) {
        std::cerr << "failed to open token selector golden data\n";
        return 1;
    }
    nlohmann::json golden;
    file >> golden;

    const int32_t codebooks =
        static_cast<int32_t>(golden.at("cond").size());
    const int32_t frames =
        static_cast<int32_t>(golden.at("cond")[0].size());
    const int32_t vocab =
        static_cast<int32_t>(golden.at("cond")[0][0].size());
    const std::vector<float> cond = flatten_floats(golden.at("cond"));
    const std::vector<float> uncond = flatten_floats(golden.at("uncond"));
    const std::vector<int32_t> current =
        flatten_i32(golden.at("current_tokens"));

    omnivoice::TokenSelectorOptions options;
    options.guidance_scale =
        golden.at("guidance_scale").get<float>();
    options.layer_penalty_factor =
        golden.at("layer_penalty_factor").get<float>();
    options.position_temperature = 0.0f;
    options.class_temperature = 0.0f;

    omnivoice::TokenSelectionInput input;
    input.codebooks = codebooks;
    input.frames = frames;
    input.vocab_size = vocab;
    input.mask_id = golden.at("mask_id").get<int32_t>();
    input.top_k = golden.at("top_k").get<int32_t>();
    input.cond_logits = &cond;
    input.uncond_logits = &uncond;
    input.current_tokens = &current;

    omnivoice::TokenSelector selector;
    omnivoice::TokenSelectionOutput output;
    omnivoice::Status status = selector.select(input, options, output);
    if (!status) {
        std::cerr << "token selection failed: "
                  << status.message() << '\n';
        return 1;
    }

    if (output.predictions !=
        flatten_i32(golden.at("predictions"))) {
        std::cerr << "predicted token mismatch\n";
        return 1;
    }
    const std::vector<float> expected_confidence =
        flatten_floats(golden.at("confidence"));
    for (size_t index = 0; index < output.confidence.size(); ++index) {
        if (!close_to(output.confidence[index], expected_confidence[index])) {
            std::cerr << "confidence mismatch at " << index << '\n';
            return 1;
        }
    }
    const std::vector<int32_t> expected_indices =
        golden.at("selected_indices").get<std::vector<int32_t>>();
    const std::vector<int32_t> expected_tokens =
        golden.at("selected_tokens").get<std::vector<int32_t>>();
    if (output.selected_count != input.top_k ||
        output.selected_mask.size() != current.size()) {
        std::cerr << "selected mask shape mismatch\n";
        return 1;
    }
    std::vector<uint8_t> expected_mask(current.size(), 0);
    for (size_t index = 0; index < expected_indices.size(); ++index) {
        const size_t cell =
            static_cast<size_t>(expected_indices[index]);
        expected_mask[cell] = 1;
        if (output.predictions[cell] != expected_tokens[index]) {
            std::cerr << "selected token mismatch\n";
            return 1;
        }
    }
    if (output.selected_mask != expected_mask) {
        std::cerr << "selected cell mismatch\n";
        return 1;
    }

    omnivoice::TokenSelectorOptions stochastic = options;
    stochastic.class_temperature = 0.7f;
    stochastic.position_temperature = 0.8f;
    omnivoice::TokenSelector selector_a;
    omnivoice::TokenSelector selector_b;
    selector_a.seed(1234);
    selector_b.seed(1234);
    omnivoice::TokenSelectionOutput output_a;
    omnivoice::TokenSelectionOutput output_b;
    status = selector_a.select(input, stochastic, output_a);
    if (status) {
        status = selector_b.select(input, stochastic, output_b);
    }
    if (!status ||
        output_a.predictions != output_b.predictions ||
        output_a.selected_mask != output_b.selected_mask ||
        output_a.selected_count != output_b.selected_count) {
        std::cerr << "seeded token selection is not reproducible\n";
        return 1;
    }

    input.top_k = 100;
    if (selector.select(input, options, output)) {
        std::cerr << "oversized top_k must be rejected\n";
        return 1;
    }
    input.top_k = 3;
    options.guidance_scale = -1.0f;
    if (selector.select(input, options, output)) {
        std::cerr << "negative guidance must be rejected\n";
        return 1;
    }
    return 0;
}
