#pragma once

#include "status.h"

#include <cstdint>
#include <random>
#include <vector>

namespace omnivoice {

struct TokenSelectorOptions {
    float guidance_scale = 2.0f;
    float layer_penalty_factor = 5.0f;
    float position_temperature = 5.0f;
    float class_temperature = 0.0f;
};

struct TokenSelectionInput {
    int32_t codebooks = 0;
    int32_t frames = 0;
    int32_t vocab_size = 0;
    int32_t mask_id = 0;
    int32_t top_k = 0;

    // All tensors are codebook-major [codebooks, frames, vocab_size].
    const std::vector<float>* cond_logits = nullptr;
    const std::vector<float>* uncond_logits = nullptr;

    // Codebook-major [codebooks, frames].
    const std::vector<int32_t>* current_tokens = nullptr;
};

struct TokenSelectionOutput {
    std::vector<int32_t> predictions;
    std::vector<float> confidence;
    std::vector<float> scores;
    // One fixed-size value per codebook/frame cell.
    std::vector<uint8_t> selected_mask;
    int32_t selected_count = 0;
};

class TokenSelector {
public:
    TokenSelector();
    ~TokenSelector();

    TokenSelector(const TokenSelector&) = delete;
    TokenSelector& operator=(const TokenSelector&) = delete;

    void seed(uint64_t value);

    Status select(
        const TokenSelectionInput& input,
        const TokenSelectorOptions& options,
        TokenSelectionOutput& output);

    Status select_scored(
        const TokenSelectionInput& input,
        const TokenSelectorOptions& options,
        const std::vector<int32_t>& predictions,
        const std::vector<float>& confidence,
        TokenSelectionOutput& output);

private:
    float uniform();
    float gumbel(float score, float temperature);

    std::mt19937_64 rng_;
};

} // namespace omnivoice
