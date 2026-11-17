#pragma once

#include "status.h"

#include "ggml-backend.h"

#include <cstdint>
#include <memory>
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

    Status init_graph(
        ggml_backend_t backend,
        int32_t codebooks,
        int32_t frames,
        int32_t vocab_size,
        int32_t mask_id,
        float guidance_scale);
    Status select(
        const TokenSelectionInput& input,
        const TokenSelectorOptions& options,
        TokenSelectionOutput& output);

    Status select_graph(
        const TokenSelectionInput& input,
        const TokenSelectorOptions& options,
        const ggml_tensor* conditional_logits,
        const ggml_tensor* unconditional_logits,
        TokenSelectionOutput& output);

private:
    class Graph;

    float uniform();
    float gumbel(float score, float temperature);

    std::mt19937_64 rng_;
    std::unique_ptr<Graph> graph_;
};

} // namespace omnivoice
