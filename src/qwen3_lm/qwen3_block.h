#pragma once

#include "config.h"

#include "ggml.h"

namespace omnivoice {

struct Qwen3BlockWeights {
    ggml_tensor* input_layernorm_weight = nullptr;
    ggml_tensor* post_attention_layernorm_weight = nullptr;

    ggml_tensor* q_proj_weight = nullptr;
    ggml_tensor* k_proj_weight = nullptr;
    ggml_tensor* v_proj_weight = nullptr;
    ggml_tensor* qkv_proj_weight = nullptr;
    ggml_tensor* o_proj_weight = nullptr;
    ggml_tensor* q_norm_weight = nullptr;
    ggml_tensor* k_norm_weight = nullptr;

    ggml_tensor* gate_proj_weight = nullptr;
    ggml_tensor* up_proj_weight = nullptr;
    ggml_tensor* down_proj_weight = nullptr;
};

// Builds one bidirectional Qwen3 transformer block.
//
// input:     [hidden_size, sequence_length]
// positions: [sequence_length] I32
// mask:      optional [sequence_length, sequence_length, 1, 1] F16
//            nullptr selects the fully visible no-mask path
// output:    [hidden_size, sequence_length]
//
// The graph has no KV cache and no causal mask.
ggml_tensor* build_qwen3_block(
    ggml_context* context,
    const Qwen3Config& config,
    const Qwen3BlockWeights& weights,
    ggml_tensor* input,
    ggml_tensor* positions,
    ggml_tensor* attention_mask);

ggml_tensor* build_qwen3_packed_block(
    ggml_context* context,
    const Qwen3Config& config,
    const Qwen3BlockWeights& weights,
    ggml_tensor* input,
    ggml_tensor* first_positions,
    ggml_tensor* second_positions,
    int64_t first_sequence_length);

} // namespace omnivoice
