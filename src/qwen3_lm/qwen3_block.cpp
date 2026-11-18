#include "qwen3_lm/qwen3_block.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace omnivoice {

namespace {

ggml_tensor* require_tensor(
    ggml_tensor* tensor,
    const char* name,
    int64_t expected_ne0,
    int64_t expected_ne1 = -1) {
    if (tensor == nullptr) {
        throw std::runtime_error(std::string("missing Qwen3 tensor: ") + name);
    }
    if (tensor->ne[0] != expected_ne0 ||
        (expected_ne1 >= 0 && tensor->ne[1] != expected_ne1)) {
        throw std::runtime_error(
            std::string("unexpected Qwen3 tensor shape: ") + name);
    }
    return tensor;
}

ggml_tensor* cast_f32(ggml_context* context, ggml_tensor* tensor) {
    return tensor->type == GGML_TYPE_F32
        ? tensor
        : ggml_cast(context, tensor, GGML_TYPE_F32);
}

ggml_tensor* linear(
    ggml_context* context,
    ggml_tensor* weight,
    ggml_tensor* input) {
    return ggml_mul_mat(context, weight, cast_f32(context, input));
}

ggml_tensor* rms_norm(
    ggml_context* context,
    ggml_tensor* input,
    ggml_tensor* weight,
    float epsilon) {
    ggml_tensor* normalized = ggml_rms_norm(context, input, epsilon);
    ggml_tensor* column = ggml_reshape_2d(
        context,
        cast_f32(context, weight),
        weight->ne[0],
        1);
    return ggml_mul(
        context,
        normalized,
        column);
}

ggml_tensor* packed_projection_slice(
    ggml_context* context,
    ggml_tensor* packed,
    int64_t head_dim,
    int64_t heads,
    int64_t sequence_length,
    int64_t channel_offset) {
    return ggml_view_3d(
        context,
        packed,
        head_dim,
        heads,
        sequence_length,
        head_dim * ggml_element_size(packed),
        packed->nb[1],
        static_cast<size_t>(channel_offset) *
            ggml_element_size(packed));
}

ggml_tensor* attention(
    ggml_context* context,
    const Qwen3Config& config,
    const Qwen3BlockWeights& weights,
    ggml_tensor* input,
    ggml_tensor* positions,
    ggml_tensor* attention_mask,
    int64_t sequence_length) {
    const int64_t head_dim = config.head_dim;
    const int64_t heads = config.num_attention_heads;
    const int64_t kv_heads = config.num_key_value_heads;
    const int64_t hidden = config.hidden_size;
    const int64_t query_dim = heads * head_dim;
    const int64_t kv_dim = kv_heads * head_dim;

    ggml_tensor* query = nullptr;
    ggml_tensor* key = nullptr;
    ggml_tensor* value = nullptr;
    if (weights.qkv_proj_weight != nullptr) {
        ggml_tensor* packed = linear(
            context,
            require_tensor(
                weights.qkv_proj_weight,
                "qkv_proj.weight",
                hidden,
                query_dim + 2 * kv_dim),
            input);
        query = packed_projection_slice(
            context,
            packed,
            head_dim,
            heads,
            sequence_length,
            0);
        key = packed_projection_slice(
            context,
            packed,
            head_dim,
            kv_heads,
            sequence_length,
            query_dim);
        value = packed_projection_slice(
            context,
            packed,
            head_dim,
            kv_heads,
            sequence_length,
            query_dim + kv_dim);
        query = ggml_cont(context, query);
        key = ggml_cont(context, key);
        value = ggml_cont(context, value);
    } else {
        query = linear(
            context,
            require_tensor(
                weights.q_proj_weight,
                "q_proj.weight",
                hidden,
                query_dim),
            input);
        key = linear(
            context,
            require_tensor(
                weights.k_proj_weight,
                "k_proj.weight",
                hidden,
                kv_dim),
            input);
        value = linear(
            context,
            require_tensor(
                weights.v_proj_weight,
                "v_proj.weight",
                hidden,
                kv_dim),
            input);
        query = ggml_reshape_3d(
            context,
            query,
            head_dim,
            heads,
            sequence_length);
        key = ggml_reshape_3d(
            context,
            key,
            head_dim,
            kv_heads,
            sequence_length);
        value = ggml_reshape_3d(
            context,
            value,
            head_dim,
            kv_heads,
            sequence_length);
    }

    query = rms_norm(
        context,
        query,
        require_tensor(
            weights.q_norm_weight,
            "q_norm.weight",
            head_dim),
        config.rms_norm_epsilon);
    key = rms_norm(
        context,
        key,
        require_tensor(
            weights.k_norm_weight,
            "k_norm.weight",
            head_dim),
        config.rms_norm_epsilon);

    if (config.rope_type != "default") {
        throw std::runtime_error("unsupported Qwen3 RoPE type");
    }
    query = ggml_rope_ext(
        context,
        query,
        positions,
        nullptr,
        head_dim,
        GGML_ROPE_TYPE_NEOX,
        config.max_position_embeddings,
        config.rope_theta,
        1.0f,
        0.0f,
        1.0f,
        0.0f,
        0.0f);
    key = ggml_rope_ext(
        context,
        key,
        positions,
        nullptr,
        head_dim,
        GGML_ROPE_TYPE_NEOX,
        config.max_position_embeddings,
        config.rope_theta,
        1.0f,
        0.0f,
        1.0f,
        0.0f,
        0.0f);

    ggml_tensor* query_rows = ggml_permute(
        context,
        query,
        0,
        2,
        1,
        3);
    ggml_tensor* key_rows = ggml_permute(
        context,
        key,
        0,
        2,
        1,
        3);
    ggml_tensor* value_rows = ggml_permute(
        context,
        value,
        0,
        2,
        1,
        3);

    ggml_tensor* output = ggml_flash_attn_ext(
        context,
        query_rows,
        key_rows,
        value_rows,
        attention_mask,
        1.0f / std::sqrt(static_cast<float>(head_dim)),
        0.0f,
        0.0f);
    ggml_flash_attn_ext_set_prec(output, GGML_PREC_F32);
    output = ggml_cont_2d(
        context,
        output,
        query_dim,
        sequence_length);
    return linear(
        context,
        require_tensor(
            weights.o_proj_weight,
            "o_proj.weight",
            query_dim,
            hidden),
        output);
}

} // namespace

ggml_tensor* build_qwen3_block(
    ggml_context* context,
    const Qwen3Config& config,
    const Qwen3BlockWeights& weights,
    ggml_tensor* input,
    ggml_tensor* positions,
    ggml_tensor* attention_mask) {
    if (context == nullptr || input == nullptr ||
        positions == nullptr) {
        throw std::runtime_error("Qwen3 block received a null tensor");
    }
    if (config.hidden_activation != "silu") {
        throw std::runtime_error("Qwen3 block only supports SiLU/SwiGLU");
    }
    if (input->ne[0] != config.hidden_size ||
        positions->ne[0] != input->ne[1]) {
        throw std::runtime_error("Qwen3 block input shape mismatch");
    }
    const int64_t sequence_length = input->ne[1];

    ggml_tensor* residual = cast_f32(context, input);
    ggml_tensor* hidden = rms_norm(
        context,
        residual,
        require_tensor(
            weights.input_layernorm_weight,
            "input_layernorm.weight",
            config.hidden_size),
        config.rms_norm_epsilon);
    hidden = attention(
        context,
        config,
        weights,
        hidden,
        positions,
        attention_mask,
        sequence_length);
    hidden = ggml_add(context, residual, hidden);

    residual = hidden;
    hidden = rms_norm(
        context,
        hidden,
        require_tensor(
            weights.post_attention_layernorm_weight,
            "post_attention_layernorm.weight",
            config.hidden_size),
        config.rms_norm_epsilon);
    ggml_tensor* gate = linear(
        context,
        require_tensor(
            weights.gate_proj_weight,
            "mlp.gate_proj.weight",
            config.hidden_size,
            config.intermediate_size),
        hidden);
    ggml_tensor* up = linear(
        context,
        require_tensor(
            weights.up_proj_weight,
            "mlp.up_proj.weight",
            config.hidden_size,
            config.intermediate_size),
        hidden);
    hidden = ggml_swiglu_split(context, gate, up);
    hidden = linear(
        context,
        require_tensor(
            weights.down_proj_weight,
            "mlp.down_proj.weight",
            config.intermediate_size,
            config.hidden_size),
        hidden);
    return ggml_add(context, residual, hidden);
}

ggml_tensor* build_qwen3_packed_block(
    ggml_context* context,
    const Qwen3Config& config,
    const Qwen3BlockWeights& weights,
    ggml_tensor* input,
    ggml_tensor* first_positions,
    ggml_tensor* second_positions,
    int64_t first_sequence_length) {
    if (context == nullptr ||
        input == nullptr ||
        first_positions == nullptr ||
        second_positions == nullptr) {
        throw std::runtime_error(
            "packed Qwen3 block received a null tensor");
    }
    const int64_t sequence_length = input->ne[1];
    const int64_t second_sequence_length =
        sequence_length - first_sequence_length;
    if (input->ne[0] != config.hidden_size ||
        first_sequence_length <= 0 ||
        second_sequence_length <= 0 ||
        first_positions->ne[0] != first_sequence_length ||
        second_positions->ne[0] != second_sequence_length) {
        throw std::runtime_error(
            "packed Qwen3 block input shape mismatch");
    }

    ggml_tensor* residual = cast_f32(context, input);
    ggml_tensor* hidden = rms_norm(
        context,
        residual,
        require_tensor(
            weights.input_layernorm_weight,
            "input_layernorm.weight",
            config.hidden_size),
        config.rms_norm_epsilon);
    ggml_tensor* first_input = ggml_cont_2d(
        context,
        ggml_view_2d(
            context,
            hidden,
            config.hidden_size,
            first_sequence_length,
            hidden->nb[1],
            0),
        config.hidden_size,
        first_sequence_length);
    ggml_tensor* second_input = ggml_cont_2d(
        context,
        ggml_view_2d(
            context,
            hidden,
            config.hidden_size,
            second_sequence_length,
            hidden->nb[1],
            static_cast<size_t>(first_sequence_length) * hidden->nb[1]),
        config.hidden_size,
        second_sequence_length);
    hidden = ggml_concat(
        context,
        attention(
            context,
            config,
            weights,
            first_input,
            first_positions,
            nullptr,
            first_sequence_length),
        attention(
            context,
            config,
            weights,
            second_input,
            second_positions,
            nullptr,
            second_sequence_length),
        1);
    hidden = ggml_add(context, residual, hidden);

    residual = hidden;
    hidden = rms_norm(
        context,
        hidden,
        require_tensor(
            weights.post_attention_layernorm_weight,
            "post_attention_layernorm.weight",
            config.hidden_size),
        config.rms_norm_epsilon);
    ggml_tensor* gate = linear(
        context,
        require_tensor(
            weights.gate_proj_weight,
            "mlp.gate_proj.weight",
            config.hidden_size,
            config.intermediate_size),
        hidden);
    ggml_tensor* up = linear(
        context,
        require_tensor(
            weights.up_proj_weight,
            "mlp.up_proj.weight",
            config.hidden_size,
            config.intermediate_size),
        hidden);
    hidden = ggml_swiglu_split(context, gate, up);
    hidden = linear(
        context,
        require_tensor(
            weights.down_proj_weight,
            "mlp.down_proj.weight",
            config.intermediate_size,
            config.hidden_size),
        hidden);
    return ggml_add(context, residual, hidden);
}

} // namespace omnivoice
