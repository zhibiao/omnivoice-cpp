#include "higgs_tokenizer/encoder.h"

#include "gguf_model.h"
#include "audio_kit.h"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace omnivoice {

namespace {

constexpr float LAYER_NORM_EPSILON = 1e-5f;
constexpr float GROUP_NORM_EPSILON = 1e-5f;
constexpr float SNAKE_EPSILON = 1e-9f;

ggml_tensor* require_weight(
    ggml_tensor* tensor,
    const std::string& name) {
    if (tensor == nullptr || tensor->data == nullptr) {
        throw std::runtime_error(
            "missing Higgs encoder tensor: " + name);
    }
    return tensor;
}

ggml_tensor* cast_f32(
    ggml_context* context,
    ggml_tensor* tensor) {
    return tensor->type == GGML_TYPE_F32
        ? tensor
        : ggml_cast(context, tensor, GGML_TYPE_F32);
}

ggml_tensor* repeat_rows(
    ggml_context* context,
    ggml_tensor* vector,
    ggml_tensor* like) {
    ggml_tensor* row = ggml_reshape_2d(
        context,
        cast_f32(context, vector),
        1,
        vector->ne[0]);
    return ggml_repeat(context, row, like);
}

ggml_tensor* repeat_columns(
    ggml_context* context,
    ggml_tensor* vector,
    ggml_tensor* like) {
    ggml_tensor* column = ggml_reshape_2d(
        context,
        cast_f32(context, vector),
        vector->ne[0],
        1);
    return ggml_repeat(context, column, like);
}

ggml_tensor* linear_channels_first(
    ggml_context* context,
    ggml_tensor* weight,
    ggml_tensor* bias,
    ggml_tensor* input) {
    ggml_tensor* output = ggml_mul_mat(
        context,
        weight,
        cast_f32(context, input));
    if (bias != nullptr) {
        output = ggml_add(
            context,
            output,
            repeat_columns(
                context,
                bias,
                output));
    }
    return output;
}

ggml_tensor* linear_time_major(
    ggml_context* context,
    ggml_tensor* weight,
    ggml_tensor* bias,
    ggml_tensor* input) {
    ggml_tensor* channels_first = ggml_cont(
        context,
        ggml_transpose(context, cast_f32(context, input)));
    ggml_tensor* output = linear_channels_first(
        context,
        weight,
        bias,
        channels_first);
    return ggml_cont(
        context,
        ggml_transpose(context, output));
}

ggml_tensor* layer_norm(
    ggml_context* context,
    ggml_tensor* input,
    ggml_tensor* weight,
    ggml_tensor* bias) {
    ggml_tensor* channels_first = ggml_cont(
        context,
        ggml_transpose(context, cast_f32(context, input)));
    ggml_tensor* normalized = ggml_norm(
        context,
        channels_first,
        LAYER_NORM_EPSILON);
    normalized = ggml_mul(
        context,
        normalized,
        repeat_columns(context, weight, normalized));
    if (bias != nullptr) {
        normalized = ggml_add(
            context,
            normalized,
            repeat_columns(context, bias, normalized));
    }
    return ggml_cont(
        context,
        ggml_transpose(context, normalized));
}

ggml_tensor* group_norm_channels(
    ggml_context* context,
    ggml_tensor* input,
    int64_t channels,
    ggml_tensor* weight,
    ggml_tensor* bias) {
    const int64_t frames = input->ne[0];
    ggml_tensor* grouped = ggml_group_norm(
        context,
        ggml_reshape_3d(
            context,
            cast_f32(context, input),
            frames,
            1,
            channels),
        static_cast<int>(channels),
        GROUP_NORM_EPSILON);
    grouped = ggml_mul(
        context,
        grouped,
        ggml_reshape_3d(
            context,
            repeat_rows(
                context,
                weight,
                ggml_reshape_2d(
                    context,
                    grouped,
                    frames,
                    channels)),
            frames,
            1,
            channels));
    grouped = ggml_add(
        context,
        grouped,
        ggml_reshape_3d(
            context,
            repeat_rows(
                context,
                bias,
                ggml_reshape_2d(
                    context,
                    grouped,
                    frames,
                    channels)),
            frames,
            1,
            channels));
    return ggml_reshape_2d(
        context,
        grouped,
        frames,
        channels);
}

ggml_tensor* conv1d(
    ggml_context* context,
    ggml_tensor* input,
    ggml_tensor* kernel,
    ggml_tensor* bias,
    int stride,
    int padding,
    int dilation,
    int64_t expected_frames = -1) {
    const int64_t input_frames = input->ne[0];
    const int64_t input_channels = input->ne[1];
    const int64_t kernel_size = kernel->ne[0];
    const int64_t output_channels = kernel->ne[2];
    if (kernel->ne[1] != input_channels ||
        kernel_size <= 0 ||
        stride <= 0 ||
        dilation <= 0) {
        throw std::runtime_error(
            "Higgs encoder convolution shape mismatch");
    }

    const int64_t calculated_frames =
        (input_frames + 2 * padding -
         dilation * (kernel_size - 1) - 1) /
            stride +
        1;
    const int64_t frames = expected_frames >= 0
        ? expected_frames
        : calculated_frames;
    if (frames <= 0 || calculated_frames < frames) {
        throw std::runtime_error(
            "Higgs encoder convolution length mismatch: expected " +
            std::to_string(frames) + ", calculated " +
            std::to_string(calculated_frames));
    }

    ggml_tensor* input_3d = ggml_reshape_3d(
        context,
        cast_f32(context, input),
        input_frames,
        input_channels,
        1);
    ggml_tensor* im2col = ggml_im2col(
        context,
        kernel,
        input_3d,
        stride,
        0,
        padding,
        0,
        dilation,
        0,
        false,
        GGML_TYPE_F32);
    ggml_tensor* raw = ggml_mul_mat(
        context,
        ggml_reshape_2d(
            context,
            im2col,
            im2col->ne[0],
            im2col->ne[2] * im2col->ne[1]),
        ggml_reshape_2d(
            context,
            kernel,
            kernel->ne[0] * kernel->ne[1],
            kernel->ne[2]));
    raw = ggml_reshape_3d(
        context,
        raw,
        im2col->ne[1],
        im2col->ne[2],
        kernel->ne[2]);
    raw = ggml_cont(
        context,
        ggml_permute(context, raw, 0, 2, 1, 3));
    if (raw->ne[0] < frames) {
        throw std::runtime_error(
            "Higgs encoder convolution produced too few frames");
    }
    ggml_tensor* output = ggml_cont_2d(
        context,
        ggml_view_2d(
            context,
            raw,
            frames,
            output_channels,
            raw->nb[1],
            0),
        frames,
        output_channels);
    if (bias != nullptr) {
        output = ggml_add(
            context,
            output,
            repeat_rows(context, bias, output));
    }
    return output;
}

ggml_tensor* grouped_position_conv(
    ggml_context* context,
    ggml_tensor* input,
    ggml_tensor* kernel,
    ggml_tensor* bias,
    int64_t groups,
    const std::function<void(
        const std::string&,
        ggml_tensor*)>& record) {
    const int64_t frames = input->ne[0];
    const int64_t channels = input->ne[1];
    if (channels % groups != 0 ||
        kernel->ne[0] != 128 ||
        kernel->ne[1] != channels / groups ||
        kernel->ne[2] != channels) {
        throw std::runtime_error(
            "Higgs positional convolution shape mismatch: input=[" +
            std::to_string(frames) + "," +
            std::to_string(channels) + "] kernel=[" +
            std::to_string(kernel->ne[0]) + "," +
            std::to_string(kernel->ne[1]) + "," +
            std::to_string(kernel->ne[2]) + "]");
    }

    ggml_tensor* joined = nullptr;
    const int64_t input_group_channels = channels / groups;
    const int64_t output_group_channels = channels / groups;
    for (int64_t group = 0; group < groups; ++group) {
        ggml_tensor* input_group = ggml_view_2d(
            context,
            cast_f32(context, input),
            frames,
            input_group_channels,
            input->nb[1],
            static_cast<size_t>(group) *
                static_cast<size_t>(input_group_channels) *
                input->nb[1]);
        input_group = ggml_cont(context, input_group);
        ggml_tensor* kernel_group = ggml_view_3d(
            context,
            kernel,
            kernel->ne[0],
            kernel->ne[1],
            output_group_channels,
            kernel->nb[1],
            kernel->nb[2],
            static_cast<size_t>(group) *
                static_cast<size_t>(output_group_channels) *
                kernel->nb[2]);
        kernel_group = ggml_cont(context, kernel_group);
        ggml_tensor* output_group = conv1d(
            context,
            input_group,
            kernel_group,
            nullptr,
            1,
            64,
            1,
            frames);
        if (group == 0) {
            record("pos_group0", output_group);
        }
        joined = joined == nullptr
            ? output_group
            : ggml_concat(context, joined, output_group, 1);
    }
    if (bias != nullptr) {
        joined = ggml_add(
            context,
            joined,
            repeat_rows(context, bias, joined));
    }
    return ggml_gelu_erf(context, joined);
}

ggml_tensor* attention(
    ggml_context* context,
    const std::string& prefix,
    ggml_tensor* input,
    ggml_tensor* mask,
    const std::unordered_map<std::string, ggml_tensor*>& weights) {
    const auto get = [&](const std::string& suffix) {
        const auto found = weights.find(prefix + suffix);
        if (found == weights.end()) {
            throw std::runtime_error(
                "missing Higgs tensor: " + prefix + suffix);
        }
        return found->second;
    };
    const int64_t hidden = input->ne[1];
    constexpr int64_t HEADS = 12;
    const int64_t head_dim = hidden / HEADS;
    const int64_t frames = input->ne[0];

    ggml_tensor* channels_first = ggml_cont(
        context,
        ggml_transpose(context, cast_f32(context, input)));
    ggml_tensor* query = linear_channels_first(
        context,
        get("q_proj.weight"),
        get("q_proj.bias"),
        channels_first);
    ggml_tensor* key = linear_channels_first(
        context,
        get("k_proj.weight"),
        get("k_proj.bias"),
        channels_first);
    ggml_tensor* value = linear_channels_first(
        context,
        get("v_proj.weight"),
        get("v_proj.bias"),
        channels_first);

    query = ggml_reshape_3d(
        context,
        query,
        head_dim,
        HEADS,
        frames);
    key = ggml_reshape_3d(
        context,
        key,
        head_dim,
        HEADS,
        frames);
    value = ggml_reshape_3d(
        context,
        value,
        head_dim,
        HEADS,
        frames);
    query = ggml_permute(context, query, 0, 2, 1, 3);
    key = ggml_permute(context, key, 0, 2, 1, 3);
    value = ggml_permute(context, value, 0, 2, 1, 3);
    ggml_tensor* output = ggml_flash_attn_ext(
        context,
        query,
        key,
        value,
        mask,
        1.0f / std::sqrt(static_cast<float>(head_dim)),
        0.0f,
        0.0f);
    ggml_flash_attn_ext_set_prec(output, GGML_PREC_F32);
    output = ggml_cont_2d(
        context,
        output,
        hidden,
        frames);
    output = linear_channels_first(
        context,
        get("out_proj.weight"),
        get("out_proj.bias"),
        output);
    return ggml_cont(
        context,
        ggml_transpose(context, output));
}

ggml_tensor* hubert_layer(
    ggml_context* context,
    const std::string& prefix,
    ggml_tensor* input,
    ggml_tensor* mask,
    const std::unordered_map<std::string, ggml_tensor*>& weights) {
    const auto get = [&](const std::string& suffix) {
        const auto found = weights.find(prefix + suffix);
        if (found == weights.end()) {
            throw std::runtime_error(
                "missing Higgs tensor: " + prefix + suffix);
        }
        return found->second;
    };
    ggml_tensor* residual = input;
    ggml_tensor* hidden = ggml_add(
        context,
        residual,
        attention(
            context,
            prefix + "attention.",
            input,
            mask,
            weights));
    hidden = layer_norm(
        context,
        hidden,
        get("ln.weight"),
        get("ln.bias"));
    residual = hidden;
    hidden = linear_time_major(
        context,
        get("ff.i.weight"),
        get("ff.i.bias"),
        hidden);
    hidden = ggml_gelu_erf(context, hidden);
    hidden = linear_time_major(
        context,
        get("ff.o.weight"),
        get("ff.o.bias"),
        hidden);
    hidden = ggml_add(context, residual, hidden);
    return layer_norm(
        context,
        hidden,
        get("final_ln.weight"),
        get("final_ln.bias"));
}

ggml_tensor* elu(
    ggml_context* context,
    ggml_tensor* input) {
    ggml_tensor* positive = ggml_relu(context, input);
    ggml_tensor* negative = ggml_sub(context, input, positive);
    negative = ggml_sub(
        context,
        ggml_exp(context, negative),
        ggml_scale_bias(
            context,
            positive,
            0.0f,
            1.0f));
    return ggml_add(context, positive, negative);
}

ggml_tensor* semantic_residual_unit(
    ggml_context* context,
    const std::string& prefix,
    ggml_tensor* input,
    int dilation,
    const std::unordered_map<std::string, ggml_tensor*>& weights) {
    const auto get = [&](const std::string& suffix) {
        const auto found = weights.find(prefix + suffix);
        if (found == weights.end()) {
            throw std::runtime_error(
                "missing Higgs tensor: " + prefix + suffix);
        }
        return found->second;
    };
    ggml_tensor* hidden = elu(context, input);
    hidden = conv1d(
        context,
        hidden,
        get("conv1.weight"),
        nullptr,
        1,
        dilation,
        dilation);
    hidden = elu(context, hidden);
    hidden = conv1d(
        context,
        hidden,
        get("conv2.weight"),
        nullptr,
        1,
        0,
        1);
    return ggml_add(context, input, hidden);
}

ggml_tensor* semantic_encoder(
    ggml_context* context,
    ggml_tensor* input,
    const std::unordered_map<std::string, ggml_tensor*>& weights) {
    const auto get = [&](const std::string& name) {
        const auto found = weights.find(name);
        if (found == weights.end()) {
            throw std::runtime_error(
                "missing Higgs tensor: " + name);
        }
        return found->second;
    };
    ggml_tensor* hidden = conv1d(
        context,
        input,
        get("es.conv.weight"),
        nullptr,
        1,
        1,
        1);
    for (int block = 0; block < 2; ++block) {
        const std::string prefix =
            "es.cb." + std::to_string(block) + ".";
        for (int unit = 0; unit < 2; ++unit) {
            hidden = semantic_residual_unit(
                context,
                prefix + "res_units." +
                    std::to_string(unit) + ".",
                hidden,
                1,
                weights);
        }
        hidden = conv1d(
            context,
            hidden,
            get(prefix + "conv.weight"),
            get(prefix + "conv.bias"),
            1,
            1,
            1);
    }
    return hidden;
}

ggml_tensor* snake(
    ggml_context* context,
    ggml_tensor* input,
    ggml_tensor* alpha) {
    ggml_tensor* x = cast_f32(context, input);
    ggml_tensor* full_alpha = repeat_rows(context, alpha, x);
    ggml_tensor* denominator = ggml_scale_bias(
        context,
        full_alpha,
        1.0f,
        SNAKE_EPSILON);
    ggml_tensor* sine = ggml_sin(
        context,
        ggml_mul(context, x, full_alpha));
    return ggml_add(
        context,
        x,
        ggml_div(
            context,
            ggml_sqr(context, sine),
            denominator));
}

ggml_tensor* dac_residual_unit(
    ggml_context* context,
    const std::string& prefix,
    ggml_tensor* input,
    int dilation,
    const std::unordered_map<std::string, ggml_tensor*>& weights) {
    const auto get = [&](const std::string& suffix) {
        const auto found = weights.find(prefix + suffix);
        if (found == weights.end()) {
            throw std::runtime_error(
                "missing Higgs tensor: " + prefix + suffix);
        }
        return found->second;
    };
    ggml_tensor* hidden = snake(
        context,
        input,
        get("snake1.alpha"));
    hidden = conv1d(
        context,
        hidden,
        get("conv1.weight"),
        get("conv1.bias"),
        1,
        3 * dilation,
        dilation);
    hidden = snake(
        context,
        hidden,
        get("snake2.alpha"));
    hidden = conv1d(
        context,
        hidden,
        get("conv2.weight"),
        get("conv2.bias"),
        1,
        0,
        1);
    return ggml_add(context, input, hidden);
}

ggml_tensor* acoustic_encoder(
    ggml_context* context,
    ggml_tensor* input,
    const std::unordered_map<std::string, ggml_tensor*>& weights,
    const std::function<void(
        const std::string&,
        ggml_tensor*)>& record) {
    const auto get = [&](const std::string& name) {
        const auto found = weights.find(name);
        if (found == weights.end()) {
            throw std::runtime_error(
                "missing Higgs tensor: " + name);
        }
        return found->second;
    };
    ggml_tensor* hidden = conv1d(
        context,
        input,
        get("ae.conv1.weight"),
        get("ae.conv1.bias"),
        1,
        3,
        1);
    record("ac_conv1", hidden);
    const int strides[] = {8, 5, 4, 2, 3};
    for (int block = 0; block < 5; ++block) {
        const std::string prefix =
            "ae.block." + std::to_string(block) + ".";
        for (int unit = 1; unit <= 3; ++unit) {
            hidden = dac_residual_unit(
                context,
                prefix + "res_unit" +
                    std::to_string(unit) + ".",
                hidden,
                unit == 1 ? 1 : (unit == 2 ? 3 : 9),
                weights);
        }
        hidden = snake(
            context,
            hidden,
            get(prefix + "snake1.alpha"));
        hidden = conv1d(
            context,
            hidden,
            get(prefix + "conv1.weight"),
            get(prefix + "conv1.bias"),
            strides[block],
            (strides[block] + 1) / 2,
            1);
        record(
            "ac_block" + std::to_string(block),
            hidden);
    }
    hidden = snake(
        context,
        hidden,
        get("ae.snake1.alpha"));
    return conv1d(
        context,
        hidden,
        get("ae.conv2.weight"),
        get("ae.conv2.bias"),
        1,
        1,
        1);
}

int64_t conv_output_length(
    int64_t length,
    int64_t kernel,
    int64_t stride,
    int64_t padding,
    int64_t dilation) {
    return (length + 2 * padding -
            dilation * (kernel - 1) - 1) /
            stride +
        1;
}

int64_t hubert_feature_frames(int64_t waveform_length) {
    int64_t length = conv_output_length(
        waveform_length,
        10,
        5,
        0,
        1);
    for (int layer = 1; layer < 7; ++layer) {
        const int kernel = layer < 5 ? 3 : 2;
        length = conv_output_length(
            length,
            kernel,
            2,
            0,
            1);
    }
    return length;
}

int64_t acoustic_encoder_frames(int64_t waveform_length) {
    int64_t length = waveform_length;
    const int strides[] = {8, 5, 4, 2, 3};
    for (int stride : strides) {
        length = conv_output_length(
            length,
            2 * stride,
            stride,
            (stride + 1) / 2,
            1);
    }
    return length;
}

HiggsEncoderStage read_stage(
    const std::string& name,
    ggml_tensor* tensor) {
    if (tensor == nullptr || tensor->type != GGML_TYPE_F32 ||
        tensor->ne[0] <= 0 || tensor->ne[1] <= 0) {
        throw std::runtime_error(
            "Higgs debug stage is not readable: " + name);
    }
    HiggsEncoderStage stage;
    stage.name = name;
    stage.frames = static_cast<int32_t>(tensor->ne[0]);
    stage.channels = static_cast<int32_t>(tensor->ne[1]);
    std::vector<float> raw(
        static_cast<size_t>(ggml_nelements(tensor)));
    ggml_backend_tensor_get(
        tensor,
        raw.data(),
        0,
        raw.size() * sizeof(float));
    stage.values = std::move(raw);
    return stage;
}

std::vector<std::string> encoder_debug_stage_names() {
    std::vector<std::string> names;
    names.push_back("semantic_input");
    names.push_back("feature_conv0");
    names.push_back("feature_extracted");
    names.push_back("normalized_features");
    names.push_back("projected");
    names.push_back("pos_group0");
    names.push_back("positional");
    names.push_back("h0");
    for (int layer = 1; layer <= 12; ++layer) {
        names.push_back("h" + std::to_string(layer));
    }
    names.push_back("hubert_mean");
    names.push_back("semantic");
    names.push_back("ac_conv1");
    for (int block = 0; block < 5; ++block) {
        names.push_back("ac_block" + std::to_string(block));
    }
    names.push_back("acoustic");
    names.push_back("concat");
    names.push_back("embeddings");
    return names;
}

std::vector<float> read_f32_tensor(const ggml_tensor* tensor) {
    if (tensor == nullptr ||
        tensor->type != GGML_TYPE_F32 ||
        tensor->data == nullptr) {
        return {};
    }
    const size_t count =
        static_cast<size_t>(ggml_nelements(tensor));
    std::vector<float> values(count);
    std::memcpy(
        values.data(),
        tensor->data,
        count * sizeof(float));
    return values;
}

void quantize_residual(
    const HiggsTokenizerConfig& config,
    const GGUFModel& model,
    const std::vector<float>& embeddings,
    int32_t frames,
    AudioCodes& codes) {
    if (frames <= 0 ||
        embeddings.size() !=
            static_cast<size_t>(frames) *
                static_cast<size_t>(config.acoustic_hidden_size +
                    config.semantic_hidden_size)) {
        throw std::runtime_error("invalid Higgs embedding shape");
    }
    std::vector<float> residual = embeddings;
    codes.frames = frames;
    codes.codebooks = config.codebook_count;
    codes.data.assign(
        static_cast<size_t>(frames) *
            static_cast<size_t>(config.codebook_count),
        0);

    const int32_t hidden =
        config.acoustic_hidden_size +
        config.semantic_hidden_size;
    for (int32_t quantizer_index = 0;
         quantizer_index < config.codebook_count;
         ++quantizer_index) {
        const std::string prefix =
            "q.quantizers." +
            std::to_string(quantizer_index) + ".";
        const std::vector<float> project_in = read_f32_tensor(
            model.tensor(prefix + "pin.weight"));
        const std::vector<float> project_in_bias = read_f32_tensor(
            model.tensor(prefix + "pin.bias"));
        const std::vector<float> project_out = read_f32_tensor(
            model.tensor(prefix + "pout.weight"));
        const std::vector<float> project_out_bias = read_f32_tensor(
            model.tensor(prefix + "pout.bias"));
        const std::vector<float> codebook_embed = read_f32_tensor(
            model.tensor(prefix + "cb.embed"));
        if (project_in.size() !=
                static_cast<size_t>(config.codebook_dim) *
                    static_cast<size_t>(hidden) ||
            project_in_bias.size() !=
                static_cast<size_t>(config.codebook_dim) ||
            project_out.size() !=
                static_cast<size_t>(config.codebook_dim) *
                    static_cast<size_t>(hidden) ||
            project_out_bias.size() !=
                static_cast<size_t>(hidden) ||
            codebook_embed.size() !=
                static_cast<size_t>(config.codebook_size) *
                    static_cast<size_t>(config.codebook_dim)) {
            throw std::runtime_error(
                "Higgs RVQ tensor is missing or not F32");
        }

        std::vector<float> projected(
            static_cast<size_t>(frames) * config.codebook_dim);
        for (int32_t frame = 0; frame < frames; ++frame) {
            for (int32_t dimension = 0;
                 dimension < config.codebook_dim;
                 ++dimension) {
                double sum = project_in_bias[
                    static_cast<size_t>(dimension)];
                const float* input =
                    residual.data() +
                    static_cast<size_t>(frame) * hidden;
                const float* weight =
                    project_in.data() +
                    static_cast<size_t>(dimension) * hidden;
                for (int32_t index = 0; index < hidden; ++index) {
                    sum +=
                        static_cast<double>(input[index]) *
                        weight[index];
                }
                projected[
                    static_cast<size_t>(frame) *
                        config.codebook_dim +
                    dimension] = static_cast<float>(sum);
            }
        }

        for (int32_t frame = 0; frame < frames; ++frame) {
            int32_t best_index = 0;
            double best_distance =
                std::numeric_limits<double>::infinity();
            const float* vector =
                projected.data() +
                static_cast<size_t>(frame) * config.codebook_dim;
            for (int32_t entry = 0;
                 entry < config.codebook_size;
                 ++entry) {
                const float* candidate =
                    codebook_embed.data() +
                    static_cast<size_t>(entry) *
                        config.codebook_dim;
                double distance = 0.0;
                for (int32_t dimension = 0;
                     dimension < config.codebook_dim;
                     ++dimension) {
                    const double difference =
                        static_cast<double>(vector[dimension]) -
                        candidate[dimension];
                    distance += difference * difference;
                }
                if (distance < best_distance) {
                    best_distance = distance;
                    best_index = entry;
                }
            }
            codes.data[
                static_cast<size_t>(quantizer_index) * frames +
                frame] = best_index;

            const float* quantized =
                codebook_embed.data() +
                static_cast<size_t>(best_index) *
                    config.codebook_dim;
            float* destination =
                residual.data() +
                static_cast<size_t>(frame) * hidden;
            for (int32_t output_dimension = 0;
                 output_dimension < hidden;
                 ++output_dimension) {
                double decoded = project_out_bias[
                    static_cast<size_t>(output_dimension)];
                const float* weight =
                    project_out.data() +
                    static_cast<size_t>(output_dimension) *
                        config.codebook_dim;
                for (int32_t dimension = 0;
                     dimension < config.codebook_dim;
                     ++dimension) {
                    decoded +=
                        static_cast<double>(quantized[dimension]) *
                        weight[dimension];
                }
                destination[output_dimension] -=
                    static_cast<float>(decoded);
            }
        }
    }
}

} // namespace

HiggsAudioEncoder::HiggsAudioEncoder() = default;

HiggsAudioEncoder::~HiggsAudioEncoder() {
    release();
}

void HiggsAudioEncoder::release() noexcept {
    loaded_ = false;
    weights_.clear();
    if (weight_buffer_ != nullptr) {
        ggml_backend_buffer_free(weight_buffer_);
        weight_buffer_ = nullptr;
    }
    if (weight_context_ != nullptr) {
        ggml_free(weight_context_);
        weight_context_ = nullptr;
    }
    backend_ = nullptr;
    model_ = nullptr;
}

ggml_tensor* HiggsAudioEncoder::weight(
    const std::string& name) const {
    const auto found = weights_.find(name);
    return found == weights_.end() ? nullptr : found->second;
}

Status HiggsAudioEncoder::load(
    const HiggsTokenizerConfig& config,
    const GGUFModel& model,
    ggml_backend_t backend) {
    release();
    if (backend == nullptr) {
        return Status::error("Higgs encoder backend is null");
    }
    if (config.sample_rate != 24000 ||
        config.semantic_sample_rate != 16000 ||
        config.hop_length != 960 ||
        config.codebook_count != 8 ||
        config.codebook_size != 1024 ||
        config.codebook_dim != 64 ||
        config.semantic_hidden_size != 768 ||
        config.semantic_attention_heads != 12 ||
        config.semantic_layer_count != 12 ||
        config.acoustic_hidden_size != 256 ||
        config.acoustic_encoder_hidden_size != 64 ||
        config.downsampling_ratios !=
            std::vector<int32_t>({8, 5, 4, 2, 3})) {
        return Status::error(
            "unsupported Higgs encoder configuration");
    }

    ggml_init_params params{};
    params.mem_size = 32 * 1024 * 1024;
    params.no_alloc = true;
    weight_context_ = ggml_init(params);
    if (weight_context_ == nullptr) {
        return Status::error(
            "failed to create Higgs encoder context");
    }

    try {
        for (int64_t index = 0;
             index < model.tensor_count();
             ++index) {
            const ggml_tensor* source = model.tensor(index);
            if (source == nullptr || source->data == nullptr) {
                continue;
            }
            const std::string name = source->name;
            const bool encoder_weight =
                name.rfind("s.", 0) == 0 ||
                name.rfind("es.", 0) == 0 ||
                name.rfind("ae.", 0) == 0 ||
                name == "fc.weight" ||
                name == "fc.bias";
            if (!encoder_weight) {
                continue;
            }
            ggml_tensor* destination = nullptr;
            if (name.size() >= 6 &&
                name.compare(
                    name.size() - 6,
                    6,
                    ".alpha") == 0) {
                destination = ggml_new_tensor_1d(
                    weight_context_,
                    GGML_TYPE_F32,
                    source->ne[1]);
            } else {
                destination = ggml_dup_tensor(
                    weight_context_,
                    source);
            }
            ggml_set_name(destination, name.c_str());
            weights_.emplace(name, destination);
        }
    } catch (const std::exception& error) {
        release();
        return Status::error(error.what());
    }

    weight_buffer_ = ggml_backend_alloc_ctx_tensors(
        weight_context_,
        backend);
    if (weight_buffer_ == nullptr) {
        release();
        return Status::error(
            "failed to allocate Higgs encoder weights");
    }
    ggml_backend_buffer_set_usage(
        weight_buffer_,
        GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (const auto& [name, destination] : weights_) {
        const ggml_tensor* source = model.tensor(name);
        if (source == nullptr) {
            release();
            return Status::error(
                "failed to reload Higgs tensor: " + name);
        }
        ggml_backend_tensor_set(
            destination,
            source->data,
            0,
            ggml_nbytes(source));
    }

    config_ = config;
    backend_ = backend;
    model_ = &model;
    loaded_ = true;
    return Status::ok();
}

bool HiggsAudioEncoder::is_loaded() const noexcept {
    return loaded_;
}

Status HiggsAudioEncoder::encode(const AudioBuffer& audio, AudioCodes& codes,
                                 std::vector<HiggsEncoderStage>* debug_stages) const {
    codes = {};
    if (!loaded_) {
        return Status::error("Higgs encoder is not loaded");
    }
    if (audio.sample_rate != 24000 || audio.channels != 1 || audio.samples.empty()) {
        return Status::error("Higgs encoder requires non-empty 24 kHz mono audio");
    }

    try {
        const std::vector<float> resampled = resample_24k_to_16k(audio.samples);
        if (resampled.empty()) {
            return Status::error("Higgs semantic resampling produced no samples");
        }
        const int64_t semantic_input_length = static_cast<int64_t>(resampled.size()) + 320;
        const int64_t hubert_frames = hubert_feature_frames(semantic_input_length);
        const int64_t semantic_frames = (hubert_frames + 1) / 2;
        const int64_t unpadded_acoustic_frames =
            acoustic_encoder_frames(static_cast<int64_t>(audio.samples.size()));
        const bool pad_acoustic = unpadded_acoustic_frames != semantic_frames;
        const int64_t acoustic_input_length =
            static_cast<int64_t>(audio.samples.size()) + (pad_acoustic ? 960 : 0);
        const int64_t acoustic_frames = acoustic_encoder_frames(acoustic_input_length);
        if (semantic_frames <= 0 || acoustic_frames != semantic_frames) {
            return Status::error("Higgs semantic/acoustic frame count mismatch");
        }

        std::vector<uint8_t> graph_metadata;
        graph_metadata.resize(ggml_tensor_overhead() * 32768 + ggml_graph_overhead());
        ggml_init_params params{};
        params.mem_size = graph_metadata.size();
        params.mem_buffer = graph_metadata.data();
        params.no_alloc = true;
        ggml_context* context = ggml_init(params);
        if (context == nullptr) {
            return Status::error("failed to create Higgs encoder graph context");
        }

        Status status = Status::ok();
        try {
            ggml_cgraph* graph = build_graph(context, semantic_input_length, acoustic_input_length,
                                             debug_stages != nullptr);
            status = run_graph(audio, resampled, semantic_input_length, acoustic_input_length,
                               pad_acoustic, graph, codes, debug_stages);
        } catch (const std::exception& error) {
            status = Status::error(error.what());
        }

        ggml_free(context);
        return status;
    } catch (const std::exception& error) {
        return Status::error(error.what());
    }
}

ggml_cgraph* HiggsAudioEncoder::build_graph(ggml_context* context, int64_t semantic_input_length,
                                            int64_t acoustic_input_length,
                                            bool record_debug_stages) const {
    const auto get = [&](const std::string& name) -> ggml_tensor* {
        return require_weight(weight(name), name);
    };

    ggml_tensor* semantic_input =
        ggml_new_tensor_2d(context, GGML_TYPE_F32, semantic_input_length, 1);
    ggml_set_name(semantic_input, "higgs_semantic_input");
    ggml_set_input(semantic_input);
    ggml_tensor* acoustic_input =
        ggml_new_tensor_2d(context, GGML_TYPE_F32, acoustic_input_length, 1);
    ggml_set_name(acoustic_input, "higgs_acoustic_input");
    ggml_set_input(acoustic_input);

    std::vector<std::tuple<std::string, ggml_tensor*>> stage_tensors;
    const auto record = [&](const std::string& name, ggml_tensor* tensor) {
        if (record_debug_stages) {
            if (name != "semantic_input" && name != "embeddings") {
                const std::string graph_name = "higgs_debug." + name;
                ggml_set_name(tensor, graph_name.c_str());
            }
            ggml_set_output(tensor);
            stage_tensors.emplace_back(name, tensor);
        }
    };
    record("semantic_input", semantic_input);

    ggml_tensor* hidden =
        conv1d(context, semantic_input, get("s.fe.cv.0.conv.weight"), nullptr, 5, 0, 1);
    record("feature_conv0", hidden);
    hidden = group_norm_channels(context, hidden, 512, get("s.fe.cv.0.ln.weight"),
                                 get("s.fe.cv.0.ln.bias"));
    hidden = ggml_gelu_erf(context, hidden);
    for (int layer = 1; layer < 7; ++layer) {
        hidden = conv1d(context, hidden, get("s.fe.cv." + std::to_string(layer) + ".conv.weight"),
                        nullptr, 2, 0, 1);
        hidden = ggml_gelu_erf(context, hidden);
    }
    record("feature_extracted", hidden);
    ggml_tensor* normalized_features =
        layer_norm(context, hidden, get("s.fp.ln.weight"), get("s.fp.ln.bias"));
    record("normalized_features", normalized_features);
    ggml_tensor* projected = linear_time_major(context, get("s.fp.projection.weight"),
                                               get("s.fp.projection.bias"), normalized_features);
    record("projected", projected);
    ggml_tensor* positional = grouped_position_conv(context, projected, get("s.enc.pce.weight"),
                                                    get("s.enc.pce.conv.bias"), 16, record);
    record("positional", positional);
    hidden = ggml_add(context, projected, positional);
    hidden = layer_norm(context, hidden, get("s.enc.ln.weight"), get("s.enc.ln.bias"));
    record("h0", hidden);

    const int64_t sequence = hidden->ne[0];
    ggml_tensor* attention_mask =
        ggml_new_tensor_4d(context, GGML_TYPE_F16, sequence, sequence, 1, 1);
    ggml_set_name(attention_mask, "higgs_attention_mask");
    ggml_set_input(attention_mask);

    std::vector<ggml_tensor*> hidden_states;
    hidden_states.reserve(13);
    hidden_states.push_back(hidden);
    for (int layer = 0; layer < 12; ++layer) {
        hidden = hubert_layer(context, "s.enc.layers." + std::to_string(layer) + ".", hidden,
                              attention_mask, weights_);
        hidden_states.push_back(hidden);
        record("h" + std::to_string(layer + 1), hidden);
    }
    ggml_tensor* semantic_mean = hidden_states[0];
    for (size_t index = 1; index < hidden_states.size(); ++index) {
        semantic_mean = ggml_add(context, semantic_mean, hidden_states[index]);
    }
    semantic_mean =
        ggml_scale(context, semantic_mean, 1.0f / static_cast<float>(hidden_states.size()));
    record("hubert_mean", semantic_mean);
    ggml_tensor* semantic_mean_cf = ggml_cont(context, ggml_transpose(context, semantic_mean));
    const int64_t semantic_frames = (semantic_mean_cf->ne[1] + 1) / 2;
    semantic_mean_cf = ggml_cont_2d(context,
                                    ggml_view_2d(context, semantic_mean_cf, semantic_mean_cf->ne[0],
                                                 semantic_frames, semantic_mean_cf->nb[1] * 2, 0),
                                    semantic_mean_cf->ne[0], semantic_frames);
    semantic_mean = ggml_cont(context, ggml_transpose(context, semantic_mean_cf));
    ggml_tensor* semantic = semantic_encoder(context, semantic_mean, weights_);
    record("semantic", semantic);

    ggml_tensor* acoustic = acoustic_encoder(context, acoustic_input, weights_, record);
    record("acoustic", acoustic);
    if (acoustic->ne[0] != semantic->ne[0]) {
        throw std::runtime_error("Higgs encoder frame alignment failed");
    }
    ggml_tensor* concatenated = ggml_concat(context, acoustic, semantic, 1);
    record("concat", concatenated);
    ggml_tensor* embeddings_channels_first =
        linear_channels_first(context, get("fc.weight"), get("fc.bias"),
                              ggml_cont(context, ggml_transpose(context, concatenated)));
    ggml_tensor* embeddings =
        ggml_cont(context, ggml_transpose(context, embeddings_channels_first));
    ggml_set_name(embeddings, "higgs_encoder_embeddings");
    ggml_set_output(embeddings);
    record("embeddings", embeddings);

    ggml_cgraph* graph = ggml_new_graph_custom(context, 32768, false);
    ggml_build_forward_expand(graph, embeddings);
    for (const auto& [name, tensor] : stage_tensors) {
        (void)name;
        ggml_build_forward_expand(graph, tensor);
    }
    return graph;
}

Status HiggsAudioEncoder::run_graph(const AudioBuffer& audio, const std::vector<float>& resampled,
                                    int64_t semantic_input_length, int64_t acoustic_input_length,
                                    bool pad_acoustic, ggml_cgraph* graph, AudioCodes& codes,
                                    std::vector<HiggsEncoderStage>* debug_stages) const {
    codes = {};
    if (graph == nullptr) {
        return Status::error("Higgs encoder graph is null");
    }

    ggml_tensor* semantic_input = ggml_graph_get_tensor(graph, "higgs_semantic_input");
    ggml_tensor* acoustic_input = ggml_graph_get_tensor(graph, "higgs_acoustic_input");
    ggml_tensor* attention_mask = ggml_graph_get_tensor(graph, "higgs_attention_mask");
    ggml_tensor* embeddings = ggml_graph_get_tensor(graph, "higgs_encoder_embeddings");
    if (semantic_input == nullptr || acoustic_input == nullptr || attention_mask == nullptr ||
        embeddings == nullptr) {
        return Status::error("Higgs encoder graph is missing tensors");
    }

    ggml_backend_t fallback = nullptr;
    ggml_backend_sched_t scheduler = nullptr;
    const auto release_scheduler = [&]() {
        if (scheduler != nullptr) {
            ggml_backend_sched_free(scheduler);
            scheduler = nullptr;
        }
        if (fallback != nullptr) {
            ggml_backend_free(fallback);
            fallback = nullptr;
        }
    };

    ggml_backend_dev_t primary_device =
        ggml_backend_get_device(backend_);
    const bool primary_is_cpu =
        primary_device != nullptr &&
        ggml_backend_dev_type(primary_device) ==
            GGML_BACKEND_DEVICE_TYPE_CPU;
    ggml_backend_t backends[2] = {backend_, nullptr};
    ggml_backend_buffer_type_t buffer_types[2] = {
        ggml_backend_get_default_buffer_type(backend_),
        nullptr};
    int backend_count = 1;
    if (!primary_is_cpu) {
        fallback = ggml_backend_init_by_name("CPU", nullptr);
        if (fallback == nullptr) {
            return Status::error(
                "failed to initialize CPU fallback backend");
        }
        backends[1] = fallback;
        buffer_types[1] =
            ggml_backend_get_default_buffer_type(fallback);
        backend_count = 2;
    }
    scheduler = ggml_backend_sched_new(
        backends,
        buffer_types,
        backend_count,
        32768,
        false,
        false);
    if (scheduler == nullptr) {
        release_scheduler();
        return Status::error("failed to create backend scheduler");
    }
    ggml_backend_sched_reset(scheduler);
    if (!ggml_backend_sched_alloc_graph(scheduler, graph)) {
        release_scheduler();
        return Status::error("failed to allocate Higgs encoder graph");
    }

    std::vector<float> semantic_values(static_cast<size_t>(semantic_input_length), 0.0f);
    std::copy(resampled.begin(), resampled.end(), semantic_values.begin() + 160);
    ggml_backend_tensor_set(semantic_input, semantic_values.data(), 0,
                            semantic_values.size() * sizeof(float));

    std::vector<float> acoustic_values(static_cast<size_t>(acoustic_input_length), 0.0f);
    const size_t acoustic_offset = pad_acoustic ? 480 : 0;
    std::copy(audio.samples.begin(), audio.samples.end(),
              acoustic_values.begin() + static_cast<ptrdiff_t>(acoustic_offset));
    ggml_backend_tensor_set(acoustic_input, acoustic_values.data(), 0,
                            acoustic_values.size() * sizeof(float));

    std::vector<ggml_fp16_t> mask(
        static_cast<size_t>(attention_mask->ne[0] * attention_mask->ne[1]),
        ggml_fp32_to_fp16(0.0f));
    ggml_backend_tensor_set(attention_mask, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));

    if (ggml_backend_sched_graph_compute(scheduler, graph) != GGML_STATUS_SUCCESS) {
        release_scheduler();
        return Status::error("Higgs encoder graph execution failed");
    }

    const int32_t frames = static_cast<int32_t>(embeddings->ne[0]);
    std::vector<float> embedding_channels_first(static_cast<size_t>(ggml_nelements(embeddings)));
    ggml_backend_tensor_get(embeddings, embedding_channels_first.data(), 0,
                            embedding_channels_first.size() * sizeof(float));
    std::vector<float> embeddings_time_major(embedding_channels_first.size());
    const int32_t hidden_size = static_cast<int32_t>(embeddings->ne[1]);
    for (int32_t frame = 0; frame < frames; ++frame) {
        for (int32_t channel = 0; channel < hidden_size; ++channel) {
            embeddings_time_major[static_cast<size_t>(frame) * hidden_size + channel] =
                embedding_channels_first[static_cast<size_t>(channel) * frames + frame];
        }
    }
    quantize_residual(config_, *model_, embeddings_time_major, frames, codes);

    if (debug_stages != nullptr) {
        debug_stages->clear();
        HiggsEncoderStage resampled_stage;
        resampled_stage.name = "resampled";
        resampled_stage.channels = 1;
        resampled_stage.frames = static_cast<int32_t>(resampled.size());
        resampled_stage.values = resampled;
        debug_stages->push_back(std::move(resampled_stage));
        const std::vector<std::string> stage_names = encoder_debug_stage_names();
        debug_stages->reserve(debug_stages->size() + stage_names.size());
        for (const std::string& name : stage_names) {
            std::string graph_name = "higgs_debug." + name;
            if (name == "semantic_input") {
                graph_name = "higgs_semantic_input";
            } else if (name == "embeddings") {
                graph_name = "higgs_encoder_embeddings";
            }
            ggml_tensor* tensor = ggml_graph_get_tensor(graph, graph_name.c_str());
            if (tensor == nullptr) {
                release_scheduler();
                return Status::error("Higgs encoder graph is missing stage: " + name);
            }
            debug_stages->push_back(read_stage(name, tensor));
        }
    }

    release_scheduler();
    return Status::ok();
}

} // namespace omnivoice
