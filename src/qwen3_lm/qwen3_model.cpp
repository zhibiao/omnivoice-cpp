#include "qwen3_lm/qwen3_model.h"

#include "gguf_model.h"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace omnivoice {

namespace {

ggml_tensor* linear(
    ggml_context* context,
    ggml_tensor* weight,
    ggml_tensor* input) {
    return ggml_mul_mat(context, weight, input);
}

ggml_tensor* rms_norm(
    ggml_context* context,
    ggml_tensor* input,
    ggml_tensor* weight,
    float epsilon) {
    ggml_tensor* normalized = ggml_rms_norm(
        context,
        input,
        epsilon);
    ggml_tensor* column = ggml_reshape_2d(
        context,
        weight,
        weight->ne[0],
        1);
    return ggml_mul(context, normalized, column);
}

ggml_tensor* repeat_vector(
    ggml_context* context,
    ggml_tensor* vector,
    ggml_tensor* like) {
    return ggml_repeat(
        context,
        ggml_reshape_2d(context, vector, 1, vector->ne[0]),
        like);
}

ggml_tensor* log_softmax_graph(
    ggml_context* context,
    ggml_tensor* logits) {
    return ggml_log(
        context,
        ggml_clamp(
            context,
            ggml_soft_max(context, logits),
            1e-30f,
            1.0f));
}

std::vector<float> to_codebook_major(
    const std::vector<float>& values,
    int32_t codebooks,
    int32_t frames,
    int32_t vocab) {
    std::vector<float> result(values.size());
    for (int32_t frame = 0; frame < frames; ++frame) {
        for (int32_t codebook = 0; codebook < codebooks; ++codebook) {
            const size_t source =
                static_cast<size_t>(frame) *
                    static_cast<size_t>(codebooks) *
                    static_cast<size_t>(vocab) +
                static_cast<size_t>(codebook) *
                    static_cast<size_t>(vocab);
            const size_t destination =
                static_cast<size_t>(codebook) *
                    static_cast<size_t>(frames) *
                    static_cast<size_t>(vocab) +
                static_cast<size_t>(frame) *
                    static_cast<size_t>(vocab);
            std::copy(
                values.begin() + static_cast<std::ptrdiff_t>(source),
                values.begin() + static_cast<std::ptrdiff_t>(source + vocab),
                result.begin() + static_cast<std::ptrdiff_t>(destination));
        }
    }
    return result;
}

std::vector<int32_t> to_codebook_major(
    const std::vector<int32_t>& values,
    int32_t codebooks,
    int32_t frames) {
    std::vector<int32_t> result(values.size());
    for (int32_t frame = 0; frame < frames; ++frame) {
        for (int32_t codebook = 0; codebook < codebooks; ++codebook) {
            const size_t source =
                static_cast<size_t>(frame) *
                    static_cast<size_t>(codebooks) +
                static_cast<size_t>(codebook);
            const size_t destination =
                static_cast<size_t>(codebook) *
                    static_cast<size_t>(frames) +
                static_cast<size_t>(frame);
            result[destination] = values[source];
        }
    }
    return result;
}

} // namespace

OmniVoiceLM::OmniVoiceLM() = default;

OmniVoiceLM::~OmniVoiceLM() {
    release();
}

void OmniVoiceLM::release() noexcept {
    release_diffusion_graph();
    loaded_ = false;
    qkv_weights_.clear();
    blocks_.clear();
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
}

ggml_tensor* OmniVoiceLM::weight(const std::string& name) const {
    const auto found = weights_.find(name);
    return found == weights_.end() ? nullptr : found->second;
}

Status OmniVoiceLM::load(
    const OmniVoiceLMConfig& config,
    const GGUFModel& model,
    ggml_backend_t backend) {
    release();
    if (backend == nullptr) {
        return Status::error("OmniVoice LM backend is null");
    }
    if (config.llm.num_hidden_layers <= 0 ||
        config.num_audio_codebooks <= 0) {
        return Status::error("OmniVoice LM configuration is incomplete");
    }

    ggml_init_params params{};
    params.mem_size = 32 * 1024 * 1024;
    params.no_alloc = true;
    weight_context_ = ggml_init(params);
    if (weight_context_ == nullptr) {
        return Status::error("failed to create OmniVoice LM weight context");
    }

    try {
        for (int64_t index = 0;
             index < model.tensor_count();
             ++index) {
            const ggml_tensor* source = model.tensor(index);
            if (source == nullptr) {
                continue;
            }
            const std::string name = source->name;
            ggml_tensor* destination =
                ggml_dup_tensor(weight_context_, source);
            ggml_set_name(destination, name.c_str());
            if (!weights_.emplace(name, destination).second) {
                throw std::runtime_error("duplicate LM tensor: " + name);
            }
        }

        qkv_weights_.assign(
            static_cast<size_t>(
                config.llm.num_hidden_layers),
            nullptr);
        for (int32_t layer = 0;
             layer < config.llm.num_hidden_layers;
             ++layer) {
            const std::string prefix =
                "llm.layers." + std::to_string(layer) + ".";
            ggml_tensor* q = weight(
                prefix + "self_attn.q_proj.weight");
            ggml_tensor* k = weight(
                prefix + "self_attn.k_proj.weight");
            ggml_tensor* v = weight(
                prefix + "self_attn.v_proj.weight");
            if (q != nullptr && k != nullptr && v != nullptr &&
                q->type == k->type &&
                q->type == v->type &&
                q->ne[0] == k->ne[0] &&
                q->ne[0] == v->ne[0]) {
                ggml_tensor* packed = ggml_new_tensor_2d(
                    weight_context_,
                    q->type,
                    q->ne[0],
                    q->ne[1] + k->ne[1] + v->ne[1]);
                const std::string packed_name =
                    prefix + "qkv_proj.weight";
                ggml_set_name(packed, packed_name.c_str());
                qkv_weights_[static_cast<size_t>(layer)] = packed;
            }
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
        return Status::error("failed to allocate OmniVoice LM weights");
    }
    ggml_backend_buffer_set_usage(
        weight_buffer_,
        GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (const auto& [name, destination] : weights_) {
        const ggml_tensor* source = model.tensor(name);
        if (source == nullptr) {
            release();
            return Status::error("failed to reload LM tensor: " + name);
        }
        ggml_backend_tensor_set(
            destination,
            source->data,
            0,
            ggml_nbytes(source));
    }

    for (int32_t layer = 0;
         layer < config.llm.num_hidden_layers;
         ++layer) {
        ggml_tensor* packed_qkv =
            qkv_weights_[static_cast<size_t>(layer)];
        const std::string prefix =
            "llm.layers." + std::to_string(layer) + ".";
        if (packed_qkv != nullptr) {
            const ggml_tensor* q =
                model.tensor(prefix + "self_attn.q_proj.weight");
            const ggml_tensor* k =
                model.tensor(prefix + "self_attn.k_proj.weight");
            const ggml_tensor* v =
                model.tensor(prefix + "self_attn.v_proj.weight");
            if (q == nullptr || k == nullptr || v == nullptr) {
                release();
                return Status::error(
                    "failed to reload packed QKV weight for layer " +
                    std::to_string(layer));
            }
            size_t offset = 0;
            ggml_backend_tensor_set(
                packed_qkv,
                q->data,
                offset,
                ggml_nbytes(q));
            offset += ggml_nbytes(q);
            ggml_backend_tensor_set(
                packed_qkv,
                k->data,
                offset,
                ggml_nbytes(k));
            offset += ggml_nbytes(k);
            ggml_backend_tensor_set(
                packed_qkv,
                v->data,
                offset,
                ggml_nbytes(v));
        }
    }

    const int64_t hidden = config.llm.hidden_size;
    const int64_t audio_flat =
        static_cast<int64_t>(config.audio_vocab_size) *
        config.num_audio_codebooks;
    const ggml_tensor* embeddings = weight("llm.embed_tokens.weight");
    const ggml_tensor* audio_embeddings =
        weight("audio_embeddings.weight");
    const ggml_tensor* audio_heads = weight("audio_heads.weight");
    const ggml_tensor* final_norm = weight("llm.norm.weight");
    if (embeddings == nullptr ||
        embeddings->ne[0] != hidden ||
        embeddings->ne[1] != config.llm.vocab_size ||
        audio_embeddings == nullptr ||
        audio_embeddings->ne[0] != hidden ||
        audio_embeddings->ne[1] != audio_flat ||
        audio_heads == nullptr ||
        audio_heads->ne[0] != hidden ||
        audio_heads->ne[1] != audio_flat ||
        final_norm == nullptr ||
        final_norm->ne[0] != hidden) {
        release();
        return Status::error("OmniVoice LM GGUF has incompatible shapes");
    }

    blocks_.reserve(
        static_cast<size_t>(config.llm.num_hidden_layers));
    for (int32_t layer = 0;
         layer < config.llm.num_hidden_layers;
         ++layer) {
        const std::string prefix =
            "llm.layers." + std::to_string(layer) + ".";
        Qwen3BlockWeights block;
        block.input_layernorm_weight =
            weight(prefix + "input_layernorm.weight");
        block.post_attention_layernorm_weight =
            weight(prefix + "post_attention_layernorm.weight");
        block.q_proj_weight =
            weight(prefix + "self_attn.q_proj.weight");
        block.k_proj_weight =
            weight(prefix + "self_attn.k_proj.weight");
        block.v_proj_weight =
            weight(prefix + "self_attn.v_proj.weight");
        block.qkv_proj_weight =
            qkv_weights_[static_cast<size_t>(layer)];
        block.o_proj_weight =
            weight(prefix + "self_attn.o_proj.weight");
        block.q_norm_weight =
            weight(prefix + "self_attn.q_norm.weight");
        block.k_norm_weight =
            weight(prefix + "self_attn.k_norm.weight");
        block.gate_proj_weight =
            weight(prefix + "mlp.gate_proj.weight");
        block.up_proj_weight =
            weight(prefix + "mlp.up_proj.weight");
        block.down_proj_weight =
            weight(prefix + "mlp.down_proj.weight");
        if (block.input_layernorm_weight == nullptr ||
            block.post_attention_layernorm_weight == nullptr ||
            block.q_proj_weight == nullptr ||
            block.k_proj_weight == nullptr ||
            block.v_proj_weight == nullptr ||
            block.o_proj_weight == nullptr ||
            block.q_norm_weight == nullptr ||
            block.k_norm_weight == nullptr ||
            block.gate_proj_weight == nullptr ||
            block.up_proj_weight == nullptr ||
            block.down_proj_weight == nullptr) {
            release();
            return Status::error(
                "OmniVoice LM is missing layer " +
                std::to_string(layer));
        }
        blocks_.push_back(block);
    }

    config_ = config;
    backend_ = backend;
    loaded_ = true;
    return Status::ok();
}

Status OmniVoiceLM::validate_input(
    const OmniVoiceLmInput& input) const {
    if (!loaded_) {
        return Status::error("OmniVoice LM is not loaded");
    }
    if (input.sequence_length <= 0 ||
        input.sequence_length > config_.llm.max_position_embeddings) {
        return Status::error("invalid OmniVoice LM sequence length");
    }
    if (input.codebooks != config_.num_audio_codebooks) {
        return Status::error("invalid OmniVoice LM codebook count");
    }
    if (input.target_start < 0 ||
        input.target_frames <= 0 ||
        input.target_start + input.target_frames >
            input.sequence_length) {
        return Status::error("invalid OmniVoice LM target region");
    }
    if (input.input_ids.size() !=
            static_cast<size_t>(input.codebooks) *
                static_cast<size_t>(input.sequence_length) ||
        input.audio_mask.size() !=
            static_cast<size_t>(input.sequence_length)) {
        return Status::error("invalid OmniVoice LM input tensor size");
    }
    for (int32_t position = 0;
         position < input.sequence_length;
         ++position) {
        if (input.audio_mask[static_cast<size_t>(position)] != 0) {
            for (int32_t codebook = 0;
                 codebook < input.codebooks;
                 ++codebook) {
                const int32_t token = input.input_ids[
                    static_cast<size_t>(codebook) *
                        static_cast<size_t>(input.sequence_length) +
                    static_cast<size_t>(position)];
                if (token < 0 || token >= config_.audio_vocab_size) {
                    return Status::error(
                        "OmniVoice LM audio token is out of range");
                }
            }
        } else {
            const int32_t text_token =
                input.input_ids[static_cast<size_t>(position)];
            if (text_token < 0 ||
                text_token >= config_.llm.vocab_size) {
                return Status::error(
                    "OmniVoice LM text token is out of range");
            }
            for (int32_t codebook = 1;
                 codebook < input.codebooks;
                 ++codebook) {
                const int32_t token = input.input_ids[
                    static_cast<size_t>(codebook) *
                        static_cast<size_t>(input.sequence_length) +
                    static_cast<size_t>(position)];
                if (token != text_token) {
                    return Status::error(
                        "OmniVoice LM text token rows are inconsistent");
                }
            }
        }
    }
    return Status::ok();
}

class OmniVoiceLM::DiffusionGraph {
public:
    DiffusionGraph() = default;

    ~DiffusionGraph() {
        release();
    }

    DiffusionGraph(const DiffusionGraph&) = delete;
    DiffusionGraph& operator=(const DiffusionGraph&) = delete;

    Status build(
        const OmniVoiceLM& model,
        const OmniVoiceLmInput& conditional,
        const OmniVoiceLmInput& unconditional,
        bool read_host_logits,
        bool expose_hidden,
        float guidance_scale);
    Status run(
        const OmniVoiceLmInput& conditional,
        const OmniVoiceLmInput& unconditional,
        OmniVoiceLmPackedOutput& output);
    void release() noexcept;

private:
    struct Key {
        int32_t conditional_length = 0;
        int32_t unconditional_length = 0;
        int32_t codebooks = 0;
        int32_t conditional_target_start = 0;
        int32_t unconditional_target_start = 0;
        int32_t target_frames = 0;
        uint32_t guidance_bits = 0;
    };

    bool matches(
        const OmniVoiceLmInput& conditional,
        const OmniVoiceLmInput& unconditional,
        float guidance_scale) const noexcept;
    ggml_cgraph* build_diffusion_graph(
        ggml_context* context,
        const OmniVoiceLmInput& conditional,
        const OmniVoiceLmInput& unconditional,
        ggml_tensor* token_mask) const;
    void upload_static_inputs(
        const OmniVoiceLmInput& conditional,
        const OmniVoiceLmInput& unconditional);
    void upload_audio_ids(
        const OmniVoiceLmInput& conditional,
        const OmniVoiceLmInput& unconditional);

    const OmniVoiceLM* model_ = nullptr;
    Key key_;
    bool read_host_logits_ = false;
    bool expose_hidden_ = false;
    std::vector<uint8_t> metadata_;
    std::vector<uint8_t> input_metadata_;
    ggml_context* context_ = nullptr;
    ggml_context* input_context_ = nullptr;
    ggml_backend_buffer_t input_buffer_ = nullptr;
    ggml_cgraph* graph_ = nullptr;
    ggml_backend_t fallback_ = nullptr;
    ggml_backend_sched_t scheduler_ = nullptr;
    ggml_tensor* conditional_text_ids_ = nullptr;
    ggml_tensor* conditional_text_weight_ = nullptr;
    ggml_tensor* conditional_audio_weight_ = nullptr;
    ggml_tensor* conditional_positions_ = nullptr;
    std::vector<ggml_tensor*> conditional_audio_ids_;
    ggml_tensor* unconditional_text_ids_ = nullptr;
    ggml_tensor* unconditional_text_weight_ = nullptr;
    ggml_tensor* unconditional_audio_weight_ = nullptr;
    ggml_tensor* unconditional_positions_ = nullptr;
    std::vector<ggml_tensor*> unconditional_audio_ids_;
    ggml_tensor* conditional_logits_ = nullptr;
    ggml_tensor* unconditional_logits_ = nullptr;
    ggml_tensor* predictions_ = nullptr;
    ggml_tensor* confidence_ = nullptr;
    ggml_tensor* hidden_ = nullptr;
    ggml_tensor* token_mask_ = nullptr;
};

Status OmniVoiceLM::DiffusionGraph::build(
    const OmniVoiceLM& model,
    const OmniVoiceLmInput& conditional,
    const OmniVoiceLmInput& unconditional,
    bool read_host_logits,
    bool expose_hidden,
    float guidance_scale) {
    release();
    Status status = model.validate_input(conditional);
    if (!status) {
        return Status::error(
            "packed conditional input: " + status.message());
    }
    status = model.validate_input(unconditional);
    if (!status) {
        return Status::error(
            "packed unconditional input: " + status.message());
    }
    if (conditional.codebooks != unconditional.codebooks ||
        conditional.target_frames != unconditional.target_frames) {
        return Status::error(
            "packed LM inputs must have matching codebooks and target frames");
    }
    if (!std::isfinite(guidance_scale) || guidance_scale < 0.0f) {
        return Status::error("packed LM guidance scale is invalid");
    }

    model_ = &model;
    read_host_logits_ = read_host_logits;
    expose_hidden_ = expose_hidden;
    uint32_t guidance_bits = 0;
    static_assert(sizeof(guidance_bits) == sizeof(guidance_scale));
    std::memcpy(&guidance_bits, &guidance_scale, sizeof(guidance_bits));
    key_ = {
        conditional.sequence_length,
        unconditional.sequence_length,
        conditional.codebooks,
        conditional.target_start,
        unconditional.target_start,
        conditional.target_frames,
        guidance_bits,
    };
    const int32_t sequence =
        conditional.sequence_length + unconditional.sequence_length;
    metadata_.resize(
        ggml_tensor_overhead() * 32768 +
        ggml_graph_overhead());
    ggml_init_params params{};
    params.mem_size = metadata_.size();
    params.mem_buffer = metadata_.data();
    params.no_alloc = true;
    context_ = ggml_init(params);
    if (context_ == nullptr) {
        metadata_.clear();
        release();
        return Status::error(
            "failed to create packed OmniVoice LM graph context");
    }

    try {
        input_metadata_.resize(1024 * 1024);
        ggml_init_params input_params{};
        input_params.mem_size = input_metadata_.size();
        input_params.mem_buffer = input_metadata_.data();
        input_params.no_alloc = true;
        input_context_ = ggml_init(input_params);
        if (input_context_ == nullptr) {
            throw std::runtime_error(
                "failed to create packed LM input context");
        }
        token_mask_ = ggml_new_tensor_2d(
            input_context_,
            GGML_TYPE_F32,
            model.config_.audio_vocab_size,
            1);
        ggml_set_name(token_mask_, "lm_token_mask");
        ggml_set_input(token_mask_);

        graph_ = build_diffusion_graph(
            context_,
            conditional,
            unconditional,
            token_mask_);
        conditional_text_ids_ =
            ggml_graph_get_tensor(graph_, "lm_conditional_text_ids");
        conditional_text_weight_ =
            ggml_graph_get_tensor(graph_, "lm_conditional_text_weight");
        conditional_audio_weight_ =
            ggml_graph_get_tensor(graph_, "lm_conditional_audio_weight");
        conditional_positions_ =
            ggml_graph_get_tensor(graph_, "lm_conditional_positions");
        unconditional_text_ids_ =
            ggml_graph_get_tensor(graph_, "lm_unconditional_text_ids");
        unconditional_text_weight_ =
            ggml_graph_get_tensor(graph_, "lm_unconditional_text_weight");
        unconditional_audio_weight_ =
            ggml_graph_get_tensor(graph_, "lm_unconditional_audio_weight");
        unconditional_positions_ =
            ggml_graph_get_tensor(graph_, "lm_unconditional_positions");
        conditional_logits_ =
            ggml_graph_get_tensor(graph_, "lm_conditional_logits");
        unconditional_logits_ =
            ggml_graph_get_tensor(graph_, "lm_unconditional_logits");
        predictions_ =
            ggml_graph_get_tensor(graph_, "lm_predictions");
        confidence_ =
            ggml_graph_get_tensor(graph_, "lm_confidence");
        if (expose_hidden_) {
            hidden_ = ggml_graph_get_tensor(graph_, "lm_hidden");
        }
        if (conditional_text_ids_ == nullptr ||
            conditional_text_weight_ == nullptr ||
            conditional_audio_weight_ == nullptr ||
            conditional_positions_ == nullptr ||
            unconditional_text_ids_ == nullptr ||
            unconditional_text_weight_ == nullptr ||
            unconditional_audio_weight_ == nullptr ||
            unconditional_positions_ == nullptr ||
            conditional_logits_ == nullptr ||
            unconditional_logits_ == nullptr ||
            predictions_ == nullptr ||
            confidence_ == nullptr ||
            (expose_hidden_ && hidden_ == nullptr)) {
            throw std::runtime_error(
                "packed OmniVoice LM graph is missing tensors");
        }

        conditional_audio_ids_.reserve(
            static_cast<size_t>(conditional.codebooks));
        unconditional_audio_ids_.reserve(
            static_cast<size_t>(conditional.codebooks));
        for (int32_t codebook = 0;
             codebook < conditional.codebooks;
             ++codebook) {
            const std::string conditional_name =
                "lm_conditional_audio_ids." + std::to_string(codebook);
            const std::string unconditional_name =
                "lm_unconditional_audio_ids." + std::to_string(codebook);
            ggml_tensor* conditional_ids =
                ggml_graph_get_tensor(graph_, conditional_name.c_str());
            ggml_tensor* unconditional_ids =
                ggml_graph_get_tensor(graph_, unconditional_name.c_str());
            if (conditional_ids == nullptr || unconditional_ids == nullptr) {
                throw std::runtime_error(
                    "packed OmniVoice LM graph is missing audio id tensors");
            }
            conditional_audio_ids_.push_back(conditional_ids);
            unconditional_audio_ids_.push_back(unconditional_ids);
        }

        input_buffer_ =
            ggml_backend_alloc_ctx_tensors(input_context_, model.backend_);
        if (input_buffer_ == nullptr) {
            throw std::runtime_error(
                "failed to allocate packed LM input buffer");
        }
        ggml_backend_buffer_set_usage(
            input_buffer_,
            GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        ggml_backend_dev_t primary_device =
            ggml_backend_get_device(model.backend_);
        const bool primary_is_cpu =
            primary_device != nullptr &&
            ggml_backend_dev_type(primary_device) ==
                GGML_BACKEND_DEVICE_TYPE_CPU;
        ggml_backend_t backends[2] = {model.backend_, nullptr};
        ggml_backend_buffer_type_t buffer_types[2] = {
            ggml_backend_get_default_buffer_type(model.backend_),
            nullptr};
        int backend_count = 1;
        if (!primary_is_cpu) {
            fallback_ = ggml_backend_init_by_name("CPU", nullptr);
            if (fallback_ == nullptr) {
                throw std::runtime_error(
                    "failed to initialize CPU fallback backend");
            }
            backends[1] = fallback_;
            buffer_types[1] =
                ggml_backend_get_default_buffer_type(fallback_);
            backend_count = 2;
        }
        scheduler_ = ggml_backend_sched_new(
            backends,
            buffer_types,
            backend_count,
            32768,
            false,
            false);
        if (scheduler_ == nullptr) {
            throw std::runtime_error("failed to create backend scheduler");
        }
        for (ggml_tensor* input : {
                 conditional_text_ids_,
                 conditional_text_weight_,
                 conditional_audio_weight_,
                 conditional_positions_,
                 unconditional_text_ids_,
                 unconditional_text_weight_,
                 unconditional_audio_weight_,
                 unconditional_positions_,
                 token_mask_}) {
            ggml_backend_sched_set_tensor_backend(
                scheduler_, input, model.backend_);
        }
        for (ggml_tensor* ids : conditional_audio_ids_) {
            ggml_backend_sched_set_tensor_backend(
                scheduler_, ids, model.backend_);
        }
        for (ggml_tensor* ids : unconditional_audio_ids_) {
            ggml_backend_sched_set_tensor_backend(
                scheduler_, ids, model.backend_);
        }
        if (!ggml_backend_sched_alloc_graph(
                scheduler_,
                graph_)) {
            throw std::runtime_error(
                "failed to allocate packed OmniVoice LM graph");
        }

        upload_static_inputs(conditional, unconditional);
        upload_audio_ids(conditional, unconditional);

        std::vector<float> token_mask_values(
            static_cast<size_t>(model.config_.audio_vocab_size),
            0.0f);
        token_mask_values[
            static_cast<size_t>(model.config_.audio_mask_id)] =
            -std::numeric_limits<float>::infinity();
        ggml_backend_tensor_set(
            token_mask_,
            token_mask_values.data(),
            0,
            token_mask_values.size() * sizeof(float));
        return Status::ok();
    } catch (const std::exception& error) {
        status = Status::error(error.what());
    }

    release();
    return status;
}

ggml_cgraph* OmniVoiceLM::DiffusionGraph::build_diffusion_graph(
    ggml_context* context,
    const OmniVoiceLmInput& conditional,
    const OmniVoiceLmInput& unconditional,
    ggml_tensor* token_mask) const {
    const int32_t codebooks = conditional.codebooks;
    const int32_t target_frames = conditional.target_frames;
    const int32_t vocab = model_->config_.audio_vocab_size;
    const int64_t cells =
        static_cast<int64_t>(codebooks) * target_frames;

    const int32_t hidden_size = model_->config_.llm.hidden_size;
    const auto build_branch = [&](
        const std::string& prefix,
        const OmniVoiceLmInput& input,
        ggml_tensor*& positions_output) {
        const int32_t sequence = input.sequence_length;
        ggml_tensor* text_ids =
            ggml_new_tensor_1d(input_context_, GGML_TYPE_I32, sequence);
        ggml_set_name(text_ids, (prefix + "text_ids").c_str());
        ggml_set_input(text_ids);
        ggml_tensor* text_weight =
            ggml_new_tensor_1d(input_context_, GGML_TYPE_F32, sequence);
        ggml_set_name(text_weight, (prefix + "text_weight").c_str());
        ggml_set_input(text_weight);
        ggml_tensor* audio_weight =
            ggml_new_tensor_1d(input_context_, GGML_TYPE_F32, sequence);
        ggml_set_name(audio_weight, (prefix + "audio_weight").c_str());
        ggml_set_input(audio_weight);
        ggml_tensor* positions =
            ggml_new_tensor_1d(input_context_, GGML_TYPE_I32, sequence);
        ggml_set_name(positions, (prefix + "positions").c_str());
        ggml_set_input(positions);
        positions_output = positions;
        std::vector<ggml_tensor*> audio_ids;
        for (int32_t codebook = 0; codebook < codebooks; ++codebook) {
            ggml_tensor* ids =
                ggml_new_tensor_1d(input_context_, GGML_TYPE_I32, sequence);
            const std::string name =
                prefix + "audio_ids." + std::to_string(codebook);
            ggml_set_name(ids, name.c_str());
            ggml_set_input(ids);
            audio_ids.push_back(ids);
        }

        ggml_tensor* text_embeddings = ggml_get_rows(
            context,
            model_->weight("llm.embed_tokens.weight"),
            text_ids);
        ggml_tensor* audio_embeddings = nullptr;
        for (int32_t codebook = 0; codebook < codebooks; ++codebook) {
            ggml_tensor* embedding = ggml_get_rows(
                context,
                model_->weight("audio_embeddings.weight"),
                audio_ids[static_cast<size_t>(codebook)]);
            audio_embeddings = audio_embeddings == nullptr
                ? embedding
                : ggml_add(context, audio_embeddings, embedding);
        }
        ggml_tensor* text_part = ggml_mul(
            context,
            text_embeddings,
            repeat_vector(context, text_weight, text_embeddings));
        ggml_tensor* audio_part = ggml_mul(
            context,
            audio_embeddings,
            repeat_vector(context, audio_weight, audio_embeddings));
        return ggml_add(context, text_part, audio_part);
    };

    ggml_tensor* conditional_positions = nullptr;
    ggml_tensor* unconditional_positions = nullptr;
    ggml_tensor* conditional_hidden =
        build_branch("lm_conditional_", conditional, conditional_positions);
    ggml_tensor* unconditional_hidden =
        build_branch(
            "lm_unconditional_",
            unconditional,
            unconditional_positions);
    ggml_tensor* hidden = ggml_concat(
        context,
        conditional_hidden,
        unconditional_hidden,
        1);
    for (const Qwen3BlockWeights& block : model_->blocks_) {
        hidden = build_qwen3_packed_block(
            context,
            model_->config_.llm,
            block,
            hidden,
            conditional_positions,
            unconditional_positions,
            conditional.sequence_length);
    }
    hidden = rms_norm(
        context,
        hidden,
        model_->weight("llm.norm.weight"),
        model_->config_.llm.rms_norm_epsilon);
    ggml_set_name(hidden, "lm_hidden");
    if (expose_hidden_) {
        ggml_set_output(hidden);
    }
    ggml_tensor* conditional_target_hidden = ggml_cont_2d(
        context,
        ggml_view_2d(
            context,
            hidden,
            hidden_size,
            target_frames,
            hidden->nb[1],
            static_cast<size_t>(conditional.target_start) * hidden->nb[1]),
        hidden_size,
        target_frames);
    ggml_tensor* unconditional_target_hidden = ggml_cont_2d(
        context,
        ggml_view_2d(
            context,
            hidden,
            hidden_size,
            target_frames,
            hidden->nb[1],
            static_cast<size_t>(
                conditional.sequence_length + unconditional.target_start) *
                hidden->nb[1]),
        hidden_size,
        target_frames);
    ggml_tensor* conditional_logits = ggml_cont_2d(
        context,
        linear(
            context,
            model_->weight("audio_heads.weight"),
            conditional_target_hidden),
        codebooks * vocab,
        target_frames);
    ggml_set_name(conditional_logits, "lm_conditional_logits");
    ggml_set_output(conditional_logits);
    ggml_tensor* unconditional_logits = ggml_cont_2d(
        context,
        linear(
            context,
            model_->weight("audio_heads.weight"),
            unconditional_target_hidden),
        codebooks * vocab,
        target_frames);
    ggml_set_name(unconditional_logits, "lm_unconditional_logits");
    ggml_set_output(unconditional_logits);

    ggml_tensor* conditional_flat = ggml_reshape_2d(
        context,
        conditional_logits,
        vocab,
        cells);
    ggml_tensor* unconditional_flat = ggml_reshape_2d(
        context,
        unconditional_logits,
        vocab,
        cells);
    ggml_tensor* conditional_log_probs =
        log_softmax_graph(context, conditional_flat);
    ggml_tensor* unconditional_log_probs =
        log_softmax_graph(context, unconditional_flat);
    ggml_tensor* guided = conditional_log_probs;
    const float guidance_scale = [](
        uint32_t bits) {
        float value = 0.0f;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }(key_.guidance_bits);
    if (guidance_scale != 0.0f) {
        guided = ggml_add(
            context,
            conditional_log_probs,
            ggml_scale(
                context,
                ggml_sub(
                    context,
                    conditional_log_probs,
                    unconditional_log_probs),
                guidance_scale));
    }
    ggml_tensor* guided_log_probs =
        log_softmax_graph(context, guided);
    ggml_tensor* masked_log_probs = ggml_add(
        context,
        guided_log_probs,
        ggml_repeat(context, token_mask, guided_log_probs));
    ggml_tensor* predictions =
        ggml_argmax(context, masked_log_probs);
    ggml_set_name(predictions, "lm_predictions");
    ggml_set_output(predictions);
    ggml_tensor* confidence = ggml_reshape_1d(
        context,
        ggml_pool_1d(
            context,
            masked_log_probs,
            GGML_OP_POOL_MAX,
            vocab,
            vocab,
            0),
        cells);
    ggml_set_name(confidence, "lm_confidence");
    ggml_set_output(confidence);

    ggml_cgraph* graph = ggml_new_graph_custom(context, 32768, false);
    if (expose_hidden_) {
        ggml_build_forward_expand(graph, hidden);
    }
    ggml_build_forward_expand(graph, predictions);
    ggml_build_forward_expand(graph, confidence);
    ggml_build_forward_expand(graph, conditional_logits);
    ggml_build_forward_expand(graph, unconditional_logits);
    return graph;
}

Status OmniVoiceLM::DiffusionGraph::run(
    const OmniVoiceLmInput& conditional,
    const OmniVoiceLmInput& unconditional,
    OmniVoiceLmPackedOutput& output) {
    output = {};
    if (model_ == nullptr || graph_ == nullptr || scheduler_ == nullptr) {
        return Status::error(
            "packed OmniVoice LM graph is not initialized");
    }
    const float guidance_scale = [](
        uint32_t bits) {
        float value = 0.0f;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }(key_.guidance_bits);
    if (!matches(conditional, unconditional, guidance_scale)) {
        return Status::error("packed OmniVoice LM graph shape mismatch");
    }

    upload_audio_ids(conditional, unconditional);
    ggml_backend_synchronize(model_->backend());
    if (ggml_backend_graph_compute(
            model_->backend_,
            graph_) != GGML_STATUS_SUCCESS) {
        return Status::error(
            "packed OmniVoice LM graph execution failed");
    }

    const int32_t codebooks = conditional.codebooks;
    const int32_t frames = conditional.target_frames;
    const int32_t vocab = model_->config_.audio_vocab_size;
    const size_t cells =
        static_cast<size_t>(codebooks) * static_cast<size_t>(frames);
    std::vector<int32_t> raw_predictions(cells);
    std::vector<float> raw_confidence(cells);
    ggml_backend_tensor_get(
        predictions_,
        raw_predictions.data(),
        0,
        raw_predictions.size() * sizeof(int32_t));
    ggml_backend_tensor_get(
        confidence_,
        raw_confidence.data(),
        0,
        raw_confidence.size() * sizeof(float));
    output.predictions =
        to_codebook_major(raw_predictions, codebooks, frames);
    output.confidence =
        to_codebook_major(raw_confidence, codebooks, frames, 1);
    output.predictions_tensor = predictions_;
    output.confidence_tensor = confidence_;
    output.conditional_target_logits_tensor = conditional_logits_;
    output.unconditional_target_logits_tensor = unconditional_logits_;

    if (expose_hidden_) {
        const size_t hidden_size =
            static_cast<size_t>(model_->config_.llm.hidden_size);
        const size_t sequence =
            static_cast<size_t>(conditional.sequence_length) +
            static_cast<size_t>(unconditional.sequence_length);
        output.hidden.resize(hidden_size * sequence);
        ggml_backend_tensor_get(
            hidden_,
            output.hidden.data(),
            0,
            output.hidden.size() * sizeof(float));
        output.hidden_tensor = hidden_;
    }

    if (read_host_logits_) {
        const size_t logits_size = cells * static_cast<size_t>(vocab);
        std::vector<float> conditional_values(logits_size);
        std::vector<float> unconditional_values(logits_size);
        ggml_backend_tensor_get(
            conditional_logits_,
            conditional_values.data(),
            0,
            conditional_values.size() * sizeof(float));
        ggml_backend_tensor_get(
            unconditional_logits_,
            unconditional_values.data(),
            0,
            unconditional_values.size() * sizeof(float));
        output.conditional_target_logits = to_codebook_major(
            conditional_values,
            codebooks,
            frames,
            vocab);
        output.unconditional_target_logits = to_codebook_major(
            unconditional_values,
            codebooks,
            frames,
            vocab);
    }
    return Status::ok();
}

bool OmniVoiceLM::DiffusionGraph::matches(
    const OmniVoiceLmInput& conditional,
    const OmniVoiceLmInput& unconditional,
    float guidance_scale) const noexcept {
    uint32_t guidance_bits = 0;
    std::memcpy(&guidance_bits, &guidance_scale, sizeof(guidance_bits));
    return key_.conditional_length == conditional.sequence_length &&
        key_.unconditional_length == unconditional.sequence_length &&
        key_.codebooks == conditional.codebooks &&
        key_.conditional_target_start == conditional.target_start &&
        key_.unconditional_target_start == unconditional.target_start &&
        key_.target_frames == conditional.target_frames &&
        key_.guidance_bits == guidance_bits;
}

void OmniVoiceLM::DiffusionGraph::upload_static_inputs(
    const OmniVoiceLmInput& conditional,
    const OmniVoiceLmInput& unconditional) {
    const auto upload_branch = [&](
        const OmniVoiceLmInput& input,
        ggml_tensor* text_ids,
        ggml_tensor* text_weight,
        ggml_tensor* audio_weight,
        ggml_tensor* positions) {
        const int32_t sequence = input.sequence_length;
        std::vector<int32_t> text_ids_value(static_cast<size_t>(sequence));
        std::vector<int32_t> position_values(static_cast<size_t>(sequence));
        std::vector<float> text_weight_value(static_cast<size_t>(sequence));
        std::vector<float> audio_weight_value(static_cast<size_t>(sequence));
        for (int32_t index = 0; index < sequence; ++index) {
            text_ids_value[static_cast<size_t>(index)] =
                input.input_ids[static_cast<size_t>(index)];
            position_values[static_cast<size_t>(index)] = index;
            const bool is_audio =
                input.audio_mask[static_cast<size_t>(index)] != 0;
            text_weight_value[static_cast<size_t>(index)] =
                is_audio ? 0.0f : 1.0f;
            audio_weight_value[static_cast<size_t>(index)] =
                is_audio ? 1.0f : 0.0f;
        }
        ggml_backend_tensor_set(
            text_ids,
            text_ids_value.data(),
            0,
            text_ids_value.size() * sizeof(int32_t));
        ggml_backend_tensor_set(
            text_weight,
            text_weight_value.data(),
            0,
            text_weight_value.size() * sizeof(float));
        ggml_backend_tensor_set(
            audio_weight,
            audio_weight_value.data(),
            0,
            audio_weight_value.size() * sizeof(float));
        ggml_backend_tensor_set(
            positions,
            position_values.data(),
            0,
            position_values.size() * sizeof(int32_t));
    };
    upload_branch(
        conditional,
        conditional_text_ids_,
        conditional_text_weight_,
        conditional_audio_weight_,
        conditional_positions_);
    upload_branch(
        unconditional,
        unconditional_text_ids_,
        unconditional_text_weight_,
        unconditional_audio_weight_,
        unconditional_positions_);
}

void OmniVoiceLM::DiffusionGraph::upload_audio_ids(
    const OmniVoiceLmInput& conditional,
    const OmniVoiceLmInput& unconditional) {
    const int32_t codebooks = conditional.codebooks;
    const auto upload_branch = [&](
        const OmniVoiceLmInput& input,
        const std::vector<ggml_tensor*>& tensors) {
        const int32_t sequence = input.sequence_length;
        for (int32_t codebook = 0; codebook < codebooks; ++codebook) {
            std::vector<int32_t> ids(static_cast<size_t>(sequence));
            for (int32_t position = 0; position < sequence; ++position) {
                const bool is_audio =
                    input.audio_mask[static_cast<size_t>(position)] != 0;
                const int32_t token = is_audio
                    ? input.input_ids[
                          static_cast<size_t>(codebook) *
                              static_cast<size_t>(sequence) +
                          static_cast<size_t>(position)]
                    : 0;
                ids[static_cast<size_t>(position)] =
                    token + codebook * model_->config_.audio_vocab_size;
            }
            ggml_backend_tensor_set(
                tensors[static_cast<size_t>(codebook)],
                ids.data(),
                0,
                ids.size() * sizeof(int32_t));
        }
    };
    upload_branch(conditional, conditional_audio_ids_);
    upload_branch(unconditional, unconditional_audio_ids_);
}

void OmniVoiceLM::DiffusionGraph::release() noexcept {
    if (scheduler_ != nullptr) {
        ggml_backend_sched_free(scheduler_);
        scheduler_ = nullptr;
    }
    if (fallback_ != nullptr) {
        ggml_backend_free(fallback_);
        fallback_ = nullptr;
    }
    if (context_ != nullptr) {
        ggml_free(context_);
        context_ = nullptr;
    }
    if (input_buffer_ != nullptr) {
        ggml_backend_buffer_free(input_buffer_);
        input_buffer_ = nullptr;
    }
    if (input_context_ != nullptr) {
        ggml_free(input_context_);
        input_context_ = nullptr;
    }
    metadata_.clear();
    input_metadata_.clear();
    graph_ = nullptr;
    model_ = nullptr;
    key_ = {};
    read_host_logits_ = false;
    conditional_text_ids_ = nullptr;
    conditional_text_weight_ = nullptr;
    conditional_audio_weight_ = nullptr;
    conditional_positions_ = nullptr;
    conditional_audio_ids_.clear();
    unconditional_text_ids_ = nullptr;
    unconditional_text_weight_ = nullptr;
    unconditional_audio_weight_ = nullptr;
    unconditional_positions_ = nullptr;
    unconditional_audio_ids_.clear();
    conditional_logits_ = nullptr;
    unconditional_logits_ = nullptr;
    predictions_ = nullptr;
    confidence_ = nullptr;
    hidden_ = nullptr;
    expose_hidden_ = false;
    token_mask_ = nullptr;
}

Status OmniVoiceLM::init_diffusion_graph(
    const OmniVoiceLmInput& conditional,
    const OmniVoiceLmInput& unconditional,
    bool read_host_logits,
    bool expose_hidden,
    float guidance_scale) {
    release_diffusion_graph();
    auto graph = std::make_unique<DiffusionGraph>();
    Status status = graph->build(
        *this,
        conditional,
        unconditional,
        read_host_logits,
        expose_hidden,
        guidance_scale);
    if (!status) {
        return status;
    }
    diffusion_graph_ = std::move(graph);
    return Status::ok();
}

Status OmniVoiceLM::run_diffusion_graph(
    const OmniVoiceLmInput& conditional,
    const OmniVoiceLmInput& unconditional,
    OmniVoiceLmPackedOutput& output) {
    if (diffusion_graph_ == nullptr) {
        return Status::error(
            "packed OmniVoice LM graph is not initialized");
    }
    return diffusion_graph_->run(conditional, unconditional, output);
}

void OmniVoiceLM::release_diffusion_graph() noexcept {
    diffusion_graph_.reset();
}

ggml_backend_t OmniVoiceLM::backend() const noexcept {
    return backend_;
}

} // namespace omnivoice
