#include "qwen3_lm/qwen3_model.h"

#include "backend_scheduler.h"
#include "gguf_model.h"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

} // namespace

OmniVoiceLM::OmniVoiceLM() = default;

OmniVoiceLM::~OmniVoiceLM() {
    release();
}

void OmniVoiceLM::release() noexcept {
    release_diffusion_graphs();
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

class OmniVoiceLM::LmGraph {
public:
    LmGraph() = default;

    ~LmGraph() {
        release();
    }

    LmGraph(const LmGraph&) = delete;
    LmGraph& operator=(const LmGraph&) = delete;

    Status build(
        const OmniVoiceLM& model,
        const OmniVoiceLmInput& input,
        bool expose_hidden);
    Status run(
        const OmniVoiceLmInput& input,
        OmniVoiceLmOutput& output,
        bool read_hidden,
        bool read_logits);
    void release() noexcept;

private:
    struct Key {
        int32_t sequence_length = 0;
        int32_t codebooks = 0;
        int32_t target_start = 0;
        int32_t target_frames = 0;
    };

    bool matches(const OmniVoiceLmInput& input) const noexcept;
    void upload_static_inputs(const OmniVoiceLmInput& input);
    void upload_audio_ids(const OmniVoiceLmInput& input);

    const OmniVoiceLM* model_ = nullptr;
    Key key_;
    bool expose_hidden_ = false;
    std::vector<uint8_t> metadata_;
    ggml_context* context_ = nullptr;
    ggml_cgraph* graph_ = nullptr;
    std::unique_ptr<BackendScheduler> scheduler_;
    ggml_tensor* text_ids_ = nullptr;
    ggml_tensor* text_weight_ = nullptr;
    ggml_tensor* audio_weight_ = nullptr;
    ggml_tensor* positions_ = nullptr;
    std::vector<ggml_tensor*> audio_ids_;
    ggml_tensor* hidden_ = nullptr;
    ggml_tensor* logits_ = nullptr;
};

Status OmniVoiceLM::forward(const OmniVoiceLmInput& input, OmniVoiceLmOutput& output) const {
    output = {};
    Status status = validate_input(input);
    if (!status) {
        return status;
    }

    LmGraph graph;
    status = graph.build(*this, input, true);
    if (!status) {
        return status;
    }
    return graph.run(input, output, true, true);
}

Status OmniVoiceLM::LmGraph::build(
    const OmniVoiceLM& model,
    const OmniVoiceLmInput& input,
    bool expose_hidden) {
    release();

    Status status = model.validate_input(input);
    if (!status) {
        return status;
    }

    model_ = &model;
    expose_hidden_ = expose_hidden;
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
            "failed to create cached OmniVoice LM graph context");
    }

    try {
        graph_ = model.build_graph(context_, input, expose_hidden);
        text_ids_ = ggml_graph_get_tensor(graph_, "lm_text_ids");
        text_weight_ = ggml_graph_get_tensor(graph_, "lm_text_weight");
        audio_weight_ = ggml_graph_get_tensor(graph_, "lm_audio_weight");
        positions_ = ggml_graph_get_tensor(graph_, "lm_positions");
        hidden_ = ggml_graph_get_tensor(graph_, "lm_hidden");
        logits_ = ggml_graph_get_tensor(graph_, "lm_audio_logits");
        if (text_ids_ == nullptr ||
            text_weight_ == nullptr ||
            audio_weight_ == nullptr ||
            positions_ == nullptr ||
            hidden_ == nullptr ||
            logits_ == nullptr) {
            throw std::runtime_error(
                "cached OmniVoice LM graph is missing tensors");
        }

        audio_ids_.reserve(static_cast<size_t>(input.codebooks));
        for (int32_t codebook = 0;
             codebook < input.codebooks;
             ++codebook) {
            const std::string name =
                "lm_audio_ids." + std::to_string(codebook);
            ggml_tensor* ids =
                ggml_graph_get_tensor(graph_, name.c_str());
            if (ids == nullptr) {
                throw std::runtime_error(
                    "cached OmniVoice LM graph is missing " + name);
            }
            audio_ids_.push_back(ids);
        }

        scheduler_ = std::make_unique<BackendScheduler>();
        status = scheduler_->create(model.backend_, 32768);
        if (!status) {
            throw std::runtime_error(status.message());
        }
        ggml_backend_sched_set_tensor_backend(
            scheduler_->get(), text_ids_, model.backend_);
        ggml_backend_sched_set_tensor_backend(
            scheduler_->get(), text_weight_, model.backend_);
        ggml_backend_sched_set_tensor_backend(
            scheduler_->get(), audio_weight_, model.backend_);
        ggml_backend_sched_set_tensor_backend(
            scheduler_->get(), positions_, model.backend_);
        for (ggml_tensor* ids : audio_ids_) {
            ggml_backend_sched_set_tensor_backend(
                scheduler_->get(), ids, model.backend_);
        }
        if (!ggml_backend_sched_alloc_graph(
                scheduler_->get(), graph_)) {
            throw std::runtime_error(
                "failed to allocate cached OmniVoice LM graph");
        }

        key_ = {
            input.sequence_length,
            input.codebooks,
            input.target_start,
            input.target_frames,
        };
        upload_static_inputs(input);
        upload_audio_ids(input);
        return Status::ok();
    } catch (const std::exception& error) {
        status = Status::error(error.what());
    }

    release();
    return status;
}

ggml_cgraph* OmniVoiceLM::build_graph(
    ggml_context* context,
    const OmniVoiceLmInput& input,
    bool expose_hidden) const {
    const int32_t sequence = input.sequence_length;
    const int32_t codebooks = input.codebooks;
    ggml_tensor* text_ids = ggml_new_tensor_1d(context, GGML_TYPE_I32, sequence);
    ggml_set_name(text_ids, "lm_text_ids");
    ggml_set_input(text_ids);
    ggml_tensor* text_weight = ggml_new_tensor_1d(context, GGML_TYPE_F32, sequence);
    ggml_set_name(text_weight, "lm_text_weight");
    ggml_set_input(text_weight);
    ggml_tensor* audio_weight = ggml_new_tensor_1d(context, GGML_TYPE_F32, sequence);
    ggml_set_name(audio_weight, "lm_audio_weight");
    ggml_set_input(audio_weight);
    ggml_tensor* positions = ggml_new_tensor_1d(context, GGML_TYPE_I32, sequence);
    ggml_set_name(positions, "lm_positions");
    ggml_set_input(positions);
    std::vector<ggml_tensor*> audio_ids;
    audio_ids.reserve(static_cast<size_t>(codebooks));
    for (int32_t codebook = 0; codebook < codebooks; ++codebook) {
        ggml_tensor* ids = ggml_new_tensor_1d(context, GGML_TYPE_I32, sequence);
        const std::string name = "lm_audio_ids." + std::to_string(codebook);
        ggml_set_name(ids, name.c_str());
        ggml_set_input(ids);
        audio_ids.push_back(ids);
    }

    ggml_tensor* text_embeddings =
        ggml_get_rows(context, weight("llm.embed_tokens.weight"), text_ids);
    ggml_tensor* audio_embeddings = nullptr;
    for (int32_t codebook = 0; codebook < codebooks; ++codebook) {
        ggml_tensor* embedding = ggml_get_rows(context, weight("audio_embeddings.weight"),
                                               audio_ids[static_cast<size_t>(codebook)]);
        audio_embeddings = audio_embeddings == nullptr
                               ? embedding
                               : ggml_add(context, audio_embeddings, embedding);
    }
    ggml_tensor* text_part =
        ggml_mul(context, text_embeddings, repeat_vector(context, text_weight, text_embeddings));
    ggml_tensor* audio_part =
        ggml_mul(context, audio_embeddings, repeat_vector(context, audio_weight, audio_embeddings));
    ggml_tensor* hidden = ggml_add(context, text_part, audio_part);

    for (const Qwen3BlockWeights& block : blocks_) {
        hidden = build_qwen3_block(
            context,
            config_.llm,
            block,
            hidden,
            positions,
            nullptr);
    }
    hidden = rms_norm(context, hidden, weight("llm.norm.weight"), config_.llm.rms_norm_epsilon);
    ggml_set_name(hidden, "lm_hidden");
    if (expose_hidden) {
        ggml_set_output(hidden);
    }

    ggml_tensor* target_hidden =
        ggml_view_2d(context, hidden, config_.llm.hidden_size, input.target_frames, hidden->nb[1],
                     static_cast<size_t>(input.target_start) * hidden->nb[1]);
    target_hidden =
        ggml_cont_2d(context, target_hidden, config_.llm.hidden_size, input.target_frames);
    ggml_tensor* logits_flat = linear(context, weight("audio_heads.weight"), target_hidden);
    ggml_set_name(logits_flat, "lm_audio_logits");
    ggml_set_output(logits_flat);

    ggml_cgraph* graph = ggml_new_graph_custom(context, 32768, false);
    if (expose_hidden) {
        ggml_build_forward_expand(graph, hidden);
    }
    ggml_build_forward_expand(graph, logits_flat);
    return graph;
}

Status OmniVoiceLM::LmGraph::run(
    const OmniVoiceLmInput& input,
    OmniVoiceLmOutput& output,
    bool read_hidden,
    bool read_logits) {
    output = {};
    if (model_ == nullptr ||
        context_ == nullptr ||
        graph_ == nullptr ||
        scheduler_ == nullptr) {
        return Status::error(
            "cached OmniVoice LM graph is not initialized");
    }
    if (!matches(input)) {
        return Status::error(
            "cached OmniVoice LM graph shape mismatch");
    }
    if (read_hidden && !expose_hidden_) {
        return Status::error(
            "cached OmniVoice LM graph does not expose hidden states");
    }

    const int32_t sequence = input.sequence_length;
    const int32_t codebooks = input.codebooks;

    upload_audio_ids(input);

    if (ggml_backend_sched_graph_compute(
            scheduler_->get(), graph_) != GGML_STATUS_SUCCESS) {
        return Status::error("OmniVoice LM graph execution failed");
    }

    if (read_hidden) {
        const size_t hidden_size =
            static_cast<size_t>(model_->config_.llm.hidden_size);
        output.hidden.resize(
            hidden_size * static_cast<size_t>(sequence));
        ggml_backend_tensor_get(
            hidden_,
            output.hidden.data(),
            0,
            output.hidden.size() * sizeof(float));
    }

    output.target_logits_tensor = logits_;
    if (read_logits) {
        const size_t target_frames =
            static_cast<size_t>(input.target_frames);
        const size_t vocab =
            static_cast<size_t>(model_->config_.audio_vocab_size);
        std::vector<float> flat_logits(
            static_cast<size_t>(codebooks) *
            vocab *
            target_frames);
        ggml_backend_tensor_get(
            logits_,
            flat_logits.data(),
            0,
            flat_logits.size() * sizeof(float));
        output.target_logits.resize(flat_logits.size());
        for (size_t frame = 0; frame < target_frames; ++frame) {
            for (size_t codebook = 0;
                 codebook < static_cast<size_t>(codebooks);
                 ++codebook) {
                const size_t source_offset =
                    frame *
                        static_cast<size_t>(codebooks) *
                        vocab +
                    codebook * vocab;
                const size_t destination_offset =
                    codebook *
                        target_frames *
                        vocab +
                    frame * vocab;
                std::copy(
                    flat_logits.begin() +
                        static_cast<std::ptrdiff_t>(source_offset),
                    flat_logits.begin() +
                        static_cast<std::ptrdiff_t>(
                            source_offset + vocab),
                    output.target_logits.begin() +
                        static_cast<std::ptrdiff_t>(
                            destination_offset));
            }
        }
    }

    return Status::ok();
}

bool OmniVoiceLM::LmGraph::matches(
    const OmniVoiceLmInput& input) const noexcept {
    return key_.sequence_length == input.sequence_length &&
        key_.codebooks == input.codebooks &&
        key_.target_start == input.target_start &&
        key_.target_frames == input.target_frames;
}

void OmniVoiceLM::LmGraph::upload_static_inputs(
    const OmniVoiceLmInput& input) {
    const int32_t sequence = input.sequence_length;
    std::vector<int32_t> text_ids_value(
        input.input_ids.begin(),
        input.input_ids.begin() + sequence);
    ggml_backend_tensor_set(
        text_ids_,
        text_ids_value.data(),
        0,
        text_ids_value.size() * sizeof(int32_t));

    std::vector<int32_t> position_values(
        static_cast<size_t>(sequence));
    std::vector<float> text_weight_value(
        static_cast<size_t>(sequence));
    std::vector<float> audio_weight_value(
        static_cast<size_t>(sequence));
    for (int32_t index = 0; index < sequence; ++index) {
        position_values[static_cast<size_t>(index)] = index;
        const bool is_audio =
            input.audio_mask[static_cast<size_t>(index)] != 0;
        text_weight_value[static_cast<size_t>(index)] =
            is_audio ? 0.0f : 1.0f;
        audio_weight_value[static_cast<size_t>(index)] =
            is_audio ? 1.0f : 0.0f;
    }
    ggml_backend_tensor_set(
        text_weight_,
        text_weight_value.data(),
        0,
        text_weight_value.size() * sizeof(float));
    ggml_backend_tensor_set(
        audio_weight_,
        audio_weight_value.data(),
        0,
        audio_weight_value.size() * sizeof(float));
    ggml_backend_tensor_set(
        positions_,
        position_values.data(),
        0,
        position_values.size() * sizeof(int32_t));

}

void OmniVoiceLM::LmGraph::upload_audio_ids(
    const OmniVoiceLmInput& input) {
    const int32_t sequence = input.sequence_length;
    const int32_t codebooks = input.codebooks;
    for (int32_t codebook = 0;
         codebook < codebooks;
         ++codebook) {
        std::vector<int32_t> ids(
            static_cast<size_t>(sequence));
        for (int32_t position = 0;
             position < sequence;
             ++position) {
            const bool is_audio =
                input.audio_mask[
                    static_cast<size_t>(position)] != 0;
            const int32_t token = is_audio
                ? input.input_ids[
                    static_cast<size_t>(codebook) *
                        static_cast<size_t>(sequence) +
                    static_cast<size_t>(position)]
                : 0;
            ids[static_cast<size_t>(position)] =
                token +
                codebook * model_->config_.audio_vocab_size;
        }
        ggml_backend_tensor_set(
            audio_ids_[static_cast<size_t>(codebook)],
            ids.data(),
            0,
            ids.size() * sizeof(int32_t));
    }
}

void OmniVoiceLM::LmGraph::release() noexcept {
    scheduler_.reset();
    if (context_ != nullptr) {
        ggml_free(context_);
        context_ = nullptr;
    }
    metadata_.clear();
    graph_ = nullptr;
    model_ = nullptr;
    key_ = {};
    expose_hidden_ = false;
    text_ids_ = nullptr;
    text_weight_ = nullptr;
    audio_weight_ = nullptr;
    positions_ = nullptr;
    audio_ids_.clear();
    hidden_ = nullptr;
    logits_ = nullptr;
}

Status OmniVoiceLM::init_diffusion_graphs(
    const OmniVoiceLmInput& conditional,
    const OmniVoiceLmInput& unconditional,
    bool read_host_logits) {
    release_diffusion_graphs();
    diffusion_read_host_logits_ = read_host_logits;

    Status status = validate_input(conditional);
    if (!status) {
        return Status::error(
            "conditional graph: " + status.message());
    }
    status = validate_input(unconditional);
    if (!status) {
        return Status::error(
            "unconditional graph: " + status.message());
    }

    std::unique_ptr<LmGraph> conditional_graph =
        std::make_unique<LmGraph>();
    status = conditional_graph->build(
        *this,
        conditional,
        false);
    if (!status) {
        return Status::error(
            "conditional graph: " + status.message());
    }

    std::unique_ptr<LmGraph> unconditional_graph =
        std::make_unique<LmGraph>();
    status = unconditional_graph->build(
        *this,
        unconditional,
        false);
    if (!status) {
        return Status::error(
            "unconditional graph: " + status.message());
    }

    conditional_graph_ = std::move(conditional_graph);
    unconditional_graph_ = std::move(unconditional_graph);
    return Status::ok();
}

Status OmniVoiceLM::run_conditional_graph(
    const OmniVoiceLmInput& input,
    OmniVoiceLmOutput& output) {
    if (conditional_graph_ == nullptr) {
        return Status::error(
            "conditional OmniVoice LM graph is not initialized");
    }
    return conditional_graph_->run(
        input,
        output,
        false,
        diffusion_read_host_logits_);
}

Status OmniVoiceLM::run_unconditional_graph(
    const OmniVoiceLmInput& input,
    OmniVoiceLmOutput& output) {
    if (unconditional_graph_ == nullptr) {
        return Status::error(
            "unconditional OmniVoice LM graph is not initialized");
    }
    return unconditional_graph_->run(
        input,
        output,
        false,
        diffusion_read_host_logits_);
}

void OmniVoiceLM::release_diffusion_graphs() noexcept {
    unconditional_graph_.reset();
    conditional_graph_.reset();
    diffusion_read_host_logits_ = true;
}

bool OmniVoiceLM::supports_selector_graph() const noexcept {
    return backend_ != nullptr;
}

ggml_backend_t OmniVoiceLM::backend() const noexcept {
    return backend_;
}

} // namespace omnivoice
