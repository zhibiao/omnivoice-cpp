#include "diffusion_decoder/token_selector.h"

#include "backend_scheduler.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace omnivoice {

namespace {

Status validate_vector(
    const std::vector<float>* values,
    size_t expected,
    const char* name) {
    if (values == nullptr || values->size() != expected) {
        return Status::error(std::string(name) + " has an invalid size");
    }
    return Status::ok();
}

std::vector<float> log_softmax(
    const float* values,
    size_t count) {
    float maximum = -std::numeric_limits<float>::infinity();
    for (size_t index = 0; index < count; ++index) {
        maximum = std::max(maximum, values[index]);
    }
    double sum = 0.0;
    std::vector<float> result(count);
    for (size_t index = 0; index < count; ++index) {
        sum += std::exp(values[index] - maximum);
    }
    for (size_t index = 0; index < count; ++index) {
        const float probability = std::max(
            std::exp(values[index] - maximum) /
                static_cast<float>(sum),
            1e-30f);
        result[index] = std::log(probability);
    }
    return result;
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

} // namespace

class TokenSelector::Graph {
public:
    Graph() = default;

    ~Graph() {
        release();
    }

    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;

    Status build(
        ggml_backend_t backend,
        int32_t codebooks,
        int32_t frames,
        int32_t vocab_size,
        int32_t mask_id,
        float guidance_scale);
    Status run(
        const ggml_tensor* conditional_logits,
        const ggml_tensor* unconditional_logits,
        std::vector<int32_t>& predictions,
        std::vector<float>& confidence);
    bool matches(
        int32_t codebooks,
        int32_t frames,
        int32_t vocab_size,
        int32_t mask_id,
        float guidance_scale) const noexcept;
    void release() noexcept;

private:
    int32_t codebooks_ = 0;
    int32_t frames_ = 0;
    int32_t vocab_size_ = 0;
    int32_t mask_id_ = 0;
    float guidance_scale_ = 0.0f;
    std::vector<uint8_t> metadata_;
    ggml_context* context_ = nullptr;
    ggml_cgraph* graph_ = nullptr;
    std::unique_ptr<BackendScheduler> scheduler_;
    ggml_tensor* conditional_input_ = nullptr;
    ggml_tensor* unconditional_input_ = nullptr;
    ggml_tensor* token_mask_ = nullptr;
    ggml_tensor* mask_select_ = nullptr;
    ggml_tensor* one_tensor_ = nullptr;
    ggml_tensor* predictions_output_ = nullptr;
    ggml_tensor* confidence_output_ = nullptr;
    std::vector<float> token_mask_values_;
    std::vector<float> mask_select_values_;
    float one_value_ = 1.0f;
    std::vector<int32_t> raw_predictions_;
    std::vector<float> raw_confidence_;
};

Status TokenSelector::Graph::build(
    ggml_backend_t backend,
    int32_t codebooks,
    int32_t frames,
    int32_t vocab_size,
    int32_t mask_id,
    float guidance_scale) {
    release();
    if (backend == nullptr ||
        codebooks <= 0 ||
        frames <= 0 ||
        vocab_size <= 1 ||
        mask_id < 0 ||
        mask_id >= vocab_size) {
        return Status::error(
            "invalid TokenSelector graph dimensions");
    }

    codebooks_ = codebooks;
    frames_ = frames;
    vocab_size_ = vocab_size;
    mask_id_ = mask_id;
    guidance_scale_ = guidance_scale;
    metadata_.resize(
        ggml_tensor_overhead() * 1024 +
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
            "failed to create TokenSelector graph context");
    }

    try {
        const int64_t cells =
            static_cast<int64_t>(codebooks) * frames;
        const int64_t audio_flat =
            static_cast<int64_t>(codebooks) * vocab_size;
        conditional_input_ = ggml_new_tensor_2d(
            context_,
            GGML_TYPE_F32,
            audio_flat,
            frames);
        ggml_set_name(
            conditional_input_,
            "selector_conditional_logits");
        ggml_set_input(conditional_input_);
        unconditional_input_ = ggml_new_tensor_2d(
            context_,
            GGML_TYPE_F32,
            audio_flat,
            frames);
        ggml_set_name(
            unconditional_input_,
            "selector_unconditional_logits");
        ggml_set_input(unconditional_input_);
        token_mask_ = ggml_new_tensor_2d(
            context_,
            GGML_TYPE_F32,
            vocab_size,
            1);
        ggml_set_name(token_mask_, "selector_token_mask");
        ggml_set_input(token_mask_);
        mask_select_ = ggml_new_tensor_2d(
            context_,
            GGML_TYPE_F32,
            vocab_size,
            1);
        ggml_set_name(mask_select_, "selector_mask_select");
        ggml_set_input(mask_select_);
        one_tensor_ = ggml_new_tensor_2d(
            context_,
            GGML_TYPE_F32,
            1,
            1);
        ggml_set_name(one_tensor_, "selector_one");
        ggml_set_input(one_tensor_);

        ggml_tensor* conditional = ggml_reshape_2d(
            context_,
            conditional_input_,
            vocab_size,
            cells);
        ggml_tensor* unconditional = ggml_reshape_2d(
            context_,
            unconditional_input_,
            vocab_size,
            cells);
        ggml_tensor* conditional_log_probs =
            log_softmax_graph(context_, conditional);
        ggml_tensor* unconditional_log_probs =
            log_softmax_graph(context_, unconditional);

        ggml_tensor* guided = conditional_log_probs;
        if (guidance_scale != 0.0f) {
            guided = ggml_add(
                context_,
                conditional_log_probs,
                ggml_scale(
                    context_,
                    ggml_sub(
                        context_,
                        conditional_log_probs,
                        unconditional_log_probs),
                    guidance_scale));
        }
        ggml_tensor* guided_with_mask = ggml_add(
            context_,
            guided,
            ggml_repeat(context_, token_mask_, guided));

        predictions_output_ = ggml_argmax(
            context_,
            guided_with_mask);
        ggml_set_name(predictions_output_, "selector_predictions");
        ggml_set_output(predictions_output_);
        ggml_tensor* unmasked_probs =
            ggml_soft_max(context_, guided);
        ggml_tensor* mask_prob = ggml_pool_1d(
            context_,
            ggml_mul(
                context_,
                unmasked_probs,
                ggml_repeat(
                    context_,
                    mask_select_,
                    unmasked_probs)),
            GGML_OP_POOL_MAX,
            vocab_size,
            vocab_size,
            0);
        ggml_tensor* selected_masked_prob = ggml_pool_1d(
            context_,
            ggml_soft_max(context_, guided_with_mask),
            GGML_OP_POOL_MAX,
            vocab_size,
            vocab_size,
            0);
        ggml_tensor* selected_prob = ggml_mul(
            context_,
            selected_masked_prob,
            ggml_sub(
                context_,
                ggml_repeat(context_, one_tensor_, selected_masked_prob),
                mask_prob));
        confidence_output_ = ggml_reshape_1d(
            context_,
            ggml_log(
                context_,
                ggml_clamp(
                    context_,
                    selected_prob,
                    1e-30f,
                    1.0f)),
            cells);
        ggml_set_name(confidence_output_, "selector_confidence");
        ggml_set_output(confidence_output_);

        graph_ = ggml_new_graph_custom(
            context_,
            1024,
            false);
        ggml_build_forward_expand(
            graph_,
            predictions_output_);
        ggml_build_forward_expand(
            graph_,
            confidence_output_);

        scheduler_ = std::make_unique<BackendScheduler>();
        Status status = scheduler_->create(backend, 1024);
        if (!status) {
            throw std::runtime_error(status.message());
        }
        ggml_backend_sched_set_tensor_backend(
            scheduler_->get(),
            conditional_input_,
            backend);
        ggml_backend_sched_set_tensor_backend(
            scheduler_->get(),
            unconditional_input_,
            backend);
        ggml_backend_sched_set_tensor_backend(
            scheduler_->get(),
            token_mask_,
            backend);
        ggml_backend_sched_set_tensor_backend(
            scheduler_->get(),
            mask_select_,
            backend);
        ggml_backend_sched_set_tensor_backend(
            scheduler_->get(),
            one_tensor_,
            backend);
        if (!ggml_backend_sched_alloc_graph(
                scheduler_->get(),
                graph_)) {
            throw std::runtime_error(
                "failed to allocate TokenSelector graph");
        }

        token_mask_values_.assign(
            static_cast<size_t>(vocab_size),
            0.0f);
        token_mask_values_[static_cast<size_t>(mask_id)] =
            -std::numeric_limits<float>::infinity();
        mask_select_values_.assign(
            static_cast<size_t>(vocab_size),
            0.0f);
        mask_select_values_[static_cast<size_t>(mask_id)] = 1.0f;
        ggml_backend_tensor_set(
            token_mask_,
            token_mask_values_.data(),
            0,
            token_mask_values_.size() * sizeof(float));
        ggml_backend_tensor_set(
            mask_select_,
            mask_select_values_.data(),
            0,
            mask_select_values_.size() * sizeof(float));
        ggml_backend_tensor_set(
            one_tensor_,
            &one_value_,
            0,
            sizeof(one_value_));
        raw_predictions_.resize(static_cast<size_t>(cells));
        raw_confidence_.resize(static_cast<size_t>(cells));
    } catch (const std::exception& error) {
        release();
        return Status::error(error.what());
    }
    return Status::ok();
}

Status TokenSelector::Graph::run(
    const ggml_tensor* conditional_logits,
    const ggml_tensor* unconditional_logits,
    std::vector<int32_t>& predictions,
    std::vector<float>& confidence) {
    if (graph_ == nullptr || scheduler_ == nullptr) {
        return Status::error(
            "TokenSelector graph is not initialized");
    }
    const int64_t audio_flat =
        static_cast<int64_t>(codebooks_) * vocab_size_;
    if (conditional_logits == nullptr ||
        unconditional_logits == nullptr ||
        conditional_logits->type != GGML_TYPE_F32 ||
        unconditional_logits->type != GGML_TYPE_F32 ||
        conditional_logits->ne[0] != audio_flat ||
        conditional_logits->ne[1] != frames_ ||
        unconditional_logits->ne[0] != audio_flat ||
        unconditional_logits->ne[1] != frames_) {
        return Status::error(
            "TokenSelector logits shape mismatch");
    }

    ggml_backend_tensor_copy(
        conditional_logits,
        conditional_input_);
    if (guidance_scale_ != 0.0f) {
        ggml_backend_tensor_copy(
            unconditional_logits,
            unconditional_input_);
    }
    ggml_backend_tensor_set(
        token_mask_,
        token_mask_values_.data(),
        0,
        token_mask_values_.size() * sizeof(float));
    ggml_backend_tensor_set(
        mask_select_,
        mask_select_values_.data(),
        0,
        mask_select_values_.size() * sizeof(float));
    ggml_backend_tensor_set(
        one_tensor_,
        &one_value_,
        0,
        sizeof(one_value_));
    if (ggml_backend_sched_graph_compute(
            scheduler_->get(),
            graph_) != GGML_STATUS_SUCCESS) {
        return Status::error(
            "TokenSelector graph execution failed");
    }
    ggml_backend_tensor_get(
        predictions_output_,
        raw_predictions_.data(),
        0,
        raw_predictions_.size() * sizeof(int32_t));
    ggml_backend_tensor_get(
        confidence_output_,
        raw_confidence_.data(),
        0,
        raw_confidence_.size() * sizeof(float));
    if (std::any_of(
            raw_predictions_.begin(),
            raw_predictions_.end(),
            [&](int32_t value) {
                return value < 0 || value >= vocab_size_;
            })) {
        return Status::error(
            "TokenSelector graph produced an invalid prediction");
    }
    predictions = raw_predictions_;
    confidence = raw_confidence_;
    return Status::ok();
}

bool TokenSelector::Graph::matches(
    int32_t codebooks,
    int32_t frames,
    int32_t vocab_size,
    int32_t mask_id,
    float guidance_scale) const noexcept {
    return codebooks_ == codebooks &&
        frames_ == frames &&
        vocab_size_ == vocab_size &&
        mask_id_ == mask_id &&
        guidance_scale_ == guidance_scale;
}

void TokenSelector::Graph::release() noexcept {
    scheduler_.reset();
    if (context_ != nullptr) {
        ggml_free(context_);
        context_ = nullptr;
    }
    metadata_.clear();
    graph_ = nullptr;
    conditional_input_ = nullptr;
    unconditional_input_ = nullptr;
    token_mask_ = nullptr;
    mask_select_ = nullptr;
    one_tensor_ = nullptr;
    predictions_output_ = nullptr;
    confidence_output_ = nullptr;
    token_mask_values_.clear();
    mask_select_values_.clear();
    one_value_ = 1.0f;
    raw_predictions_.clear();
    raw_confidence_.clear();
    codebooks_ = 0;
    frames_ = 0;
    vocab_size_ = 0;
    mask_id_ = 0;
    guidance_scale_ = 0.0f;
}

TokenSelector::TokenSelector()
    : rng_(0x4f4d4e49564f4943ULL) {
}

TokenSelector::~TokenSelector() = default;

Status TokenSelector::init_graph(
    ggml_backend_t backend,
    int32_t codebooks,
    int32_t frames,
    int32_t vocab_size,
    int32_t mask_id,
    float guidance_scale) {
    graph_ = std::make_unique<Graph>();
    Status status = graph_->build(
        backend,
        codebooks,
        frames,
        vocab_size,
        mask_id,
        guidance_scale);
    if (!status) {
        graph_.reset();
    }
    return status;
}

void TokenSelector::seed(uint64_t value) {
    rng_.seed(value);
}

float TokenSelector::uniform() {
    return std::generate_canonical<float, 24>(rng_);
}

float TokenSelector::gumbel(float score, float temperature) {
    const float u = uniform();
    const float noise =
        -std::log(-std::log(u + 1e-10f) + 1e-10f);
    return score / temperature + noise;
}

Status TokenSelector::select(
    const TokenSelectionInput& input,
    const TokenSelectorOptions& options,
    TokenSelectionOutput& output) {
    if (input.codebooks <= 0 ||
        input.frames <= 0 ||
        input.vocab_size <= 1) {
        return Status::error("token selector dimensions must be positive");
    }
    if (input.mask_id < 0 || input.mask_id >= input.vocab_size) {
        return Status::error("token selector MASK ID is out of range");
    }
    const size_t cells =
        static_cast<size_t>(input.codebooks) *
        static_cast<size_t>(input.frames);
    const size_t logits_size =
        cells * static_cast<size_t>(input.vocab_size);
    Status status = validate_vector(
        input.cond_logits,
        logits_size,
        "conditional logits");
    if (!status) {
        return status;
    }
    status = validate_vector(
        input.uncond_logits,
        logits_size,
        "unconditional logits");
    if (!status) {
        return status;
    }
    if (input.current_tokens == nullptr ||
        input.current_tokens->size() != cells) {
        return Status::error("current token tensor has an invalid size");
    }
    if (!std::isfinite(options.guidance_scale) ||
        options.guidance_scale < 0.0f ||
        !std::isfinite(options.layer_penalty_factor) ||
        options.layer_penalty_factor < 0.0f ||
        !std::isfinite(options.position_temperature) ||
        options.position_temperature < 0.0f ||
        !std::isfinite(options.class_temperature) ||
        options.class_temperature < 0.0f) {
        return Status::error("invalid token selector options");
    }

    size_t pending = 0;
    for (int32_t token : *input.current_tokens) {
        if (token == input.mask_id) {
            ++pending;
        }
    }
    if (input.top_k < 0 ||
        static_cast<size_t>(input.top_k) > pending) {
        return Status::error("token selector top_k exceeds pending cells");
    }

    output.predictions.assign(cells, 0);
    output.confidence.assign(cells, 0.0f);
    output.scores.assign(cells, 0.0f);
    output.selected_mask.assign(cells, 0);
    output.selected_count = 0;

    const size_t vocab = static_cast<size_t>(input.vocab_size);
    std::vector<float> combined(vocab);
    std::vector<float> uncond_probs;
    for (size_t cell = 0; cell < cells; ++cell) {
        const float* cond = input.cond_logits->data() + cell * vocab;
        const std::vector<float> cond_probs =
            log_softmax(cond, vocab);
        std::vector<float> combined_values(vocab);
        if (options.guidance_scale != 0.0f) {
            const float* uncond =
                input.uncond_logits->data() + cell * vocab;
            uncond_probs = log_softmax(uncond, vocab);
            for (size_t token = 0; token < vocab; ++token) {
                combined_values[token] =
                    cond_probs[token] +
                    options.guidance_scale *
                        (cond_probs[token] - uncond_probs[token]);
            }
            combined = log_softmax(
                combined_values.data(),
                vocab);
        } else {
            combined = cond_probs;
        }
        combined[static_cast<size_t>(input.mask_id)] =
            -std::numeric_limits<float>::infinity();

        size_t predicted = 0;
        if (options.class_temperature > 0.0f) {
            const size_t keep = std::max<size_t>(
                1,
                static_cast<size_t>(
                    std::ceil(0.1 * static_cast<double>(vocab))));
            std::vector<size_t> indices(vocab);
            for (size_t token = 0; token < vocab; ++token) {
                indices[token] = token;
            }
            std::partial_sort(
                indices.begin(),
                indices.begin() + static_cast<std::ptrdiff_t>(keep),
                indices.end(),
                [&](size_t lhs, size_t rhs) {
                    return combined[lhs] > combined[rhs];
                });
            std::vector<float> filtered(
                vocab,
                -std::numeric_limits<float>::infinity());
            for (size_t index = 0; index < keep; ++index) {
                const size_t token = indices[index];
                filtered[token] = combined[token];
            }
            float best = -std::numeric_limits<float>::infinity();
            for (size_t token = 0; token < vocab; ++token) {
                if (!std::isfinite(filtered[token])) {
                    continue;
                }
                const float sampled = gumbel(
                    filtered[token],
                    options.class_temperature);
                if (sampled > best) {
                    best = sampled;
                    predicted = token;
                }
            }
        } else {
            predicted = static_cast<size_t>(
                std::max_element(combined.begin(), combined.end()) -
                combined.begin());
        }
        output.predictions[cell] = static_cast<int32_t>(predicted);

        const float confidence =
            combined[static_cast<size_t>(predicted)];
        output.confidence[cell] = confidence;
        const int32_t codebook = static_cast<int32_t>(
            cell / static_cast<size_t>(input.frames));
        output.scores[cell] =
            confidence -
            static_cast<float>(codebook) *
                options.layer_penalty_factor;
    }

    if (options.position_temperature > 0.0f) {
        for (float& score : output.scores) {
            if (std::isfinite(score)) {
                score = gumbel(score, options.position_temperature);
            }
        }
    }
    for (size_t cell = 0; cell < cells; ++cell) {
        if ((*input.current_tokens)[cell] != input.mask_id) {
            output.scores[cell] =
                -std::numeric_limits<float>::infinity();
        }
    }

    if (input.top_k > 0) {
        std::vector<std::pair<float, size_t>> candidates;
        candidates.reserve(pending);
        for (size_t cell = 0; cell < cells; ++cell) {
            if ((*input.current_tokens)[cell] == input.mask_id) {
                candidates.emplace_back(output.scores[cell], cell);
            }
        }
        std::partial_sort(
            candidates.begin(),
            candidates.begin() +
                static_cast<std::ptrdiff_t>(input.top_k),
            candidates.end(),
            [](const auto& lhs, const auto& rhs) {
                if (lhs.first != rhs.first) {
                    return lhs.first > rhs.first;
                }
                return lhs.second < rhs.second;
            });
        for (int32_t index = 0; index < input.top_k; ++index) {
            const size_t cell =
                candidates[static_cast<size_t>(index)].second;
            output.selected_mask[cell] = 1;
        }
        output.selected_count = input.top_k;
    }
    return Status::ok();
}

Status TokenSelector::select_graph(
    const TokenSelectionInput& input,
    const TokenSelectorOptions& options,
    const ggml_tensor* conditional_logits,
    const ggml_tensor* unconditional_logits,
    TokenSelectionOutput& output) {
    if (graph_ == nullptr) {
        return Status::error(
            "TokenSelector graph is not initialized");
    }
    if (input.codebooks <= 0 ||
        input.frames <= 0 ||
        input.vocab_size <= 1 ||
        input.mask_id < 0 ||
        input.mask_id >= input.vocab_size) {
        return Status::error(
            "graph TokenSelector dimensions must be positive");
    }
    if (input.current_tokens == nullptr ||
        input.current_tokens->size() !=
            static_cast<size_t>(input.codebooks) *
                static_cast<size_t>(input.frames)) {
        return Status::error(
            "graph TokenSelector current token tensor has an invalid size");
    }
    if (!std::isfinite(options.guidance_scale) ||
        options.guidance_scale < 0.0f ||
        !std::isfinite(options.layer_penalty_factor) ||
        options.layer_penalty_factor < 0.0f ||
        !std::isfinite(options.position_temperature) ||
        options.position_temperature < 0.0f ||
        !std::isfinite(options.class_temperature) ||
        options.class_temperature != 0.0f) {
        return Status::error(
            "graph TokenSelector supports class_temperature=0 only");
    }
    if (!graph_->matches(
            input.codebooks,
            input.frames,
            input.vocab_size,
            input.mask_id,
            options.guidance_scale)) {
        return Status::error(
            "graph TokenSelector dimensions or guidance changed");
    }

    const size_t cells =
        static_cast<size_t>(input.codebooks) *
        static_cast<size_t>(input.frames);
    size_t pending = 0;
    for (int32_t token : *input.current_tokens) {
        if (token == input.mask_id) {
            ++pending;
        }
    }
    if (input.top_k < 0 ||
        static_cast<size_t>(input.top_k) > pending) {
        return Status::error(
            "graph TokenSelector top_k exceeds pending cells");
    }

    std::vector<int32_t> raw_predictions;
    std::vector<float> raw_confidence;
    Status status = graph_->run(
        conditional_logits,
        unconditional_logits,
        raw_predictions,
        raw_confidence);
    if (!status) {
        return status;
    }

    output.predictions.assign(cells, 0);
    output.confidence.assign(cells, 0.0f);
    output.scores.assign(cells, 0.0f);
    output.selected_mask.assign(cells, 0);
    output.selected_count = 0;
    for (int32_t frame = 0; frame < input.frames; ++frame) {
        for (int32_t codebook = 0;
             codebook < input.codebooks;
             ++codebook) {
            const size_t source =
                static_cast<size_t>(frame) *
                    static_cast<size_t>(input.codebooks) +
                static_cast<size_t>(codebook);
            const size_t destination =
                static_cast<size_t>(codebook) *
                    static_cast<size_t>(input.frames) +
                static_cast<size_t>(frame);
            output.predictions[destination] =
                raw_predictions[source];
            output.confidence[destination] =
                raw_confidence[source];
            output.scores[destination] =
                output.confidence[destination] -
                static_cast<float>(codebook) *
                    options.layer_penalty_factor;
        }
    }
    if (options.position_temperature > 0.0f) {
        for (float& score : output.scores) {
            score = gumbel(
                score,
                options.position_temperature);
        }
    }
    for (size_t cell = 0; cell < cells; ++cell) {
        if ((*input.current_tokens)[cell] != input.mask_id) {
            output.scores[cell] =
                -std::numeric_limits<float>::infinity();
        }
    }

    if (input.top_k > 0) {
        std::vector<std::pair<float, size_t>> candidates;
        candidates.reserve(pending);
        for (size_t cell = 0; cell < cells; ++cell) {
            if ((*input.current_tokens)[cell] == input.mask_id) {
                candidates.emplace_back(output.scores[cell], cell);
            }
        }
        std::partial_sort(
            candidates.begin(),
            candidates.begin() +
                static_cast<std::ptrdiff_t>(input.top_k),
            candidates.end(),
            [](const auto& lhs, const auto& rhs) {
                if (lhs.first != rhs.first) {
                    return lhs.first > rhs.first;
                }
                return lhs.second < rhs.second;
            });
        for (int32_t index = 0; index < input.top_k; ++index) {
            const size_t cell =
                candidates[static_cast<size_t>(index)].second;
            output.selected_mask[cell] = 1;
        }
        output.selected_count = input.top_k;
    }
    return Status::ok();
}

} // namespace omnivoice
