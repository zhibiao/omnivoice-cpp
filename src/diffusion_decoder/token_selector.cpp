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

TokenSelector::TokenSelector()
    : rng_(0x4f4d4e49564f4943ULL) {
}

TokenSelector::~TokenSelector() = default;

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

Status TokenSelector::select_scored(
    const TokenSelectionInput& input,
    const TokenSelectorOptions& options,
    const std::vector<int32_t>& predictions,
    const std::vector<float>& confidence,
    TokenSelectionOutput& output) {
    if (input.codebooks <= 0 ||
        input.frames <= 0 ||
        input.vocab_size <= 1 ||
        input.mask_id < 0 ||
        input.mask_id >= input.vocab_size) {
        return Status::error("scored token selector dimensions are invalid");
    }
    if (options.class_temperature != 0.0f) {
        return Status::error(
            "scored token selector requires class_temperature=0");
    }
    if (!std::isfinite(options.layer_penalty_factor) ||
        options.layer_penalty_factor < 0.0f ||
        !std::isfinite(options.position_temperature) ||
        options.position_temperature < 0.0f) {
        return Status::error("scored token selector options are invalid");
    }
    const size_t cells =
        static_cast<size_t>(input.codebooks) *
        static_cast<size_t>(input.frames);
    if (input.current_tokens == nullptr ||
        input.current_tokens->size() != cells ||
        predictions.size() != cells ||
        confidence.size() != cells) {
        return Status::error("scored token selector tensor size mismatch");
    }
    size_t pending = 0;
    for (int32_t token : *input.current_tokens) {
        if (token == input.mask_id) {
            ++pending;
        }
    }
    if (input.top_k < 0 ||
        static_cast<size_t>(input.top_k) > pending) {
        return Status::error("scored token selector top_k exceeds pending cells");
    }
    for (size_t cell = 0; cell < cells; ++cell) {
        if (predictions[cell] < 0 ||
            predictions[cell] >= input.vocab_size ||
            predictions[cell] == input.mask_id ||
            !std::isfinite(confidence[cell])) {
            return Status::error(
                "scored token selector received an invalid prediction");
        }
    }

    output.predictions = predictions;
    output.confidence = confidence;
    output.scores.assign(cells, 0.0f);
    output.selected_mask.assign(cells, 0);
    output.selected_count = 0;
    for (size_t cell = 0; cell < cells; ++cell) {
        const int32_t codebook = static_cast<int32_t>(
            cell / static_cast<size_t>(input.frames));
        output.scores[cell] =
            confidence[cell] -
            static_cast<float>(codebook) *
                options.layer_penalty_factor;
    }
    if (options.position_temperature > 0.0f) {
        for (size_t cell = 0; cell < cells; ++cell) {
            if ((*input.current_tokens)[cell] == input.mask_id) {
                output.scores[cell] = gumbel(
                    output.scores[cell],
                    options.position_temperature);
            } else {
                output.scores[cell] =
                    -std::numeric_limits<float>::infinity();
            }
        }
    } else {
        for (size_t cell = 0; cell < cells; ++cell) {
            if ((*input.current_tokens)[cell] != input.mask_id) {
                output.scores[cell] =
                    -std::numeric_limits<float>::infinity();
            }
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
            output.selected_mask[
                candidates[static_cast<size_t>(index)].second] = 1;
        }
        output.selected_count = input.top_k;
    }
    return Status::ok();
}

} // namespace omnivoice
