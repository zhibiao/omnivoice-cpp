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

omnivoice::Status test_graph_selector(
    const omnivoice::TokenSelectionInput& input,
    const omnivoice::TokenSelectorOptions& options,
    const std::vector<int32_t>& expected_predictions,
    const std::vector<float>& expected_confidence,
    const std::vector<uint8_t>& expected_mask) {
    static const bool backends_loaded = []() {
        ggml_backend_load_all();
        return true;
    }();
    (void)backends_loaded;
    const char* device_name =
        std::getenv("OMNIVOICE_TEST_DEVICE");
    if (device_name == nullptr || device_name[0] == '\0') {
        device_name = "CPU";
    }
    ggml_backend_dev_t device =
        ggml_backend_dev_by_name(device_name);
    ggml_backend_t backend = device == nullptr
        ? nullptr
        : ggml_backend_dev_init(device, nullptr);
    if (backend == nullptr) {
        return omnivoice::Status::error(
            "CPU backend unavailable for selector graph test");
    }

    const int64_t audio_flat =
        static_cast<int64_t>(input.codebooks) *
        input.vocab_size;
    ggml_init_params params{};
    params.mem_size = 16 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context* context = ggml_init(params);
    ggml_backend_buffer_t buffer = nullptr;
    auto cleanup = [&]() {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
        if (context != nullptr) {
            ggml_free(context);
        }
        ggml_backend_free(backend);
    };
    if (context == nullptr) {
        cleanup();
        return omnivoice::Status::error(
            "failed to create selector graph test context");
    }

    ggml_tensor* conditional = ggml_new_tensor_2d(
        context,
        GGML_TYPE_F32,
        audio_flat,
        input.frames);
    ggml_tensor* unconditional = ggml_new_tensor_2d(
        context,
        GGML_TYPE_F32,
        audio_flat,
        input.frames);
    buffer = ggml_backend_alloc_ctx_tensors(
        context,
        backend);
    if (buffer == nullptr) {
        cleanup();
        return omnivoice::Status::error(
            "failed to allocate selector graph test tensors");
    }

    const std::vector<float> conditional_data =
        to_frame_major(
            *input.cond_logits,
            input.codebooks,
            input.frames,
            input.vocab_size);
    const std::vector<float> unconditional_data =
        to_frame_major(
            *input.uncond_logits,
            input.codebooks,
            input.frames,
            input.vocab_size);
    ggml_backend_tensor_set(
        conditional,
        conditional_data.data(),
        0,
        conditional_data.size() * sizeof(float));
    ggml_backend_tensor_set(
        unconditional,
        unconditional_data.data(),
        0,
        unconditional_data.size() * sizeof(float));

    omnivoice::Status status = omnivoice::Status::ok();
    {
        omnivoice::TokenSelector selector;
        selector.seed(1234);
        status = selector.init_graph(
            backend,
            input.codebooks,
            input.frames,
            input.vocab_size,
            input.mask_id,
            options.guidance_scale);
        omnivoice::TokenSelectionOutput output;
        for (int32_t run = 0; status && run < 4; ++run) {
            status = selector.select_graph(
                input,
                options,
                conditional,
                unconditional,
                output);
            if (status &&
                output.predictions != expected_predictions) {
                for (size_t index = 0;
                     index < output.predictions.size();
                     ++index) {
                    if (output.predictions[index] !=
                        expected_predictions[index]) {
                        std::cerr
                            << "graph prediction mismatch at " << index
                            << " run " << run
                            << ": actual=" << output.predictions[index]
                            << " expected=" << expected_predictions[index]
                            << " actual_confidence="
                            << output.confidence[index]
                            << " expected_confidence="
                            << expected_confidence[index]
                            << '\n';
                    }
                }
                status = omnivoice::Status::error(
                    "selector graph predicted token mismatch");
            }
            if (status) {
                float maximum_error = 0.0f;
                size_t maximum_error_index = 0;
                for (size_t index = 0;
                     index < output.confidence.size();
                     ++index) {
                    const float error = std::abs(
                        output.confidence[index] -
                        expected_confidence[index]);
                    if (error > maximum_error) {
                        maximum_error = error;
                        maximum_error_index = index;
                    }
                }
                if (maximum_error > 2e-4f) {
                    std::cerr
                        << "graph confidence maximum error: "
                        << maximum_error << " run " << run
                        << " shape=" << input.codebooks << 'x'
                        << input.frames << 'x' << input.vocab_size
                        << " index=" << maximum_error_index
                        << " actual="
                        << output.confidence[maximum_error_index]
                        << " expected="
                        << expected_confidence[maximum_error_index]
                        << '\n';
                    for (size_t index = 0;
                         index < output.confidence.size();
                         ++index) {
                        std::cerr
                            << " confidence[" << index << "] actual="
                            << output.confidence[index]
                            << " expected="
                            << expected_confidence[index]
                            << '\n';
                    }
                    status = omnivoice::Status::error(
                        "selector graph confidence mismatch");
                }
            }
            if (status &&
                (output.selected_mask != expected_mask ||
                 output.selected_count != input.top_k)) {
                status = omnivoice::Status::error(
                    "selector graph selected mask mismatch");
            }
        }
    }

    cleanup();
    return status;
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

    status = test_graph_selector(
        input,
        options,
        flatten_i32(golden.at("predictions")),
        expected_confidence,
        expected_mask);
    if (!status) {
        std::cerr << status.message() << '\n';
        return 1;
    }

    omnivoice::TokenSelectionInput production_shape;
    production_shape.codebooks = 8;
    production_shape.frames = 25;
    production_shape.vocab_size = 1025;
    production_shape.mask_id = 1024;
    production_shape.top_k = 20;
    std::vector<int32_t> production_current(
        static_cast<size_t>(production_shape.codebooks) *
            production_shape.frames,
        production_shape.mask_id);
    std::vector<float> production_cond(
        production_current.size() * production_shape.vocab_size);
    std::vector<float> production_uncond(production_cond.size());
    const size_t production_cells = production_current.size();
    const size_t production_vocab = production_shape.vocab_size;
    for (size_t cell = 0; cell < production_cells; ++cell) {
        const size_t dominant =
            (cell * 37 + 11) % (production_vocab - 1);
        for (size_t token = 0; token < production_vocab; ++token) {
            const size_t index = cell * production_vocab + token;
            production_cond[index] =
                std::sin(static_cast<float>(index) * 0.017f) * 0.25f;
            production_uncond[index] =
                std::cos(static_cast<float>(index) * 0.013f) * 0.20f;
            if (token == dominant) {
                production_cond[index] +=
                    3.0f + static_cast<float>(cell) * 0.01f;
            }
        }
    }
    production_shape.cond_logits = &production_cond;
    production_shape.uncond_logits = &production_uncond;
    production_shape.current_tokens = &production_current;
    omnivoice::TokenSelector production_cpu;
    omnivoice::TokenSelectionOutput production_expected;
    status = production_cpu.select(
        production_shape,
        options,
        production_expected);
    if (status) {
        status = test_graph_selector(
            production_shape,
            options,
            production_expected.predictions,
            production_expected.confidence,
            production_expected.selected_mask);
    }
    if (!status) {
        std::cerr << "production-shape CUDA selector: "
                  << status.message() << '\n';
        return 1;
    }
    for (float guidance : {0.0f, 3.0f}) {
        omnivoice::TokenSelectorOptions variant = options;
        variant.guidance_scale = guidance;
        production_expected = {};
        status = production_cpu.select(
            production_shape,
            variant,
            production_expected);
        if (status) {
            status = test_graph_selector(
                production_shape,
                variant,
                production_expected.predictions,
                production_expected.confidence,
                production_expected.selected_mask);
        }
        if (!status) {
            std::cerr << "selector guidance variant failed: "
                      << status.message() << '\n';
            return 1;
        }
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
