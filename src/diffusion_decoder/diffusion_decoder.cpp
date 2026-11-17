#include "diffusion_decoder/diffusion_decoder.h"

#include "diffusion_decoder/mask_schedule.h"
#include "diffusion_decoder/token_selector.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

namespace omnivoice {

namespace {

using Clock = std::chrono::steady_clock;

std::string elapsed_milliseconds(Clock::time_point start) {
    const double milliseconds = std::chrono::duration<double, std::milli>(
        Clock::now() - start).count();
    char text[32];
    std::snprintf(text, sizeof(text), "%.2f", milliseconds);
    return text;
}

Status validate_prompt(
    const PromptPlan& prompt,
    const OmniVoiceLMConfig& config) {
    if (prompt.codebooks != config.num_audio_codebooks) {
        return Status::error("diffusion prompt codebook count mismatch");
    }
    if (prompt.layout.sequence_length <= 0 ||
        prompt.layout.target_frames <= 0 ||
        prompt.layout.target_start < 0 ||
        prompt.layout.target_start + prompt.layout.target_frames >
            prompt.layout.sequence_length) {
        return Status::error("diffusion prompt layout is invalid");
    }
    if (prompt.input_ids.size() !=
            static_cast<size_t>(prompt.codebooks) *
                static_cast<size_t>(prompt.layout.sequence_length) ||
        prompt.audio_mask.size() !=
            static_cast<size_t>(prompt.layout.sequence_length)) {
        return Status::error("diffusion prompt tensor has an invalid size");
    }

    const size_t sequence =
        static_cast<size_t>(prompt.layout.sequence_length);
    for (int32_t codebook = 0; codebook < prompt.codebooks; ++codebook) {
        const size_t row =
            static_cast<size_t>(codebook) * sequence;
        for (int32_t frame = 0;
             frame < prompt.layout.target_frames;
             ++frame) {
            if (prompt.input_ids[
                    row +
                    static_cast<size_t>(
                        prompt.layout.target_start + frame)] !=
                config.audio_mask_id) {
                return Status::error(
                    "diffusion target region must start as MASK");
            }
        }
    }
    return Status::ok();
}

} // namespace

Status run_diffusion_decoder(
    const PromptPlan& prompt,
    const OmniVoiceLMConfig& config,
    const VoiceCloneConfig& options,
    uint64_t seed,
    DiffusionLmRunner& runner,
    DiffusionDecodeResult& result,
    const LogCallback& log) {
    result = {};
    Status status = validate_prompt(prompt, config);
    if (!status) {
        return status;
    }
    if (options.num_step <= 0) {
        return Status::error("diffusion decoder requires at least one step");
    }

    const int32_t codebooks = prompt.codebooks;
    const int32_t frames = prompt.layout.target_frames;
    const size_t cell_count =
        static_cast<size_t>(codebooks) *
        static_cast<size_t>(frames);
    log_message(
        log,
        "diffusion: " + std::to_string(options.num_step) +
            " steps, " + std::to_string(codebooks) + " x " +
            std::to_string(frames) + " target cells");

    MaskSchedule schedule;
    status = build_mask_schedule(
        frames,
        codebooks,
        options.num_step,
        options.t_shift,
        schedule);
    if (!status) {
        return status;
    }

    std::vector<int32_t> current(
        cell_count,
        config.audio_mask_id);
    OmniVoiceLmInput conditional;
    conditional.sequence_length = prompt.layout.sequence_length;
    conditional.codebooks = codebooks;
    conditional.target_start = prompt.layout.target_start;
    conditional.target_frames = frames;
    conditional.input_ids = prompt.input_ids;
    conditional.audio_mask = prompt.audio_mask;

    OmniVoiceLmInput unconditional;
    unconditional.sequence_length = frames;
    unconditional.codebooks = codebooks;
    unconditional.target_start = 0;
    unconditional.target_frames = frames;
    unconditional.audio_mask.assign(
        static_cast<size_t>(frames),
        1);
    unconditional.input_ids = current;

    TokenSelector selector;
    selector.seed(seed);
    TokenSelectorOptions selector_options;
    selector_options.guidance_scale = options.guidance_scale;
    selector_options.layer_penalty_factor =
        options.layer_penalty_factor;
    selector_options.position_temperature =
        options.position_temperature;
    selector_options.class_temperature =
        options.class_temperature;
    bool use_selector_graph =
        runner.supports_selector_graph() &&
        selector_options.class_temperature == 0.0f;

    if (use_selector_graph) {
        status = selector.init_graph(
            runner.backend(),
            codebooks,
            frames,
            config.audio_vocab_size,
            config.audio_mask_id,
            selector_options.guidance_scale);
        if (!status) {
            log_message(
                log,
                "selector: backend graph unavailable (" +
                    status.message() +
                    "), using CPU selector");
            use_selector_graph = false;
        } else {
            const char* backend_name =
                ggml_backend_name(runner.backend());
            log_message(
                log,
                "selector: " +
                    std::string(
                        backend_name == nullptr
                            ? "backend"
                            : backend_name) +
                    " GGML graph, fixed output mask");
        }
    } else {
        log_message(
            log,
            "selector: CPU path, fixed output mask");
    }
    status = runner.init_graphs(
        conditional,
        unconditional,
        !use_selector_graph);
    if (!status) {
        return status;
    }

    result.counts_per_step = schedule.counts;
    size_t selected_total = 0;
    TokenSelectionOutput selection;
    for (int32_t step = 0; step < options.num_step; ++step) {
        const auto step_start = Clock::now();
        OmniVoiceLmOutput conditional_output;
        const auto conditional_start = Clock::now();
        status = runner.run_conditional(conditional, conditional_output);
        if (!status) {
            return status;
        }
        const std::string conditional_time =
            elapsed_milliseconds(conditional_start);
        if (use_selector_graph
                ? conditional_output.target_logits_tensor == nullptr
                : conditional_output.target_logits.size() !=
                    cell_count * config.audio_vocab_size) {
            return Status::error(
                "conditional LM produced invalid logits");
        }

        const auto unconditional_start = Clock::now();
        OmniVoiceLmInput step_unconditional = unconditional;
        step_unconditional.input_ids = current;
        OmniVoiceLmOutput unconditional_output;
        status = runner.run_unconditional(step_unconditional, unconditional_output);
        if (!status) {
            return status;
        }
        const std::string unconditional_time =
            elapsed_milliseconds(unconditional_start);
        if (use_selector_graph
                ? unconditional_output.target_logits_tensor == nullptr
                : unconditional_output.target_logits.size() !=
                    conditional_output.target_logits.size()) {
            return Status::error(
                "unconditional LM produced invalid logits");
        }

        const auto selection_start = Clock::now();
        TokenSelectionInput selection_input;
        selection_input.codebooks = codebooks;
        selection_input.frames = frames;
        selection_input.vocab_size = config.audio_vocab_size;
        selection_input.mask_id = config.audio_mask_id;
        selection_input.top_k = schedule.counts[
            static_cast<size_t>(step)];
        selection_input.current_tokens = &current;

        if (use_selector_graph) {
            status = selector.select_graph(
                selection_input,
                selector_options,
                conditional_output.target_logits_tensor,
                unconditional_output.target_logits_tensor,
                selection);
        } else {
            selection_input.cond_logits =
                &conditional_output.target_logits;
            selection_input.uncond_logits =
                &unconditional_output.target_logits;
            status = selector.select(
                selection_input,
                selector_options,
                selection);
        }
        if (!status) {
            return status;
        }
        if (selection.selected_count != selection_input.top_k ||
            selection.selected_mask.size() != cell_count) {
            return Status::error("token selector returned an invalid top-k");
        }
        const std::string selection_time =
            elapsed_milliseconds(selection_start);

        const size_t prompt_sequence =
            static_cast<size_t>(prompt.layout.sequence_length);
        size_t selected_this_step = 0;
        for (size_t cell = 0; cell < cell_count; ++cell) {
            if (selection.selected_mask[cell] == 0) {
                continue;
            }
            const int32_t token = selection.predictions[cell];
            if (token < 0 ||
                token >= config.audio_vocab_size ||
                token == config.audio_mask_id) {
                return Status::error(
                    "token selector produced invalid token " +
                    std::to_string(token) +
                    " at codebook " +
                    std::to_string(static_cast<int32_t>(cell) / frames) +
                    ", frame " +
                    std::to_string(static_cast<int32_t>(cell) % frames));
            }
            current[cell] = token;
            const int32_t codebook =
                static_cast<int32_t>(cell) / frames;
            const int32_t frame =
                static_cast<int32_t>(cell) % frames;
            conditional.input_ids[
                static_cast<size_t>(codebook) * prompt_sequence +
                static_cast<size_t>(prompt.layout.target_start) +
                static_cast<size_t>(frame)] = token;
            ++selected_this_step;
        }
        if (selected_this_step !=
            static_cast<size_t>(selection.selected_count)) {
            return Status::error(
                "token selector mask count mismatch");
        }
        selected_total += selected_this_step;
        const size_t remaining = cell_count - selected_total;
        log_message(
            log,
            "diffusion step " + std::to_string(step + 1) + "/" +
                std::to_string(options.num_step) +
                ": selected=" +
                std::to_string(selected_this_step) +
                " remaining=" + std::to_string(remaining) +
                " cond=" + conditional_time + " ms" +
                " uncond=" + unconditional_time + " ms" +
                " select=" + selection_time + " ms" +
                " total=" + elapsed_milliseconds(step_start) + " ms");
    }

    for (int32_t token : current) {
        if (token == config.audio_mask_id) {
            return Status::error(
                "diffusion decoder left target cells masked");
        }
    }
    result.target_codes.codebooks = codebooks;
    result.target_codes.frames = frames;
    result.target_codes.data = std::move(current);
    return Status::ok();
}

} // namespace omnivoice
