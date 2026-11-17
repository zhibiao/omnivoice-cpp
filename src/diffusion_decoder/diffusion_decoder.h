#pragma once

#include "config.h"
#include "prompt_builder.h"
#include "qwen3_lm/qwen3_model.h"
#include "status.h"
#include "types.h"

#include <cstdint>
#include <vector>

namespace omnivoice {

class DiffusionLmRunner {
public:
    virtual ~DiffusionLmRunner() = default;

    virtual Status init_graphs(
        const OmniVoiceLmInput& conditional,
        const OmniVoiceLmInput& unconditional,
        bool read_host_logits) = 0;

    virtual bool supports_selector_graph() const = 0;
    virtual ggml_backend_t backend() const = 0;

    virtual Status run_conditional(
        const OmniVoiceLmInput& input,
        OmniVoiceLmOutput& output) = 0;

    virtual Status run_unconditional(
        const OmniVoiceLmInput& input,
        OmniVoiceLmOutput& output) = 0;
};

struct DiffusionDecodeResult {
    AudioCodes target_codes;
    std::vector<int32_t> counts_per_step;
};

Status run_diffusion_decoder(
    const PromptPlan& prompt,
    const OmniVoiceLMConfig& config,
    const VoiceCloneConfig& options,
    uint64_t seed,
    DiffusionLmRunner& runner,
    DiffusionDecodeResult& result,
    const LogCallback& log = {});

} // namespace omnivoice
