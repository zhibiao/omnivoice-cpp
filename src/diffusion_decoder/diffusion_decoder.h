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

    virtual Status init_graph(
        const OmniVoiceLmInput& conditional,
        const OmniVoiceLmInput& unconditional,
        bool read_host_logits,
        float guidance_scale) = 0;

    virtual Status run_diffusion(
        const OmniVoiceLmInput& conditional,
        const OmniVoiceLmInput& unconditional,
        OmniVoiceLmPackedOutput& output) = 0;
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
