#pragma once

#include "config.h"
#include "prompt_builder.h"
#include "qwen3_lm/qwen3_model.h"
#include "status.h"
#include "types.h"

#include <cstdint>
#include <vector>

namespace omnivoice {

struct DiffusionDecodeResult {
    AudioCodes target_codes;
    std::vector<int32_t> counts_per_step;
};

Status run_diffusion_decoder(
    const PromptPlan& prompt,
    const OmniVoiceLMConfig& config,
    const VoiceCloneConfig& options,
    uint64_t seed,
    OmniVoiceLM& lm,
    DiffusionDecodeResult& result,
    const LogCallback& log = {});

} // namespace omnivoice
