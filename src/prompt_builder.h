#pragma once

#include "config.h"
#include "status.h"
#include "text_tokenizer.h"
#include "types.h"

#include <cstdint>
#include <string>
#include <vector>

namespace omnivoice {

struct VoiceClonePromptInput {
    std::string reference_text;
    std::string target_text;
    std::string language;
    AudioCodes reference_codes;
    int32_t target_frames = 0;
    bool denoise = true;
};

struct PromptLayout {
    int32_t style_frames = 0;
    int32_t text_frames = 0;
    int32_t reference_start = 0;
    int32_t target_start = 0;
    int32_t sequence_length = 0;
    int32_t target_frames = 0;
};

struct PromptPlan {
    int32_t codebooks = 0;
    PromptLayout layout;
    std::vector<int32_t> style_tokens;
    std::vector<int32_t> text_tokens;
    // Codebook-major [codebooks, sequence_length].
    std::vector<int32_t> input_ids;
    // One value per sequence position; audio starts at reference_start.
    std::vector<uint8_t> audio_mask;
};

Status combine_reference_target_text(
    const std::string& reference_text,
    const std::string& target_text,
    std::string& combined_text);

Status build_voice_clone_prompt(
    const TextTokenizer& tokenizer,
    const OmniVoiceConfig& config,
    const VoiceClonePromptInput& input,
    PromptPlan& plan);

} // namespace omnivoice
