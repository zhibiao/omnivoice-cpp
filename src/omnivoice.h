#pragma once

#include "config.h"
#include "prompt_builder.h"
#include "status.h"
#include "types.h"
#include "voice_clone_prompt.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace omnivoice {

struct VoiceClonePromptRequest {
    AudioBuffer reference_audio;
    std::string reference_text;
    std::string language = "None";
    bool denoise = true;
    bool preprocess_reference = true;
    LogCallback log;
};

struct VoiceCloneRequest {
    std::string target_text;
    std::string language = "None";
    bool denoise = true;
    float speed = 1.0f;
    int32_t target_frames = 0;
    float duration_seconds = 0.0f;
    uint64_t seed = 0;
    std::optional<VoiceCloneConfig> generation_config;
    LogCallback log;
};

struct VoiceCloneOutput {
    VoiceClonePrompt prompt;
    PromptPlan prompt_plan;
    AudioCodes target_codes;
    std::vector<PromptPlan> chunk_prompt_plans;
    std::vector<AudioCodes> chunk_codes;
    AudioBuffer audio;
    std::vector<int32_t> counts_per_step;
};

struct OmniVoiceModelParams {
    std::string device = "CPU";
    LogCallback log;
};

class OmniVoiceModel {
public:
    OmniVoiceModel();
    ~OmniVoiceModel();

    OmniVoiceModel(const OmniVoiceModel&) = delete;
    OmniVoiceModel& operator=(const OmniVoiceModel&) = delete;

    // Not thread-safe: the ggml graphs cached inside the LM and the audio
    // decoder are mutated on first use, so callers must serialize access to
    // a single instance (see omnivoice_server for the worker pattern).
    Status load(
        const std::string& lm_gguf_path,
        const std::string& audio_tokenizer_gguf_path,
        const OmniVoiceModelParams& params = {});

    Status create_prompt(
        const VoiceClonePromptRequest& request,
        VoiceClonePrompt& prompt) const;

    Status generate_voice_clone(
        const VoiceClonePrompt& prompt,
        const VoiceCloneRequest& request,
        VoiceCloneOutput& output) const;

    Status decode_codes(
        const AudioCodes& codes,
        AudioBuffer& audio,
        const LogCallback& log = {}) const;

    VoiceCloneConfig generation_defaults() const noexcept;
    bool is_loaded() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace omnivoice
