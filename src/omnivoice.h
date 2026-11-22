#pragma once

#include "config.h"
#include "diffusion_decoder/diffusion_decoder.h"
#include "ggml_runtime.h"
#include "higgs_tokenizer/decoder.h"
#include "higgs_tokenizer/encoder.h"
#include "prompt_builder.h"
#include "qwen3_lm/qwen3_model.h"
#include "status.h"
#include "text_tokenizer/text_tokenizer.h"
#include "types.h"
#include "voice_clone_prompt.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace omnivoice {

class GGUFModel;

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

class OmniVoiceModel {
public:
    OmniVoiceModel();
    ~OmniVoiceModel();

    OmniVoiceModel(const OmniVoiceModel&) = delete;
    OmniVoiceModel& operator=(const OmniVoiceModel&) = delete;

    // The model borrows the runtime and is released before it.
    Status load(
        GGMLRuntime& runtime,
        const LogCallback& log = {});

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
    void release() noexcept;

    GGMLRuntime* runtime_ = nullptr;
    OmniVoiceConfig config_;
    TextTokenizer tokenizer_;
    std::unique_ptr<OmniVoiceLM> lm_;
    std::unique_ptr<HiggsAudioEncoder> encoder_;
    std::unique_ptr<HiggsAudioDecoder> decoder_;
    bool loaded_ = false;
};

} // namespace omnivoice
