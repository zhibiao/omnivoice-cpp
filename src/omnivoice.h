#pragma once

#include "config.h"
#include "diffusion_decoder/diffusion_decoder.h"
#include "higgs_tokenizer/decoder.h"
#include "higgs_tokenizer/encoder.h"
#include "prompt_builder.h"
#include "qwen3_lm/qwen3_model.h"
#include "status.h"
#include "text_tokenizer.h"
#include "types.h"
#include "voice_clone_prompt.h"

#include "ggml-backend.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace omnivoice {

class GGUFModel;

struct VoiceClonePromptOptions {
    bool denoise = true;
    bool preprocess_reference = true;
    LogCallback log;
};

struct VoiceCloneRequest {
    AudioBuffer reference_audio;
    std::string reference_text;
    std::string target_text;
    std::string language = "None";
    const VoiceClonePrompt* voice_clone_prompt = nullptr;
    bool preprocess_reference = true;
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

struct OmniVoiceContextParams {
    std::string device = "CPU";
    LogCallback log;
};

class OmniVoiceContext {
public:
    OmniVoiceContext();
    ~OmniVoiceContext();

    OmniVoiceContext(const OmniVoiceContext&) = delete;
    OmniVoiceContext& operator=(const OmniVoiceContext&) = delete;

    Status init_runtime(const OmniVoiceContextParams& params);
    Status load(
        const std::string& lm_gguf_path,
        const std::string& audio_tokenizer_gguf_path,
        const OmniVoiceContextParams& params = {});

    bool is_ready() const noexcept;
    const OmniVoiceConfig& config() const noexcept;
    ggml_backend_t backend() const noexcept;
    const GGUFModel& lm_gguf() const;
    const GGUFModel& audio_gguf() const;

private:
    void release() noexcept;

    OmniVoiceConfig config_;
    std::unique_ptr<GGUFModel> lm_gguf_;
    std::unique_ptr<GGUFModel> audio_gguf_;
    ggml_backend_t backend_ = nullptr;
    bool runtime_ready_ = false;
    bool gguf_ready_ = false;
};

class OmniVoiceModel {
public:
    OmniVoiceModel();
    ~OmniVoiceModel();

    OmniVoiceModel(const OmniVoiceModel&) = delete;
    OmniVoiceModel& operator=(const OmniVoiceModel&) = delete;

    // The model borrows the context and is released before it.
    Status load(
        OmniVoiceContext& context,
        const LogCallback& log = {});

    Status create_prompt(
        const AudioBuffer& reference_audio,
        const std::string& reference_text,
        const std::string& language,
        const VoiceClonePromptOptions& options,
        VoiceClonePrompt& prompt) const;

    Status create_prompt(
        const AudioBuffer& reference_audio,
        const std::string& reference_text,
        const std::string& language,
        bool denoise,
        bool preprocess_reference,
        VoiceClonePrompt& prompt) const;

    Status generate_voice_clone(
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

    OmniVoiceContext* context_ = nullptr;
    OmniVoiceConfig config_;
    TextTokenizer tokenizer_;
    std::unique_ptr<OmniVoiceLM> lm_;
    std::unique_ptr<HiggsAudioEncoder> encoder_;
    std::unique_ptr<HiggsAudioDecoder> decoder_;
    bool loaded_ = false;
};

} // namespace omnivoice
