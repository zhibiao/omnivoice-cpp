#pragma once

#include "config.h"
#include "status.h"
#include "types.h"

#include <memory>
#include <string>

namespace omnivoice {

class GGUFModel;

struct GGMLModelParams {
    std::string lm_gguf_path;
    std::string audio_tokenizer_gguf_path;
    LogCallback log;
};

// Owns the two GGUF files and the parsed OmniVoice configuration. This is an
// internal resource bundle; callers interact through OmniVoiceModel.
class GGMLModel {
public:
    GGMLModel();
    ~GGMLModel();

    GGMLModel(const GGMLModel&) = delete;
    GGMLModel& operator=(const GGMLModel&) = delete;

    Status load(const GGMLModelParams& params);

    bool is_ready() const noexcept;
    const OmniVoiceConfig& config() const noexcept;
    const GGUFModel& lm_gguf() const noexcept;
    const GGUFModel& audio_gguf() const noexcept;

private:
    void release() noexcept;

    OmniVoiceConfig config_;
    std::unique_ptr<GGUFModel> lm_gguf_;
    std::unique_ptr<GGUFModel> audio_gguf_;
};

} // namespace omnivoice
