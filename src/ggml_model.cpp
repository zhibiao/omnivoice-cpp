#include "ggml_model.h"

#include "gguf_model.h"

#include <memory>
#include <string>

namespace omnivoice {

GGMLModel::GGMLModel() = default;

GGMLModel::~GGMLModel() {
    release();
}

void GGMLModel::release() noexcept {
    audio_gguf_.reset();
    lm_gguf_.reset();
    config_ = {};
}

Status GGMLModel::load(const GGMLModelParams& params) {
    if (is_ready()) {
        return Status::error("GGML model is already loaded");
    }
    if (params.lm_gguf_path.empty() ||
        params.audio_tokenizer_gguf_path.empty()) {
        return Status::error("GGML model paths must not be empty");
    }

    release();
    log_message(
        params.log,
        "loading LM GGUF: " + params.lm_gguf_path);
    std::unique_ptr<GGUFModel> lm_gguf;
    Status status = GGUFModel::open(params.lm_gguf_path, lm_gguf);
    if (!status) {
        return status;
    }
    log_message(
        params.log,
        "LM GGUF ready: " + std::to_string(lm_gguf->tensor_count()) +
            " tensors");

    log_message(
        params.log,
        "loading audio tokenizer GGUF: " +
            params.audio_tokenizer_gguf_path);
    std::unique_ptr<GGUFModel> audio_gguf;
    status = GGUFModel::open(
        params.audio_tokenizer_gguf_path,
        audio_gguf);
    if (!status) {
        return status;
    }
    log_message(
        params.log,
        "audio tokenizer GGUF ready: " +
            std::to_string(audio_gguf->tensor_count()) + " tensors");

    log_message(params.log, "loading OmniVoice configuration");
    OmniVoiceConfig config;
    status = load_omnivoice_config(*lm_gguf, *audio_gguf, config);
    if (!status) {
        return status;
    }

    lm_gguf_ = std::move(lm_gguf);
    audio_gguf_ = std::move(audio_gguf);
    config_ = std::move(config);
    log_message(
        params.log,
        "configuration ready: LM hidden=" +
            std::to_string(config_.lm.llm.hidden_size) +
            ", layers=" +
            std::to_string(config_.lm.llm.num_hidden_layers) +
            ", audio=" +
            std::to_string(config_.audio_tokenizer.frame_rate) +
            " Hz x " +
            std::to_string(config_.lm.num_audio_codebooks) +
            " codebooks");
    return Status::ok();
}

bool GGMLModel::is_ready() const noexcept {
    return lm_gguf_ != nullptr && audio_gguf_ != nullptr;
}

const OmniVoiceConfig& GGMLModel::config() const noexcept {
    return config_;
}

const GGUFModel& GGMLModel::lm_gguf() const noexcept {
    return *lm_gguf_;
}

const GGUFModel& GGMLModel::audio_gguf() const noexcept {
    return *audio_gguf_;
}

} // namespace omnivoice
