#include "ggml_runtime.h"

#include "gguf_model.h"

#include "ggml-backend.h"

#include <memory>
#include <string>

namespace omnivoice {

namespace {

Status initialize_device(
    const std::string& name,
    ggml_backend_t& backend) {
    ggml_backend_dev_t device =
        ggml_backend_dev_by_name(name.c_str());
    if (device == nullptr) {
        return Status::error(
            "backend device not found: " + name);
    }
    backend = ggml_backend_dev_init(device, nullptr);
    if (backend == nullptr) {
        return Status::error(
            "failed to initialize backend device: " + name);
    }
    return Status::ok();
}

} // namespace

GGMLRuntime::GGMLRuntime() = default;

GGMLRuntime::~GGMLRuntime() {
    release();
}

void GGMLRuntime::release() noexcept {
    if (backend_ != nullptr) {
        ggml_backend_free(backend_);
        backend_ = nullptr;
    }
    audio_gguf_.reset();
    lm_gguf_.reset();
    config_ = {};
    runtime_ready_ = false;
    gguf_ready_ = false;
}

Status GGMLRuntime::init(const GGMLRuntimeParams& params) {
    if (runtime_ready_) {
        return Status::error(
            "GGML runtime is already initialized");
    }
    if (params.device.empty()) {
        return Status::error(
            "GGML runtime device name must not be empty");
    }

    log_message(params.log, "loading backend: " + params.device);
    ggml_backend_load_all();
    Status status = initialize_device(params.device, backend_);
    if (!status) {
        release();
        return status;
    }
    runtime_ready_ = true;
    log_message(
        params.log,
        "backend ready: " + std::string(ggml_backend_name(backend_)));
    return Status::ok();
}

Status GGMLRuntime::load(
    const std::string& lm_gguf_path,
    const std::string& audio_tokenizer_gguf_path,
    const GGMLRuntimeParams& params) {
    if (gguf_ready_) {
        return Status::error("GGML runtime is already loaded");
    }
    if (!runtime_ready_) {
        Status status = init(params);
        if (!status) {
            return status;
        }
    }

    log_message(params.log, "loading LM GGUF: " + lm_gguf_path);
    std::unique_ptr<GGUFModel> lm_gguf;
    Status status = GGUFModel::open(lm_gguf_path, lm_gguf);
    if (!status) {
        return status;
    }
    log_message(
        params.log,
        "LM GGUF ready: " + std::to_string(lm_gguf->tensor_count()) +
            " tensors");

    log_message(
        params.log,
        "loading audio tokenizer GGUF: " + audio_tokenizer_gguf_path);
    std::unique_ptr<GGUFModel> audio_gguf;
    status = GGUFModel::open(audio_tokenizer_gguf_path, audio_gguf);
    if (!status) {
        return status;
    }
    log_message(
        params.log,
        "audio tokenizer GGUF ready: " +
            std::to_string(audio_gguf->tensor_count()) + " tensors");

    log_message(params.log, "loading OmniVoice configuration");
    OmniVoiceConfig config;
    status = load_omnivoice_config(
        *lm_gguf,
        *audio_gguf,
        config);
    if (!status) {
        return status;
    }

    lm_gguf_ = std::move(lm_gguf);
    audio_gguf_ = std::move(audio_gguf);
    config_ = std::move(config);
    gguf_ready_ = true;
    log_message(
        params.log,
        "configuration ready: LM hidden=" +
            std::to_string(config_.lm.llm.hidden_size) +
            ", layers=" + std::to_string(config_.lm.llm.num_hidden_layers) +
            ", audio=" +
            std::to_string(config_.audio_tokenizer.frame_rate) +
            " Hz x " +
            std::to_string(config_.lm.num_audio_codebooks) +
            " codebooks");
    return Status::ok();
}

bool GGMLRuntime::is_ready() const noexcept {
    return runtime_ready_ &&
        gguf_ready_ &&
        backend_ != nullptr &&
        lm_gguf_ != nullptr &&
        audio_gguf_ != nullptr;
}

const OmniVoiceConfig& GGMLRuntime::config() const noexcept {
    return config_;
}

ggml_backend_t GGMLRuntime::backend() const noexcept {
    return backend_;
}

const GGUFModel& GGMLRuntime::lm_gguf() const {
    return *lm_gguf_;
}

const GGUFModel& GGMLRuntime::audio_gguf() const {
    return *audio_gguf_;
}

} // namespace omnivoice
