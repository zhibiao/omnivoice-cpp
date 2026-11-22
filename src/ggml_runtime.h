#pragma once

#include "config.h"
#include "status.h"
#include "types.h"

#include "ggml-backend.h"

#include <memory>
#include <string>

namespace omnivoice {

class GGUFModel;

struct GGMLRuntimeParams {
    std::string device = "CPU";
    LogCallback log;
};

// Owns the selected GGML backend, the two GGUF model files and the parsed
// OmniVoice configuration. Model objects borrow from this runtime, so the
// runtime must outlive every model loaded from it.
class GGMLRuntime {
public:
    GGMLRuntime();
    ~GGMLRuntime();

    GGMLRuntime(const GGMLRuntime&) = delete;
    GGMLRuntime& operator=(const GGMLRuntime&) = delete;

    // Initializes the backend only. Calling load() afterwards reuses it.
    Status init(const GGMLRuntimeParams& params);

    // Initializes the backend when needed, then opens and validates both GGUF
    // files.
    Status load(
        const std::string& lm_gguf_path,
        const std::string& audio_tokenizer_gguf_path,
        const GGMLRuntimeParams& params = {});

    // True when the backend and both GGUF files are ready.
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

} // namespace omnivoice
