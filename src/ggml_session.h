#pragma once

#include "status.h"
#include "types.h"

#include "ggml-backend.h"

#include <string>

namespace omnivoice {

struct GGMLSessionParams {
    std::string device = "CPU";
    LogCallback log;
};

// Owns the selected GGML backend. Models loaded through OmniVoiceModel borrow
// this backend, so the session must outlive all backend resources.
class GGMLSession {
public:
    GGMLSession();
    ~GGMLSession();

    GGMLSession(const GGMLSession&) = delete;
    GGMLSession& operator=(const GGMLSession&) = delete;

    Status init(const GGMLSessionParams& params);

    bool is_ready() const noexcept;

    ggml_backend_t backend() const noexcept;

private:
    void release() noexcept;

    ggml_backend_t backend_ = nullptr;
};

} // namespace omnivoice
