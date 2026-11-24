#include "ggml_session.h"

#include "ggml-backend.h"

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

GGMLSession::GGMLSession() = default;

GGMLSession::~GGMLSession() {
    release();
}

void GGMLSession::release() noexcept {
    if (backend_ != nullptr) {
        ggml_backend_free(backend_);
        backend_ = nullptr;
    }
}

Status GGMLSession::init(const GGMLSessionParams& params) {
    if (backend_ != nullptr) {
        return Status::error(
            "GGML session is already initialized");
    }
    if (params.device.empty()) {
        return Status::error(
            "GGML session device name must not be empty");
    }

    log_message(params.log, "loading backend: " + params.device);
    ggml_backend_load_all();
    Status status = initialize_device(params.device, backend_);
    if (!status) {
        release();
        return status;
    }
    log_message(
        params.log,
        "backend ready: " + std::string(ggml_backend_name(backend_)));
    return Status::ok();
}

bool GGMLSession::is_ready() const noexcept {
    return backend_ != nullptr;
}

ggml_backend_t GGMLSession::backend() const noexcept {
    return backend_;
}

} // namespace omnivoice
