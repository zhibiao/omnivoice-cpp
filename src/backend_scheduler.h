#pragma once

#include "status.h"

#include "ggml-backend.h"

#include <cstddef>

namespace omnivoice {

// Creates a scheduler with the selected primary backend and, when needed, a
// CPU fallback. ggml requires CPU to be the final backend in the scheduler.
class BackendScheduler {
public:
    BackendScheduler() = default;
    ~BackendScheduler() {
        reset();
    }

    BackendScheduler(const BackendScheduler&) = delete;
    BackendScheduler& operator=(const BackendScheduler&) = delete;

    Status create(
        ggml_backend_t primary,
        int max_nodes,
        bool parallel = false,
        bool op_offload = false) {
        reset();
        if (primary == nullptr || max_nodes <= 0) {
            return Status::error("invalid backend scheduler arguments");
        }
        ggml_backend_dev_t primary_device =
            ggml_backend_get_device(primary);
        const bool primary_is_cpu =
            primary_device != nullptr &&
            ggml_backend_dev_type(primary_device) ==
                GGML_BACKEND_DEVICE_TYPE_CPU;

        backends_[0] = primary;
        buffer_types_[0] =
            ggml_backend_get_default_buffer_type(primary);
        count_ = 1;
        if (!primary_is_cpu) {
            fallback_ = ggml_backend_init_by_name("CPU", nullptr);
            if (fallback_ == nullptr) {
                reset();
                return Status::error(
                    "failed to initialize CPU fallback backend");
            }
            backends_[1] = fallback_;
            buffer_types_[1] =
                ggml_backend_get_default_buffer_type(fallback_);
            count_ = 2;
        }

        scheduler_ = ggml_backend_sched_new(
            backends_,
            buffer_types_,
            count_,
            max_nodes,
            parallel,
            op_offload);
        if (scheduler_ == nullptr) {
            reset();
            return Status::error("failed to create backend scheduler");
        }
        return Status::ok();
    }

    void reset() noexcept {
        if (scheduler_ != nullptr) {
            ggml_backend_sched_free(scheduler_);
            scheduler_ = nullptr;
        }
        if (fallback_ != nullptr) {
            ggml_backend_free(fallback_);
            fallback_ = nullptr;
        }
        backends_[0] = nullptr;
        backends_[1] = nullptr;
        buffer_types_[0] = nullptr;
        buffer_types_[1] = nullptr;
        count_ = 0;
    }

    ggml_backend_sched_t get() const noexcept {
        return scheduler_;
    }

private:
    ggml_backend_t backends_[2] = {nullptr, nullptr};
    ggml_backend_buffer_type_t buffer_types_[2] = {nullptr, nullptr};
    ggml_backend_t fallback_ = nullptr;
    ggml_backend_sched_t scheduler_ = nullptr;
    int count_ = 0;
};

} // namespace omnivoice
