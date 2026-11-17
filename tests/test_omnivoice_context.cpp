#include "omnivoice.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

int main() {
    omnivoice::OmniVoiceContextParams cpu;
    cpu.device = "CPU";

    omnivoice::OmniVoiceContext runtime_only;
    omnivoice::Status status = runtime_only.init_runtime(cpu);
    if (!status ||
        !runtime_only.backend() ||
        runtime_only.is_ready()) {
        std::cerr << "runtime-only initialization failed\n";
        return 1;
    }
    if (runtime_only.init_runtime(cpu)) {
        std::cerr << "duplicate runtime initialization must fail\n";
        return 1;
    }

    omnivoice::OmniVoiceContext invalid;
    omnivoice::OmniVoiceContextParams invalid_params;
    invalid_params.device = "OmniVoiceDeviceThatDoesNotExist";
    if (invalid.init_runtime(invalid_params) || invalid.is_ready()) {
        std::cerr << "invalid device must fail\n";
        return 1;
    }

    omnivoice::OmniVoiceModel unloaded_model;
    if (unloaded_model.load(runtime_only)) {
        std::cerr << "model must reject a context without GGUF files\n";
        return 1;
    }

    if (!std::filesystem::is_regular_file(OMNIVOICE_LM_GGUF) ||
        !std::filesystem::is_regular_file(OMNIVOICE_AUDIO_GGUF)) {
        std::cerr << "OmniVoice model GGUF is unavailable; skipping VM\n";
        return 77;
    }

    const char* device = std::getenv("OMNIVOICE_TEST_DEVICE");
    omnivoice::OmniVoiceContextParams params;
    params.device = device == nullptr ? "CPU" : device;
    std::vector<std::string> logs;
    params.log = [&](const std::string& line) {
        logs.push_back(line);
    };

    omnivoice::OmniVoiceContext context;
    status = context.load(
        OMNIVOICE_LM_GGUF,
        OMNIVOICE_AUDIO_GGUF,
        params);
    if (!status) {
        if (device != nullptr) {
            std::cerr << "requested backend unavailable; skipping\n";
            return 77;
        }
        std::cerr << status.message() << '\n';
        return 1;
    }
    if (!context.is_ready() || !context.backend()) {
        std::cerr << "loaded context is not ready\n";
        return 1;
    }
    if (context.load(
            OMNIVOICE_LM_GGUF,
            OMNIVOICE_AUDIO_GGUF,
            params)) {
        std::cerr << "duplicate context load must fail\n";
        return 1;
    }

    omnivoice::OmniVoiceModel model;
    status = model.load(context, params.log);
    if (!status || !model.is_loaded()) {
        std::cerr << status.message() << '\n';
        return 1;
    }
    bool model_ready_logged = false;
    for (const std::string& line : logs) {
        if (line.find("OmniVoice model ready") != std::string::npos) {
            model_ready_logged = true;
        }
    }
    if (!model_ready_logged) {
        std::cerr << "model load did not emit readiness log\n";
        return 1;
    }
    if (model.load(context)) {
        std::cerr << "duplicate model load must fail\n";
        return 1;
    }
    if (model.generation_defaults().num_step != 32) {
        std::cerr << "unexpected generation defaults\n";
        return 1;
    }
    return 0;
}
