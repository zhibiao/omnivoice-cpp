#include "omnivoice.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

int main() {
    omnivoice::GGMLRuntimeParams cpu;
    cpu.device = "CPU";

    omnivoice::GGMLRuntime runtime_only;
    omnivoice::Status status = runtime_only.init(cpu);
    if (!status ||
        !runtime_only.backend() ||
        runtime_only.is_ready()) {
        std::cerr << "runtime-only initialization failed\n";
        return 1;
    }
    if (runtime_only.init(cpu)) {
        std::cerr << "duplicate runtime initialization must fail\n";
        return 1;
    }

    omnivoice::GGMLRuntime invalid;
    omnivoice::GGMLRuntimeParams invalid_params;
    invalid_params.device = "OmniVoiceDeviceThatDoesNotExist";
    if (invalid.init(invalid_params) || invalid.is_ready()) {
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
    omnivoice::GGMLRuntimeParams params;
    params.device = device == nullptr ? "CPU" : device;
    std::vector<std::string> logs;
    params.log = [&](const std::string& line) {
        logs.push_back(line);
    };

    omnivoice::GGMLRuntime runtime;
    status = runtime.load(
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
    if (!runtime.is_ready() || !runtime.backend()) {
        std::cerr << "loaded runtime is not ready\n";
        return 1;
    }
    if (runtime.load(
            OMNIVOICE_LM_GGUF,
            OMNIVOICE_AUDIO_GGUF,
            params)) {
        std::cerr << "duplicate runtime load must fail\n";
        return 1;
    }

    omnivoice::OmniVoiceModel model;
    status = model.load(runtime, params.log);
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
    if (model.load(runtime)) {
        std::cerr << "duplicate model load must fail\n";
        return 1;
    }
    if (model.generation_defaults().num_step != 32) {
        std::cerr << "unexpected generation defaults\n";
        return 1;
    }
    return 0;
}
