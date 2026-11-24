#include "omnivoice.h"

#include "ggml_session.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

int main() {
    omnivoice::GGMLSessionParams cpu;
    cpu.device = "CPU";

    omnivoice::GGMLSession session;
    omnivoice::Status status = session.init(cpu);
    if (!status ||
        !session.backend() ||
        !session.is_ready()) {
        std::cerr << "session initialization failed\n";
        return 1;
    }
    if (session.init(cpu)) {
        std::cerr << "duplicate session initialization must fail\n";
        return 1;
    }

    omnivoice::GGMLSession invalid_session;
    omnivoice::GGMLSessionParams invalid_params;
    invalid_params.device = "OmniVoiceDeviceThatDoesNotExist";
    if (invalid_session.init(invalid_params) ||
        invalid_session.is_ready()) {
        std::cerr << "invalid device must fail\n";
        return 1;
    }

    omnivoice::OmniVoiceModel empty_paths_model;
    omnivoice::OmniVoiceModelParams empty_paths_params;
    if (empty_paths_model.load("", "", empty_paths_params) ||
        empty_paths_model.is_loaded()) {
        std::cerr << "empty GGUF paths must fail\n";
        return 1;
    }

    omnivoice::OmniVoiceModel unloaded_model;
    omnivoice::OmniVoiceModelParams unloaded_params;
    if (unloaded_model.load(
            "missing-lm.gguf",
            "missing-audio.gguf",
            unloaded_params)) {
        std::cerr << "model must reject missing GGUF files\n";
        return 1;
    }

    if (!std::filesystem::is_regular_file(OMNIVOICE_LM_GGUF) ||
        !std::filesystem::is_regular_file(OMNIVOICE_AUDIO_GGUF)) {
        std::cerr << "OmniVoice model GGUF is unavailable; skipping VM\n";
        return 77;
    }

    const char* device = std::getenv("OMNIVOICE_TEST_DEVICE");
    omnivoice::GGMLSessionParams params;
    params.device = device == nullptr ? "CPU" : device;
    std::vector<std::string> logs;
    params.log = [&](const std::string& line) {
        logs.push_back(line);
    };

    omnivoice::GGMLSession loaded_session;
    status = loaded_session.init(params);
    if (!status) {
        if (device != nullptr) {
            std::cerr << "requested backend unavailable; skipping\n";
            return 77;
        }
        std::cerr << status.message() << '\n';
        return 1;
    }

    omnivoice::OmniVoiceModel model;
    omnivoice::OmniVoiceModelParams model_load_params;
    model_load_params.device = params.device;
    model_load_params.log = params.log;
    status = model.load(
        OMNIVOICE_LM_GGUF,
        OMNIVOICE_AUDIO_GGUF,
        model_load_params);
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
    if (model.load(
            OMNIVOICE_LM_GGUF,
            OMNIVOICE_AUDIO_GGUF,
            model_load_params)) {
        std::cerr << "duplicate model load must fail\n";
        return 1;
    }
    if (model.generation_defaults().num_step != 32) {
        std::cerr << "unexpected generation defaults\n";
        return 1;
    }
    return 0;
}
