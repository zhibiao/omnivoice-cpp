#include "audio_kit.h"
#include "omnivoice.h"
#include "voice_clone_prompt.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#ifndef OMNIVOICE_VERSION
#define OMNIVOICE_VERSION "unknown"
#endif

namespace {

using Clock = std::chrono::steady_clock;

std::string elapsed_text(Clock::time_point start) {
    const double seconds = std::chrono::duration<double>(
        Clock::now() - start).count();
    char text[32];
    std::snprintf(text, sizeof(text), "%.2fs", seconds);
    return text;
}

omnivoice::LogCallback make_logger(Clock::time_point start) {
    return [start](const std::string& line) {
        std::cerr << '[' << elapsed_text(start) << "] "
                  << line << '\n';
    };
}

struct Arguments {
    std::string model;
    std::string audio_tokenizer;
    std::string reference_audio;
    std::string reference_text;
    std::string text;
    std::string language = "None";
    std::string output = "clone.wav";
    std::string prompt_input;
    std::string prompt_output;
    std::string device = "CPU";
    float speed = 1.0f;
    std::optional<float> duration;
    std::optional<int32_t> num_steps;
    std::optional<bool> denoise;
    uint64_t seed = 0;
    bool preprocess_reference = true;
    bool postprocess_output = true;
};

bool parse_float(
    const std::string& value,
    float& parsed) {
    errno = 0;
    char* end = nullptr;
    parsed = std::strtof(value.c_str(), &end);
    return end != value.c_str() &&
        end != nullptr &&
        *end == '\0' &&
        errno != ERANGE &&
        std::isfinite(parsed);
}

bool parse_i32(
    const std::string& value,
    int32_t& parsed) {
    errno = 0;
    char* end = nullptr;
    const long result = std::strtol(value.c_str(), &end, 10);
    if (end == value.c_str() ||
        end == nullptr ||
        *end != '\0' ||
        errno == ERANGE ||
        result < INT32_MIN ||
        result > INT32_MAX) {
        return false;
    }
    parsed = static_cast<int32_t>(result);
    return true;
}

bool parse_u64(
    const std::string& value,
    uint64_t& parsed) {
    if (value.empty()) {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long result =
        std::strtoull(value.c_str(), &end, 10);
    if (end == value.c_str() ||
        end == nullptr ||
        *end != '\0' ||
        errno == ERANGE ||
        value.front() == '-') {
        return false;
    }
    parsed = static_cast<uint64_t>(result);
    return true;
}

bool parse_bool(
    const std::string& value,
    bool& parsed) {
    std::string lower = value;
    std::transform(
        lower.begin(),
        lower.end(),
        lower.begin(),
        [](unsigned char character) {
            return static_cast<char>(std::tolower(character));
        });
    if (lower == "1" || lower == "true" ||
        lower == "on" || lower == "yes") {
        parsed = true;
        return true;
    }
    if (lower == "0" || lower == "false" ||
        lower == "off" || lower == "no") {
        parsed = false;
        return true;
    }
    return false;
}

bool has_txt_extension(const std::filesystem::path& path) {
    std::string extension = path.extension().u8string();
    for (char& value : extension) {
        if (value >= 'A' && value <= 'Z') {
            value = static_cast<char>(value - 'A' + 'a');
        }
    }
    return extension == ".txt";
}

bool parse_reference_text(
    const std::string& value,
    std::string& reference_text) {
    const std::filesystem::path path =
        std::filesystem::u8path(value);
    if (!has_txt_extension(path)) {
        reference_text = value;
        return true;
    }

    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error)) {
        std::cerr << "reference text file does not exist: "
                  << value << '\n';
        return false;
    }
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "failed to open reference text file: "
                  << value << '\n';
        return false;
    }
    reference_text.assign(
        std::istreambuf_iterator<char>(file),
        std::istreambuf_iterator<char>());
    if (reference_text.size() >= 3 &&
        static_cast<unsigned char>(reference_text[0]) == 0xef &&
        static_cast<unsigned char>(reference_text[1]) == 0xbb &&
        static_cast<unsigned char>(reference_text[2]) == 0xbf) {
        reference_text.erase(0, 3);
    }
    while (!reference_text.empty() &&
           (reference_text.back() == '\r' ||
            reference_text.back() == '\n')) {
        reference_text.pop_back();
    }
    if (reference_text.empty()) {
        std::cerr << "reference text file is empty: "
                  << value << '\n';
        return false;
    }
    return true;
}

bool parse_arguments(
    int argc,
    char** argv,
    Arguments& args) {
    for (int index = 2; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--no-preprocess") {
            args.preprocess_reference = false;
            continue;
        }
        if (option == "--no-postprocess") {
            args.postprocess_output = false;
            continue;
        }
        if (option == "--denoise") {
            args.denoise = true;
            if (index + 1 < argc) {
                bool parsed = false;
                if (parse_bool(argv[index + 1], parsed)) {
                    args.denoise = parsed;
                    ++index;
                }
            }
            continue;
        }
        if (option == "--no-denoise") {
            args.denoise = false;
            continue;
        }
        if (index + 1 >= argc) {
            std::cerr << option << " requires a value\n";
            return false;
        }
        const std::string value = argv[++index];
        if (option == "--model") {
            args.model = value;
        } else if (option == "--audio-tokenizer") {
            args.audio_tokenizer = value;
        } else if (option == "--ref-audio") {
            args.reference_audio = value;
        } else if (option == "--ref-text") {
            if (!parse_reference_text(value, args.reference_text)) {
                return false;
            }
        } else if (option == "--text") {
            args.text = value;
        } else if (option == "--language") {
            args.language = value;
        } else if (option == "--output") {
            args.output = value;
        } else if (option == "--prompt-input") {
            args.prompt_input = value;
        } else if (option == "--prompt-output") {
            args.prompt_output = value;
        } else if (option == "--device") {
            args.device = value;
        } else if (option == "--speed") {
            float parsed = 0.0f;
            if (!parse_float(value, parsed) || parsed <= 0.0f) {
                std::cerr << "invalid speed: " << value << '\n';
                return false;
            }
            args.speed = parsed;
        } else if (option == "--duration") {
            float parsed = 0.0f;
            if (!parse_float(value, parsed) || parsed <= 0.0f) {
                std::cerr << "invalid duration: " << value << '\n';
                return false;
            }
            args.duration = parsed;
        } else if (option == "--num-step") {
            int32_t parsed = 0;
            if (!parse_i32(value, parsed) || parsed <= 0) {
                std::cerr << "invalid num-step: " << value << '\n';
                return false;
            }
            args.num_steps = parsed;
        } else if (option == "--seed") {
            if (!parse_u64(value, args.seed)) {
                std::cerr << "invalid seed: " << value << '\n';
                return false;
            }
        } else {
            std::cerr << "unknown option: " << option << '\n';
            return false;
        }
    }
    return true;
}

void print_usage() {
    std::cout
        << "OmniVoice C++/GGML Voice Clone\n\n"
        << "Usage:\n"
        << "  omnivoice-cli --version\n"
        << "  omnivoice-cli --list-devices\n"
        << "  omnivoice-cli --audio-info FILE.wav\n"
        << "  omnivoice-cli --create-prompt --model LM.gguf "
           "--audio-tokenizer AUDIO.gguf --ref-audio REF.wav|REF.mp3 "
           "--ref-text TEXT_OR_FILE.txt --prompt-output FILE.ovprompt "
           "[--device CPU|CUDA0] [--no-preprocess] "
           "[--denoise [true|false]] [--no-denoise]\n"
        << "  omnivoice-cli --clone --model LM.gguf "
           "--audio-tokenizer AUDIO.gguf "
           "(--prompt-input FILE.ovprompt | "
           "--ref-audio REF.wav|REF.mp3 --ref-text TEXT_OR_FILE.txt) "
           "--text TEXT [--output OUT.wav] [--language LANG] "
           "[--device CPU|CUDA0] [--speed F] [--duration SECONDS] "
           "[--num-step N] [--seed N] "
           "[--denoise [true|false]] [--no-denoise] "
           "[--prompt-output FILE.ovprompt]\n"
        << "  omnivoice-cli --help\n\n"
        << "Only Voice Clone is implemented. Reference audio must be "
           "24 kHz mono WAV or MP3.\n"
        << "The runtime consumes GGUF models only and does not require "
           "Python.\n";
}

const char* device_type_name(enum ggml_backend_dev_type type) {
    switch (type) {
    case GGML_BACKEND_DEVICE_TYPE_CPU:
        return "CPU";
    case GGML_BACKEND_DEVICE_TYPE_GPU:
        return "GPU";
    case GGML_BACKEND_DEVICE_TYPE_IGPU:
        return "IGPU";
    case GGML_BACKEND_DEVICE_TYPE_ACCEL:
        return "ACCEL";
    case GGML_BACKEND_DEVICE_TYPE_META:
        return "META";
    default:
        return "UNKNOWN";
    }
}

void list_devices() {
    ggml_backend_load_all();

    const size_t count = ggml_backend_dev_count();
    std::cout << "Devices: " << count << '\n';
    for (size_t index = 0; index < count; ++index) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        ggml_backend_dev_props props{};
        ggml_backend_dev_get_props(device, &props);

        constexpr double MIB = 1024.0 * 1024.0;
        std::cout << index << ": " << props.name
                  << " | " << device_type_name(props.type)
                  << " | " << props.description;
        if (props.memory_total > 0) {
            std::cout << " | " << std::fixed << std::setprecision(1)
                      << static_cast<double>(props.memory_free) / MIB
                      << " / "
                      << static_cast<double>(props.memory_total) / MIB
                      << " MiB";
        }
        if (props.device_id != nullptr && props.device_id[0] != '\0') {
            std::cout << " | " << props.device_id;
        }
        std::cout << '\n';
    }
}

int show_audio_info(const std::string& path) {
    omnivoice::AudioBuffer audio;
    omnivoice::WavInfo info;
    const omnivoice::Status status = omnivoice::load_wav_24k_mono(
        path,
        audio,
        &info);
    if (!status) {
        std::cerr << "error: " << status.message() << '\n';
        return 1;
    }

    std::cout << "format_tag: " << info.format_tag << '\n'
              << "bits_per_sample: " << info.bits_per_sample << '\n'
              << "sample_rate: " << info.sample_rate << '\n'
              << "channels: " << info.channels << '\n'
              << "frames: " << info.frames << '\n'
              << "duration_seconds: " << std::fixed
              << std::setprecision(6)
              << audio.duration_seconds() << '\n'
              << "rms: " << std::setprecision(8)
              << omnivoice::compute_rms(audio) << '\n';
    return 0;
}

omnivoice::Status load_model(
    const Arguments& args,
    const omnivoice::OmniVoiceModelParams& params,
    omnivoice::OmniVoiceModel& model) {
    return model.load(
        args.model,
        args.audio_tokenizer,
        params);
}

int create_prompt_command(const Arguments& args) {
    if (args.model.empty() ||
        args.audio_tokenizer.empty() ||
        args.reference_audio.empty() ||
        args.reference_text.empty() ||
        args.prompt_output.empty()) {
        std::cerr
            << "--create-prompt requires --model, --audio-tokenizer, "
               "--ref-audio, --ref-text and --prompt-output\n";
        return 2;
    }

    const auto command_start = Clock::now();
    const omnivoice::LogCallback log = make_logger(command_start);
    omnivoice::OmniVoiceModelParams params;
    params.device = args.device;
    params.log = log;
    omnivoice::OmniVoiceModel model;
    log("loading models");
    const auto load_start = Clock::now();
    omnivoice::Status status = load_model(
        args,
        params,
        model);
    if (!status) {
        std::cerr << "error: " << status.message() << '\n';
        return 1;
    }
    log("models loaded in " + elapsed_text(load_start));

    log("reading reference WAV: " + args.reference_audio);
    const auto audio_start = Clock::now();
    omnivoice::AudioBuffer audio;
    status = omnivoice::load_audio_24k_mono(
        args.reference_audio,
        audio);
    if (!status) {
        std::cerr << "error: " << status.message() << '\n';
        return 1;
    }
    log(
        "reference WAV ready: " +
        std::to_string(audio.samples.size()) + " samples, " +
        std::to_string(audio.duration_seconds()) + " s in " +
        elapsed_text(audio_start));

    omnivoice::VoiceClonePromptRequest prompt_request;
    prompt_request.reference_audio = std::move(audio);
    prompt_request.reference_text = args.reference_text;
    prompt_request.language = args.language;
    prompt_request.denoise = args.denoise.value_or(
        model.generation_defaults().denoise);
    prompt_request.preprocess_reference = args.preprocess_reference;
    prompt_request.log = log;
    omnivoice::VoiceClonePrompt prompt;
    status = model.create_prompt(prompt_request, prompt);
    if (!status) {
        std::cerr << "error: " << status.message() << '\n';
        return 1;
    }

    log("writing voice prompt: " + args.prompt_output);
    const auto save_start = Clock::now();
    status = omnivoice::save_voice_clone_prompt(
        args.prompt_output,
        prompt);
    if (!status) {
        std::cerr << "error: " << status.message() << '\n';
        return 1;
    }
    log("voice prompt written in " + elapsed_text(save_start));
    std::cout << "wrote voice prompt: " << args.prompt_output << '\n'
              << "codebooks: " << prompt.reference_codes.codebooks << '\n'
              << "frames: " << prompt.reference_codes.frames << '\n'
              << "elapsed: " << elapsed_text(command_start) << '\n';
    return 0;
}

int clone_command(const Arguments& args) {
    if (args.model.empty() ||
        args.audio_tokenizer.empty() ||
        args.text.empty()) {
        std::cerr
            << "--clone requires --model, --audio-tokenizer and --text\n";
        return 2;
    }
    if (args.prompt_input.empty() &&
        (args.reference_audio.empty() ||
         args.reference_text.empty())) {
        std::cerr
            << "--clone requires --ref-audio and --ref-text when "
               "--prompt-input is not provided\n";
        return 2;
    }

    const auto command_start = Clock::now();
    const omnivoice::LogCallback log = make_logger(command_start);
    omnivoice::OmniVoiceModelParams params;
    params.device = args.device;
    params.log = log;
    omnivoice::OmniVoiceModel model;
    log("loading models");
    const auto load_start = Clock::now();
    omnivoice::Status status = load_model(
        args,
        params,
        model);
    if (!status) {
        std::cerr << "error: " << status.message() << '\n';
        return 1;
    }
    log("models loaded in " + elapsed_text(load_start));

    omnivoice::VoiceClonePrompt prompt;
    if (!args.prompt_input.empty()) {
        log("loading reusable voice prompt: " + args.prompt_input);
        const auto prompt_start = Clock::now();
        status = omnivoice::load_voice_clone_prompt(
            args.prompt_input,
            prompt);
        if (!status) {
            std::cerr << "error: " << status.message() << '\n';
            return 1;
        }
        log(
            "voice prompt ready: " +
            std::to_string(prompt.reference_codes.codebooks) + " x " +
            std::to_string(prompt.reference_codes.frames) + " codes in " +
            elapsed_text(prompt_start));
    } else {
        log("reading reference WAV: " + args.reference_audio);
        const auto audio_start = Clock::now();
        omnivoice::AudioBuffer reference_audio;
        status = omnivoice::load_audio_24k_mono(
            args.reference_audio,
            reference_audio);
        if (!status) {
            std::cerr << "error: " << status.message() << '\n';
            return 1;
        }
        log(
            "reference WAV ready: " +
            std::to_string(reference_audio.samples.size()) + " samples, " +
            std::to_string(reference_audio.duration_seconds()) + " s in " +
            elapsed_text(audio_start));

        omnivoice::VoiceClonePromptRequest prompt_request;
        prompt_request.reference_audio = std::move(reference_audio);
        prompt_request.reference_text = args.reference_text;
        prompt_request.language = args.language;
        prompt_request.denoise = args.denoise.value_or(
            model.generation_defaults().denoise);
        prompt_request.preprocess_reference = args.preprocess_reference;
        prompt_request.log = log;
        status = model.create_prompt(prompt_request, prompt);
        if (!status) {
            std::cerr << "error: " << status.message() << '\n';
            return 1;
        }
    }

    omnivoice::VoiceCloneRequest request;
    request.target_text = args.text;
    request.language = args.language;
    request.speed = args.speed;
    request.seed = args.seed;
    request.duration_seconds = args.duration.value_or(0.0f);
    request.log = log;

    omnivoice::VoiceCloneConfig options =
        model.generation_defaults();
    if (args.num_steps) {
        options.num_step = *args.num_steps;
    }
    options.postprocess_output = args.postprocess_output;
    request.denoise = args.denoise.value_or(options.denoise);
    options.denoise = request.denoise;
    request.generation_config = options;

    omnivoice::VoiceCloneOutput output;
    log("generating target speech codes and audio");
    const auto generation_start = Clock::now();
    status = model.generate_voice_clone(prompt, request, output);
    if (!status) {
        std::cerr << "error: " << status.message() << '\n';
        return 1;
    }
    const double generation_seconds = std::chrono::duration<double>(
        Clock::now() - generation_start).count();
    const double audio_seconds = output.audio.duration_seconds();
    const double rtf = audio_seconds > 0.0
        ? generation_seconds / audio_seconds
        : 0.0;
    log(
        "generation complete in " + elapsed_text(generation_start) +
        ", audio=" + std::to_string(audio_seconds) +
        " s, RTF=" + std::to_string(rtf));

    omnivoice::AudioBuffer wav_audio = output.audio;
    for (float& sample : wav_audio.samples) {
        sample = std::clamp(sample, -1.0f, 1.0f);
    }
    log("writing cloned WAV: " + args.output);
    const auto write_start = Clock::now();
    status = omnivoice::save_wav_24k_mono(args.output, wav_audio);
    if (!status) {
        std::cerr << "error: " << status.message() << '\n';
        return 1;
    }
    log("cloned WAV written in " + elapsed_text(write_start));
    if (!args.prompt_output.empty()) {
        log("writing reusable voice prompt: " + args.prompt_output);
        const auto prompt_start = Clock::now();
        status = omnivoice::save_voice_clone_prompt(
            args.prompt_output,
            output.prompt);
        if (!status) {
            std::cerr << "error: " << status.message() << '\n';
            return 1;
        }
        log("voice prompt written in " + elapsed_text(prompt_start));
    }

    std::cout << "wrote cloned audio: " << args.output << '\n';
    return 0;
}

} // namespace

int run_cli(int argc, char** argv) {
    if (argc <= 1) {
        print_usage();
        return 0;
    }

    const std::string option = argv[1];
    if (option == "--help" || option == "-h") {
        print_usage();
        return 0;
    }
    if (option == "--version") {
        std::cout << "omnivoice-cpp " << OMNIVOICE_VERSION << '\n';
        return 0;
    }
    if (option == "--list-devices") {
        list_devices();
        return 0;
    }
    if (option == "--audio-info") {
        if (argc != 3) {
            std::cerr << "--audio-info requires a WAV path\n";
            return 2;
        }
        return show_audio_info(argv[2]);
    }
    if (option != "--create-prompt" && option != "--clone") {
        std::cerr << "unknown option: " << option << "\n\n";
        print_usage();
        return 2;
    }

    Arguments args;
    if (!parse_arguments(argc, argv, args)) {
        return 2;
    }
    return option == "--create-prompt"
        ? create_prompt_command(args)
        : clone_command(args);
}

#if defined(_WIN32)

namespace {

bool wide_to_utf8(
    const wchar_t* input,
    std::string& output) {
    if (input == nullptr || input[0] == L'\0') {
        output.clear();
        return true;
    }
    const int size = WideCharToMultiByte(
        CP_UTF8,
        0,
        input,
        -1,
        nullptr,
        0,
        nullptr,
        nullptr);
    if (size <= 0) {
        return false;
    }
    std::vector<char> buffer(static_cast<size_t>(size));
    if (WideCharToMultiByte(
            CP_UTF8,
            0,
            input,
            -1,
            buffer.data(),
            size,
            nullptr,
            nullptr) <= 0) {
        return false;
    }
    output.assign(buffer.data());
    return true;
}

} // namespace

int wmain(int argc, wchar_t** wide_argv) {
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    std::vector<std::string> storage;
    std::vector<char*> argv;
    storage.reserve(static_cast<size_t>(argc));
    argv.reserve(static_cast<size_t>(argc));
    for (int index = 0; index < argc; ++index) {
        std::string value;
        if (!wide_to_utf8(wide_argv[index], value)) {
            std::cerr
                << "failed to convert command-line argument to UTF-8\n";
            return 2;
        }
        storage.push_back(std::move(value));
    }
    for (std::string& value : storage) {
        argv.push_back(value.data());
    }
    return run_cli(argc, argv.data());
}

#else

int main(int argc, char** argv) {
    return run_cli(argc, argv);
}

#endif
