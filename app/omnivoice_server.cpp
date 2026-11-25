#include "audio_toolkit/audio_writer.h"
#include "omnivoice.h"
#include "voice_clone_prompt.h"

#include "httplib.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <exception>
#include <filesystem>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef OMNIVOICE_VERSION
#define OMNIVOICE_VERSION "unknown"
#endif

namespace {

using Clock = std::chrono::steady_clock;

constexpr size_t kMaxInputCharacters = 4096;
constexpr size_t kMaxBodyBytes = 1u << 20;

std::string format_seconds(double seconds) {
    char text[32];
    std::snprintf(text, sizeof(text), "%.2f", seconds);
    return text;
}

std::string trim(const std::string& text) {
    const size_t begin = text.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return {};
    }
    const size_t end = text.find_last_not_of(" \t\r\n");
    return text.substr(begin, end - begin + 1);
}

size_t utf8_length(const std::string& text) {
    size_t count = 0;
    for (char byte : text) {
        if ((static_cast<uint8_t>(byte) & 0xC0) != 0x80) {
            ++count;
        }
    }
    return count;
}

class Logger {
public:
    explicit Logger(bool verbose)
        : verbose_(verbose) {
    }

    void info(const std::string& message) {
        write(message);
    }

    void debug(const std::string& message) {
        if (verbose_) {
            write(message);
        }
    }

private:
    void write(const std::string& message) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::cerr << message << '\n';
    }

    bool verbose_;
    std::mutex mutex_;
};

struct ServerOptions {
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string model;
    std::string audio_tokenizer;
    std::string device = "CPU";
    std::string voice_dir = "voices";
    std::string default_voice;
    std::string language = "None";
    std::string api_key;
    std::string model_name = "omnivoice";
    size_t max_queue = 2;
    size_t threads = 4;
    bool verbose = false;
};

void print_usage() {
    std::cout
        << "OmniVoice OpenAI-compatible speech server\n\n"
        << "  omnivoice-server --model LM.gguf "
           "--audio-tokenizer TOKENIZER.gguf [options]\n"
        << "  omnivoice-server --version\n"
        << "  omnivoice-server --help\n\n"
        << "Endpoints:\n"
        << "  POST /v1/audio/speech    text to speech, wav or pcm output\n"
        << "  GET  /v1/models          lists the configured model name\n"
        << "  GET  /health             liveness probe\n\n"
        << "Options:\n"
        << "  --host HOST              bind address (default 127.0.0.1)\n"
        << "  --port PORT              bind port (default 8080)\n"
        << "  --device DEVICE          GGML device, e.g. CPU or CUDA0\n"
        << "  --voice-dir DIR          scan DIR for *.ovprompt (default voices)\n"
        << "  --default-voice NAME     voice used when a request omits it\n"
        << "  --language LANG          default language tag (default None)\n"
        << "  --model-name NAME        model id reported by /v1/models\n"
        << "  --api-key KEY            require Authorization: Bearer KEY\n"
        << "  --max-queue N            queued requests before 429 (default 2)\n"
        << "  --threads N              HTTP worker threads (default 4)\n"
        << "  --verbose                log generation progress\n";
}

enum class ParseResult {
    Ok,
    Help,
    Version,
    Error,
};

ParseResult parse_arguments(
    int argc,
    char** argv,
    ServerOptions& options,
    std::string& error) {
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        const auto take_value = [&](std::string& target) {
            if (index + 1 >= argc) {
                error = option + " requires a value";
                return false;
            }
            target = argv[++index];
            return true;
        };
        const auto take_size = [&](size_t& target, size_t minimum, size_t maximum) {
            std::string value;
            if (!take_value(value)) {
                return false;
            }
            try {
                const long long parsed = std::stoll(value);
                if (parsed < static_cast<long long>(minimum) ||
                    parsed > static_cast<long long>(maximum)) {
                    error = option + " must be between " +
                        std::to_string(minimum) + " and " +
                        std::to_string(maximum);
                    return false;
                }
                target = static_cast<size_t>(parsed);
                return true;
            } catch (const std::exception&) {
                error = option + " requires an integer value";
                return false;
            }
        };

        if (option == "--help" || option == "-h") {
            return ParseResult::Help;
        }
        if (option == "--version") {
            return ParseResult::Version;
        }
        if (option == "--verbose") {
            options.verbose = true;
            continue;
        }
        if (option == "--host") {
            if (!take_value(options.host)) {
                return ParseResult::Error;
            }
            continue;
        }
        if (option == "--port") {
            size_t port = static_cast<size_t>(options.port);
            if (!take_size(port, 1, 65535)) {
                return ParseResult::Error;
            }
            options.port = static_cast<int>(port);
            continue;
        }
        if (option == "--model") {
            if (!take_value(options.model)) {
                return ParseResult::Error;
            }
            continue;
        }
        if (option == "--audio-tokenizer") {
            if (!take_value(options.audio_tokenizer)) {
                return ParseResult::Error;
            }
            continue;
        }
        if (option == "--device") {
            if (!take_value(options.device)) {
                return ParseResult::Error;
            }
            continue;
        }
        if (option == "--voice-dir") {
            if (!take_value(options.voice_dir)) {
                return ParseResult::Error;
            }
            continue;
        }
        if (option == "--default-voice") {
            if (!take_value(options.default_voice)) {
                return ParseResult::Error;
            }
            continue;
        }
        if (option == "--language") {
            if (!take_value(options.language)) {
                return ParseResult::Error;
            }
            continue;
        }
        if (option == "--model-name") {
            if (!take_value(options.model_name)) {
                return ParseResult::Error;
            }
            continue;
        }
        if (option == "--api-key") {
            if (!take_value(options.api_key)) {
                return ParseResult::Error;
            }
            continue;
        }
        if (option == "--max-queue") {
            if (!take_size(options.max_queue, 1, 64)) {
                return ParseResult::Error;
            }
            continue;
        }
        if (option == "--threads") {
            if (!take_size(options.threads, 1, 64)) {
                return ParseResult::Error;
            }
            continue;
        }
        error = "unknown option: " + option;
        return ParseResult::Error;
    }
    return ParseResult::Ok;
}

using PromptPtr = std::shared_ptr<const omnivoice::VoiceClonePrompt>;

struct VoiceEntry {
    std::string name;
    std::string path;
    PromptPtr prompt;
};

class VoiceRegistry {
public:
    omnivoice::Status add_file(
        const std::string& name,
        const std::string& path,
        Logger& log) {
        if (name.empty()) {
            return omnivoice::Status::error(
                "voice name must not be empty: " + path);
        }
        auto prompt = std::make_shared<omnivoice::VoiceClonePrompt>();
        omnivoice::Status status =
            omnivoice::load_voice_clone_prompt(path, *prompt);
        if (!status) {
            return status;
        }
        const bool replaced = entries_.count(name) != 0;
        VoiceEntry entry;
        entry.name = name;
        entry.path = path;
        entry.prompt = std::move(prompt);
        entries_[name] = std::move(entry);
        log.info(
            std::string(replaced ? "replaced voice " : "loaded voice ") +
            name + " (" +
            std::to_string(entries_[name].prompt->reference_codes.frames) +
            " frames, " +
            format_seconds(entries_[name].prompt->reference_codes.frames *
                static_cast<double>(omnivoice::OMNIVOICE_HOP_LENGTH) /
                omnivoice::OMNIVOICE_SAMPLE_RATE) +
            " s)");
        return omnivoice::Status::ok();
    }

    omnivoice::Status add_directory(
        const std::string& directory,
        Logger& log) {
        if (directory.empty()) {
            return omnivoice::Status::ok();
        }
        std::error_code error;
        const std::filesystem::path path =
            std::filesystem::u8path(directory);
        if (!std::filesystem::exists(path, error)) {
            log.info("voice directory not found, skipping: " + directory);
            return omnivoice::Status::ok();
        }
        if (!std::filesystem::is_directory(path, error)) {
            return omnivoice::Status::error(
                "voice directory is not a directory: " + directory);
        }
        std::vector<std::filesystem::path> files;
        for (const std::filesystem::directory_entry& entry :
             std::filesystem::directory_iterator(path, error)) {
            if (!entry.is_regular_file(error)) {
                continue;
            }
            if (entry.path().extension() != ".ovprompt") {
                continue;
            }
            files.push_back(entry.path());
        }
        if (error) {
            return omnivoice::Status::error(
                "failed to scan voice directory: " + directory);
        }
        std::sort(files.begin(), files.end());
        for (const std::filesystem::path& file : files) {
            const std::string name = file.stem().u8string();
            omnivoice::Status status =
                add_file(name, file.u8string(), log);
            if (!status) {
                return status;
            }
        }
        return omnivoice::Status::ok();
    }

    PromptPtr find(const std::string& name) const {
        const auto entry = entries_.find(name);
        if (entry == entries_.end()) {
            return nullptr;
        }
        return entry->second.prompt;
    }

    bool empty() const noexcept {
        return entries_.empty();
    }

    std::string single_name() const {
        if (entries_.size() != 1) {
            return {};
        }
        return entries_.begin()->first;
    }

    std::string names() const {
        std::vector<std::string> sorted;
        sorted.reserve(entries_.size());
        for (const auto& entry : entries_) {
            sorted.push_back(entry.first);
        }
        std::sort(sorted.begin(), sorted.end());
        std::string joined;
        for (size_t index = 0; index < sorted.size(); ++index) {
            if (index != 0) {
                joined += ", ";
            }
            joined += sorted[index];
        }
        return joined;
    }

private:
    std::unordered_map<std::string, VoiceEntry> entries_;
};

struct JobResult {
    bool ok = false;
    std::string error;
    omnivoice::AudioBuffer audio;
    double generation_seconds = 0.0;
};

struct GenerationJob {
    PromptPtr prompt;
    omnivoice::VoiceCloneRequest request;
    std::promise<JobResult> promise;
};

// Serializes inference on the single OmniVoiceModel instance. The ggml graphs
// cached inside the model are mutable, so the model must never be entered from
// two threads at once; HTTP threads only enqueue work and wait for the result.
class GenerationWorker {
public:
    GenerationWorker(
        const omnivoice::OmniVoiceModel& model,
        size_t max_queue,
        Logger& log)
        : model_(model),
          max_queue_(max_queue),
          log_(log),
          thread_(&GenerationWorker::run, this) {
    }

    ~GenerationWorker() {
        stop();
    }

    GenerationWorker(const GenerationWorker&) = delete;
    GenerationWorker& operator=(const GenerationWorker&) = delete;

    // Returns an invalid future when the queue is full.
    std::future<JobResult> submit(GenerationJob job) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_ || queue_.size() >= max_queue_) {
            return {};
        }
        std::future<JobResult> future = job.promise.get_future();
        queue_.push_back(std::move(job));
        condition_.notify_one();
        return future;
    }

    size_t max_queue() const noexcept {
        return max_queue_;
    }

private:
    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) {
                return;
            }
            stopping_ = true;
        }
        condition_.notify_all();
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    void run() {
        for (;;) {
            GenerationJob job;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                condition_.wait(lock, [this] {
                    return stopping_ || !queue_.empty();
                });
                if (queue_.empty()) {
                    if (stopping_) {
                        return;
                    }
                    continue;
                }
                job = std::move(queue_.front());
                queue_.pop_front();
            }

            JobResult result;
            try {
                omnivoice::VoiceCloneOutput output;
                const Clock::time_point start = Clock::now();
                const omnivoice::Status status =
                    model_.generate_voice_clone(
                        *job.prompt,
                        job.request,
                        output);
                result.generation_seconds =
                    std::chrono::duration<double>(Clock::now() - start).count();
                if (status) {
                    result.ok = true;
                    result.audio = std::move(output.audio);
                } else {
                    result.error = status.message();
                }
            } catch (const std::exception& error) {
                result.error = error.what();
            } catch (...) {
                result.error = "unknown generation error";
            }
            log_.debug(
                "generation finished in " +
                format_seconds(result.generation_seconds) + " s");
            try {
                job.promise.set_value(std::move(result));
            } catch (...) {
            }
        }
    }

    const omnivoice::OmniVoiceModel& model_;
    size_t max_queue_;
    Logger& log_;
    std::mutex mutex_;
    std::condition_variable condition_;
    std::deque<GenerationJob> queue_;
    bool stopping_ = false;
    std::thread thread_;
};

void send_error(
    httplib::Response& response,
    int status,
    const std::string& message,
    const std::string& type = "invalid_request_error",
    const std::string& param = {}) {
    const nlohmann::json body = {
        {"error",
         {
             {"message", message},
             {"type", type},
             {"param", param.empty() ? nlohmann::json() : nlohmann::json(param)},
             {"code", nullptr},
         }},
    };
    response.status = status;
    response.set_content(body.dump(), "application/json");
}

struct ServerContext {
    const ServerOptions& options;
    VoiceRegistry& voices;
    GenerationWorker& worker;
    Logger& log;
    omnivoice::VoiceCloneConfig defaults;
    std::string default_voice;
    int64_t created_at = 0;
};

void handle_speech(
    const httplib::Request& request,
    httplib::Response& response,
    ServerContext& context) {
    nlohmann::json body;
    try {
        body = nlohmann::json::parse(request.body);
    } catch (const std::exception&) {
        send_error(response, 400, "request body must be valid JSON");
        return;
    }
    if (!body.is_object()) {
        send_error(response, 400, "request body must be a JSON object");
        return;
    }

    if (!body.contains("input") || !body["input"].is_string()) {
        send_error(response, 400, "input is required and must be a string", "invalid_request_error", "input");
        return;
    }
    const std::string input = body["input"].get<std::string>();
    if (trim(input).empty()) {
        send_error(response, 400, "input must not be empty", "invalid_request_error", "input");
        return;
    }
    if (utf8_length(input) > kMaxInputCharacters) {
        send_error(
            response,
            400,
            "input must be at most " +
                std::to_string(kMaxInputCharacters) + " characters",
            "invalid_request_error",
            "input");
        return;
    }

    std::string voice_name;
    if (body.contains("voice") && !body["voice"].is_null()) {
        if (!body["voice"].is_string()) {
            send_error(response, 400, "voice must be a string", "invalid_request_error", "voice");
            return;
        }
        voice_name = body["voice"].get<std::string>();
    }
    if (voice_name.empty()) {
        voice_name = context.default_voice;
    }
    if (voice_name.empty()) {
        send_error(
            response,
            400,
            "voice is required; available voices: " + context.voices.names(),
            "invalid_request_error",
            "voice");
        return;
    }
    const PromptPtr prompt = context.voices.find(voice_name);
    if (!prompt) {
        send_error(
            response,
            400,
            "unknown voice '" + voice_name + "'; available voices: " +
                context.voices.names(),
            "invalid_request_error",
            "voice");
        return;
    }

    std::string response_format = "wav";
    if (body.contains("response_format")) {
        if (!body["response_format"].is_string()) {
            send_error(response, 400, "response_format must be a string", "invalid_request_error", "response_format");
            return;
        }
        response_format = body["response_format"].get<std::string>();
    }
    if (response_format != "wav" && response_format != "pcm") {
        send_error(
            response,
            400,
            "unsupported response_format '" + response_format +
                "'; supported formats: wav, pcm",
            "invalid_request_error",
            "response_format");
        return;
    }

    float speed = 1.0f;
    if (body.contains("speed") && !body["speed"].is_null()) {
        if (!body["speed"].is_number()) {
            send_error(response, 400, "speed must be a number", "invalid_request_error", "speed");
            return;
        }
        const double requested_speed = body["speed"].get<double>();
        if (!std::isfinite(requested_speed) ||
            requested_speed < 0.25 ||
            requested_speed > 4.0) {
            send_error(response, 400, "speed must be between 0.25 and 4.0", "invalid_request_error", "speed");
            return;
        }
        speed = static_cast<float>(requested_speed);
    }

    int32_t num_step = context.defaults.num_step;
    if (body.contains("num_step") && !body["num_step"].is_null()) {
        const nlohmann::json& value = body["num_step"];
        int64_t requested_step = 0;
        if (value.is_number_unsigned()) {
            const uint64_t raw_step = value.get<uint64_t>();
            requested_step =
                raw_step <= 256 ? static_cast<int64_t>(raw_step) : 0;
        } else if (value.is_number_integer()) {
            requested_step = value.get<int64_t>();
        }
        if (requested_step < 1 || requested_step > 256) {
            send_error(
                response,
                400,
                "num_step must be an integer between 1 and 256",
                "invalid_request_error",
                "num_step");
            return;
        }
        num_step = static_cast<int32_t>(requested_step);
    }

    std::string language = context.options.language;
    if (body.contains("language") && body["language"].is_string()) {
        language = body["language"].get<std::string>();
    }

    bool denoise = context.defaults.denoise;
    if (body.contains("denoise") && body["denoise"].is_boolean()) {
        denoise = body["denoise"].get<bool>();
    }

    uint64_t seed = 0;
    if (body.contains("seed") && body["seed"].is_number_unsigned()) {
        seed = body["seed"].get<uint64_t>();
    }

    omnivoice::VoiceCloneRequest clone_request;
    clone_request.target_text = input;
    clone_request.language = language;
    clone_request.speed = speed;
    clone_request.seed = seed;
    clone_request.denoise = denoise;
    omnivoice::VoiceCloneConfig config = context.defaults;
    config.num_step = num_step;
    config.denoise = denoise;
    clone_request.generation_config = config;
    Logger* logger = &context.log;
    clone_request.log = [logger](const std::string& message) {
        logger->debug(message);
    };

    GenerationJob job;
    job.prompt = prompt;
    job.request = std::move(clone_request);
    std::future<JobResult> future = context.worker.submit(std::move(job));
    if (!future.valid()) {
        response.set_header("Retry-After", "1");
        send_error(
            response,
            429,
            "server is busy; " + std::to_string(context.worker.max_queue()) +
                " requests are already queued",
            "rate_limit_error");
        return;
    }

    const JobResult result = future.get();
    if (!result.ok) {
        send_error(response, 500, result.error, "server_error");
        return;
    }

    omnivoice::AudioBuffer audio = result.audio;
    for (float& sample : audio.samples) {
        sample = std::clamp(sample, -1.0f, 1.0f);
    }
    std::vector<uint8_t> bytes;
    const omnivoice::Status status = response_format == "pcm"
        ? omnivoice::encode_pcm_s16le(audio, bytes)
        : omnivoice::encode_wav_24k_mono(audio, bytes);
    if (!status) {
        send_error(response, 500, status.message(), "server_error");
        return;
    }

    const double audio_seconds = audio.duration_seconds();
    const double rtf = audio_seconds > 0.0
        ? result.generation_seconds / audio_seconds
        : 0.0;
    context.log.info(
        "speech: voice=" + voice_name + ", format=" + response_format +
        ", characters=" + std::to_string(utf8_length(input)) +
        ", steps=" + std::to_string(num_step) +
        ", audio=" + format_seconds(audio_seconds) + " s, RTF=" +
        format_seconds(rtf));

    response.set_content(
        std::string(
            reinterpret_cast<const char*>(bytes.data()),
            bytes.size()),
        response_format == "pcm" ? "audio/pcm" : "audio/wav");
    response.set_header("X-Omnivoice-Voice", voice_name);
    if (response_format == "pcm") {
        response.set_header(
            "X-Omnivoice-Sample-Format",
            "s16le,24000,mono");
    }
}

void handle_models(
    const httplib::Request&,
    httplib::Response& response,
    ServerContext& context) {
    const nlohmann::json model_entry = {
        {"id", context.options.model_name},
        {"object", "model"},
        {"created", context.created_at},
        {"owned_by", "omnivoice"},
    };
    const nlohmann::json body = {
        {"object", "list"},
        {"data", nlohmann::json::array({model_entry})},
    };
    response.set_content(body.dump(), "application/json");
}

void handle_health(
    const httplib::Request&,
    httplib::Response& response) {
    response.set_content(
        nlohmann::json{{"status", "ok"}}.dump(),
        "application/json");
}

} // namespace

int main(int argc, char** argv) {
    ServerOptions options;
    std::string parse_error;
    switch (parse_arguments(argc, argv, options, parse_error)) {
    case ParseResult::Help:
        print_usage();
        return 0;
    case ParseResult::Version:
        std::cout << "omnivoice-server " << OMNIVOICE_VERSION << '\n';
        return 0;
    case ParseResult::Error:
        std::cerr << "error: " << parse_error << "\n\n";
        print_usage();
        return 2;
    case ParseResult::Ok:
        break;
    }

    if (options.model.empty() || options.audio_tokenizer.empty()) {
        std::cerr
            << "--model and --audio-tokenizer are required\n\n";
        print_usage();
        return 2;
    }
    options.threads = std::max(options.threads, options.max_queue + 2);

    Logger log(options.verbose);

    VoiceRegistry voices;
    omnivoice::Status status = voices.add_directory(options.voice_dir, log);
    if (!status) {
        std::cerr << "error: " << status.message() << '\n';
        return 1;
    }
    if (voices.empty()) {
        std::cerr
            << "error: no voice prompts found in '" << options.voice_dir
            << "'; add .ovprompt files or point --voice-dir at them\n";
        return 1;
    }

    std::string default_voice = options.default_voice;
    if (default_voice.empty()) {
        default_voice = voices.single_name();
    }
    if (!default_voice.empty() && !voices.find(default_voice)) {
        std::cerr << "error: unknown default voice: " << default_voice << '\n';
        return 1;
    }
    if (!default_voice.empty()) {
        log.info("default voice: " + default_voice);
    }

    omnivoice::OmniVoiceModelParams params;
    params.device = options.device;
    params.log = [&log](const std::string& message) {
        log.info(message);
    };
    omnivoice::OmniVoiceModel model;
    log.info("loading models");
    status = model.load(options.model, options.audio_tokenizer, params);
    if (!status) {
        std::cerr << "error: " << status.message() << '\n';
        return 1;
    }

    const int64_t created_at =
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

    GenerationWorker worker(model, options.max_queue, log);
    ServerContext context{
        options,
        voices,
        worker,
        log,
        model.generation_defaults(),
        default_voice,
        created_at,
    };

    httplib::Server server;
    server.set_payload_max_length(kMaxBodyBytes);
    server.set_read_timeout(60);
    server.set_write_timeout(120);
    server.set_keep_alive_timeout(60);
    server.new_task_queue = [threads = options.threads] {
        return new httplib::ThreadPool(threads);
    };

    if (!options.api_key.empty()) {
        server.set_pre_routing_handler(
            [&context](
                const httplib::Request& request,
                httplib::Response& response) {
                if (request.path == "/health") {
                    return httplib::Server::HandlerResponse::Unhandled;
                }
                const std::string expected =
                    "Bearer " + context.options.api_key;
                if (request.get_header_value("Authorization") == expected) {
                    return httplib::Server::HandlerResponse::Unhandled;
                }
                send_error(response, 401, "invalid API key");
                return httplib::Server::HandlerResponse::Handled;
            });
    }

    server.set_exception_handler(
        [](const httplib::Request&,
           httplib::Response& response,
           std::exception_ptr exception) {
            std::string message = "internal server error";
            try {
                if (exception) {
                    std::rethrow_exception(exception);
                }
            } catch (const std::exception& error) {
                message = error.what();
            } catch (...) {
            }
            send_error(response, 500, message, "server_error");
        });

    server.set_error_handler(
        [](const httplib::Request& request, httplib::Response& response) {
            if (response.status == 404) {
                send_error(
                    response,
                    404,
                    "unknown endpoint: " + request.method + " " +
                        request.path);
            } else if (response.status == 413) {
                send_error(
                    response,
                    413,
                    "request body must be at most " +
                        std::to_string(kMaxBodyBytes) + " bytes");
            }
        });

    server.Post(
        "/v1/audio/speech",
        [&context](
            const httplib::Request& request,
            httplib::Response& response) {
            handle_speech(request, response, context);
        });
    server.Get(
        "/v1/models",
        [&context](
            const httplib::Request& request,
            httplib::Response& response) {
            handle_models(request, response, context);
        });
    server.Get("/health", handle_health);

    log.info(
        "listening on http://" + options.host + ":" +
        std::to_string(options.port) + " with " +
        std::to_string(options.threads) + " HTTP threads, queue=" +
        std::to_string(options.max_queue));
    if (!server.listen(options.host, options.port)) {
        std::cerr << "error: failed to listen on " << options.host << ':'
                  << options.port << '\n';
        return 1;
    }
    return 0;
}
