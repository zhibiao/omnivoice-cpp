#include "higgs_tokenizer/decoder.h"

#include "backend_scheduler.h"
#include "gguf_model.h"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace omnivoice {

namespace {

ggml_tensor* require(
    ggml_tensor* tensor,
    const std::string& name,
    int64_t ne0,
    int64_t ne1 = -1,
    int64_t ne2 = -1) {
    if (tensor == nullptr) {
        throw std::runtime_error("missing Higgs tensor: " + name);
    }
    if (tensor->ne[0] != ne0 ||
        (ne1 >= 0 && tensor->ne[1] != ne1) ||
        (ne2 >= 0 && tensor->ne[2] != ne2)) {
        throw std::runtime_error("unexpected Higgs tensor shape: " + name);
    }
    return tensor;
}

ggml_tensor* cast_f32(ggml_context* context, ggml_tensor* tensor) {
    return tensor->type == GGML_TYPE_F32
        ? tensor
        : ggml_cast(context, tensor, GGML_TYPE_F32);
}

ggml_tensor* repeat_rows(
    ggml_context* context,
    ggml_tensor* vector,
    ggml_tensor* like) {
    ggml_tensor* row = ggml_reshape_2d(
        context,
        vector,
        1,
        vector->ne[0]);
    if (like->ne[0] % row->ne[0] != 0 ||
        like->ne[1] % row->ne[1] != 0) {
        throw std::runtime_error(
            "Higgs repeat shape mismatch: row=[" +
            std::to_string(row->ne[0]) + "," +
            std::to_string(row->ne[1]) + "] like=[" +
            std::to_string(like->ne[0]) + "," +
            std::to_string(like->ne[1]) + "]");
    }
    return ggml_repeat(
        context,
        row,
        like);
}

ggml_tensor* repeat_columns(
    ggml_context* context,
    ggml_tensor* vector,
    ggml_tensor* like) {
    ggml_tensor* column = ggml_reshape_2d(
        context,
        vector,
        vector->ne[0],
        1);
    if (like->ne[0] % column->ne[0] != 0 ||
        like->ne[1] % column->ne[1] != 0) {
        throw std::runtime_error(
            "Higgs repeat-columns shape mismatch");
    }
    return ggml_repeat(context, column, like);
}

ggml_tensor* linear(
    ggml_context* context,
    ggml_tensor* weight,
    ggml_tensor* bias,
    ggml_tensor* input) {
    ggml_tensor* output = ggml_mul_mat(
        context,
        weight,
        cast_f32(context, input));
    if (bias != nullptr) {
        output = ggml_add(
            context,
            output,
            repeat_columns(context, cast_f32(context, bias), output));
    }
    return output;
}

ggml_tensor* transpose_2d(
    ggml_context* context,
    ggml_tensor* input) {
    return ggml_cont(
        context,
        ggml_transpose(context, input));
}

ggml_tensor* conv1d(
    ggml_context* context,
    ggml_tensor* input,
    ggml_tensor* kernel,
    ggml_tensor* bias,
    int dilation = 1) {
    const int64_t input_length = input->ne[0];
    const int64_t input_channels = input->ne[1];
    const int64_t kernel_size = kernel->ne[0];
    const int64_t output_channels = kernel->ne[2];
    const int padding = static_cast<int>(
        ((kernel_size - 1) * dilation) / 2);
    if (kernel->ne[1] != input_channels) {
        throw std::runtime_error("Higgs convolution channel mismatch");
    }

    if (kernel_size == 1) {
        ggml_tensor* input_transposed = transpose_2d(
            context,
            cast_f32(context, input));
        ggml_tensor* output = ggml_mul_mat(
            context,
            ggml_reshape_2d(
                context,
                kernel,
                input_channels,
                output_channels),
            input_transposed);
        if (bias != nullptr) {
            output = ggml_add(
                context,
                output,
                repeat_columns(
                    context,
                    cast_f32(context, bias),
                    output));
        }
        return transpose_2d(context, output);
    }

    ggml_tensor* input_3d = ggml_reshape_3d(
        context,
        cast_f32(context, input),
        input_length,
        input_channels,
        1);
    ggml_tensor* im2col = ggml_im2col(
        context,
        kernel,
        input_3d,
        1,
        0,
        padding,
        0,
        dilation,
        0,
        false,
        GGML_TYPE_F32);
    ggml_tensor* raw = ggml_mul_mat(
        context,
        ggml_reshape_2d(
            context,
            im2col,
            im2col->ne[0],
            im2col->ne[2] * im2col->ne[1]),
        ggml_reshape_2d(
            context,
            kernel,
            kernel->ne[0] * kernel->ne[1],
            kernel->ne[2]));
    raw = ggml_reshape_3d(
        context,
        raw,
        im2col->ne[1],
        im2col->ne[2],
        output_channels);
    raw = ggml_cont(
        context,
        ggml_permute(context, raw, 0, 2, 1, 3));
    ggml_tensor* output = ggml_reshape_2d(
        context,
        raw,
        raw->ne[0],
        output_channels);
    if (bias != nullptr) {
        output = ggml_add(
            context,
            output,
            repeat_rows(context, cast_f32(context, bias), output));
    }
    return output;
}

ggml_tensor* transposed_conv(
    ggml_context* context,
    ggml_tensor* input,
    ggml_tensor* kernel,
    ggml_tensor* bias,
    int stride,
    bool matrix_path) {
    const int64_t input_length = input->ne[0];
    const int64_t input_channels = input->ne[1];
    const int64_t output_length = input_length * stride;
    const int64_t output_channels = kernel->ne[1];
    const int64_t kernel_size = matrix_path
        ? kernel->ne[2]
        : kernel->ne[0];
    if (matrix_path
            ? (kernel->ne[0] != input_channels ||
               kernel_size != 2 * stride)
            : (kernel_size != 2 * stride ||
               kernel->ne[2] != input_channels)) {
        throw std::runtime_error(
            "Higgs transposed convolution shape mismatch");
    }

    if (matrix_path) {
        ggml_tensor* input_transposed = transpose_2d(
            context,
            cast_f32(context, input));
        ggml_tensor* phases = nullptr;
        for (int phase = 0; phase < stride; ++phase) {
            const auto kernel_tap = [&](int tap) {
                return ggml_view_2d(
                    context,
                    kernel,
                    input_channels,
                    output_channels,
                    kernel->nb[1],
                    static_cast<size_t>(tap) * kernel->nb[2]);
            };
            ggml_tensor* first = ggml_mul_mat(
                context,
                kernel_tap(phase),
                input_transposed);
            first = ggml_pad_ext(
                context,
                first,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0);
            ggml_tensor* second = ggml_mul_mat(
                context,
                kernel_tap(phase + stride),
                input_transposed);
            second = ggml_pad_ext(
                context,
                second,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0);
            ggml_tensor* phase_output =
                transpose_2d(
                    context,
                    ggml_add(context, first, second));
            phases = phases == nullptr
                ? phase_output
                : ggml_concat(
                    context,
                    phases,
                    phase_output,
                    2);
        }
        ggml_tensor* raw = ggml_cont(
            context,
            ggml_permute(context, phases, 1, 2, 0, 3));
        raw = ggml_reshape_2d(
            context,
            raw,
            static_cast<int64_t>(input_length + 1) * stride,
            output_channels);
        const int left_crop = (stride + 1) / 2;
        ggml_tensor* output = ggml_cont_2d(
            context,
            ggml_view_2d(
                context,
                raw,
                output_length,
                output_channels,
                raw->nb[1],
                static_cast<size_t>(left_crop) *
                    ggml_element_size(raw)),
            output_length,
            output_channels);
        if (bias != nullptr) {
            output = ggml_add(
                context,
                output,
                repeat_rows(
                    context,
                    cast_f32(context, bias),
                    output));
        }
        return output;
    }

    ggml_tensor* raw = ggml_conv_transpose_1d(
        context,
        kernel,
        cast_f32(context, input),
        stride,
        0,
        1);
    const int left_crop = (stride + 1) / 2;
    const int right_crop = stride - left_crop;
    if (raw->ne[0] < left_crop + right_crop + output_length) {
        throw std::runtime_error(
            "Higgs transposed convolution output is too short");
    }
    ggml_tensor* output = ggml_cont_2d(
        context,
        ggml_view_2d(
            context,
            raw,
            output_length,
            output_channels,
            raw->nb[1],
            static_cast<size_t>(left_crop) * ggml_element_size(raw)),
        output_length,
        output_channels);
    if (bias != nullptr) {
        output = ggml_add(
            context,
            output,
            repeat_rows(context, cast_f32(context, bias), output));
    }
    return output;
}

ggml_tensor* snake(
    ggml_context* context,
    ggml_tensor* input,
    ggml_tensor* alpha) {
    ggml_tensor* x = cast_f32(context, input);
    ggml_tensor* alpha_full = repeat_rows(
        context,
        cast_f32(context, alpha),
        x);
    ggml_tensor* denominator = ggml_scale_bias(
        context,
        alpha_full,
        1.0f,
        1e-9f);
    ggml_tensor* sine = ggml_sin(
        context,
        ggml_mul(context, x, alpha_full));
    ggml_tensor* periodic = ggml_div(
        context,
        ggml_sqr(context, sine),
        denominator);
    return ggml_add(context, x, periodic);
}

ggml_tensor* residual_unit(
    ggml_context* context,
    ggml_tensor* input,
    const std::string& prefix,
    int dilation,
    const std::unordered_map<std::string, ggml_tensor*>& weights) {
    const auto get = [&](const std::string& suffix) -> ggml_tensor* {
        const auto found = weights.find(prefix + suffix);
        if (found == weights.end()) {
            throw std::runtime_error(
                "missing Higgs tensor: " + prefix + suffix);
        }
        return found->second;
    };
    ggml_tensor* residual = cast_f32(context, input);
    ggml_tensor* hidden = snake(
        context,
        residual,
        get("snake1.alpha"));
    hidden = conv1d(
        context,
        hidden,
        get("conv1.weight"),
        get("conv1.bias"),
        dilation);
    hidden = snake(
        context,
        hidden,
        get("snake2.alpha"));
    hidden = conv1d(
        context,
        hidden,
        get("conv2.weight"),
        get("conv2.bias"));
    return ggml_add(context, residual, hidden);
}

HiggsDecoderStage read_debug_stage(
    const std::string& name,
    ggml_tensor* tensor,
    bool time_major) {
    if (tensor == nullptr || tensor->type != GGML_TYPE_F32) {
        throw std::runtime_error(
            "Higgs debug stage is not readable: " + name);
    }
    HiggsDecoderStage stage;
    stage.name = name;
    stage.channels = static_cast<int32_t>(
        time_major ? tensor->ne[1] : tensor->ne[0]);
    stage.frames = static_cast<int32_t>(
        time_major ? tensor->ne[0] : tensor->ne[1]);
    std::vector<float> raw(
        static_cast<size_t>(ggml_nelements(tensor)));
    ggml_backend_tensor_get(
        tensor,
        raw.data(),
        0,
        raw.size() * sizeof(float));
    stage.values.resize(raw.size());
    if (!time_major) {
        for (int32_t frame = 0; frame < stage.frames; ++frame) {
            for (int32_t channel = 0;
                 channel < stage.channels;
                 ++channel) {
                stage.values[
                    static_cast<size_t>(channel) *
                        static_cast<size_t>(stage.frames) +
                    static_cast<size_t>(frame)] =
                    raw[
                        static_cast<size_t>(frame) *
                            static_cast<size_t>(stage.channels) +
                        static_cast<size_t>(channel)];
            }
        }
    } else {
        stage.values = std::move(raw);
    }
    return stage;
}

struct DecoderDebugStageSpec {
    std::string name;
    bool time_major = false;
};

std::vector<DecoderDebugStageSpec> decoder_debug_stage_specs(
    const HiggsTokenizerConfig& config) {
    std::vector<DecoderDebugStageSpec> stages;
    stages.push_back({"quantized", false});
    stages.push_back({"fc2", false});
    stages.push_back({"conv1", true});
    for (size_t block = 0;
         block < config.upsampling_ratios.size();
         ++block) {
        stages.push_back({
            "block" + std::to_string(block),
            true});
    }
    stages.push_back({"output", true});
    return stages;
}

bool is_transposed_conv_weight(const std::string& name) {
    static const std::string suffix = ".conv_t1.weight";
    return name.rfind("ad.block.", 0) == 0 &&
        name.size() >= suffix.size() &&
        name.compare(
            name.size() - suffix.size(),
            suffix.size(),
            suffix) == 0;
}

} // namespace

class HiggsAudioDecoder::DecodeGraph {
public:
    ~DecodeGraph() {
        release();
    }

    Status build(
        const HiggsAudioDecoder& decoder,
        int32_t frames,
        bool record_debug_stages) {
        release();
        decoder_ = &decoder;
        frames_ = frames;
        record_debug_stages_ = record_debug_stages;
        metadata_.resize(
            ggml_tensor_overhead() * 8192 +
            ggml_graph_overhead());
        ggml_init_params params{};
        params.mem_size = metadata_.size();
        params.mem_buffer = metadata_.data();
        params.no_alloc = true;
        context_ = ggml_init(params);
        if (context_ == nullptr) {
            release();
            return Status::error(
                "failed to create Higgs decoder graph context");
        }

        try {
            graph_ = decoder_->build_graph(
                context_,
                frames_,
                record_debug_stages_);
            code_inputs_.reserve(
                static_cast<size_t>(
                    decoder_->config_.codebook_count));
            for (int32_t codebook = 0;
                 codebook < decoder_->config_.codebook_count;
                 ++codebook) {
                const std::string name =
                    "higgs_codes." + std::to_string(codebook);
                ggml_tensor* input =
                    ggml_graph_get_tensor(graph_, name.c_str());
                if (input == nullptr) {
                    throw std::runtime_error(
                        "Higgs decoder graph is missing " + name);
                }
                code_inputs_.push_back(input);
            }
            output_ = ggml_graph_get_tensor(
                graph_,
                "higgs_decoder_output");
            if (output_ == nullptr) {
                throw std::runtime_error(
                    "Higgs decoder graph is missing output");
            }
            if (record_debug_stages_) {
                debug_specs_ = decoder_debug_stage_specs(
                    decoder_->config_);
                debug_tensors_.reserve(debug_specs_.size());
                for (const DecoderDebugStageSpec& spec :
                     debug_specs_) {
                    const std::string graph_name =
                        spec.name == "output"
                            ? "higgs_decoder_output"
                            : "higgs_debug." + spec.name;
                    ggml_tensor* tensor =
                        ggml_graph_get_tensor(
                            graph_,
                            graph_name.c_str());
                    if (tensor == nullptr) {
                        throw std::runtime_error(
                            "Higgs decoder graph is missing stage: " +
                            spec.name);
                    }
                    debug_tensors_.push_back(tensor);
                }
            }

            scheduler_ = std::make_unique<BackendScheduler>();
            Status status = scheduler_->create(
                decoder_->backend_,
                8192);
            if (!status) {
                throw std::runtime_error(status.message());
            }
            for (ggml_tensor* input : code_inputs_) {
                ggml_backend_sched_set_tensor_backend(
                    scheduler_->get(),
                    input,
                    decoder_->backend_);
            }
            if (!ggml_backend_sched_alloc_graph(
                    scheduler_->get(),
                    graph_)) {
                throw std::runtime_error(
                    "failed to allocate Higgs decoder graph");
            }
        } catch (const std::exception& error) {
            release();
            return Status::error(error.what());
        }
        return Status::ok();
    }

    bool matches(
        int32_t frames,
        bool record_debug_stages) const noexcept {
        return context_ != nullptr &&
            frames_ == frames &&
            record_debug_stages_ == record_debug_stages;
    }

    Status run(
        const AudioCodes& codes,
        AudioBuffer& audio,
        std::vector<HiggsDecoderStage>* debug_stages) {
        audio = {};
        if (decoder_ == nullptr ||
            graph_ == nullptr ||
            scheduler_ == nullptr ||
            output_ == nullptr) {
            return Status::error(
                "Higgs decoder graph is not initialized");
        }
        if (codes.codebooks !=
                decoder_->config_.codebook_count ||
            codes.frames != frames_ ||
            codes.data.size() != codes.size()) {
            return Status::error(
                "Higgs decoder graph shape mismatch");
        }

        for (int32_t codebook = 0;
             codebook < codes.codebooks;
             ++codebook) {
            const int32_t* values =
                codes.data.data() +
                static_cast<size_t>(codebook) *
                    static_cast<size_t>(codes.frames);
            ggml_backend_tensor_set(
                code_inputs_[static_cast<size_t>(codebook)],
                values,
                0,
                static_cast<size_t>(codes.frames) *
                    sizeof(int32_t));
        }
        if (ggml_backend_sched_graph_compute(
                scheduler_->get(),
                graph_) != GGML_STATUS_SUCCESS) {
            return Status::error(
                "Higgs decoder graph execution failed");
        }

        std::vector<float> waveform(
            static_cast<size_t>(ggml_nelements(output_)));
        ggml_backend_tensor_get(
            output_,
            waveform.data(),
            0,
            waveform.size() * sizeof(float));
        audio.samples = std::move(waveform);
        audio.sample_rate = decoder_->config_.sample_rate;
        audio.channels = 1;
        if (debug_stages != nullptr) {
            if (!record_debug_stages_) {
                return Status::error(
                    "Higgs decoder graph does not expose debug stages");
            }
            debug_stages->clear();
            debug_stages->reserve(debug_specs_.size());
            for (size_t index = 0;
                 index < debug_specs_.size();
                 ++index) {
                debug_stages->push_back(read_debug_stage(
                    debug_specs_[index].name,
                    debug_tensors_[index],
                    debug_specs_[index].time_major));
            }
        }
        return Status::ok();
    }

    void release() noexcept {
        scheduler_.reset();
        if (context_ != nullptr) {
            ggml_free(context_);
            context_ = nullptr;
        }
        metadata_.clear();
        graph_ = nullptr;
        output_ = nullptr;
        code_inputs_.clear();
        debug_specs_.clear();
        debug_tensors_.clear();
        decoder_ = nullptr;
        frames_ = 0;
        record_debug_stages_ = false;
    }

private:
    const HiggsAudioDecoder* decoder_ = nullptr;
    int32_t frames_ = 0;
    bool record_debug_stages_ = false;
    std::vector<uint8_t> metadata_;
    ggml_context* context_ = nullptr;
    ggml_cgraph* graph_ = nullptr;
    std::unique_ptr<BackendScheduler> scheduler_;
    std::vector<ggml_tensor*> code_inputs_;
    ggml_tensor* output_ = nullptr;
    std::vector<DecoderDebugStageSpec> debug_specs_;
    std::vector<ggml_tensor*> debug_tensors_;
};

HiggsAudioDecoder::HiggsAudioDecoder() = default;

HiggsAudioDecoder::~HiggsAudioDecoder() {
    release();
}

void HiggsAudioDecoder::release() noexcept {
    loaded_ = false;
    decode_graph_.reset();
    use_matrix_transposed_conv_ = false;
    weights_.clear();
    if (weight_buffer_ != nullptr) {
        ggml_backend_buffer_free(weight_buffer_);
        weight_buffer_ = nullptr;
    }
    if (weight_context_ != nullptr) {
        ggml_free(weight_context_);
        weight_context_ = nullptr;
    }
    backend_ = nullptr;
}

ggml_tensor* HiggsAudioDecoder::weight(
    const std::string& name) const {
    const auto found = weights_.find(name);
    return found == weights_.end() ? nullptr : found->second;
}

Status HiggsAudioDecoder::load(
    const HiggsTokenizerConfig& config,
    const GGUFModel& model,
    ggml_backend_t backend) {
    release();
    if (backend == nullptr) {
        return Status::error("Higgs decoder backend is null");
    }
    if (config.codebook_count != 8 ||
        config.codebook_size != 1024 ||
        config.codebook_dim != 64 ||
        config.acoustic_hidden_size != 256 ||
        config.acoustic_decoder_hidden_size != 1024 ||
        config.upsampling_ratios.size() != 5) {
        return Status::error("unsupported Higgs decoder configuration");
    }
    const bool gpu_backend =
        ggml_backend_dev_type(
            ggml_backend_get_device(backend)) !=
        GGML_BACKEND_DEVICE_TYPE_CPU;
    use_matrix_transposed_conv_ = gpu_backend;

    ggml_init_params params{};
    params.mem_size = 16 * 1024 * 1024;
    params.no_alloc = true;
    weight_context_ = ggml_init(params);
    if (weight_context_ == nullptr) {
        return Status::error("failed to create Higgs decoder context");
    }

    try {
        for (int64_t index = 0;
             index < model.tensor_count();
             ++index) {
            const ggml_tensor* source = model.tensor(index);
            if (source == nullptr) {
                continue;
            }
            const std::string name = source->name;
            const bool decoder_weight =
                name.rfind("ad.", 0) == 0 ||
                name.rfind("fc2.", 0) == 0 ||
                (name.rfind("q.quantizers.", 0) == 0 &&
                 (name.find(".cb.embed") != std::string::npos ||
                  name.find(".pout.") != std::string::npos));
            if (!decoder_weight) {
                continue;
            }
            ggml_tensor* destination = nullptr;
            if (name.size() >= 6 &&
                name.compare(name.size() - 6, 6, ".alpha") == 0) {
                destination = ggml_new_tensor_1d(
                    weight_context_,
                    GGML_TYPE_F32,
                    source->ne[1]);
            } else if (
                use_matrix_transposed_conv_ &&
                is_transposed_conv_weight(name)) {
                if (source->type != GGML_TYPE_F32) {
                    throw std::runtime_error(
                        "Higgs transposed convolution weight must be F32: " +
                        name);
                }
                destination = ggml_new_tensor_3d(
                    weight_context_,
                    GGML_TYPE_F32,
                    source->ne[2],
                    source->ne[1],
                    source->ne[0]);
            } else {
                destination = ggml_dup_tensor(
                    weight_context_,
                    source);
            }
            ggml_set_name(destination, name.c_str());
            weights_.emplace(name, destination);
        }
    } catch (const std::exception& error) {
        release();
        return Status::error(error.what());
    }

    weight_buffer_ = ggml_backend_alloc_ctx_tensors(
        weight_context_,
        backend);
    if (weight_buffer_ == nullptr) {
        release();
        return Status::error("failed to allocate Higgs decoder weights");
    }
    ggml_backend_buffer_set_usage(
        weight_buffer_,
        GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (const auto& [name, destination] : weights_) {
        const ggml_tensor* source = model.tensor(name);
        if (source == nullptr) {
            release();
            return Status::error("failed to reload Higgs tensor: " + name);
        }
        if (use_matrix_transposed_conv_ &&
            is_transposed_conv_weight(name)) {
            const int64_t kernel_size = source->ne[0];
            const int64_t output_channels = source->ne[1];
            const int64_t input_channels = source->ne[2];
            const float* source_values =
                static_cast<const float*>(source->data);
            std::vector<float> transformed(
                static_cast<size_t>(ggml_nelements(source)));
            for (int64_t kernel = 0;
                 kernel < kernel_size;
                 ++kernel) {
                for (int64_t output = 0;
                     output < output_channels;
                     ++output) {
                    for (int64_t input = 0;
                         input < input_channels;
                         ++input) {
                        const size_t source_index =
                            static_cast<size_t>(
                                kernel +
                                kernel_size *
                                    (output +
                                     output_channels * input));
                        const size_t destination_index =
                            static_cast<size_t>(
                                input +
                                input_channels *
                                    (output +
                                     output_channels * kernel));
                        transformed[destination_index] =
                            source_values[source_index];
                    }
                }
            }
            ggml_backend_tensor_set(
                destination,
                transformed.data(),
                0,
                transformed.size() * sizeof(float));
        } else {
            ggml_backend_tensor_set(
                destination,
                source->data,
                0,
                ggml_nbytes(source));
        }
    }

    config_ = config;
    backend_ = backend;
    loaded_ = true;
    return Status::ok();
}

Status HiggsAudioDecoder::decode(const AudioCodes& codes, AudioBuffer& audio,
                                 std::vector<HiggsDecoderStage>* debug_stages) const {
    audio = {};
    if (!loaded_) {
        return Status::error("Higgs decoder is not loaded");
    }
    if (codes.codebooks != config_.codebook_count || codes.frames <= 0 ||
        codes.data.size() != codes.size()) {
        return Status::error("invalid Higgs audio code shape");
    }
    for (int32_t code : codes.data) {
        if (code < 0 || code >= config_.codebook_size) {
            return Status::error("Higgs audio code is out of range");
        }
    }

    const bool record_debug_stages =
        debug_stages != nullptr;
    if (decode_graph_ == nullptr ||
        !decode_graph_->matches(
            codes.frames,
            record_debug_stages)) {
        std::unique_ptr<DecodeGraph> graph =
            std::make_unique<DecodeGraph>();
        Status status = graph->build(
            *this,
            codes.frames,
            record_debug_stages);
        if (!status) {
            return status;
        }
        decode_graph_ = std::move(graph);
    }
    try {
        return decode_graph_->run(
            codes,
            audio,
            debug_stages);
    } catch (const std::exception& error) {
        return Status::error(error.what());
    }
}

ggml_cgraph* HiggsAudioDecoder::build_graph(ggml_context* context, int32_t frames,
                                            bool record_debug_stages) const {
    const auto record_stage = [&](const std::string& name, ggml_tensor* tensor, bool time_major) {
        (void)time_major;
        if (record_debug_stages) {
            if (name != "output") {
                const std::string graph_name = "higgs_debug." + name;
                ggml_set_name(tensor, graph_name.c_str());
            }
            ggml_set_output(tensor);
        }
    };
    std::vector<ggml_tensor*> code_inputs;
    code_inputs.reserve(static_cast<size_t>(config_.codebook_count));
    for (int32_t codebook = 0; codebook < config_.codebook_count; ++codebook) {
        ggml_tensor* input = ggml_new_tensor_1d(context, GGML_TYPE_I32, frames);
        const std::string name = "higgs_codes." + std::to_string(codebook);
        ggml_set_name(input, name.c_str());
        ggml_set_input(input);
        code_inputs.push_back(input);
    }

    ggml_tensor* quantized = nullptr;
    for (int32_t codebook = 0; codebook < config_.codebook_count; ++codebook) {
        const std::string prefix = "q.quantizers." + std::to_string(codebook) + ".";
        ggml_tensor* embedding = ggml_get_rows(context, weight(prefix + "cb.embed"),
                                               code_inputs[static_cast<size_t>(codebook)]);
        ggml_tensor* projected = linear(context, weight(prefix + "pout.weight"),
                                        weight(prefix + "pout.bias"), embedding);
        quantized = quantized == nullptr ? projected : ggml_add(context, quantized, projected);
    }
    record_stage("quantized", quantized, false);

    ggml_tensor* hidden = linear(context, weight("fc2.weight"), weight("fc2.bias"), quantized);
    record_stage("fc2", hidden, false);
    hidden = ggml_cont(context, ggml_transpose(context, hidden));
    hidden = conv1d(context, hidden, weight("ad.conv1.weight"), weight("ad.conv1.bias"));
    record_stage("conv1", hidden, true);

    for (size_t block = 0; block < config_.upsampling_ratios.size(); ++block) {
        const std::string prefix = "ad.block." + std::to_string(block) + ".";
        hidden = snake(context, hidden, weight(prefix + "snake1.alpha"));
        hidden = transposed_conv(context, hidden, weight(prefix + "conv_t1.weight"),
                                 weight(prefix + "conv_t1.bias"),
                                 config_.upsampling_ratios[block],
                                 use_matrix_transposed_conv_);
        for (int unit = 1; unit <= 3; ++unit) {
            hidden =
                residual_unit(context, hidden, prefix + "res_unit" + std::to_string(unit) + ".",
                              unit == 1 ? 1 : (unit == 2 ? 3 : 9), weights_);
        }
        record_stage("block" + std::to_string(block), hidden, true);
    }
    hidden = snake(context, hidden, weight("ad.snake1.alpha"));
    hidden = conv1d(context, hidden, weight("ad.conv2.weight"), weight("ad.conv2.bias"));
    record_stage("output", hidden, true);
    ggml_set_name(hidden, "higgs_decoder_output");
    ggml_set_output(hidden);

    ggml_cgraph* graph = ggml_new_graph_custom(context, 8192, false);
    ggml_build_forward_expand(graph, hidden);
    return graph;
}

} // namespace omnivoice
