#include "qwen3_lm/qwen3_block.h"

#include "backend_scheduler.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Fixture {
    int32_t hidden = 0;
    int32_t heads = 0;
    int32_t kv_heads = 0;
    int32_t head_dim = 0;
    int32_t intermediate = 0;
    int32_t sequence = 0;

    std::vector<float> input;
    std::vector<float> q_proj;
    std::vector<float> k_proj;
    std::vector<float> v_proj;
    std::vector<float> o_proj;
    std::vector<float> q_norm;
    std::vector<float> k_norm;
    std::vector<float> input_norm;
    std::vector<float> post_norm;
    std::vector<float> gate;
    std::vector<float> up;
    std::vector<float> down;
    std::vector<float> expected;
    std::vector<int32_t> positions;
    std::vector<float> mask;
};

std::vector<float> read_f32(std::ifstream& file, size_t count) {
    std::vector<float> values(count);
    file.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(count * sizeof(float)));
    if (!file) {
        throw std::runtime_error("failed to read fixture float data");
    }
    return values;
}

Fixture read_fixture() {
    std::ifstream file(
        std::string(OMNIVOICE_TEST_DATA_DIR) +
            "/qwen3_block_fixture.bin",
        std::ios::binary);
    if (!file) {
        throw std::runtime_error("failed to open Qwen3 block fixture");
    }

    char magic[8] = {};
    std::array<int32_t, 7> header{};
    file.read(magic, sizeof(magic));
    file.read(
        reinterpret_cast<char*>(header.data()),
        static_cast<std::streamsize>(
            header.size() * sizeof(int32_t)));
    if (!file ||
        std::memcmp(magic, "QVBLK01\0", 8) != 0 ||
        header[6] != 13) {
        throw std::runtime_error("invalid Qwen3 block fixture");
    }

    Fixture fixture;
    fixture.hidden = header[0];
    fixture.heads = header[1];
    fixture.kv_heads = header[2];
    fixture.head_dim = header[3];
    fixture.intermediate = header[4];
    fixture.sequence = header[5];

    const size_t hidden = static_cast<size_t>(fixture.hidden);
    const size_t sequence = static_cast<size_t>(fixture.sequence);
    const size_t q_dim =
        static_cast<size_t>(fixture.heads * fixture.head_dim);
    const size_t kv_dim =
        static_cast<size_t>(fixture.kv_heads * fixture.head_dim);
    const size_t intermediate =
        static_cast<size_t>(fixture.intermediate);

    fixture.input = read_f32(file, sequence * hidden);
    fixture.q_proj = read_f32(file, q_dim * hidden);
    fixture.k_proj = read_f32(file, kv_dim * hidden);
    fixture.v_proj = read_f32(file, kv_dim * hidden);
    fixture.o_proj = read_f32(file, hidden * q_dim);
    fixture.q_norm = read_f32(file, static_cast<size_t>(fixture.head_dim));
    fixture.k_norm = read_f32(file, static_cast<size_t>(fixture.head_dim));
    fixture.input_norm = read_f32(file, hidden);
    fixture.post_norm = read_f32(file, hidden);
    fixture.gate = read_f32(file, intermediate * hidden);
    fixture.up = read_f32(file, intermediate * hidden);
    fixture.down = read_f32(file, hidden * intermediate);
    fixture.expected = read_f32(file, sequence * hidden);

    fixture.positions.resize(sequence);
    file.read(
        reinterpret_cast<char*>(fixture.positions.data()),
        static_cast<std::streamsize>(
            sequence * sizeof(int32_t)));
    fixture.mask = read_f32(file, sequence * sequence);
    if (!file) {
        throw std::runtime_error("truncated Qwen3 block fixture");
    }
    return fixture;
}

ggml_tensor* make_matrix(
    ggml_context* context,
    const char* name,
    int64_t columns,
    int64_t rows,
    const std::vector<float>& values) {
    ggml_tensor* tensor =
        ggml_new_tensor_2d(context, GGML_TYPE_F32, columns, rows);
    ggml_set_name(tensor, name);
    if (ggml_nelements(tensor) !=
        static_cast<int64_t>(values.size())) {
        throw std::runtime_error("fixture matrix size mismatch");
    }
    return tensor;
}

ggml_tensor* make_vector(
    ggml_context* context,
    const char* name,
    int64_t size,
    const std::vector<float>& values) {
    ggml_tensor* tensor =
        ggml_new_tensor_1d(context, GGML_TYPE_F32, size);
    ggml_set_name(tensor, name);
    if (ggml_nelements(tensor) !=
        static_cast<int64_t>(values.size())) {
        throw std::runtime_error("fixture vector size mismatch");
    }
    return tensor;
}

void copy_float_tensor(ggml_tensor* tensor, const std::vector<float>& data) {
    ggml_backend_tensor_set(
        tensor,
        data.data(),
        0,
        data.size() * sizeof(float));
}

} // namespace

int main() {
    try {
        Fixture fixture = read_fixture();
        omnivoice::Qwen3Config config;
        config.hidden_size = fixture.hidden;
        config.intermediate_size = fixture.intermediate;
        config.num_hidden_layers = 1;
        config.num_attention_heads = fixture.heads;
        config.num_key_value_heads = fixture.kv_heads;
        config.head_dim = fixture.head_dim;
        config.max_position_embeddings = fixture.sequence;
        config.rms_norm_epsilon = 1e-6f;
        config.rope_theta = 10000.0f;
        config.hidden_activation = "silu";
        config.rope_type = "default";

        ggml_backend_load_all();
        const char* device_name =
            std::getenv("OMNIVOICE_TEST_DEVICE");
        if (device_name == nullptr || device_name[0] == '\0') {
            device_name = "CPU";
        }
        ggml_backend_dev_t device =
            ggml_backend_dev_by_name(device_name);
        ggml_backend_t backend = device == nullptr
            ? nullptr
            : ggml_backend_dev_init(device, nullptr);
        if (backend == nullptr) {
            std::cerr << "backend device unavailable; skipping\n";
            return 77;
        }

        ggml_init_params weight_params{};
        weight_params.mem_size = 16 * 1024 * 1024;
        weight_params.no_alloc = true;
        ggml_context* weight_context = ggml_init(weight_params);
        if (weight_context == nullptr) {
            throw std::runtime_error("failed to create weight context");
        }

        omnivoice::Qwen3BlockWeights weights;
        weights.q_proj_weight = make_matrix(
            weight_context,
            "q_proj",
            fixture.hidden,
            fixture.heads * fixture.head_dim,
            fixture.q_proj);
        weights.k_proj_weight = make_matrix(
            weight_context,
            "k_proj",
            fixture.hidden,
            fixture.kv_heads * fixture.head_dim,
            fixture.k_proj);
        weights.v_proj_weight = make_matrix(
            weight_context,
            "v_proj",
            fixture.hidden,
            fixture.kv_heads * fixture.head_dim,
            fixture.v_proj);
        std::vector<float> packed_qkv = fixture.q_proj;
        packed_qkv.insert(
            packed_qkv.end(),
            fixture.k_proj.begin(),
            fixture.k_proj.end());
        packed_qkv.insert(
            packed_qkv.end(),
            fixture.v_proj.begin(),
            fixture.v_proj.end());
        weights.qkv_proj_weight = make_matrix(
            weight_context,
            "qkv_proj",
            fixture.hidden,
            fixture.heads * fixture.head_dim +
                2 * fixture.kv_heads * fixture.head_dim,
            packed_qkv);
        weights.o_proj_weight = make_matrix(
            weight_context,
            "o_proj",
            fixture.heads * fixture.head_dim,
            fixture.hidden,
            fixture.o_proj);
        weights.q_norm_weight = make_vector(
            weight_context,
            "q_norm",
            fixture.head_dim,
            fixture.q_norm);
        weights.k_norm_weight = make_vector(
            weight_context,
            "k_norm",
            fixture.head_dim,
            fixture.k_norm);
        weights.input_layernorm_weight = make_vector(
            weight_context,
            "input_norm",
            fixture.hidden,
            fixture.input_norm);
        weights.post_attention_layernorm_weight = make_vector(
            weight_context,
            "post_norm",
            fixture.hidden,
            fixture.post_norm);
        weights.gate_proj_weight = make_matrix(
            weight_context,
            "gate",
            fixture.hidden,
            fixture.intermediate,
            fixture.gate);
        weights.up_proj_weight = make_matrix(
            weight_context,
            "up",
            fixture.hidden,
            fixture.intermediate,
            fixture.up);
        weights.down_proj_weight = make_matrix(
            weight_context,
            "down",
            fixture.intermediate,
            fixture.hidden,
            fixture.down);

        ggml_backend_buffer_t weight_buffer =
            ggml_backend_alloc_ctx_tensors(weight_context, backend);
        if (weight_buffer == nullptr) {
            throw std::runtime_error("failed to allocate weight buffer");
        }
        copy_float_tensor(weights.q_proj_weight, fixture.q_proj);
        copy_float_tensor(weights.k_proj_weight, fixture.k_proj);
        copy_float_tensor(weights.v_proj_weight, fixture.v_proj);
        copy_float_tensor(
            weights.qkv_proj_weight,
            packed_qkv);
        copy_float_tensor(weights.o_proj_weight, fixture.o_proj);
        copy_float_tensor(weights.q_norm_weight, fixture.q_norm);
        copy_float_tensor(weights.k_norm_weight, fixture.k_norm);
        copy_float_tensor(
            weights.input_layernorm_weight,
            fixture.input_norm);
        copy_float_tensor(
            weights.post_attention_layernorm_weight,
            fixture.post_norm);
        copy_float_tensor(weights.gate_proj_weight, fixture.gate);
        copy_float_tensor(weights.up_proj_weight, fixture.up);
        copy_float_tensor(weights.down_proj_weight, fixture.down);

        std::vector<uint8_t> graph_metadata(
            ggml_tensor_overhead() * 1024 + ggml_graph_overhead());
        ggml_init_params graph_params{};
        graph_params.mem_size = graph_metadata.size();
        graph_params.mem_buffer = graph_metadata.data();
        graph_params.no_alloc = true;
        ggml_context* graph_context = ggml_init(graph_params);
        if (graph_context == nullptr) {
            throw std::runtime_error("failed to create graph context");
        }

        ggml_tensor* input = ggml_new_tensor_2d(
            graph_context,
            GGML_TYPE_F32,
            fixture.hidden,
            fixture.sequence);
        ggml_set_name(input, "block_input");
        ggml_set_input(input);
        ggml_tensor* positions = ggml_new_tensor_1d(
            graph_context,
            GGML_TYPE_I32,
            fixture.sequence);
        ggml_set_name(positions, "positions");
        ggml_set_input(positions);
        ggml_tensor* attention_mask = ggml_new_tensor_4d(
            graph_context,
            GGML_TYPE_F16,
            fixture.sequence,
            fixture.sequence,
            1,
            1);
        ggml_set_name(attention_mask, "attention_mask");
        ggml_set_input(attention_mask);
        ggml_tensor* output = omnivoice::build_qwen3_block(
            graph_context,
            config,
            weights,
            input,
            positions,
            attention_mask);
        ggml_set_output(output);
        ggml_cgraph* graph = ggml_new_graph_custom(
            graph_context,
            4096,
            false);
        ggml_build_forward_expand(graph, output);

        omnivoice::BackendScheduler scheduler;
        omnivoice::Status scheduler_status =
            scheduler.create(backend, 4096);
        if (!scheduler_status) {
            throw std::runtime_error(scheduler_status.message());
        }
        ggml_backend_sched_reset(scheduler.get());
        if (!ggml_backend_sched_alloc_graph(scheduler.get(), graph)) {
            throw std::runtime_error("failed to allocate graph");
        }

        copy_float_tensor(input, fixture.input);
        ggml_backend_tensor_set(
            positions,
            fixture.positions.data(),
            0,
            fixture.positions.size() * sizeof(int32_t));
        std::vector<ggml_fp16_t> mask_half(fixture.mask.size());
        for (size_t index = 0; index < fixture.mask.size(); ++index) {
            mask_half[index] =
                ggml_fp32_to_fp16(fixture.mask[index]);
        }
        ggml_backend_tensor_set(
            attention_mask,
            mask_half.data(),
            0,
            mask_half.size() * sizeof(ggml_fp16_t));
        if (ggml_backend_sched_graph_compute(scheduler.get(), graph) !=
            GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Qwen3 block graph failed");
        }

        std::vector<float> actual(fixture.expected.size());
        ggml_backend_tensor_get(
            output,
            actual.data(),
            0,
            actual.size() * sizeof(float));
        float max_error = 0.0f;
        for (size_t index = 0; index < actual.size(); ++index) {
            max_error = std::max(
                max_error,
                std::abs(actual[index] - fixture.expected[index]));
        }
        if (!std::isfinite(max_error) || max_error > 1.5e-3f) {
            std::cerr << "Qwen3 block max error: " << max_error << '\n';
            ggml_free(graph_context);
            ggml_backend_buffer_free(weight_buffer);
            ggml_free(weight_context);
            ggml_backend_free(backend);
            return 1;
        }
        std::cout << "qwen3_block_max_error: " << max_error << '\n';

        ggml_free(graph_context);
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(weight_context);
        ggml_backend_free(backend);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Qwen3 block test failed: " << error.what() << '\n';
        return 1;
    }
}
