#!/usr/bin/env python3
"""Convert OmniVoice HF safetensors and tokenizer files to GGUF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from gguf import GGMLQuantizationType, GGUFWriter
from safetensors import safe_open


class ConversionError(RuntimeError):
    pass


AUDIO_NAME_REPLACEMENTS = (
    ("semantic_model.feature_extractor.", "s.fe."),
    ("semantic_model.feature_projection.", "s.fp."),
    ("semantic_model.encoder.", "s.enc."),
    ("acoustic_encoder.", "ae."),
    ("acoustic_decoder.", "ad."),
    ("encoder_semantic.", "es."),
    ("decoder_semantic.", "ds."),
    ("quantizer.", "q."),
    ("feed_forward.intermediate_dense.", "ff.i."),
    ("feed_forward.output_dense.", "ff.o."),
    ("feed_forward.", "ff."),
    ("self_attn.", "attn."),
    ("layer_norm.", "ln."),
    ("conv_layers.", "cv."),
    ("conv_blocks.", "cb."),
    ("pos_conv_embed.", "pce."),
    (
        "conv.parametrizations.weight.original0",
        "conv.wn0",
    ),
    (
        "conv.parametrizations.weight.original1",
        "conv.wn1",
    ),
    ("codebook.", "cb."),
    ("project_in.", "pin."),
    ("project_out.", "pout."),
)

SKIP_LM_TENSORS = {
    # Derived at runtime: arange(8) * audio_vocab_size.
    "codebook_layer_offsets",
}

SKIP_AUDIO_TENSORS = {
    "semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original0",
    "semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original1",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def build_tokenizer(
    tokenizer: dict[str, Any],
) -> tuple[list[str], list[str], list[str], list[int]]:
    model = tokenizer.get("model")
    if not isinstance(model, dict) or model.get("type") != "BPE":
        raise ValueError("only Qwen BPE tokenizer.json files are supported")

    vocab = model.get("vocab")
    merges = model.get("merges")
    added_tokens = tokenizer.get("added_tokens")
    if not isinstance(vocab, dict):
        raise ValueError("tokenizer model has no BPE vocabulary")
    if not isinstance(merges, list):
        raise ValueError("tokenizer model has no BPE merge list")
    if not isinstance(added_tokens, list):
        raise ValueError("tokenizer has no added_tokens array")

    added: list[tuple[str, int]] = []
    max_id = -1
    for token, token_id in vocab.items():
        if not isinstance(token, str) or not isinstance(token_id, int):
            raise ValueError("BPE vocabulary must contain string/int pairs")
        if token_id < 0:
            raise ValueError("BPE vocabulary contains a negative ID")
        max_id = max(max_id, token_id)
    for item in added_tokens:
        if not isinstance(item, dict):
            raise ValueError("invalid added token descriptor")
        token = item.get("content")
        token_id = item.get("id")
        if not isinstance(token, str) or not isinstance(token_id, int):
            raise ValueError("invalid added token content or ID")
        if token_id < 0:
            raise ValueError("added token ID is negative")
        max_id = max(max_id, token_id)
        added.append((token, token_id))

    if max_id < 0:
        raise ValueError("tokenizer vocabulary is empty")

    tokens: list[str | None] = [None] * (max_id + 1)
    for token, token_id in vocab.items():
        if tokens[token_id] is not None:
            raise ValueError(f"duplicate BPE token ID: {token_id}")
        tokens[token_id] = token
    for token, token_id in added:
        if tokens[token_id] is not None:
            raise ValueError(
                f"added token conflicts with BPE ID: {token_id}")
        tokens[token_id] = token
    for token_id, token in enumerate(tokens):
        if token is None or not token:
            raise ValueError(f"missing tokenizer token at ID {token_id}")

    converted_merges: list[str] = []
    for merge in merges:
        if isinstance(merge, str):
            converted_merges.append(merge)
        elif (
            isinstance(merge, list)
            and len(merge) == 2
            and all(isinstance(item, str) for item in merge)
        ):
            converted_merges.append(f"{merge[0]} {merge[1]}")
        else:
            raise ValueError("invalid BPE merge entry")

    normalizer = tokenizer.get("normalizer")
    if not isinstance(normalizer, dict) or normalizer.get("type") != "NFC":
        raise ValueError("only the NFC normalizer is supported")

    return (
        [token for token in tokens if token is not None],
        converted_merges,
        [token for token, _ in added],
        [token_id for _, token_id in added],
    )


def add_lm_metadata(
    writer: GGUFWriter,
    model_config: dict[str, Any],
    tokenizer_json: dict[str, Any],
    tokenizer_config: dict[str, Any],
    name: str,
) -> None:
    llm = model_config["llm_config"]
    rope = llm.get("rope_parameters", {})
    tokens, merges, added_names, added_ids = build_tokenizer(tokenizer_json)
    token_to_id = {
        token: token_id
        for token_id, token in enumerate(tokens)
    }

    pad_token = tokenizer_config.get("pad_token")
    eos_token = tokenizer_config.get("eos_token")
    if not isinstance(pad_token, str) or pad_token not in token_to_id:
        raise ValueError("tokenizer pad_token is invalid")
    if not isinstance(eos_token, str) or eos_token not in token_to_id:
        raise ValueError("tokenizer eos_token is invalid")

    writer.add_name(name)
    writer.add_file_type(0)
    writer.add_quantization_version(2)
    writer.add_uint32("omnivoice.schema_version", 1)
    writer.add_string("omnivoice.file_role", "lm")

    llm_fields = {
        "hidden_size": "omnivoice.llm.hidden_size",
        "intermediate_size": "omnivoice.llm.intermediate_size",
        "num_hidden_layers": "omnivoice.llm.block_count",
        "num_attention_heads": "omnivoice.llm.attention_head_count",
        "num_key_value_heads": "omnivoice.llm.attention_head_count_kv",
        "head_dim": "omnivoice.llm.attention_key_length",
        "max_position_embeddings": "omnivoice.llm.context_length",
        "vocab_size": "omnivoice.llm.vocab_size",
    }
    for source, destination in llm_fields.items():
        writer.add_uint32(destination, int(llm[source]))
    writer.add_float32(
        "omnivoice.llm.rms_norm_epsilon",
        float(llm["rms_norm_eps"]),
    )
    writer.add_float32(
        "omnivoice.llm.rope_theta",
        float(rope["rope_theta"]),
    )
    writer.add_string(
        "omnivoice.llm.hidden_activation",
        str(llm["hidden_act"]),
    )
    writer.add_string(
        "omnivoice.llm.rope_type",
        str(rope.get("rope_type", "default")),
    )
    writer.add_bool(
        "omnivoice.llm.attention_bias",
        bool(llm.get("attention_bias", False)),
    )
    writer.add_bool(
        "omnivoice.llm.tie_word_embeddings",
        bool(llm.get("tie_word_embeddings", False)),
    )
    writer.add_bool(
        "omnivoice.llm.use_sliding_window",
        bool(llm.get("use_sliding_window", False)),
    )

    writer.add_uint32(
        "omnivoice.audio.vocab_size",
        int(model_config["audio_vocab_size"]),
    )
    writer.add_uint32(
        "omnivoice.audio.mask_id",
        int(model_config["audio_mask_id"]),
    )
    writer.add_uint32(
        "omnivoice.audio.codebook_count",
        int(model_config["num_audio_codebook"]),
    )
    writer.add_uint32("omnivoice.audio.frame_rate", 25)
    writer.add_array(
        "omnivoice.audio.codebook_weights",
        [int(value) for value in model_config["audio_codebook_weights"]],
    )

    writer.add_string("tokenizer.ggml.model", "qwen2")
    writer.add_uint32("omnivoice.text.vocab_size", len(tokens))
    writer.add_int32(
        "tokenizer.ggml.padding_token_id",
        token_to_id[pad_token],
    )
    writer.add_int32(
        "tokenizer.ggml.eos_token_id",
        token_to_id[eos_token],
    )
    writer.add_bool("omnivoice.text.nfc_normalization", True)
    writer.add_array("tokenizer.ggml.tokens", tokens)
    writer.add_array("tokenizer.ggml.merges", merges)
    writer.add_array("tokenizer.ggml.added_token_names", added_names)
    writer.add_array("tokenizer.ggml.added_token_ids", added_ids)

    generation_defaults = {
        "omnivoice.generation.num_step": (32, "int32"),
        "omnivoice.generation.guidance_scale": (2.0, "float32"),
        "omnivoice.generation.t_shift": (0.1, "float32"),
        "omnivoice.generation.layer_penalty_factor": (5.0, "float32"),
        "omnivoice.generation.position_temperature": (5.0, "float32"),
        "omnivoice.generation.class_temperature": (0.0, "float32"),
        "omnivoice.generation.denoise": (True, "bool"),
        "omnivoice.generation.preprocess_prompt": (True, "bool"),
        "omnivoice.generation.postprocess_output": (True, "bool"),
        "omnivoice.generation.audio_chunk_duration": (15.0, "float32"),
        "omnivoice.generation.audio_chunk_threshold": (30.0, "float32"),
        "omnivoice.generation.pad_duration": (0.1, "float32"),
        "omnivoice.generation.fade_duration": (0.1, "float32"),
    }
    for key, (value, value_type) in generation_defaults.items():
        if value_type == "int32":
            writer.add_int32(key, int(value))
        elif value_type == "float32":
            writer.add_float32(key, float(value))
        else:
            writer.add_bool(key, bool(value))


def add_audio_metadata(
    writer: GGUFWriter,
    audio_config: dict[str, Any],
    preprocessor_config: dict[str, Any],
    name: str,
) -> None:
    semantic = audio_config["semantic_model_config"]
    acoustic = audio_config["acoustic_model_config"]
    sample_rate = int(audio_config["sample_rate"])
    hop_length = int(preprocessor_config["hop_length"])

    writer.add_name(name)
    writer.add_file_type(0)
    writer.add_quantization_version(2)
    writer.add_uint32("omnivoice.schema_version", 1)
    writer.add_string("omnivoice.file_role", "audio_tokenizer")
    writer.add_string(
        "omnivoice.audio_tokenizer.model_type",
        str(audio_config["model_type"]),
    )
    writer.add_uint32(
        "omnivoice.audio_tokenizer.sample_rate",
        sample_rate,
    )
    writer.add_uint32(
        "omnivoice.audio_tokenizer.semantic_sample_rate",
        int(audio_config["semantic_sample_rate"]),
    )
    writer.add_uint32("omnivoice.audio_tokenizer.hop_length", hop_length)
    writer.add_uint32(
        "omnivoice.audio_tokenizer.frame_rate",
        sample_rate // hop_length,
    )
    writer.add_uint32("omnivoice.audio_tokenizer.codebook_count", 8)
    writer.add_uint32(
        "omnivoice.audio_tokenizer.codebook_size",
        int(audio_config["codebook_size"]),
    )
    writer.add_uint32(
        "omnivoice.audio_tokenizer.codebook_dim",
        int(audio_config["codebook_dim"]),
    )
    writer.add_uint32(
        "omnivoice.audio_tokenizer.semantic_hidden_size",
        int(semantic["hidden_size"]),
    )
    writer.add_uint32(
        "omnivoice.audio_tokenizer.semantic_attention_heads",
        int(semantic["num_attention_heads"]),
    )
    writer.add_uint32(
        "omnivoice.audio_tokenizer.semantic_layer_count",
        int(semantic["num_hidden_layers"]),
    )
    writer.add_uint32(
        "omnivoice.audio_tokenizer.acoustic_hidden_size",
        int(acoustic["hidden_size"]),
    )
    writer.add_uint32(
        "omnivoice.audio_tokenizer.acoustic_encoder_hidden_size",
        int(acoustic["encoder_hidden_size"]),
    )
    writer.add_uint32(
        "omnivoice.audio_tokenizer.acoustic_decoder_hidden_size",
        int(acoustic["decoder_hidden_size"]),
    )
    writer.add_array(
        "omnivoice.audio_tokenizer.semantic_conv_dim",
        [int(value) for value in semantic["conv_dim"]],
    )
    writer.add_array(
        "omnivoice.audio_tokenizer.semantic_conv_kernel",
        [int(value) for value in semantic["conv_kernel"]],
    )
    writer.add_array(
        "omnivoice.audio_tokenizer.semantic_conv_stride",
        [int(value) for value in semantic["conv_stride"]],
    )
    writer.add_array(
        "omnivoice.audio_tokenizer.downsampling_ratios",
        [int(value) for value in acoustic["downsampling_ratios"]],
    )
    writer.add_array(
        "omnivoice.audio_tokenizer.upsampling_ratios",
        [int(value) for value in acoustic["upsampling_ratios"]],
    )


def map_lm_tensor_name(name: str) -> str:
    return name


def map_audio_tensor_name(name: str) -> str:
    mapped = name
    for source, destination in AUDIO_NAME_REPLACEMENTS:
        mapped = mapped.replace(source, destination)
    return mapped


def tensor_dtype(
    tensor: torch.Tensor,
) -> tuple[Any, GGMLQuantizationType | None]:
    if tensor.dtype == torch.bfloat16:
        return tensor.view(torch.int16).numpy(), GGMLQuantizationType.BF16
    if tensor.dtype in (torch.float16, torch.float32):
        return tensor.numpy(), None
    raise ConversionError(f"unsupported safetensors dtype: {tensor.dtype}")


def add_safetensors(
    writer: GGUFWriter,
    source: Path,
    map_name,
    skip_names: set[str] | None = None,
) -> int:
    skip_names = skip_names or set()
    with safe_open(str(source), framework="pt", device="cpu") as tensors:
        names = [
            name for name in tensors.keys()
            if name not in skip_names
        ]
        mapped_names = [map_name(name) for name in names]
        if len(set(mapped_names)) != len(mapped_names):
            raise ConversionError(
                f"tensor name collision while converting {source}")

        for source_name, destination_name in zip(names, mapped_names):
            if len(destination_name) >= 64:
                raise ConversionError(
                    f"GGUF tensor name exceeds 63 characters: "
                    f"{destination_name}")
            tensor = tensors.get_tensor(source_name)
            tensor = tensor.detach().cpu().contiguous()
            array, raw_dtype = tensor_dtype(tensor)
            writer.add_tensor(
                destination_name,
                array,
                raw_dtype=raw_dtype,
            )
    return len(names)


def add_higgs_fused_tensors(
    writer: GGUFWriter,
    source: Path,
) -> None:
    original0 = (
        "semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original0"
    )
    original1 = (
        "semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original1"
    )
    with safe_open(str(source), framework="pt", device="cpu") as tensors:
        weight_g = tensors.get_tensor(original0).float()
        weight_v = tensors.get_tensor(original1).float()
        # Match the exact parametrization used when the HF model evaluates
        # its positional convolution. torch._weight_norm(dim=2) is not the
        # same reduction as v * g / norm(v, dim=2).
        effective = torch._weight_norm(
            weight_v,
            weight_g,
            2,
        )
        grouped = effective.contiguous().numpy()
        writer.add_tensor("s.enc.pce.weight", grouped)


def write_gguf(writer: GGUFWriter, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def convert(args: argparse.Namespace) -> None:
    model_dir: Path = args.model_dir
    output_dir: Path = args.output_dir

    model_path = model_dir / "model.safetensors"
    config_path = model_dir / "config.json"
    tokenizer_json_path = model_dir / "tokenizer.json"
    tokenizer_config_path = model_dir / "tokenizer_config.json"
    audio_dir = model_dir / "audio_tokenizer"
    audio_model_path = audio_dir / "model.safetensors"
    audio_config_path = audio_dir / "config.json"
    audio_preprocessor_path = audio_dir / "preprocessor_config.json"

    required = [
        model_path,
        config_path,
        tokenizer_json_path,
        tokenizer_config_path,
        audio_model_path,
        audio_config_path,
        audio_preprocessor_path,
    ]
    for path in required:
        if not path.is_file():
            raise ConversionError(f"missing input file: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    lm_output = output_dir / "omnivoice-lm-f32.gguf"
    audio_output = output_dir / "omnivoice-tokenizer-f32.gguf"
    if not args.overwrite:
        for path in (lm_output, audio_output):
            if path.exists():
                raise ConversionError(
                    f"output already exists: {path}; use --overwrite")

    model_config = load_json(config_path)
    tokenizer_json = load_json(tokenizer_json_path)
    tokenizer_config = load_json(tokenizer_config_path)
    audio_config = load_json(audio_config_path)
    audio_preprocessor = load_json(audio_preprocessor_path)

    print(f"converting LM: {model_path}")
    lm_writer = GGUFWriter(str(lm_output), "omnivoice")
    add_lm_metadata(
        lm_writer,
        model_config,
        tokenizer_json,
        tokenizer_config,
        name="OmniVoice LM",
    )
    lm_count = add_safetensors(
        lm_writer,
        model_path,
        map_lm_tensor_name,
        SKIP_LM_TENSORS,
    )
    write_gguf(lm_writer, lm_output)
    print(f"wrote {lm_output} ({lm_count} tensors)")

    print(f"converting audio tokenizer: {audio_model_path}")
    audio_writer = GGUFWriter(str(audio_output), "omnivoice")
    add_audio_metadata(
        audio_writer,
        audio_config,
        audio_preprocessor,
        name="OmniVoice Higgs audio tokenizer",
    )
    add_higgs_fused_tensors(audio_writer, audio_model_path)
    audio_count = add_safetensors(
        audio_writer,
        audio_model_path,
        map_audio_tensor_name,
        SKIP_AUDIO_TENSORS,
    )
    write_gguf(audio_writer, audio_output)
    print(f"wrote {audio_output} ({audio_count} tensors)")

    print("conversion complete")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="OmniVoice Hugging Face model directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    convert(parse_args())


if __name__ == "__main__":
    main()
