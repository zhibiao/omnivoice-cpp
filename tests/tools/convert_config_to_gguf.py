#!/usr/bin/env python3
"""Build OmniVoice config-only GGUF fixtures from model JSON files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from gguf import GGUFWriter

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.convert_hf_to_gguf import build_tokenizer


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def add_lm_metadata(
    writer: GGUFWriter,
    model_config: dict[str, Any],
    tokenizer_json: dict[str, Any],
    tokenizer_config: dict[str, Any],
    name: str = "OmniVoice LM config fixture",
) -> None:
    llm = model_config["llm_config"]
    rope = llm.get("rope_parameters", {})
    tokens, merges, added_names, added_ids = build_tokenizer(tokenizer_json)
    token_to_id = {token: token_id for token_id, token in enumerate(tokens)}

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
    name: str = "OmniVoice Higgs tokenizer config fixture",
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
    writer.add_uint32("omnivoice.audio_tokenizer.sample_rate", sample_rate)
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
def write_gguf(writer: GGUFWriter, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def write_lm_config(
    output: Path,
    model_config: dict[str, Any],
    tokenizer_json: dict[str, Any],
    tokenizer_config: dict[str, Any],
) -> None:
    writer = GGUFWriter(str(output), "omnivoice")
    add_lm_metadata(
        writer,
        model_config,
        tokenizer_json,
        tokenizer_config,
    )
    write_gguf(writer, output)
    print(f"wrote LM config: {output}")


def write_audio_config(
    output: Path,
    audio_config: dict[str, Any],
    preprocessor_config: dict[str, Any],
) -> None:
    writer = GGUFWriter(str(output), "omnivoice")
    add_audio_metadata(writer, audio_config, preprocessor_config)
    write_gguf(writer, output)
    print(f"wrote audio config: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--tokenizer-config", type=Path, required=True)
    parser.add_argument("--audio-config", type=Path, required=True)
    parser.add_argument("--audio-preprocessor-config", type=Path, required=True)
    parser.add_argument("--output-lm", type=Path, required=True)
    parser.add_argument("--output-audio", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_lm_config(
        args.output_lm,
        load_json(args.model_config),
        load_json(args.tokenizer_json),
        load_json(args.tokenizer_config),
    )
    write_audio_config(
        args.output_audio,
        load_json(args.audio_config),
        load_json(args.audio_preprocessor_config),
    )


if __name__ == "__main__":
    main()
