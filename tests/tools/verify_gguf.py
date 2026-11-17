#!/usr/bin/env python3
"""Verify converted OmniVoice GGUF tensors against source safetensors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFReader
from safetensors import safe_open

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.convert_hf_to_gguf import (  # noqa: E402
    SKIP_AUDIO_TENSORS,
    SKIP_LM_TENSORS,
    map_audio_tensor_name,
    map_lm_tensor_name,
)


FUSED_POSITIONAL_CONV = "s.enc.pce.weight"


def verify_fused_positional_conv(source: Path, gguf_path: Path) -> None:
    original0 = (
        "semantic_model.encoder.pos_conv_embed.conv."
        "parametrizations.weight.original0"
    )
    original1 = (
        "semantic_model.encoder.pos_conv_embed.conv."
        "parametrizations.weight.original1"
    )
    reader = GGUFReader(str(gguf_path))
    converted = next(
        (tensor for tensor in reader.tensors
         if tensor.name == FUSED_POSITIONAL_CONV),
        None,
    )
    if converted is None:
        raise RuntimeError(f"{gguf_path}: missing {FUSED_POSITIONAL_CONV}")

    with safe_open(str(source), framework="pt", device="cpu") as tensors:
        weight_g = tensors.get_tensor(original0).float()
        weight_v = tensors.get_tensor(original1).float()
    expected = torch._weight_norm(
        weight_v,
        weight_g,
        2,
    ).contiguous().numpy()

    actual = np.asarray(converted.data)
    if actual.shape != expected.shape:
        raise RuntimeError(
            f"{FUSED_POSITIONAL_CONV}: shape {actual.shape} "
            f"!= {expected.shape}")
    if actual.dtype != expected.dtype:
        raise RuntimeError(
            f"{FUSED_POSITIONAL_CONV}: dtype {actual.dtype} "
            f"!= {expected.dtype}")
    if not np.array_equal(actual, expected):
        raise RuntimeError(
            f"{FUSED_POSITIONAL_CONV}: fused values differ from source")


def verify_file(
    source: Path,
    gguf_path: Path,
    map_name,
    skip_names: set[str] | None = None,
    extra_names: set[str] | None = None,
) -> int:
    skip_names = skip_names or set()
    extra_names = extra_names or set()
    reader = GGUFReader(str(gguf_path))
    gguf_tensors = {tensor.name: tensor for tensor in reader.tensors}
    checked = 0

    with safe_open(str(source), framework="pt", device="cpu") as source_tensors:
        expected_names = [
            name for name in source_tensors.keys()
            if name not in skip_names
        ]
        mapped_names = [map_name(name) for name in expected_names]
        all_expected_names = set(mapped_names) | extra_names
        if len(gguf_tensors) != len(all_expected_names):
            raise RuntimeError(
                f"{gguf_path}: expected {len(all_expected_names)} tensors, "
                f"found {len(gguf_tensors)}")
        if set(gguf_tensors) != all_expected_names:
            missing = sorted(all_expected_names - set(gguf_tensors))
            extra = sorted(set(gguf_tensors) - all_expected_names)
            raise RuntimeError(
                f"{gguf_path}: tensor-name mismatch; "
                f"missing={missing[:5]}, extra={extra[:5]}")

        for source_name, destination_name in zip(expected_names, mapped_names):
            source_tensor = (
                source_tensors.get_tensor(source_name)
                .detach()
                .cpu()
                .contiguous()
            )
            converted = np.asarray(gguf_tensors[destination_name].data)
            if tuple(converted.shape) != tuple(source_tensor.shape):
                raise RuntimeError(
                    f"{destination_name}: shape "
                    f"{converted.shape} != {tuple(source_tensor.shape)}")
            source_array = source_tensor.numpy()
            if converted.dtype != source_array.dtype:
                raise RuntimeError(
                    f"{destination_name}: dtype "
                    f"{converted.dtype} != {source_array.dtype}")
            if not np.array_equal(converted, source_array):
                raise RuntimeError(
                    f"{destination_name}: converted values differ from source")
            checked += 1
    return checked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--lm-gguf", type=Path, required=True)
    parser.add_argument("--audio-gguf", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lm_count = verify_file(
        args.model_dir / "model.safetensors",
        args.lm_gguf,
        map_lm_tensor_name,
        SKIP_LM_TENSORS,
    )
    audio_count = verify_file(
        args.model_dir / "audio_tokenizer" / "model.safetensors",
        args.audio_gguf,
        map_audio_tensor_name,
        SKIP_AUDIO_TENSORS,
        {FUSED_POSITIONAL_CONV},
    )
    verify_fused_positional_conv(
        args.model_dir / "audio_tokenizer" / "model.safetensors",
        args.audio_gguf,
    )
    print(f"verified LM tensors: {lm_count}")
    print(f"verified audio tensors: {audio_count}")


if __name__ == "__main__":
    main()
