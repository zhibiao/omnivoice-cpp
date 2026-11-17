#!/usr/bin/env python3
"""Generate an OmniVoice LM reference fixture from the HF checkpoint."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


def rms_norm(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    variance = input_tensor.pow(2).mean(dim=-1, keepdim=True)
    return input_tensor * torch.rsqrt(variance + epsilon) * weight


def rotate_half(input_tensor: torch.Tensor) -> torch.Tensor:
    half = input_tensor.shape[-1] // 2
    return torch.cat(
        (-input_tensor[..., half:], input_tensor[..., :half]),
        dim=-1,
    )


def rotary(
    input_tensor: torch.Tensor,
    positions: torch.Tensor,
    theta: float,
) -> torch.Tensor:
    head_dim = input_tensor.shape[-1]
    inverse_frequency = 1.0 / (
        theta ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32)
            / head_dim
        )
    )
    frequencies = torch.outer(
        positions.to(torch.float32),
        inverse_frequency,
    )
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    cosine = embedding.cos().unsqueeze(1)
    sine = embedding.sin().unsqueeze(1)
    return input_tensor * cosine + rotate_half(input_tensor) * sine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model_path = args.model_dir / "model.safetensors"
    sequence = 6
    target_start = 3
    target_frames = 3
    audio_vocab = 1025
    codebooks = 8
    hidden_size = 1024
    heads = 16
    kv_heads = 8
    head_dim = 128
    intermediate = 3072
    layers = 28
    epsilon = 1e-6
    rope_theta = 1000000.0
    text_vocab = 151676

    input_ids = torch.zeros(codebooks, sequence, dtype=torch.int32)
    text_ids = [151674, 9707, 1879]
    input_ids[:, :target_start] = torch.tensor(
        text_ids,
        dtype=torch.int32,
    )
    for codebook in range(codebooks):
        for frame in range(target_frames):
            input_ids[
                codebook,
                target_start + frame,
            ] = (codebook * 37 + frame * 11 + 3) % 1024
    audio_mask = torch.tensor(
        [0.0] * target_start + [1.0] * target_frames,
        dtype=torch.float32,
    )
    positions = torch.arange(sequence, dtype=torch.int32)

    with safe_open(
        str(model_path),
        framework="pt",
        device="cpu",
    ) as source:
        def get(name: str) -> torch.Tensor:
            return source.get_tensor(name).detach().cpu().contiguous()

        token_embeddings = get("llm.embed_tokens.weight")
        audio_embeddings = get("audio_embeddings.weight")
        audio_heads = get("audio_heads.weight")

        text_values = token_embeddings[input_ids[0].long()]
        audio_values = torch.zeros(sequence, hidden_size)
        audio_input_ids = torch.where(
            audio_mask.bool().unsqueeze(0),
            input_ids,
            torch.zeros_like(input_ids),
        )
        for codebook in range(codebooks):
            audio_values += audio_embeddings[
                audio_input_ids[codebook].long() + codebook * audio_vocab
            ]
        mask = audio_mask.unsqueeze(-1)
        hidden = text_values * (1.0 - mask) + audio_values * mask

        with torch.no_grad():
            for layer in range(layers):
                prefix = f"llm.layers.{layer}."
                residual = hidden
                normalized = rms_norm(
                    hidden,
                    get(prefix + "input_layernorm.weight"),
                    epsilon,
                )
                query = F.linear(
                    normalized,
                    get(prefix + "self_attn.q_proj.weight"),
                ).reshape(sequence, heads, head_dim)
                key = F.linear(
                    normalized,
                    get(prefix + "self_attn.k_proj.weight"),
                ).reshape(sequence, kv_heads, head_dim)
                value = F.linear(
                    normalized,
                    get(prefix + "self_attn.v_proj.weight"),
                ).reshape(sequence, kv_heads, head_dim)
                query = rms_norm(
                    query,
                    get(prefix + "self_attn.q_norm.weight"),
                    epsilon,
                )
                key = rms_norm(
                    key,
                    get(prefix + "self_attn.k_norm.weight"),
                    epsilon,
                )
                query = rotary(query, positions, rope_theta)
                key = rotary(key, positions, rope_theta)

                key = key.repeat_interleave(
                    heads // kv_heads,
                    dim=1,
                )
                value = value.repeat_interleave(
                    heads // kv_heads,
                    dim=1,
                )
                scores = torch.einsum(
                    "qhd,khd->hqk",
                    query,
                    key,
                ) / (head_dim ** 0.5)
                probabilities = torch.softmax(scores, dim=-1)
                context = torch.einsum(
                    "hqk,khd->qhd",
                    probabilities,
                    value,
                ).reshape(sequence, heads * head_dim)
                attention_output = F.linear(
                    context,
                    get(prefix + "self_attn.o_proj.weight"),
                )
                hidden = residual + attention_output

                residual = hidden
                normalized = rms_norm(
                    hidden,
                    get(prefix + "post_attention_layernorm.weight"),
                    epsilon,
                )
                gate = F.linear(
                    normalized,
                    get(prefix + "mlp.gate_proj.weight"),
                )
                up = F.linear(
                    normalized,
                    get(prefix + "mlp.up_proj.weight"),
                )
                hidden = residual + F.linear(
                    F.silu(gate) * up,
                    get(prefix + "mlp.down_proj.weight"),
                )

            hidden = rms_norm(
                hidden,
                get("llm.norm.weight"),
                epsilon,
            )
            target_hidden = hidden[target_start:target_start + target_frames]
            logits = F.linear(target_hidden, audio_heads)
            logits = logits.reshape(
                target_frames,
                codebooks,
                audio_vocab,
            )
            logits = logits.permute(1, 0, 2).contiguous()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as file:
        file.write(
            struct.pack(
                "<8s7i",
                b"OVLM0001",
                sequence,
                codebooks,
                target_start,
                target_frames,
                hidden_size,
                audio_vocab,
                4,
            )
        )
        input_ids.numpy().astype(np.int32).tofile(file)
        audio_mask.numpy().astype(np.float32).tofile(file)
        hidden.numpy().astype(np.float32).tofile(file)
        logits.numpy().astype(np.float32).tofile(file)
    print(f"wrote OmniVoice LM fixture: {args.output}")


if __name__ == "__main__":
    main()
