#!/usr/bin/env python3
"""Generate a deterministic Qwen3 block reference fixture."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


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


def linear(weight: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
    return F.linear(input_tensor, weight)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(12345)
    hidden = 64
    heads = 4
    kv_heads = 2
    head_dim = 16
    intermediate = 96
    sequence = 5
    epsilon = 1e-6
    rope_theta = 10000.0

    def matrix(out_features: int, in_features: int) -> torch.Tensor:
        return (
            torch.rand(out_features, in_features, dtype=torch.float32)
            - 0.5
        ) * 0.2

    def norm(shape: tuple[int, ...]) -> torch.Tensor:
        return 1.0 + (
            torch.rand(shape, dtype=torch.float32) - 0.5
        ) * 0.04

    input_norm = norm((hidden,))
    post_norm = norm((hidden,))
    q_proj = matrix(heads * head_dim, hidden)
    k_proj = matrix(kv_heads * head_dim, hidden)
    v_proj = matrix(kv_heads * head_dim, hidden)
    o_proj = matrix(hidden, heads * head_dim)
    q_norm = norm((head_dim,))
    k_norm = norm((head_dim,))
    gate = matrix(intermediate, hidden)
    up = matrix(intermediate, hidden)
    down = matrix(hidden, intermediate)
    input_value = torch.randn(sequence, hidden) * 0.2
    positions = torch.arange(sequence, dtype=torch.int32)

    mask = torch.zeros(sequence, sequence, dtype=torch.float32)
    mask[0, -1] = float("-inf")

    hidden_value = rms_norm(input_value, input_norm, epsilon)
    query = linear(q_proj, hidden_value).reshape(
        sequence, heads, head_dim)
    key = linear(k_proj, hidden_value).reshape(
        sequence, kv_heads, head_dim)
    value = linear(v_proj, hidden_value).reshape(
        sequence, kv_heads, head_dim)
    query = rms_norm(query, q_norm, epsilon)
    key = rms_norm(key, k_norm, epsilon)
    query = rotary(query, positions, rope_theta)
    key = rotary(key, positions, rope_theta)

    key = key.repeat_interleave(heads // kv_heads, dim=1)
    value = value.repeat_interleave(heads // kv_heads, dim=1)
    scores = torch.einsum("qhd,khd->hqk", query, key)
    scores = scores / (head_dim ** 0.5)
    scores = scores + mask.unsqueeze(0)
    probabilities = torch.softmax(scores, dim=-1)
    context = torch.einsum("hqk,khd->qhd", probabilities, value)
    context = context.reshape(sequence, heads * head_dim)
    attention_output = linear(o_proj, context)
    hidden_value = input_value + attention_output

    residual = hidden_value
    hidden_value = rms_norm(hidden_value, post_norm, epsilon)
    hidden_value = F.silu(linear(gate, hidden_value)) * linear(
        up,
        hidden_value,
    )
    hidden_value = linear(down, hidden_value)
    output = residual + hidden_value

    arrays = [
        input_value.numpy(),
        q_proj.numpy(),
        k_proj.numpy(),
        v_proj.numpy(),
        o_proj.numpy(),
        q_norm.numpy(),
        k_norm.numpy(),
        input_norm.numpy(),
        post_norm.numpy(),
        gate.numpy(),
        up.numpy(),
        down.numpy(),
        output.detach().numpy(),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as file:
        file.write(
            struct.pack(
                "<8s7i",
                b"QVBLK01\0",
                hidden,
                heads,
                kv_heads,
                head_dim,
                intermediate,
                sequence,
                len(arrays),
            )
        )
        for array in arrays:
            np.asarray(array, dtype=np.float32).tofile(file)
        positions.numpy().tofile(file)
        mask.numpy().tofile(file)
    print(f"wrote Qwen3 block fixture: {args.output}")


if __name__ == "__main__":
    main()
