#!/usr/bin/env python3
"""Generate a Higgs audio decoder reference fixture."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


def snake(input_tensor: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    return input_tensor + (
        (alpha + 1e-9).reciprocal()
        * torch.sin(alpha * input_tensor).pow(2)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stages-output", type=Path, required=True)
    args = parser.parse_args()

    model_path = args.model_dir / "audio_tokenizer" / "model.safetensors"
    codebooks = 8
    frames = 5
    codes = torch.zeros(codebooks, frames, dtype=torch.long)
    for codebook in range(codebooks):
        for frame in range(frames):
            codes[codebook, frame] = (
                codebook * 97 + frame * 31 + 7
            ) % 1024

    with safe_open(
        str(model_path),
        framework="pt",
        device="cpu",
    ) as source:
        def get(name: str) -> torch.Tensor:
            return source.get_tensor(name).detach().cpu().contiguous()

        stages: list[tuple[str, torch.Tensor]] = []
        quantized = torch.zeros(1024, frames)
        for codebook in range(codebooks):
            prefix = f"quantizer.quantizers.{codebook}."
            embed = F.embedding(
                codes[codebook],
                get(prefix + "codebook.embed"),
            ).transpose(0, 1)
            quantized += F.linear(
                embed.transpose(0, 1),
                get(prefix + "project_out.weight"),
                get(prefix + "project_out.bias"),
            ).transpose(0, 1)

        fc2_hidden = F.linear(
            quantized.transpose(0, 1),
            get("fc2.weight"),
            get("fc2.bias"),
        ).transpose(0, 1)
        stages.append(("quantized", quantized.detach().clone()))
        stages.append(("fc2", fc2_hidden.detach().clone()))
        hidden = fc2_hidden.unsqueeze(0)

        hidden = F.conv1d(
            hidden,
            get("acoustic_decoder.conv1.weight"),
            get("acoustic_decoder.conv1.bias"),
            padding=3,
        )
        stages.append(("conv1", hidden.squeeze(0).detach().clone()))

        for block in range(5):
            prefix = f"acoustic_decoder.block.{block}."
            hidden = snake(
                hidden,
                get(prefix + "snake1.alpha"),
            )
            stride = int(
                get(prefix + "conv_t1.weight").shape[-1] // 2
            )
            hidden = F.conv_transpose1d(
                hidden,
                get(prefix + "conv_t1.weight"),
                get(prefix + "conv_t1.bias"),
                stride=stride,
                padding=(stride + 1) // 2,
                output_padding=stride % 2,
            )

            for unit in range(1, 4):
                unit_prefix = prefix + f"res_unit{unit}."
                dilation = (1, 3, 9)[unit - 1]
                residual = hidden
                hidden = snake(
                    hidden,
                    get(unit_prefix + "snake1.alpha"),
                )
                hidden = F.conv1d(
                    hidden,
                    get(unit_prefix + "conv1.weight"),
                    get(unit_prefix + "conv1.bias"),
                    dilation=dilation,
                    padding=3 * dilation,
                )
                hidden = snake(
                    hidden,
                    get(unit_prefix + "snake2.alpha"),
                )
                hidden = F.conv1d(
                    hidden,
                    get(unit_prefix + "conv2.weight"),
                    get(unit_prefix + "conv2.bias"),
                )
                hidden = residual + hidden
            stages.append((
                f"block{block}",
                hidden.squeeze(0).detach().clone(),
            ))

        hidden = snake(
            hidden,
            get("acoustic_decoder.snake1.alpha"),
        )
        waveform = F.conv1d(
            hidden,
            get("acoustic_decoder.conv2.weight"),
            get("acoustic_decoder.conv2.bias"),
            padding=3,
        ).squeeze(0).squeeze(0)
        stages.append(("output", waveform.unsqueeze(0).detach().clone()))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as file:
        file.write(
            struct.pack(
                "<8s4i",
                b"HIGDEC01",
                codebooks,
                frames,
                int(waveform.numel()),
                1,
            )
        )
        codes.numpy().astype(np.int32).tofile(file)
        waveform.detach().numpy().astype(np.float32).tofile(file)
    with args.stages_output.open("wb") as file:
        file.write(struct.pack("<8sI", b"HIGSTG01", len(stages)))
        for name, values in stages:
            channels = int(values.shape[0])
            length = int(values.shape[1])
            file.write(
                struct.pack(
                    "<16sii",
                    name.encode("ascii"),
                    channels,
                    length,
                )
            )
            values.numpy().astype(np.float32).tofile(file)
    print(f"wrote Higgs decoder fixture: {args.output}")
    print(f"wrote Higgs decoder stages: {args.stages_output}")
    print(f"output samples: {waveform.numel()}")


if __name__ == "__main__":
    main()
