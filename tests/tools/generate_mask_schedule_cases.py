#!/usr/bin/env python3
"""Generate diffusion mask schedule golden cases."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    configurations = [
        (5, 8, 4, 0.1),
        (25, 8, 32, 0.1),
        (221, 8, 32, 0.1),
        (7, 3, 1, 1.0),
        (100, 8, 10, 2.0),
    ]
    cases = []
    for target_frames, codebooks, num_steps, t_shift in configurations:
        timesteps = torch.linspace(
            0.0,
            1.0,
            num_steps + 1,
            dtype=torch.float32,
        )
        timesteps = t_shift * timesteps / (1 + (t_shift - 1) * timesteps)
        total = target_frames * codebooks
        remaining = total
        counts = []
        for step in range(num_steps):
            count = remaining
            if step != num_steps - 1:
                delta = float(timesteps[step + 1] - timesteps[step])
                count = min(remaining, math.ceil(total * delta))
            counts.append(count)
            remaining -= count
        cases.append({
            "target_frames": target_frames,
            "codebooks": codebooks,
            "num_steps": num_steps,
            "t_shift": t_shift,
            "counts": counts,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote mask schedule cases: {args.output}")


if __name__ == "__main__":
    main()
