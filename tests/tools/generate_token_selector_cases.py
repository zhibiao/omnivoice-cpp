#!/usr/bin/env python3
"""Generate token selector golden data from the reference scoring formula."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as functional


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    codebooks = 2
    frames = 4
    vocab = 10
    mask_id = 9
    top_k = 3
    guidance_scale = 2.0
    layer_penalty_factor = 0.75

    conditional = torch.empty(codebooks, frames, vocab)
    unconditional = torch.empty_like(conditional)
    for codebook in range(codebooks):
        for frame in range(frames):
            for token in range(vocab):
                index = codebook * frames * vocab + frame * vocab + token
                conditional[codebook, frame, token] = (
                    math.sin(index * 0.17) * 0.35
                )
                unconditional[codebook, frame, token] = (
                    math.cos(index * 0.11) * 0.25
                )
            dominant = (codebook * 3 + frame * 2 + 1) % (vocab - 1)
            conditional[codebook, frame, dominant] += 2.0

    current = [mask_id] * (codebooks * frames)
    current[0] = 4
    current[5] = 2

    conditional_log_probs = functional.log_softmax(conditional, dim=-1)
    unconditional_log_probs = functional.log_softmax(unconditional, dim=-1)
    if guidance_scale != 0.0:
        guided_values = (
            conditional_log_probs
            + guidance_scale
            * (conditional_log_probs - unconditional_log_probs)
        )
        log_probs = functional.log_softmax(guided_values, dim=-1)
    else:
        log_probs = conditional_log_probs
    log_probs[..., mask_id] = -torch.inf

    predictions = log_probs.argmax(dim=-1)
    confidence = log_probs.max(dim=-1).values
    flat_current = torch.tensor(current, dtype=torch.int64)
    flat_predictions = predictions.reshape(-1)
    flat_confidence = confidence.reshape(-1)
    scores = flat_confidence.clone()
    for cell in range(codebooks * frames):
        codebook = cell // frames
        scores[cell] -= codebook * layer_penalty_factor
        if flat_current[cell].item() != mask_id:
            scores[cell] = -torch.inf
    selected_indices = sorted(
        range(codebooks * frames),
        key=lambda cell: (-float(scores[cell]), cell),
    )[:top_k]
    selected_tokens = [
        int(flat_predictions[cell]) for cell in selected_indices
    ]

    payload = {
        "guidance_scale": guidance_scale,
        "layer_penalty_factor": layer_penalty_factor,
        "mask_id": mask_id,
        "top_k": top_k,
        "cond": conditional.tolist(),
        "uncond": unconditional.tolist(),
        "current_tokens": current,
        "predictions": predictions.tolist(),
        "confidence": confidence.tolist(),
        "selected_indices": selected_indices,
        "selected_tokens": selected_tokens,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote token selector cases: {args.output}")


if __name__ == "__main__":
    main()
