#!/usr/bin/env python3
"""Generate RuleDurationEstimator golden cases."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location(
        "omnivoice_duration_reference",
        args.duration_module,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load reference duration module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    estimator = module.RuleDurationEstimator()
    reference_text = "Hello, world."
    texts = [
        "Hello, world.",
        "今天，中国人民正在推进现代化建设。",
        "One, two, three; four?",
        "Привет, мир!",
        "مرحبا بالعالم",
        "こんにちは世界",
        "안녕하세요 세계",
        "नमस्ते दुनिया",
        "12345",
        "A",
    ]
    cases = []
    for text in texts:
        cases.append({
            "text": text,
            "weight": estimator.calculate_total_weight(text),
            "duration_ref5": estimator.estimate_duration(
                text,
                reference_text,
                5.0,
            ),
            "duration_ref50": estimator.estimate_duration(
                text,
                reference_text,
                50.0,
            ),
        })

    payload = {
        "reference_text": reference_text,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote duration cases: {args.output}")


if __name__ == "__main__":
    main()
