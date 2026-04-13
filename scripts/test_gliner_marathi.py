#!/usr/bin/env python3
"""Run a quick GLiNER NER smoke test on Marathi text."""

from __future__ import annotations

import argparse
import json
import sys


DEFAULT_TEXT = (
    "इराणसोबतची चर्चा फिस्कटल्यानंतर डोनाल्ड ट्रम्प पुन्हा आक्रमक झाले आहेत. होर्मुझ सामुद्रधुनीच बंद करण्याची धमकी त्यांनी दिली आहे."
)

DEFAULT_LABELS = ("person", "organization", "location", "date")


def parse_labels(value: str) -> list[str]:
    labels = [label.strip() for label in value.split(",") if label.strip()]
    if not labels:
        raise argparse.ArgumentTypeError("provide at least one comma-separated label")
    return labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="urchade/gliner_multi-v2.1",
        help="GLiNER model name or local checkpoint path.",
    )
    parser.add_argument(
        "--text",
        default=DEFAULT_TEXT,
        help="Marathi text to run NER on.",
    )
    parser.add_argument(
        "--labels",
        type=parse_labels,
        default=list(DEFAULT_LABELS),
        help="Comma-separated entity labels to extract.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.15,
        help="Prediction confidence threshold. Lower values return more entities.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full entity dictionaries as JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from gliner import GLiNER
    except ImportError:
        print(
            "Missing dependency: install GLiNER with `pip install gliner`.",
            file=sys.stderr,
        )
        return 1

    model = GLiNER.from_pretrained(args.model)
    entities = model.predict_entities(args.text, args.labels, threshold=args.threshold)

    if args.json:
        print(json.dumps(entities, indent=2, ensure_ascii=False))
        return 0

    print(f"Text: {args.text}")
    print(f"Labels: {', '.join(args.labels)}")
    print()

    if not entities:
        print("No entities found. Try lowering --threshold or changing --labels.")
        return 0

    for entity in entities:
        text = entity.get("text", "")
        label = entity.get("label", "")
        score = entity.get("score")
        start = entity.get("start")
        end = entity.get("end")
        score_text = f"{score:.3f}" if isinstance(score, float) else "n/a"
        print(f"{text}\t{label}\t{score_text}\t[{start}, {end}]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
