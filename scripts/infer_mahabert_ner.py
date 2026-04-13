#!/usr/bin/env python3
"""Run Marathi NER inference with an L3Cube MahaBERT token-classification model."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


DEFAULT_TEXT = (
    "सीमा पाटील सोमवारी पुण्यात मायक्रोसॉफ्टच्या कार्यालयात गेली. "
    "तिने महाराष्ट्र बँकेतील अमित देशमुख यांची भेट घेतली."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="l3cube-pune/marathi-ner",
        help=(
            "Hugging Face model id or local checkpoint path. Examples: "
            "l3cube-pune/marathi-ner, l3cube-pune/marathi-ner-iob, "
            "l3cube-pune/marathi-social-ner, l3cube-pune/marathi-mixed-ner"
        ),
    )
    parser.add_argument(
        "--text",
        default=DEFAULT_TEXT,
        help="Marathi text to run NER on.",
    )
    parser.add_argument(
        "--aggregation",
        default="simple",
        choices=("none", "simple", "first", "average", "max"),
        help="Transformers token aggregation strategy for word/entity spans.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="Pipeline device: -1 for CPU, 0 for first CUDA GPU.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw pipeline output as JSON.",
    )
    return parser.parse_args()


def load_pipeline(model_name: str, aggregation: str, device: int) -> Any:
    try:
        from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
    except ImportError:
        print(
            "Missing dependency: install Transformers with `pip install transformers torch`.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    kwargs: dict[str, Any] = {
        "task": "token-classification",
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
    }
    if aggregation != "none":
        kwargs["aggregation_strategy"] = aggregation

    return pipeline(**kwargs)


def normalize_entity(entity: dict[str, Any]) -> dict[str, Any]:
    label = entity.get("entity_group") or entity.get("entity")
    return {
        "text": entity.get("word", ""),
        "label": label,
        "score": entity.get("score"),
        "start": entity.get("start"),
        "end": entity.get("end"),
    }


def main() -> int:
    args = parse_args()
    ner = load_pipeline(args.model, args.aggregation, args.device)
    entities = ner(args.text)

    if args.json:
        print(json.dumps(entities, indent=2, ensure_ascii=False))
        return 0

    print(f"Model: {args.model}")
    print(f"Text: {args.text}")
    print()

    if not entities:
        print("No entities found.")
        return 0

    for entity in entities:
        item = normalize_entity(entity)
        score = item["score"]
        score_text = f"{score:.3f}" if isinstance(score, float) else "n/a"
        print(f"{item['text']}\t{item['label']}\t{score_text}\t[{item['start']}, {item['end']}]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
