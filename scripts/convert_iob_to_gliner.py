#!/usr/bin/env python3
"""Convert L3Cube Marathi IOB NER files to GLiNER JSONL format."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Literal


Split = Literal["train", "valid", "test"]


LABEL_MAP = {
    "NEP": "person",
    "NEL": "location",
    "NEO": "organization",
    "NEM": "measure",
    "NETI": "time",
    "NED": "date",
    "ED": "designation",
}


@dataclasses.dataclass(frozen=True)
class SourceSpec:
    name: str
    root: Path
    files: dict[Split, str]


@dataclasses.dataclass
class TokenRow:
    token: str
    tag: str
    sentence_id: str


@dataclasses.dataclass
class ConvertStats:
    examples: int = 0
    tokens: int = 0
    entities: int = 0
    labels: Counter[str] = dataclasses.field(default_factory=Counter)


SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        name="l3cube-mahaner",
        root=Path("L3Cube-MahaNER/IOB"),
        files={
            "train": "train_iob.txt",
            "valid": "valid_iob.txt",
            "test": "test_iob.txt",
        },
    ),
    SourceSpec(
        name="l3cube-mahasocialner",
        root=Path("L3Cube-MahaSocialNER/IOB"),
        files={
            "train": "train_iob.txt",
            "valid": "valid_iob.txt",
            "test": "test_iob.txt",
        },
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw/upstream/l3cube-marathinlp"),
        help="Root directory containing the downloaded L3Cube MarathiNLP checkout.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/gliner"),
        help="Directory where GLiNER JSONL files and metadata are written.",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Source name to convert. May be repeated. Defaults to all configured sources.",
    )
    parser.add_argument(
        "--split",
        action="append",
        choices=("train", "valid", "test"),
        default=[],
        help="Split to convert. May be repeated. Defaults to train, valid, and test.",
    )
    parser.add_argument(
        "--include-provenance",
        action="store_true",
        help="Include source metadata in each JSONL example.",
    )
    parser.add_argument(
        "--strict-iob",
        action="store_true",
        help="Fail on malformed I-* tags instead of repairing them as new spans.",
    )
    parser.add_argument(
        "--indent",
        action="store_true",
        help="Pretty-print JSONL records. Useful for debugging, not recommended for training.",
    )
    return parser.parse_args()


def selected_sources(names: Iterable[str]) -> list[SourceSpec]:
    source_by_name = {source.name: source for source in SOURCES}
    names = list(names)
    if not names:
        return list(SOURCES)

    unknown = sorted(set(names) - set(source_by_name))
    if unknown:
        available = ", ".join(sorted(source_by_name))
        raise SystemExit(f"Unknown source(s): {', '.join(unknown)}. Available: {available}")

    return [source_by_name[name] for name in names]


def normalize_iob_tag(tag: str) -> tuple[str, str | None]:
    if tag == "O":
        return "O", None

    if tag.startswith(("B-", "I-")):
        prefix, raw_label = tag.split("-", 1)
    else:
        prefix, raw_label = tag[:1], tag[1:]

    if prefix not in {"B", "I"}:
        raise ValueError(f"unsupported IOB tag prefix in {tag!r}")

    label = LABEL_MAP.get(raw_label)
    if label is None:
        known = ", ".join(sorted(LABEL_MAP))
        raise ValueError(f"unknown raw label {raw_label!r} in tag {tag!r}; known labels: {known}")

    return prefix, label


def read_iob(path: Path) -> list[tuple[str, list[TokenRow]]]:
    if not path.is_file():
        raise SystemExit(f"Missing input file: {path}")

    grouped: list[tuple[str, list[TokenRow]]] = []
    current_id: str | None = None
    current_rows: list[TokenRow] = []

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = {"words", "labels", "sentence_id"}
        if not reader.fieldnames or set(reader.fieldnames) != expected:
            raise SystemExit(f"{path} must contain TSV columns: words, labels, sentence_id")

        for line_number, row in enumerate(reader, start=2):
            token = row["words"].strip()
            tag = row["labels"].strip()
            sentence_id = row["sentence_id"].strip()
            if not token:
                raise SystemExit(f"{path}:{line_number}: empty token")
            if not tag:
                raise SystemExit(f"{path}:{line_number}: empty tag")
            if not sentence_id:
                raise SystemExit(f"{path}:{line_number}: empty sentence_id")

            if current_id is None:
                current_id = sentence_id
            elif sentence_id != current_id:
                grouped.append((current_id, current_rows))
                current_id = sentence_id
                current_rows = []

            current_rows.append(TokenRow(token=token, tag=tag, sentence_id=sentence_id))

    if current_id is not None:
        grouped.append((current_id, current_rows))

    return grouped


def rows_to_gliner_ner(
    rows: list[TokenRow],
    source_name: str,
    sentence_id: str,
    strict_iob: bool,
) -> list[list[int | str]]:
    spans: list[list[int | str]] = []
    active_start: int | None = None
    active_label: str | None = None

    def close_span(end_index: int) -> None:
        nonlocal active_start, active_label
        if active_start is not None and active_label is not None:
            spans.append([active_start, end_index, active_label])
        active_start = None
        active_label = None

    for token_index, row in enumerate(rows):
        try:
            prefix, label = normalize_iob_tag(row.tag)
        except ValueError as exc:
            raise SystemExit(f"{source_name}:{sentence_id}:{token_index}: {exc}") from exc

        if prefix == "O":
            close_span(token_index - 1)
            continue

        if prefix == "B":
            close_span(token_index - 1)
            active_start = token_index
            active_label = label
            continue

        if prefix == "I" and active_start is not None and active_label == label:
            continue

        if strict_iob:
            raise SystemExit(
                f"{source_name}:{sentence_id}:{token_index}: malformed {row.tag!r}; "
                "rerun without --strict-iob to treat it as a new span"
            )

        close_span(token_index - 1)
        active_start = token_index
        active_label = label
        continue

    close_span(len(rows) - 1)
    return spans


def make_example(
    source_name: str,
    split: Split,
    sentence_id: str,
    rows: list[TokenRow],
    include_provenance: bool,
    strict_iob: bool,
) -> dict[str, Any]:
    example: dict[str, Any] = {
        "tokenized_text": [row.token for row in rows],
        "ner": rows_to_gliner_ner(rows, source_name, sentence_id, strict_iob),
    }
    if include_provenance:
        example["metadata"] = {
            "source": source_name,
            "source_split": split,
            "source_sentence_id": sentence_id,
        }
    return example


def validate_example(example: dict[str, Any], source_name: str, sentence_id: str) -> None:
    tokens = example["tokenized_text"]
    if not isinstance(tokens, list) or not tokens:
        raise SystemExit(f"{source_name}:{sentence_id}: example has no tokens")

    seen_spans: set[tuple[int, int, str]] = set()
    for span in example["ner"]:
        if len(span) != 3:
            raise SystemExit(f"{source_name}:{sentence_id}: invalid span shape {span!r}")
        start, end, label = span
        if not isinstance(start, int) or not isinstance(end, int) or not isinstance(label, str):
            raise SystemExit(f"{source_name}:{sentence_id}: invalid span types {span!r}")
        if start < 0 or end < start or end >= len(tokens):
            raise SystemExit(
                f"{source_name}:{sentence_id}: span {span!r} outside token range 0..{len(tokens) - 1}"
            )
        key = (start, end, label)
        if key in seen_spans:
            raise SystemExit(f"{source_name}:{sentence_id}: duplicate span {span!r}")
        seen_spans.add(key)


def write_jsonl(path: Path, examples: list[dict[str, Any]], indent: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            if indent:
                handle.write(json.dumps(example, ensure_ascii=False, indent=2))
            else:
                handle.write(json.dumps(example, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def update_stats(stats: ConvertStats, example: dict[str, Any]) -> None:
    stats.examples += 1
    stats.tokens += len(example["tokenized_text"])
    stats.entities += len(example["ner"])
    stats.labels.update(span[2] for span in example["ner"])


def convert_source_split(
    raw_root: Path,
    source: SourceSpec,
    split: Split,
    include_provenance: bool,
    strict_iob: bool,
) -> tuple[list[dict[str, Any]], ConvertStats]:
    input_path = raw_root / source.root / source.files[split]
    examples: list[dict[str, Any]] = []
    stats = ConvertStats()

    for sentence_id, rows in read_iob(input_path):
        example = make_example(
            source_name=source.name,
            split=split,
            sentence_id=sentence_id,
            rows=rows,
            include_provenance=include_provenance,
            strict_iob=strict_iob,
        )
        validate_example(example, source.name, sentence_id)
        update_stats(stats, example)
        examples.append(example)

    return examples, stats


def stats_to_json(stats: ConvertStats) -> dict[str, Any]:
    return {
        "examples": stats.examples,
        "tokens": stats.tokens,
        "entities": stats.entities,
        "labels": dict(sorted(stats.labels.items())),
    }


def main() -> int:
    args = parse_args()
    sources = selected_sources(args.source)
    splits: list[Split] = args.split or ["train", "valid", "test"]
    metadata: dict[str, Any] = {
        "format": "gliner-jsonl",
        "label_map": LABEL_MAP,
        "include_provenance": args.include_provenance,
        "sources": {},
        "splits": {},
    }

    for split in splits:
        split_examples: list[dict[str, Any]] = []
        split_stats = ConvertStats()
        metadata["splits"][split] = {"sources": {}}

        for source in sources:
            examples, stats = convert_source_split(
                raw_root=args.raw_root,
                source=source,
                split=split,
                include_provenance=args.include_provenance,
                strict_iob=args.strict_iob,
            )
            split_examples.extend(examples)
            split_stats.examples += stats.examples
            split_stats.tokens += stats.tokens
            split_stats.entities += stats.entities
            split_stats.labels.update(stats.labels)
            metadata["sources"].setdefault(source.name, str(source.root))
            metadata["splits"][split]["sources"][source.name] = stats_to_json(stats)

        output_path = args.output_dir / f"{split}.jsonl"
        write_jsonl(output_path, split_examples, args.indent)
        metadata["splits"][split]["total"] = stats_to_json(split_stats)
        print(
            f"Wrote {output_path} "
            f"({split_stats.examples} examples, {split_stats.entities} entities)"
        )

    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
