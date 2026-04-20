#!/usr/bin/env python3
"""Streamlit app for visualizing Marathi MahaBERT NER predictions."""

from __future__ import annotations

import html
import json
import re
import time
from typing import Any

import streamlit as st

from infer_mahabert_ner import DEFAULT_TEXT, load_pipeline, normalize_entity


LABEL_COLORS = [
    ("#0f766e", "#ccfbf1"),
    ("#b45309", "#fef3c7"),
    ("#1d4ed8", "#dbeafe"),
    ("#be123c", "#ffe4e6"),
    ("#047857", "#d1fae5"),
    ("#7c3aed", "#ede9fe"),
    ("#a21caf", "#fae8ff"),
    ("#4d7c0f", "#ecfccb"),
]


LABEL_ALIASES = {
    "nep": "Person",
    "per": "Person",
    "person": "Person",
    "nel": "Location",
    "loc": "Location",
    "location": "Location",
    "neo": "Organization",
    "org": "Organization",
    "organization": "Organization",
    "nem": "Measure",
    "measure": "Measure",
    "neti": "Time",
    "time": "Time",
    "ned": "Date",
    "date": "Date",
    "ed": "Designation",
    "designation": "Designation",
    "misc": "Misc",
    "other": "Other",
    "o": "Other",
}


def get_label_style(label: str) -> tuple[str, str]:
    index = sum(ord(char) for char in label) % len(LABEL_COLORS)
    return LABEL_COLORS[index]


@st.cache_resource(show_spinner=False)
def get_ner_pipeline(model_name: str, aggregation: str, device: int) -> Any:
    return load_pipeline(model_name, aggregation, device)


def get_token_count(tokenizer: Any, text: str) -> int:
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    return len(encoded["input_ids"])


def split_piece_by_token_limit(
    piece: str, start_offset: int, tokenizer: Any, max_tokens: int
) -> list[tuple[int, str]]:
    chunks = []
    cursor = 0
    while cursor < len(piece):
        low = cursor + 1
        high = len(piece)
        best = low
        while low <= high:
            mid = (low + high) // 2
            candidate = piece[cursor:mid]
            if get_token_count(tokenizer, candidate) <= max_tokens:
                best = mid
                low = mid + 1
            else:
                high = mid - 1

        chunks.append((start_offset + cursor, piece[cursor:best]))
        cursor = best

    return chunks


def split_text_by_token_limit(
    text: str, tokenizer: Any, max_tokens: int
) -> list[tuple[int, str]]:
    chunks: list[tuple[int, str]] = []
    current_start: int | None = None
    current_text = ""

    for match in re.finditer(r"\S+\s*", text):
        piece = match.group(0)
        piece_start = match.start()

        if not current_text:
            if get_token_count(tokenizer, piece) <= max_tokens:
                current_start = piece_start
                current_text = piece
            else:
                chunks.extend(
                    split_piece_by_token_limit(piece, piece_start, tokenizer, max_tokens)
                )
            continue

        candidate = current_text + piece
        if get_token_count(tokenizer, candidate) <= max_tokens:
            current_text = candidate
            continue

        if current_start is not None:
            chunks.append((current_start, current_text))

        if get_token_count(tokenizer, piece) <= max_tokens:
            current_start = piece_start
            current_text = piece
        else:
            current_start = None
            current_text = ""
            chunks.extend(
                split_piece_by_token_limit(piece, piece_start, tokenizer, max_tokens)
            )

    if current_text and current_start is not None:
        chunks.append((current_start, current_text))

    return chunks or [(0, text)]


def shift_entity_offsets(entity: dict[str, Any], offset: int) -> dict[str, Any]:
    shifted = dict(entity)
    if isinstance(shifted.get("start"), int):
        shifted["start"] += offset
    if isinstance(shifted.get("end"), int):
        shifted["end"] += offset
    return shifted


def run_ner_with_chunking(
    ner: Any, text: str, max_tokens: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    chunks = split_text_by_token_limit(text, ner.tokenizer, max_tokens)
    raw_entities: list[dict[str, Any]] = []
    chunk_details: list[dict[str, Any]] = []

    for index, (start_offset, chunk_text) in enumerate(chunks, start=1):
        if not chunk_text.strip():
            continue
        token_count = get_token_count(ner.tokenizer, chunk_text)
        started_at = time.perf_counter()
        chunk_entities = ner(chunk_text)
        latency_seconds = time.perf_counter() - started_at
        raw_entities.extend(
            shift_entity_offsets(entity, start_offset) for entity in chunk_entities
        )
        chunk_details.append(
            {
                "chunk": index,
                "start": start_offset,
                "end": start_offset + len(chunk_text),
                "characters": len(chunk_text),
                "tokens": token_count,
                "latency_ms": round(latency_seconds * 1000, 2),
                "raw_entities": len(chunk_entities),
            }
        )

    return raw_entities, chunk_details


def clean_label(label: Any) -> str:
    value = str(label or "").strip()
    if "-" in value:
        prefix, remainder = value.split("-", 1)
        if prefix.upper() in {"B", "I", "E", "S"}:
            value = remainder

    key = value.strip().lower().replace("_", " ")
    return LABEL_ALIASES.get(key, value.title() if value.isupper() else value)


def prepare_entities(text: str, raw_entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entities = []
    for raw_entity in raw_entities:
        entity = normalize_entity(raw_entity)
        entity["label"] = clean_label(entity.get("label"))

        start = entity.get("start")
        end = entity.get("end")
        if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text):
            entity["text"] = text[start:end]

        score = entity.get("score")
        if score is not None:
            entity["score"] = float(score)
        entities.append(entity)
    return entities


def is_other_label(label: Any) -> bool:
    return clean_label(label).lower() == "other"


def can_merge_entities(text: str, current: dict[str, Any], candidate: dict[str, Any]) -> bool:
    if current.get("label") != candidate.get("label"):
        return False

    current_end = current.get("end")
    candidate_start = candidate.get("start")
    if not isinstance(current_end, int) or not isinstance(candidate_start, int):
        return False
    if candidate_start < current_end:
        return False

    gap = text[current_end:candidate_start]
    if not gap:
        return True

    # Marathi token classifiers can split consonant clusters, matras, and virama
    # pieces. Merge same-label spans across tiny non-space gaps to reconstruct words.
    return len(gap) <= 4 and not any(char.isspace() for char in gap)


def is_word_boundary(char: str) -> bool:
    return char.isspace() or char in ",.;:!?()[]{}\"'‘’“”`|/\\"


def expand_to_word_boundaries(text: str, entity: dict[str, Any]) -> dict[str, Any]:
    start = entity.get("start")
    end = entity.get("end")
    if not isinstance(start, int) or not isinstance(end, int):
        return entity

    expanded = dict(entity)
    while start > 0 and not is_word_boundary(text[start - 1]):
        start -= 1
    while end < len(text) and not is_word_boundary(text[end]):
        end += 1

    expanded["start"] = start
    expanded["end"] = end
    expanded["text"] = text[start:end]
    return expanded


def remove_nested_entities(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for entity in entities:
        start = entity.get("start")
        end = entity.get("end")
        label = entity.get("label")
        if not isinstance(start, int) or not isinstance(end, int):
            kept.append(entity)
            continue

        is_nested = False
        for other in kept:
            other_start = other.get("start")
            other_end = other.get("end")
            if (
                other.get("label") == label
                and isinstance(other_start, int)
                and isinstance(other_end, int)
                and other_start <= start
                and end <= other_end
            ):
                is_nested = True
                break
        if not is_nested:
            kept.append(entity)
    return kept


def merge_fragmented_entities(
    text: str, entities: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    valid_entities = [
        entity
        for entity in entities
        if not is_other_label(entity.get("label"))
        and isinstance(entity.get("start"), int)
        and isinstance(entity.get("end"), int)
        and 0 <= entity["start"] < entity["end"] <= len(text)
    ]
    valid_entities = [expand_to_word_boundaries(text, entity) for entity in valid_entities]
    valid_entities.sort(key=lambda item: (item["start"], item["end"]))

    merged: list[dict[str, Any]] = []
    for entity in valid_entities:
        if merged and can_merge_entities(text, merged[-1], entity):
            previous = merged[-1]
            previous["end"] = entity["end"]
            previous["text"] = text[previous["start"] : previous["end"]]
            previous_score = previous.get("score")
            entity_score = entity.get("score")
            if isinstance(previous_score, float) and isinstance(entity_score, float):
                previous["score"] = round((previous_score + entity_score) / 2, 6)
            continue
        merged.append(dict(entity))

    return remove_nested_entities(merged)


def filter_entities(
    entities: list[dict[str, Any]], show_other_labels: bool
) -> list[dict[str, Any]]:
    if show_other_labels:
        return entities
    return [entity for entity in entities if not is_other_label(entity.get("label"))]


def render_highlighted_text(text: str, entities: list[dict[str, Any]]) -> str:
    valid_entities = [
        entity
        for entity in entities
        if isinstance(entity.get("start"), int)
        and isinstance(entity.get("end"), int)
        and 0 <= entity["start"] < entity["end"] <= len(text)
    ]
    valid_entities.sort(key=lambda item: (item["start"], item["end"]))

    pieces = []
    cursor = 0
    for entity in valid_entities:
        start = entity["start"]
        end = entity["end"]
        if start < cursor:
            continue

        pieces.append(html.escape(text[cursor:start]))

        label = str(entity.get("label") or "ENTITY")
        score = entity.get("score")
        title = label if score is None else f"{label} | {score:.3f}"
        border_color, background_color = get_label_style(label)
        entity_text = html.escape(text[start:end])
        pieces.append(
            '<mark class="entity" '
            f'style="border-color: {border_color}; background: {background_color};" '
            f'title="{html.escape(title)}">'
            f"{entity_text}"
            "</mark>"
        )
        cursor = end

    pieces.append(html.escape(text[cursor:]))
    return "".join(pieces).replace("\n", "<br>")


def render_entities_table(entities: list[dict[str, Any]]) -> str:
    rows = []
    for entity in entities:
        label = str(entity.get("label") or "")
        score = entity.get("score")
        score_text = f"{score:.3f}" if isinstance(score, float) else ""
        rows.append(
            "<tr>"
            f'<td class="entity-output-text">{html.escape(str(entity.get("text") or ""))}</td>'
            f"<td>{html.escape(label)}</td>"
            f"<td>{html.escape(score_text)}</td>"
            f'<td class="offset">{html.escape("" if entity.get("start") is None else str(entity.get("start")))}</td>'
            f'<td class="offset">{html.escape("" if entity.get("end") is None else str(entity.get("end")))}</td>'
            "</tr>"
        )

    return (
        '<table class="entities-table">'
        "<thead><tr>"
        "<th>Text</th><th>Label</th><th>Score</th><th>Start</th><th>End</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def main() -> None:
    st.set_page_config(page_title="Marathi NER", layout="wide")

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Devanagari:wght@400;500;700&display=swap');

        :root {
            --marathi-font: "Noto Sans Devanagari", "Nirmala UI", Mangal,
                "Kohinoor Devanagari", "Lohit Devanagari", sans-serif;
        }

        html, body, .stApp, .stMarkdown, .stTextArea, .stDataFrame,
        textarea, input, button {
            font-family: var(--marathi-font) !important;
        }

        .entity {
            border: 1px solid;
            border-radius: 6px;
            color: #111827;
            display: inline-block;
            font-family: var(--marathi-font);
            line-height: 1.8;
            margin: 0 2px;
            padding: 0 4px;
        }
        .ner-text {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            font-family: var(--marathi-font);
            font-size: 1.1rem;
            line-height: 2;
            overflow-wrap: break-word;
            padding: 1rem;
            white-space: pre-wrap;
            word-break: normal;
        }
        .entities-table {
            border-collapse: collapse;
            font-family: var(--marathi-font);
            width: 100%;
        }
        .entities-table th,
        .entities-table td {
            border-bottom: 1px solid #e5e7eb;
            color: #111827;
            padding: 0.6rem 0.75rem;
            text-align: left;
            vertical-align: top;
        }
        .entities-table th {
            background: #f9fafb;
            font-weight: 700;
        }
        .entities-table .entity-output-text {
            font-family: var(--marathi-font);
            font-size: 1.05rem;
            line-height: 1.8;
            overflow-wrap: break-word;
            word-break: normal;
        }
        .entities-table .offset {
            font-variant-numeric: tabular-nums;
            white-space: nowrap;
        }
        textarea {
            font-size: 1.05rem !important;
            line-height: 1.8 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Marathi NER")

    with st.sidebar:
        st.header("Model")
        model_name = st.text_input("Model", value="l3cube-pune/marathi-ner")
        aggregation = st.selectbox(
            "Aggregation",
            options=("simple", "first", "average", "max", "none"),
            index=0,
        )
        cleanup_output = st.checkbox("Clean fragmented output", value=True)
        show_other_labels = st.checkbox("Show Other/O labels", value=False)
        max_tokens = st.slider(
            "Max tokens per chunk",
            min_value=128,
            max_value=512,
            value=480,
            step=16,
            help="BERT models support up to 512 tokens. Keep this below 512 to leave room for special tokens.",
        )
        device_label = st.selectbox("Device", options=("CPU", "CUDA GPU 0"), index=0)
        device = -1 if device_label == "CPU" else 0
        run_inference = st.button("Run NER", type="primary", use_container_width=True)

    text = st.text_area("Text", value=DEFAULT_TEXT, height=180)

    if not run_inference:
        st.info("Enter Marathi text and run NER.")
        return

    if not text.strip():
        st.warning("Enter text before running inference.")
        return

    with st.spinner("Loading model and running inference..."):
        run_started_at = time.perf_counter()
        model_started_at = time.perf_counter()
        ner = get_ner_pipeline(model_name, aggregation, device)
        model_latency_ms = round((time.perf_counter() - model_started_at) * 1000, 2)

        raw_entities, chunk_details = run_ner_with_chunking(ner, text, max_tokens)
        inference_latency_ms = round(
            sum(chunk["latency_ms"] for chunk in chunk_details), 2
        )

        postprocess_started_at = time.perf_counter()
        entities = prepare_entities(text, raw_entities)
        if cleanup_output:
            entities = merge_fragmented_entities(text, entities)
        visible_entities = filter_entities(entities, show_other_labels)
        postprocess_latency_ms = round(
            (time.perf_counter() - postprocess_started_at) * 1000, 2
        )
        total_latency_ms = round((time.perf_counter() - run_started_at) * 1000, 2)

    chunk_count = len(chunk_details)
    total_tokens = sum(chunk["tokens"] for chunk in chunk_details)

    st.subheader("Run Details")
    metric_columns = st.columns(5)
    metric_columns[0].metric("Chunks", chunk_count)
    metric_columns[1].metric("Tokens", total_tokens)
    metric_columns[2].metric("Entities", len(visible_entities))
    metric_columns[3].metric("Inference", f"{inference_latency_ms:.0f} ms")
    metric_columns[4].metric("Total", f"{total_latency_ms:.0f} ms")

    with st.expander("Latency and token breakdown"):
        st.write(
            {
                "model_access_ms": model_latency_ms,
                "inference_ms": inference_latency_ms,
                "postprocess_ms": postprocess_latency_ms,
                "total_ms": total_latency_ms,
                "max_tokens_per_chunk": max_tokens,
            }
        )
        st.dataframe(chunk_details, hide_index=True, use_container_width=True)

    st.subheader("Highlighted Text")
    if visible_entities:
        highlighted_text = render_highlighted_text(text, visible_entities)
        st.markdown(
            f'<div class="ner-text">{highlighted_text}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.write(text)
        st.info("No entities found.")

    st.subheader("Entities")
    if visible_entities:
        st.markdown(
            render_entities_table(visible_entities),
            unsafe_allow_html=True,
        )

    st.download_button(
        "Download JSON",
        data=json.dumps(visible_entities, indent=2, ensure_ascii=False),
        file_name="mahabert_ner_predictions.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()
