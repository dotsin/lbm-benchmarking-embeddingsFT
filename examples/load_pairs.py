"""
Tiny helper for loading the JSONL sample data shapes documented in
`examples/README.md`. Used by `code/extended_benchmark.py` and
`code/compare_finetuned.py` when `--data-dir` is passed.

Each loader returns a list of dicts in the shape declared in
`examples/README.md`. The functions accept either a Path/str pointing at a
JSONL file, or pointing at a directory — in which case the canonical
filename is appended automatically.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CANONICAL_FILES = {
    "connected_disconnected": "connected_disconnected_pairs.jsonl",
    "hard_negatives": "hard_negative_triplets.jsonl",
    "biosses": "biosses_style_pairs.jsonl",
    "domain": "domain_sentences.jsonl",
    "finetuning": "finetuning_triplets.jsonl",
}


def _resolve(path: str | Path, key: str) -> Path:
    p = Path(path)
    if p.is_dir():
        p = p / CANONICAL_FILES[key]
    if not p.exists():
        raise FileNotFoundError(f"Expected JSONL at {p}")
    return p


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{ln} invalid JSON: {exc}") from exc
    return out


def load_connected_disconnected(path: str | Path) -> list[dict[str, Any]]:
    """Records: {sentence_a, sentence_b, label: 'connected'|'disconnected'}."""
    rows = _read_jsonl(_resolve(path, "connected_disconnected"))
    for r in rows:
        assert {"sentence_a", "sentence_b", "label"} <= r.keys(), r
        assert r["label"] in {"connected", "disconnected"}, r
    return rows


def load_hard_negatives(path: str | Path) -> list[dict[str, Any]]:
    """Records: {anchor, positive, hard_negative, [domain]}."""
    rows = _read_jsonl(_resolve(path, "hard_negatives"))
    for r in rows:
        assert {"anchor", "positive", "hard_negative"} <= r.keys(), r
    return rows


def load_biosses_style(path: str | Path) -> list[dict[str, Any]]:
    """Records: {sentence_a, sentence_b, human_score: float in [0,1]}."""
    rows = _read_jsonl(_resolve(path, "biosses"))
    for r in rows:
        assert {"sentence_a", "sentence_b", "human_score"} <= r.keys(), r
        assert 0.0 <= float(r["human_score"]) <= 1.0, r
    return rows


def load_domain_sentences(path: str | Path) -> dict[str, list[str]]:
    """Records: {domain, sentence}. Returned grouped by domain."""
    rows = _read_jsonl(_resolve(path, "domain"))
    grouped: dict[str, list[str]] = {}
    for r in rows:
        assert {"domain", "sentence"} <= r.keys(), r
        grouped.setdefault(r["domain"], []).append(r["sentence"])
    return grouped


def load_finetuning_triplets(path: str | Path) -> list[dict[str, Any]]:
    """Records: {anchor, positive, negative, [level], [domain_anchor], [domain_negative]}."""
    rows = _read_jsonl(_resolve(path, "finetuning"))
    for r in rows:
        assert {"anchor", "positive", "negative"} <= r.keys(), r
    return rows
