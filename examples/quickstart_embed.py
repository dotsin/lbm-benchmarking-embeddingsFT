"""
Minimal end-to-end example: load PubMedBERT BODHI, embed four biomedical
sentences, and print the cosine-similarity matrix.

Run from the repo root:

    python examples/quickstart_embed.py
    python examples/quickstart_embed.py --backend openvino
    python examples/quickstart_embed.py --model pubmedbert_pass1

The model directory is resolved relative to this file so the script works
both inside the repo and after `pip install` if the package is ever
restructured.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_ROOT = REPO_ROOT / "models"

DEFAULT_SENTENCES = [
    "HbA1c 9.2% sustained hyperglycaemia poor glycaemic control",
    "Patient describes fatigue, low mood and difficulty concentrating daily",
    "BRCA1 pathogenic variant detected in germline DNA",
    "Stock market volatility increased investor anxiety this quarter",
]


def mean_pool(last_hidden, mask):
    mask = mask[..., np.newaxis].astype(np.float32)
    summed = (last_hidden * mask).sum(axis=1)
    counts = mask.sum(axis=1).clip(min=1e-9)
    pooled = summed / counts
    return pooled / np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-8)


def embed_pytorch(model_path: Path, texts: list[str]) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModel.from_pretrained(str(model_path), torch_dtype=torch.bfloat16).eval()
    enc = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.inference_mode():
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            out = model(**enc)
    return mean_pool(out.last_hidden_state.float().numpy(), enc["attention_mask"].numpy())


def embed_openvino(model_path: Path, texts: list[str]) -> np.ndarray:
    import openvino as ov
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_path))
    core = ov.Core()
    compiled = core.compile_model(
        str(model_path / "openvino_model.xml"),
        "CPU",
        {"PERFORMANCE_HINT": "THROUGHPUT", "INFERENCE_PRECISION_HINT": "bf16"},
    )
    valid = {i.get_any_name() for i in compiled.inputs}
    enc = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="np")
    out = compiled({k: v for k, v in enc.items() if k in valid})
    last_hidden = next(iter(out.values()))
    return mean_pool(last_hidden, enc["attention_mask"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model",
        default="pubmedbert_bodhi",
        choices=["pubmedbert_bodhi", "pubmedbert_pass1", "biobert_finetuned"],
    )
    parser.add_argument("--backend", default="pytorch", choices=["pytorch", "openvino"])
    parser.add_argument("--sentences", nargs="+", default=None,
                        help="Override the four built-in example sentences.")
    args = parser.parse_args()

    texts = args.sentences or DEFAULT_SENTENCES

    if args.backend == "pytorch":
        model_path = MODEL_ROOT / "pytorch" / args.model
        vecs = embed_pytorch(model_path, texts)
    else:
        suffix = "_bf16"
        model_path = MODEL_ROOT / "openvino" / f"{args.model}{suffix}"
        vecs = embed_openvino(model_path, texts)

    sim = vecs @ vecs.T
    print(f"Model        : {args.model} ({args.backend})")
    print(f"Embedding dim: {vecs.shape[1]}")
    print()
    print("Cosine similarity matrix:")
    print("        " + "  ".join(f"  S{i}" for i in range(len(texts))))
    for i, row in enumerate(sim):
        print(f"  S{i}  " + "  ".join(f"{v:+.3f}" for v in row))
    print()
    for i, t in enumerate(texts):
        print(f"  S{i}: {t}")


if __name__ == "__main__":
    main()
