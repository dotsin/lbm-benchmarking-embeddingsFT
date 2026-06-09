"""
Accuracy evaluation — cosine similarity of quantized embeddings vs FP32 PyTorch reference.

For each variant, computes embeddings on the 30-sentence corpus and measures:
  - Mean cosine similarity vs reference
  - Min cosine similarity
  - Std deviation
  - % of sentences within 0.99 cosine similarity

Usage:
  python accuracy_eval.py [--model pubmed]
"""
import argparse, json, os, sys, time
import numpy as np
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["KMP_AFFINITY"]    = "granularity=fine,compact,1,0"

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="all", choices=["pubmed","bodhi","biobert","all"])
ap.add_argument("--out",   default=None)
args = ap.parse_args()

sys.path.insert(0, str(Path(__file__).parent))
from config import PT_PATHS, OV_PATHS, SENTENCES, MODELS, RESULTS_DIR

MODELS_TO_RUN = list(MODELS.keys()) if args.model == "all" else [args.model]

def cosine_sim(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return (a * b).sum(axis=1)

def mean_pool(hidden, mask):
    m = mask[:, :, np.newaxis].astype(np.float32)
    return (hidden * m).sum(1) / m.sum(1).clip(min=1e-9)


def embed_pytorch(model_path, sentences, dtype_str="fp32"):
    import torch
    from transformers import AutoTokenizer, AutoModel

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[dtype_str]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModel.from_pretrained(model_path, torch_dtype=dtype)
    model.eval()

    enc = tokenizer(sentences, padding=True, truncation=True,
                    max_length=512, return_tensors="pt")
    enc = {k: v for k, v in enc.items()}

    with torch.no_grad():
        if dtype != torch.float32:
            enc = {k: v for k, v in enc.items()}
        out = model(**enc)

    hidden = out.last_hidden_state.float().numpy()
    mask   = enc["attention_mask"].numpy()
    return mean_pool(hidden, mask)


def embed_openvino(model_path, sentences):
    import openvino as ov
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    xml = str(Path(model_path) / "openvino_model.xml")
    core = ov.Core()
    compiled = core.compile_model(xml, "CPU", {
        "PERFORMANCE_HINT": "LATENCY",
        "INFERENCE_PRECISION_HINT": "f32",  # use f32 for accurate comparison
    })
    valid = {i.get_any_name() for i in compiled.inputs}

    enc = tokenizer(sentences, padding=True, truncation=True,
                    max_length=512, return_tensors="np")
    inp = {k: v for k, v in enc.items() if k in valid}
    out = compiled(inp)
    key = "last_hidden_state" if "last_hidden_state" in out else list(out.keys())[0]
    return mean_pool(out[key].astype(np.float32), enc["attention_mask"])


all_results = {}

for model_key in MODELS_TO_RUN:
    label   = MODELS[model_key]
    pt_path = PT_PATHS[model_key]
    results = {}

    print(f"\n{'='*60}")
    print(f"  Accuracy eval: {label}")
    print(f"{'='*60}")

    # Reference: PyTorch FP32
    print("  [ref] PyTorch FP32...", end=" ", flush=True)
    t0  = time.perf_counter()
    ref = embed_pytorch(pt_path, SENTENCES, "fp32")
    print(f"{time.perf_counter()-t0:.1f}s")

    # PyTorch variants
    for dtype in ["fp16", "bf16"]:
        print(f"  [pytorch-{dtype}]...", end=" ", flush=True)
        try:
            t0   = time.perf_counter()
            emb  = embed_pytorch(pt_path, SENTENCES, dtype)
            sims = cosine_sim(ref, emb)
            elapsed = time.perf_counter() - t0
            results[f"pytorch_{dtype}"] = {
                "mean_cosine": float(sims.mean()),
                "min_cosine":  float(sims.min()),
                "std_cosine":  float(sims.std()),
                "pct_above_99": float((sims >= 0.99).mean() * 100),
                "inference_s": round(elapsed, 3),
            }
            print(f"mean_cos={sims.mean():.5f}  min={sims.min():.5f}")
        except Exception as e:
            print(f"ERROR: {e}")
            results[f"pytorch_{dtype}"] = {"error": str(e)}

    # OpenVINO variants
    for prec in ["bf16", "int8", "int4"]:
        ov_path = OV_PATHS[f"{model_key}_{prec}"]
        print(f"  [ov-{prec}]...", end=" ", flush=True)
        if not (Path(ov_path) / "openvino_model.xml").exists():
            print(f"SKIP (not quantized yet)")
            results[f"ov_{prec}"] = {"error": "not quantized"}
            continue
        try:
            t0   = time.perf_counter()
            emb  = embed_openvino(ov_path, SENTENCES)
            sims = cosine_sim(ref, emb)
            elapsed = time.perf_counter() - t0
            results[f"ov_{prec}"] = {
                "mean_cosine": float(sims.mean()),
                "min_cosine":  float(sims.min()),
                "std_cosine":  float(sims.std()),
                "pct_above_99": float((sims >= 0.99).mean() * 100),
                "inference_s": round(elapsed, 3),
            }
            print(f"mean_cos={sims.mean():.5f}  min={sims.min():.5f}")
        except Exception as e:
            print(f"ERROR: {e}")
            results[f"ov_{prec}"] = {"error": str(e)}

    all_results[model_key] = results

    # Print summary table
    print(f"\n  {'Variant':<18} {'Mean Cos':>10} {'Min Cos':>10} {'Pct≥0.99':>10}")
    print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*10}")
    for variant, r in results.items():
        if "error" in r:
            print(f"  {variant:<18} {'ERROR':>10}")
        else:
            print(f"  {variant:<18} {r['mean_cosine']:>10.5f} {r['min_cosine']:>10.5f} {r['pct_above_99']:>9.1f}%")

out_path = args.out or str(RESULTS_DIR / "accuracy_results.json")
Path(out_path).write_text(json.dumps(all_results, indent=2))
print(f"\n[saved] {out_path}")
