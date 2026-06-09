"""
bench_all.py — Single-variant throughput + similarity bench on Intel Xeon 6737P.
Pinned to one NUMA node (via outer numactl) or both. Reports SPS and TPS.

Usage:
  python bench_all.py --variant ov-int8 --model all
  python bench_all.py --variant pytorch-bf16 --model biobert --bs 256
"""
import argparse, json, os, sys, time, warnings
from datetime import datetime
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ap = argparse.ArgumentParser()
ap.add_argument("--variant", required=True,
    choices=["pytorch-fp32","pytorch-fp16","pytorch-bf16",
             "ov-bf16","ov-int8","ov-int4"])
ap.add_argument("--model", default="all",
    choices=["pubmed","bodhi","biobert","all"])
ap.add_argument("--bs", type=int, default=None)
ap.add_argument("--out-dir", default=None)
args = ap.parse_args()

from config import (PT_PATHS, OV_PATHS, SENTENCES, MODELS,
                    TOKENS_PER_SENTENCE)

N_CORES = len(os.sched_getaffinity(0))
os.environ["OMP_NUM_THREADS"] = str(N_CORES)
os.environ["MKL_NUM_THREADS"] = str(N_CORES)
os.environ["KMP_BLOCKTIME"]   = "1"
os.environ["KMP_AFFINITY"]    = "granularity=fine,compact,1,0"
os.environ["ONEDNN_VERBOSE"]  = "0"

OUT_DIR = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent.parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZES = [args.bs] if args.bs else [64, 128, 256, 384, 512]
N_THROUGHPUT = 1024
MODELS_TO_RUN = list(MODELS.keys()) if args.model == "all" else [args.model]

CONNECTED = [
    ("BRCA1 pathogenic variant detected in germline DNA sequencing",
     "Patient diagnosed with hereditary breast and ovarian cancer syndrome"),
    ("TP53 R175H missense mutation dominant negative effect identified",
     "Li-Fraumeni syndrome: multiple early-onset malignancies in family history"),
    ("EGFR exon 19 deletion driver mutation non-small-cell lung adenocarcinoma",
     "Patient started on erlotinib targeted therapy for lung cancer"),
    ("KRAS G12C activating mutation colorectal cancer",
     "Oncologist noted tumour progressed despite standard chemotherapy regimen"),
    ("Serum cortisol 32 μg/dL markedly elevated HPA axis dysregulation",
     "Patient reports persistent anxiety, sleep disruption and mood instability"),
    ("HbA1c 9.2% sustained hyperglycaemia poor glycaemic control",
     "Patient describes fatigue, low mood and difficulty concentrating daily"),
    ("Troponin I 3.8 ng/mL myocardial necrosis acute coronary syndrome",
     "Patient admitted with crushing chest pain, diaphoresis and fear of death"),
    ("TSH suppressed free T4 elevated clinical hyperthyroidism",
     "Reported palpitations, anxiety, heat intolerance and weight loss"),
    ("Patient with intellectual disability and seizures since early childhood",
     "FMR1 CGG repeat expansion confirmed fragile X syndrome genetic testing"),
    ("Progressive muscle weakness proximal limb girdle pattern onset age 5",
     "DMD frameshift deletion dystrophin absent confirmed Duchenne muscular dystrophy"),
    ("PHQ-9 score 18 severe depression, suicidal ideation present",
     "Patient admitted to psychiatry ward, started on SSRI and CBT protocol"),
    ("Journaled: feeling disconnected from reality, visual disturbances, paranoia",
     "Clinical assessment consistent with first-episode psychosis, initiated antipsychotics"),
    ("LDL 185 mg/dL HDL 38 mg/dL triglycerides 290 mg/dL",
     "Initiated high-intensity statin therapy and dietary modification for dyslipidaemia"),
    ("BNP 820 pg/mL markedly elevated congestive heart failure decompensated",
     "Admitted for IV diuresis and optimisation of heart failure medical therapy"),
    ("Cortisol elevation adrenal hyperactivity Cushing syndrome",
     "Patient presents with central obesity, striae, hypertension, glucose intolerance"),
    ("APOE e4 allele amyloid accumulation Alzheimer disease risk",
     "Cognitive decline progressive memory loss executive dysfunction in 65yo"),
]

DISCONNECTED = [
    ("BRCA1 pathogenic variant detected in germline DNA sequencing",
     "Patient reports work-related stress and difficulty sleeping for two weeks"),
    ("KRAS G12C mutation colorectal cancer specimen NGS panel",
     "Mood score 3/10 today, feeling hopeless, journaling recommended"),
    ("MLH1 promoter hypermethylation microsatellite instability Lynch syndrome",
     "Therapist noted patient shows avoidant attachment style in relationships"),
    ("HER2 amplification breast cancer trastuzumab targeted therapy",
     "Annual dental check-up: no cavities, gum health satisfactory"),
    ("CFTR delta-F508 mutation cystic fibrosis chloride channel",
     "Patient presents with acute ankle sprain following sports injury"),
    ("HbA1c 9.2% poor glycaemic control insulin resistance",
     "Stock market volatility increased investor anxiety this quarter"),
    ("BNP 820 pg/mL congestive heart failure decompensated",
     "Software deployment completed successfully with zero downtime"),
    ("Patient exhibits hallucinations paranoid ideation psychosis",
     "Post-operative wound healing satisfactory, sutures removed day 10"),
    ("PHQ-9 score 18 severe depression suicidal ideation",
     "Bone density scan DEXA normal T-score bilateral hip and spine"),
    ("Lumbar puncture elevated protein lymphocytic pleocytosis meningitis",
     "Patient enrolled in weight loss programme, BMI reduced from 34 to 31"),
    ("Spirometry FEV1/FVC 0.62 obstructive airway disease bronchodilator",
     "Annual performance review completed, employee met all KPIs this year"),
    ("Serum cortisol elevated HPA axis adrenal dysregulation",
     "Quarterly financial audit found minor discrepancies in expense reports"),
    ("EGFR mutation lung adenocarcinoma tyrosine kinase inhibitor",
     "Weather forecast predicts heavy rain and strong winds this weekend"),
]

THROUGHPUT_TEXTS = (SENTENCES * ((N_THROUGHPUT // len(SENTENCES)) + 1))[:N_THROUGHPUT]


def resolve_path(variant, model_key):
    if variant.startswith("pytorch"):
        return PT_PATHS[model_key]
    prec = variant.split("-")[1]
    return OV_PATHS[f"{model_key}_{prec}"]


# ── Backends ────────────────────────────────────────────────────────────────
_pt_cache, _ov_cache, _tok_cache = {}, {}, {}

def _tok(path):
    if path not in _tok_cache:
        from transformers import AutoTokenizer
        _tok_cache[path] = AutoTokenizer.from_pretrained(path)
    return _tok_cache[path]


def _load_pt(path, dtype_str):
    key = (path, dtype_str)
    if key in _pt_cache:
        return _pt_cache[key]
    import torch
    from transformers import AutoModel
    torch.set_num_threads(N_CORES)
    dmap = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    print(f"  [pt-load] {Path(path).name} ({dtype_str})", flush=True)
    mdl = AutoModel.from_pretrained(path, torch_dtype=dmap[dtype_str]).eval()
    _pt_cache[key] = mdl
    return mdl


def _load_ov(path):
    if path in _ov_cache:
        return _ov_cache[path]
    import openvino as ov
    core = ov.Core()
    cfg = {"PERFORMANCE_HINT": "THROUGHPUT",
           "INFERENCE_NUM_THREADS": str(N_CORES),
           "INFERENCE_PRECISION_HINT": "bf16"}
    print(f"  [ov-compile] {Path(path).name}", flush=True)
    compiled = core.compile_model(str(Path(path) / "openvino_model.xml"), "CPU", cfg)
    _ov_cache[path] = compiled
    return compiled


def embed(variant, path, texts, batch_size=128):
    tok = _tok(path)
    dtype_str = variant.split("-")[1]
    all_vecs = []
    if variant.startswith("pytorch"):
        import torch
        mdl = _load_pt(path, dtype_str)
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i+batch_size], padding=True, truncation=True,
                      max_length=512, return_tensors="pt")
            with torch.inference_mode():
                if dtype_str in ("fp16", "bf16"):
                    dmap = {"fp16": torch.float16, "bf16": torch.bfloat16}
                    with torch.autocast(device_type="cpu", dtype=dmap[dtype_str]):
                        out = mdl(**enc)
                else:
                    out = mdl(**enc)
            lhs = out.last_hidden_state.float().numpy()
            mask = enc["attention_mask"].numpy()[..., np.newaxis].astype(np.float32)
            pooled = (lhs * mask).sum(1) / mask.sum(1).clip(min=1e-9)
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            all_vecs.append(pooled / np.clip(norms, 1e-8, None))
    else:
        compiled = _load_ov(path)
        valid = {i.get_any_name() for i in compiled.inputs}
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i+batch_size], padding=True, truncation=True,
                      max_length=512, return_tensors="np")
            out = compiled({k: v for k, v in enc.items() if k in valid})
            lhs = list(out.values())[0]
            mask = enc["attention_mask"][..., np.newaxis].astype(np.float32)
            pooled = (lhs * mask).sum(1) / mask.sum(1).clip(min=1e-9)
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            all_vecs.append(pooled / np.clip(norms, 1e-8, None))
    return np.vstack(all_vecs)


def bench_throughput(variant, path, texts, bs):
    tok = _tok(path)
    dtype_str = variant.split("-")[1]
    # warmup
    if variant.startswith("pytorch"):
        import torch
        mdl = _load_pt(path, dtype_str)
        for _ in range(2):
            enc = tok(texts[:bs], padding=True, truncation=True,
                      max_length=512, return_tensors="pt")
            with torch.inference_mode():
                if dtype_str in ("fp16","bf16"):
                    dmap = {"fp16": torch.float16, "bf16": torch.bfloat16}
                    with torch.autocast(device_type="cpu", dtype=dmap[dtype_str]):
                        mdl(**enc)
                else:
                    mdl(**enc)
        t0 = time.perf_counter()
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i+bs], padding=True, truncation=True,
                      max_length=512, return_tensors="pt")
            with torch.inference_mode():
                if dtype_str in ("fp16","bf16"):
                    dmap = {"fp16": torch.float16, "bf16": torch.bfloat16}
                    with torch.autocast(device_type="cpu", dtype=dmap[dtype_str]):
                        mdl(**enc)
                else:
                    mdl(**enc)
    else:
        compiled = _load_ov(path)
        valid = {i.get_any_name() for i in compiled.inputs}
        for _ in range(2):
            enc = tok(texts[:bs], padding=True, truncation=True,
                      max_length=512, return_tensors="np")
            compiled({k: v for k, v in enc.items() if k in valid})
        t0 = time.perf_counter()
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i+bs], padding=True, truncation=True,
                      max_length=512, return_tensors="np")
            compiled({k: v for k, v in enc.items() if k in valid})
    elapsed = time.perf_counter() - t0
    return len(texts) / elapsed, elapsed


# ── Main ────────────────────────────────────────────────────────────────────
print(f"\n[platform] variant={args.variant}  cores_visible={N_CORES}")
print(f"[batch sizes] {BATCH_SIZES}\n")

results = {
    "run_at":       datetime.now().isoformat(),
    "variant":      args.variant,
    "n_cores":      N_CORES,
    "batch_sizes":  BATCH_SIZES,
    "models":       {},
}

print("=" * 72)
print("  SIMILARITY (connected vs disconnected)")
print("=" * 72)

for mkey in MODELS_TO_RUN:
    path = resolve_path(args.variant, mkey)
    if not Path(path).exists() or (
        args.variant.startswith("ov") and not (Path(path) / "openvino_model.xml").exists()
    ):
        print(f"  [{mkey}] SKIP — not found at {path}")
        continue
    print(f"\n  [{mkey}]  {Path(path).name}")
    ec = embed(args.variant, path, [p[0] for p in CONNECTED])
    ep = embed(args.variant, path, [p[1] for p in CONNECTED])
    conn = [float(np.dot(ec[i], ep[i])) for i in range(len(CONNECTED))]
    ed = embed(args.variant, path, [p[0] for p in DISCONNECTED])
    eq = embed(args.variant, path, [p[1] for p in DISCONNECTED])
    disc = [float(np.dot(ed[i], eq[i])) for i in range(len(DISCONNECTED))]
    gap = float(np.mean(conn) - np.mean(disc))
    print(f"  conn mean={np.mean(conn):.3f}  disc mean={np.mean(disc):.3f}  gap={gap:+.3f}")
    results["models"][mkey] = {
        "connected_mean":    float(np.mean(conn)),
        "disconnected_mean": float(np.mean(disc)),
        "discrimination_gap": gap,
        "connected_sims":    conn,
        "disconnected_sims": disc,
    }

print(f"\n{'='*72}")
print(f"  THROUGHPUT  (sent/sec | tok/sec)  n={N_THROUGHPUT}")
print(f"  {'Model':<14}  " + "  ".join(f"bs={b:>4}" for b in BATCH_SIZES))
print(f"  {'-'*72}")

for mkey in MODELS_TO_RUN:
    path = resolve_path(args.variant, mkey)
    if mkey not in results["models"]:
        continue
    row_sps, row_tps = [], []
    results["models"][mkey].setdefault("throughput", {})
    for bs in BATCH_SIZES:
        sps, elapsed = bench_throughput(args.variant, path, THROUGHPUT_TEXTS, bs)
        tps = sps * TOKENS_PER_SENTENCE[mkey]
        row_sps.append(f"{sps:>7.1f}")
        row_tps.append(f"{tps:>9.0f}")
        results["models"][mkey]["throughput"][bs] = {
            "sentences_per_sec": round(sps, 1),
            "tokens_per_sec":    round(tps, 1),
            "elapsed_s":         round(elapsed, 3),
        }
    print(f"  {mkey:<14}  " + "  ".join(row_sps) + "   sps")
    print(f"  {mkey:<14}  " + "  ".join(row_tps) + "   tps")

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out = OUT_DIR / f"bench_all_{args.variant.replace('-','_')}_{ts}.json"
json.dump(results, out.open("w"), indent=2)
print(f"\n[saved] {out}")
