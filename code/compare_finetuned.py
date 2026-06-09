"""
compare_finetuned.py
====================
Compare PyTorch FP32 base vs finetuned variants for a single model family on
connected/disconnected pairs + throughput.

Usage:
  python compare_finetuned.py --model biobert
  python compare_finetuned.py --model pubmed  --variants ov-bf16 ov-int8
"""
import argparse, json, os, sys, time, warnings
from datetime import datetime
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ALL_VARIANTS = ["pytorch-fp32","pytorch-fp16","pytorch-bf16",
                "ov-bf16","ov-int8","ov-int4"]

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="biobert", choices=["pubmed","bodhi","biobert"])
ap.add_argument("--variants", nargs="+", default=ALL_VARIANTS, choices=ALL_VARIANTS)
ap.add_argument("--out-dir", default=None)
ap.add_argument("--data-file", default=None,
                help="Optional JSONL of connected/disconnected pairs "
                     "(schema: examples/README.md §1). Overrides the built-in literals.")
args = ap.parse_args()

from config import PT_PATHS, OV_PATHS, SENTENCES

N_CORES = len(os.sched_getaffinity(0))
os.environ["OMP_NUM_THREADS"] = str(N_CORES)
os.environ["MKL_NUM_THREADS"] = str(N_CORES)
os.environ["KMP_BLOCKTIME"]   = "1"
os.environ["KMP_AFFINITY"]    = "granularity=fine,compact,1,0"
os.environ["ONEDNN_VERBOSE"]  = "0"

OUT_DIR = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent.parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZES = [128, 256, 384, 512]
N_THROUGHPUT = 512
THROUGHPUT_TEXTS = (SENTENCES * ((N_THROUGHPUT // len(SENTENCES)) + 1))[:N_THROUGHPUT]

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
_DEFAULT_CONNECTED = CONNECTED
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
_DEFAULT_DISCONNECTED = DISCONNECTED

if args.data_file:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from examples.load_pairs import load_connected_disconnected
    _rows = load_connected_disconnected(args.data_file)
    CONNECTED    = [(r["sentence_a"], r["sentence_b"]) for r in _rows if r["label"] == "connected"]
    DISCONNECTED = [(r["sentence_a"], r["sentence_b"]) for r in _rows if r["label"] == "disconnected"]
    print(f"[data] override: {len(CONNECTED)} connected + {len(DISCONNECTED)} disconnected "
          f"from {args.data_file}", flush=True)

# ── Backends ────────────────────────────────────────────────────────────────
_pt_cache, _ov_cache, _tok_cache = {}, {}, {}

def _tok(path):
    if path not in _tok_cache:
        from transformers import AutoTokenizer
        _tok_cache[path] = AutoTokenizer.from_pretrained(path)
    return _tok_cache[path]

def _pt(path, dtype_str):
    key = (path, dtype_str)
    if key in _pt_cache: return _pt_cache[key]
    import torch
    from transformers import AutoModel
    torch.set_num_threads(N_CORES)
    dmap = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    print(f"  [pt-load] {Path(path).name} ({dtype_str})", flush=True)
    _pt_cache[key] = AutoModel.from_pretrained(path, torch_dtype=dmap[dtype_str]).eval()
    return _pt_cache[key]

def _ov(path):
    if path in _ov_cache: return _ov_cache[path]
    import openvino as ov
    core = ov.Core()
    cfg = {"PERFORMANCE_HINT": "THROUGHPUT",
           "INFERENCE_NUM_THREADS": str(N_CORES),
           "INFERENCE_PRECISION_HINT": "bf16"}
    print(f"  [ov-compile] {Path(path).name}", flush=True)
    _ov_cache[path] = core.compile_model(str(Path(path)/"openvino_model.xml"), "CPU", cfg)
    return _ov_cache[path]

def _resolve(variant, model_key):
    if variant.startswith("pytorch"):
        return PT_PATHS[model_key]
    return OV_PATHS[f"{model_key}_{variant.split('-')[1]}"]

def embed(variant, path, texts, bs=128):
    tok = _tok(path); dtype_str = variant.split("-")[1]; out = []
    if variant.startswith("pytorch"):
        import torch
        mdl = _pt(path, dtype_str)
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i+bs], padding=True, truncation=True,
                      max_length=512, return_tensors="pt")
            with torch.inference_mode():
                if dtype_str in ("fp16","bf16"):
                    dmap = {"fp16": torch.float16, "bf16": torch.bfloat16}
                    with torch.autocast(device_type="cpu", dtype=dmap[dtype_str]):
                        o = mdl(**enc)
                else:
                    o = mdl(**enc)
            lhs = o.last_hidden_state.float().numpy()
            mask = enc["attention_mask"].numpy()[..., np.newaxis].astype(np.float32)
            pooled = (lhs * mask).sum(1) / mask.sum(1).clip(min=1e-9)
            out.append(pooled / np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-8, None))
    else:
        compiled = _ov(path)
        valid = {i.get_any_name() for i in compiled.inputs}
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i+bs], padding=True, truncation=True,
                      max_length=512, return_tensors="np")
            o = compiled({k: v for k, v in enc.items() if k in valid})
            lhs = list(o.values())[0]
            mask = enc["attention_mask"][..., np.newaxis].astype(np.float32)
            pooled = (lhs * mask).sum(1) / mask.sum(1).clip(min=1e-9)
            out.append(pooled / np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-8, None))
    return np.vstack(out)

def pair_cos(variant, path, pairs):
    a = embed(variant, path, [p[0] for p in pairs])
    b = embed(variant, path, [p[1] for p in pairs])
    return [float(np.dot(a[i], b[i])) for i in range(len(pairs))]

def bench_tp(variant, path, texts, bs):
    tok = _tok(path); dtype_str = variant.split("-")[1]
    if variant.startswith("pytorch"):
        import torch
        mdl = _pt(path, dtype_str)
        # warmup
        enc = tok(texts[:bs], padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.inference_mode(): mdl(**enc)
        t0 = time.perf_counter()
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i+bs], padding=True, truncation=True, max_length=512, return_tensors="pt")
            with torch.inference_mode():
                if dtype_str in ("fp16","bf16"):
                    dmap = {"fp16": torch.float16, "bf16": torch.bfloat16}
                    with torch.autocast(device_type="cpu", dtype=dmap[dtype_str]):
                        mdl(**enc)
                else:
                    mdl(**enc)
    else:
        compiled = _ov(path)
        valid = {i.get_any_name() for i in compiled.inputs}
        enc = tok(texts[:bs], padding=True, truncation=True, max_length=512, return_tensors="np")
        compiled({k: v for k, v in enc.items() if k in valid})
        t0 = time.perf_counter()
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i+bs], padding=True, truncation=True, max_length=512, return_tensors="np")
            compiled({k: v for k, v in enc.items() if k in valid})
    return round(len(texts) / (time.perf_counter() - t0), 1)


# ── Main ────────────────────────────────────────────────────────────────────
print(f"\n[platform] model={args.model}  cores={N_CORES}")
print(f"[variants] {args.variants}\n")

runs = {"base_fp32": ("pytorch-fp32", PT_PATHS["pubmed"])}  # untuned reference
for v in args.variants:
    path = _resolve(v, args.model)
    if v.startswith("ov") and not (Path(path)/"openvino_model.xml").exists():
        print(f"  [skip] {v} — not found at {path}")
        continue
    runs[f"{args.model}_{v}"] = (v, path)

results = {"run_at": datetime.now().isoformat(), "model": args.model,
           "n_cores": N_CORES, "variants": list(runs.keys()), "results": {}}

print("=" * 72)
print("  SIMILARITY")
print("=" * 72)

sims = {}
for key, (variant, path) in runs.items():
    print(f"\n  [{key}]")
    conn = pair_cos(variant, path, CONNECTED)
    disc = pair_cos(variant, path, DISCONNECTED)
    sims[key] = {"connected": conn, "disconnected": disc}
    print(f"  conn  mean={np.mean(conn):.3f} min={min(conn):.3f} max={max(conn):.3f}")
    print(f"  disc  mean={np.mean(disc):.3f} min={min(disc):.3f} max={max(disc):.3f}")

base_gap = np.mean(sims["base_fp32"]["connected"]) - np.mean(sims["base_fp32"]["disconnected"])
print(f"\n  {'Variant':<24} {'conn':>7} {'disc':>7} {'gap':>7} {'Δgap':>7}")
for key in runs:
    cg = float(np.mean(sims[key]["connected"]))
    dg = float(np.mean(sims[key]["disconnected"]))
    gap = cg - dg
    print(f"  {key:<24} {cg:>7.3f} {dg:>7.3f} {gap:>+7.3f} {gap-base_gap:>+7.3f}")

print(f"\n{'='*72}\n  THROUGHPUT (sent/sec)  n={N_THROUGHPUT}")
print(f"  {'Variant':<24}  " + "  ".join(f"bs={b:>3}" for b in BATCH_SIZES))
print(f"  {'-'*72}")

tp_data = {}
for key, (variant, path) in runs.items():
    row, tps = [], {}
    for bs in BATCH_SIZES:
        sps = bench_tp(variant, path, THROUGHPUT_TEXTS, bs)
        tps[bs] = sps
        row.append(f"{sps:>8.1f}")
    tp_data[key] = tps
    print(f"  {key:<24}  " + "  ".join(row))

for key in runs:
    results["results"][key] = {
        "connected_scores":    sims[key]["connected"],
        "disconnected_scores": sims[key]["disconnected"],
        "connected_mean":      float(np.mean(sims[key]["connected"])),
        "disconnected_mean":   float(np.mean(sims[key]["disconnected"])),
        "discrimination_gap":  float(np.mean(sims[key]["connected"]) - np.mean(sims[key]["disconnected"])),
        "throughput":          tp_data[key],
    }

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out = OUT_DIR / f"compare_finetuned_{args.model}_{ts}.json"
json.dump(results, open(out, "w"), indent=2)
print(f"\n[saved] {out}")
