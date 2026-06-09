"""
eval_full.py
============
Full evaluation of base vs finetuned models for a given variant.

Metrics:
  1. BIOSSES Spearman
  2. Cross-domain connected accuracy (> 0.50)
  3. Pairwise classification accuracy (t=0.50)
  4. Hard-negative detection (< 0.40)
  5. Intra/inter-domain ratio

Usage:
  python eval_full.py --variant ov-bf16 --model biobert
  python eval_full.py --variant pytorch-bf16 --model pubmed
"""
import argparse, json, os, sys, warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from datasets import load_dataset

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ap = argparse.ArgumentParser()
ap.add_argument("--variant", required=True,
    choices=["pytorch-fp32","pytorch-fp16","pytorch-bf16",
             "ov-bf16","ov-int8","ov-int4"])
ap.add_argument("--model", default="biobert",
    choices=["pubmed","bodhi","biobert"])
ap.add_argument("--out-dir", default=None)
args = ap.parse_args()

from config import PT_PATHS, OV_PATHS, MODELS

N_CORES = len(os.sched_getaffinity(0))
os.environ["OMP_NUM_THREADS"] = str(N_CORES)
os.environ["MKL_NUM_THREADS"] = str(N_CORES)
os.environ["KMP_BLOCKTIME"]   = "1"
os.environ["KMP_AFFINITY"]    = "granularity=fine,compact,1,0"
os.environ["ONEDNN_VERBOSE"]  = "0"

OUT_DIR = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent.parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL = f"{args.variant}_{args.model}"

# ── Resolve paths: base (pubmed pass1 PyTorch) vs finetuned ──────────────────
def resolve(variant, model_key):
    if variant.startswith("pytorch"):
        return PT_PATHS[model_key]
    prec = variant.split("-")[1]
    return OV_PATHS[f"{model_key}_{prec}"]

# Base = pubmed-pass1 PyTorch fp32 reference; tuned = requested (variant, model)
BASE_PATH = PT_PATHS["pubmed"]
BASE_VARIANT = "pytorch-fp32"
TUNED_PATH = resolve(args.variant, args.model)
TUNED_VARIANT = args.variant

if args.variant.startswith("ov") and not (Path(TUNED_PATH) / "openvino_model.xml").exists():
    print(f"ERROR: OV model not found: {TUNED_PATH}")
    sys.exit(1)

# ── Backends ────────────────────────────────────────────────────────────────
_pt_cache, _ov_cache, _tok_cache = {}, {}, {}

def _tok(path):
    if path not in _tok_cache:
        from transformers import AutoTokenizer
        _tok_cache[path] = AutoTokenizer.from_pretrained(path)
    return _tok_cache[path]

def _pt(path, dtype_str):
    key = (path, dtype_str)
    if key in _pt_cache:
        return _pt_cache[key]
    import torch
    from transformers import AutoModel
    torch.set_num_threads(N_CORES)
    dmap = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    print(f"  [pt-load] {Path(path).name} ({dtype_str})", flush=True)
    _pt_cache[key] = AutoModel.from_pretrained(path, torch_dtype=dmap[dtype_str]).eval()
    return _pt_cache[key]

def _ov(path):
    if path in _ov_cache:
        return _ov_cache[path]
    import openvino as ov
    core = ov.Core()
    cfg = {"PERFORMANCE_HINT": "THROUGHPUT",
           "INFERENCE_NUM_THREADS": str(N_CORES),
           "INFERENCE_PRECISION_HINT": "bf16"}
    print(f"  [ov-compile] {Path(path).name}", flush=True)
    _ov_cache[path] = core.compile_model(str(Path(path)/"openvino_model.xml"), "CPU", cfg)
    return _ov_cache[path]

def embed(variant, path, texts, bs=128):
    tok = _tok(path); dtype_str = variant.split("-")[1]
    out = []
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


# ── BIOSSES ─────────────────────────────────────────────────────────────────
print("  Loading BIOSSES ...", flush=True)
biosses = load_dataset("mteb/biosses-sts", split="test")
biosses_pairs = list(zip(biosses["sentence1"], biosses["sentence2"]))
biosses_gold  = np.array(biosses["score"])

# ── Pair sets (kept identical to original) ──────────────────────────────────
CROSS_CONNECTED = [
    ("BRCA1 pathogenic variant detected in germline DNA sequencing",
     "Patient diagnosed with hereditary breast and ovarian cancer syndrome"),
    ("TP53 R175H missense mutation dominant negative effect",
     "Li-Fraumeni syndrome: multiple early-onset malignancies in family"),
    ("EGFR exon 19 deletion driver mutation lung adenocarcinoma",
     "Patient started on erlotinib targeted therapy for lung cancer"),
    ("KRAS G12C activating mutation colorectal cancer",
     "Tumour progressed despite standard chemotherapy regimen"),
    ("Serum cortisol 32 μg/dL markedly elevated HPA axis dysregulation",
     "Patient reports persistent anxiety, sleep disruption and mood instability"),
    ("HbA1c 9.2% sustained hyperglycaemia poor glycaemic control",
     "Patient describes fatigue, low mood and difficulty concentrating daily"),
    ("Troponin I 3.8 ng/mL myocardial necrosis acute coronary syndrome",
     "Patient admitted with crushing chest pain, diaphoresis and fear of death"),
    ("TSH suppressed free T4 elevated clinical hyperthyroidism",
     "Reported palpitations, anxiety, heat intolerance and weight loss"),
    ("FMR1 CGG repeat expansion fragile X syndrome",
     "Patient with intellectual disability and seizures since early childhood"),
    ("DMD frameshift deletion dystrophin absent",
     "Progressive proximal muscle weakness onset age five"),
    ("PHQ-9 score 18 severe depression suicidal ideation",
     "Patient admitted to psychiatry ward, started on SSRI and CBT"),
    ("Journaled: disconnected from reality, visual disturbances, paranoia",
     "First-episode psychosis initiated antipsychotics inpatient"),
    ("LDL 185 mg/dL HDL 38 mg/dL triglycerides 290 mg/dL",
     "High-intensity statin therapy and dietary modification for dyslipidaemia"),
    ("BNP 820 pg/mL decompensated congestive heart failure",
     "Admitted for IV diuresis optimisation of heart failure therapy"),
    ("Cortisol elevation Cushing syndrome adrenal hyperactivity",
     "Central obesity striae hypertension glucose intolerance"),
    ("APOE e4 allele amyloid accumulation Alzheimer risk",
     "Cognitive decline progressive memory loss executive dysfunction"),
]

HARD_NEGATIVES = [
    ("BRCA1 variant hereditary breast cancer genetic counselling",
     "BRCA2 ovarian cancer risk-reducing salpingo-oophorectomy decision"),
    ("HbA1c 8.5% type 2 diabetes glycaemic control review",
     "HbA1c testing frequency monitoring guideline laboratory protocol"),
    ("TSH elevated hypothyroidism levothyroxine dose adjustment",
     "TSH suppressed hyperthyroidism radioiodine ablation thyroid"),
    ("Cortisol morning serum measurement adrenal insufficiency",
     "Cortisol salivary diurnal rhythm testing Cushing investigation"),
    ("PHQ-9 depression screening primary care threshold",
     "PHQ-9 scoring administration instructions patient handout"),
    ("EGFR mutation targeted therapy erlotinib response",
     "EGFR pathway inhibitor resistance mechanism preclinical model"),
    ("Troponin elevation myocardial infarction emergency pathway",
     "Troponin assay high-sensitivity analytical performance laboratory"),
    ("BNP cardiac biomarker heart failure diagnosis threshold",
     "BNP NT-proBNP analytical variation preanalytical confounders"),
    ("Patient reports low mood fatigue anhedonia two weeks",
     "Ferritin 6 ng/mL microcytic anaemia iron deficiency treatment"),
    ("KRAS mutation colorectal cancer oncology tumour board",
     "Patient anxiety about chemotherapy counselling support referral"),
    ("Lumbar puncture CSF pleocytosis meningitis infectious",
     "Lumbar disc herniation L4-L5 radiculopathy physiotherapy"),
    ("Spirometry obstructive pattern COPD bronchodilator reversibility",
     "Spirometry restrictive pattern interstitial lung fibrosis"),
]

INTRA_DOMAIN = {
    "genomics": [
        ("BRCA1 pathogenic variant germline hereditary breast ovarian cancer",
         "BRCA2 frameshift mutation familial cancer syndrome genetic testing"),
        ("TP53 missense mutation dominant negative tumour suppressor",
         "CDKN2A deletion loss of function cell cycle checkpoint"),
        ("EGFR exon 19 deletion lung adenocarcinoma driver mutation",
         "KRAS G12C activating mutation colorectal cancer oncogene"),
    ],
    "psychology": [
        ("PHQ-9 score 18 severe depression suicidal ideation anhedonia",
         "GAD-7 score 15 generalised anxiety disorder rumination worry"),
        ("Journaled: low mood hopelessness social withdrawal fatigue",
         "Mood score 2/10 feeling disconnected unable to concentrate"),
        ("First-episode psychosis hallucinations paranoid ideation",
         "Schizophrenia positive symptoms delusions auditory hallucinations"),
    ],
    "biomarkers": [
        ("HbA1c 9.2% sustained hyperglycaemia insulin resistance",
         "Fasting glucose 11.4 mmol/L type 2 diabetes poor control"),
        ("Troponin I 3.8 ng/mL myocardial necrosis acute MI",
         "CK-MB elevated creatine kinase cardiac ischaemia injury"),
        ("BNP 820 pg/mL congestive heart failure decompensated",
         "NT-proBNP 4200 pg/mL volume overload cardiac stretch marker"),
    ],
    "clinical": [
        ("Patient admitted chest pain diaphoresis ECG ST elevation",
         "Emergency PCI performed left anterior descending artery stent"),
        ("Lumbar puncture protein elevated lymphocytic pleocytosis",
         "CSF culture gram-positive diplococci bacterial meningitis"),
        ("Spirometry FEV1/FVC 0.62 obstructive airway bronchodilator",
         "COPD GOLD stage III exacerbation inhaler escalation review"),
    ],
}

INTER_DOMAIN = [
    ("BRCA1 pathogenic variant hereditary breast cancer",
     "PHQ-9 severe depression suicidal ideation CBT"),
    ("HbA1c 9.2% diabetes insulin resistance glycaemic",
     "TP53 missense mutation Li-Fraumeni tumour suppressor"),
    ("Troponin MI acute coronary syndrome emergency",
     "Journaled: low mood hopeless social withdrawal"),
    ("EGFR lung adenocarcinoma erlotinib targeted therapy",
     "BNP congestive heart failure IV diuresis"),
    ("KRAS colorectal cancer mutation oncology",
     "TSH hypothyroidism levothyroxine thyroid"),
    ("Cortisol adrenal dysregulation HPA axis",
     "Spirometry COPD obstructive airway FEV1"),
    ("APOE Alzheimer amyloid cognitive decline",
     "LDL dyslipidaemia statin therapy cardiovascular"),
    ("Lumbar puncture CSF meningitis pleocytosis",
     "PHQ-9 depression screening primary care"),
    ("DMD frameshift muscular dystrophy dystrophin",
     "Fasting glucose insulin resistance diabetes"),
    ("FMR1 fragile X intellectual disability seizures",
     "Troponin cardiac biomarker myocardial infarction"),
    ("Schizophrenia antipsychotics hallucinations psychosis",
     "EGFR mutation lung cancer targeted therapy"),
    ("CDKN2A melanoma cell cycle checkpoint",
     "BNP heart failure volume overload cardiac"),
]
all_intra = [p for pairs in INTRA_DOMAIN.values() for p in pairs]

# ── Run ─────────────────────────────────────────────────────────────────────
RUNS = {
    "base_fp32":     (BASE_VARIANT,  BASE_PATH),
    f"tuned_{args.variant}": (TUNED_VARIANT, TUNED_PATH),
}
all_results = {}

print(f"\n{'='*72}")
print(f"  FULL EVALUATION — {LABEL}")
print(f"{'='*72}")

for mkey, (variant, path) in RUNS.items():
    print(f"\n  ── {mkey} ({Path(path).name}) ──")
    bio_sims = pair_cos(variant, path, biosses_pairs)
    spear_r, spear_p = spearmanr(bio_sims, biosses_gold)
    print(f"  [1] BIOSSES Spearman      : {spear_r:.4f}  (p={spear_p:.2e})")

    cc = pair_cos(variant, path, CROSS_CONNECTED)
    cd_acc = float(np.mean([s > 0.50 for s in cc]))
    print(f"  [2] Cross-domain acc      : {cd_acc*100:.1f}%  (mean={np.mean(cc):.3f})")

    hn = pair_cos(variant, path, HARD_NEGATIVES)
    pw_correct = sum(s > 0.50 for s in cc) + sum(s < 0.50 for s in hn)
    pw_total = len(cc) + len(hn)
    pw_acc = pw_correct / pw_total
    print(f"  [3] Pairwise sim accuracy : {pw_acc*100:.1f}%  ({pw_correct}/{pw_total})")

    hn_det = float(np.mean([s < 0.40 for s in hn]))
    print(f"  [4] Hard-negative detect  : {hn_det*100:.1f}%  (mean={np.mean(hn):.3f})")

    intra = pair_cos(variant, path, all_intra)
    inter = pair_cos(variant, path, INTER_DOMAIN)
    ratio = float(np.mean(intra) / np.mean(inter)) if np.mean(inter) > 0 else float("inf")
    print(f"  [5] Intra={np.mean(intra):.3f}  Inter={np.mean(inter):.3f}  Ratio={ratio:.3f}x")

    all_results[mkey] = {
        "biosses_spearman":  round(float(spear_r), 4),
        "biosses_p":         float(spear_p),
        "cross_domain_acc":  round(float(cd_acc), 4),
        "cross_domain_mean": round(float(np.mean(cc)), 4),
        "pairwise_acc":      round(float(pw_acc), 4),
        "hard_neg_detect":   round(float(hn_det), 4),
        "hard_neg_mean":     round(float(np.mean(hn)), 4),
        "intra_domain_mean": round(float(np.mean(intra)), 4),
        "inter_domain_mean": round(float(np.mean(inter)), 4),
        "intra_inter_ratio": round(ratio, 4),
        "intra_by_domain":   {d: round(float(np.mean(pair_cos(variant, path, p))), 4)
                              for d, p in INTRA_DOMAIN.items()},
    }

print(f"\n{'='*72}")
print(f"  SUMMARY — {LABEL}")
print(f"{'='*72}")
keys = ["biosses_spearman","cross_domain_acc","pairwise_acc","hard_neg_detect","intra_inter_ratio"]
for k in keys:
    print(f"  {k:<22}  base={all_results['base_fp32'][k]}  tuned={all_results[f'tuned_{args.variant}'][k]}")

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out = OUT_DIR / f"eval_full_{LABEL}_{ts}.json"
json.dump({"label": LABEL, "variant": args.variant, "model": args.model,
           "run_at": datetime.now().isoformat(), "models": all_results},
          open(out, "w"), indent=2)
print(f"\n[saved] {out}")
