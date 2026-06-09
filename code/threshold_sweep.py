"""
threshold_sweep.py
- Sweeps classification threshold 0.25→0.65 for all finetuned models
- Reports TPR, TNR, pairwise accuracy, F1 at each threshold
- Highlights optimal threshold
- Reports BODHI-pass2 prediction accuracy at all thresholds
"""
import os, sys, json, warnings
import numpy as np
import openvino as ov
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")
N_CORES = len(os.sched_getaffinity(0))
os.environ.update({
    "OMP_NUM_THREADS": str(N_CORES), "MKL_NUM_THREADS": str(N_CORES),
    "KMP_BLOCKTIME": "1", "KMP_AFFINITY": "granularity=fine,compact,1,0",
    "ONEDNN_VERBOSE": "0",
})

from config import OV_PATHS, PT_PATHS  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OV_CORE = ov.Core()
_compiled_cache, _tok_cache = {}, {}

def compile_ov(d: Path):
    k = str(d)
    if k not in _compiled_cache:
        cfg = {"PERFORMANCE_HINT": "LATENCY",
               "INFERENCE_NUM_THREADS": str(N_CORES),
               "INFERENCE_PRECISION_HINT": "bf16"}
        print(f"  [compile] {d.name}", flush=True)
        _compiled_cache[k] = OV_CORE.compile_model(str(d/"openvino_model.xml"), "CPU", cfg)
    return _compiled_cache[k]

def get_tok(d: Path):
    k = str(d)
    if k not in _tok_cache:
        _tok_cache[k] = AutoTokenizer.from_pretrained(k)
    return _tok_cache[k]

def embed(model_dir: Path, texts: list, bs=128) -> np.ndarray:
    m   = compile_ov(model_dir)
    tok = get_tok(model_dir)
    valid = [i.get_any_name() for i in m.inputs]
    parts = []
    for i in range(0, len(texts), bs):
        enc  = tok(texts[i:i+bs], padding=True, truncation=True, max_length=512, return_tensors="np")
        inp  = {k: v for k, v in enc.items() if k in valid}
        lhs  = list(m(inp).values())[0]
        mask = enc["attention_mask"][..., np.newaxis].astype(np.float32)
        pooled = (lhs * mask).sum(1) / mask.sum(1)
        parts.append(pooled)
    emb = np.vstack(parts)
    return emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8, None)

def cosines(a, b):
    return [float(np.dot(a[i], b[i])) for i in range(len(a))]

# ── Pair sets ─────────────────────────────────────────────────────────────────
CROSS_CONNECTED = [
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
HARD_NEGATIVES = [
    ("BRCA1 pathogenic variant hereditary cancer syndrome",
     "BRCA2 counselling risk-reduction surgery options discussed"),
    ("HbA1c 9.2% poor glycaemic control type 2 diabetes",
     "HbA1c monitoring protocol adjusted insulin dose titration"),
    ("TSH suppressed clinical hyperthyroidism Graves disease",
     "TSH receptor antibody positive autoimmune thyroid disease screening"),
    ("PHQ-9 score 18 severe depression suicidal ideation",
     "PHQ-9 follow-up score 12 partial response antidepressant therapy"),
    ("KRAS G12C mutation colorectal cancer chemotherapy resistance",
     "NRAS mutation metastatic colorectal cancer EGFR inhibitor contraindicated"),
    ("Troponin I elevated acute myocardial infarction STEMI",
     "Troponin trend serial measurement rule-out NSTEMI protocol"),
    ("BNP elevated decompensated heart failure fluid overload",
     "NT-proBNP elevated chronic heart failure outpatient diuretic optimisation"),
    ("Lumbar spine MRI disc herniation L4-L5 nerve root compression",
     "Cervical spine stenosis C5-C6 myelopathy upper limb symptoms"),
    ("FEV1/FVC ratio 0.62 COPD GOLD stage II bronchodilator therapy",
     "Restrictive lung disease TLC reduced interstitial fibrosis pattern"),
    ("Serum cortisol elevated adrenal insufficiency exclusion protocol",
     "ACTH stimulation test cortisol response blunted adrenal dysfunction"),
    ("EGFR exon 19 deletion NSCLC first-line osimertinib therapy",
     "ALK rearrangement NSCLC crizotinib targeted therapy initiated"),
    ("DMD frameshift mutation absent dystrophin Duchenne diagnosis",
     "BMD in-frame mutation reduced dystrophin Becker muscular dystrophy"),
]

MODELS = {
    "PubMedBERT pass1 (bf16)":     Path(OV_PATHS["pubmed_bf16"]),
    "PubMedBERT BODHI (bf16)":     Path(OV_PATHS["bodhi_bf16"]),
    "BioBERT finetuned (bf16)":    Path(OV_PATHS["biobert_bf16"]),
    "PubMedBERT pass1 (int8)":     Path(OV_PATHS["pubmed_int8"]),
    "PubMedBERT BODHI (int8)":     Path(OV_PATHS["bodhi_int8"]),
    "BioBERT finetuned (int8)":    Path(OV_PATHS["biobert_int8"]),
}
MODELS = {k: v for k, v in MODELS.items() if (v / "openvino_model.xml").exists()}

THRESHOLDS = np.arange(0.25, 0.66, 0.05)

print(f"\n[platform] {N_CORES} cores | OV {ov.__version__.split('-')[0]}")
print("=" * 80)
print("  THRESHOLD SWEEP  (positive=CROSS_CONNECTED, negative=HARD_NEGATIVES)")
print("=" * 80)

all_results = {}

for mname, mdir in MODELS.items():
    label = mname.replace("\n  ", " ")
    print(f"\n  ── {label} ──")
    ea_pos = embed(mdir, [p[0] for p in CROSS_CONNECTED])
    eb_pos = embed(mdir, [p[1] for p in CROSS_CONNECTED])
    pos_sims = np.array(cosines(ea_pos, eb_pos))

    ea_neg = embed(mdir, [p[0] for p in HARD_NEGATIVES])
    eb_neg = embed(mdir, [p[1] for p in HARD_NEGATIVES])
    neg_sims = np.array(cosines(ea_neg, eb_neg))

    n_pos, n_neg = len(pos_sims), len(neg_sims)
    print(f"  Pos sims: mean={pos_sims.mean():.3f}  min={pos_sims.min():.3f}  max={pos_sims.max():.3f}")
    print(f"  Neg sims: mean={neg_sims.mean():.3f}  min={neg_sims.min():.3f}  max={neg_sims.max():.3f}")
    print(f"\n  {'Thresh':>7}  {'TPR':>6}  {'TNR':>6}  {'Acc':>6}  {'Prec':>6}  {'F1':>6}  {'Overlap':>8}")
    print(f"  {'─'*60}")

    sweep = []
    best_f1, best_t = 0, 0
    for t in THRESHOLDS:
        tp = (pos_sims >= t).sum()
        tn = (neg_sims <  t).sum()
        fp = (neg_sims >= t).sum()
        fn = (pos_sims <  t).sum()
        tpr  = tp / n_pos
        tnr  = tn / n_neg
        acc  = (tp + tn) / (n_pos + n_neg)
        prec = tp / max(tp + fp, 1)
        rec  = tpr
        f1   = 2 * prec * rec / max(prec + rec, 1e-8)
        errors = fn + fp
        mark = " ◄ best F1" if f1 > best_f1 else ""
        print(f"  t={t:.2f}   {tpr*100:>5.1f}%  {tnr*100:>5.1f}%  {acc*100:>5.1f}%  "
              f"{prec*100:>5.1f}%  {f1*100:>5.1f}%  err={errors:>2}/{n_pos+n_neg}{mark}")
        sweep.append({"t": round(float(t),2), "tpr": round(tpr,4), "tnr": round(tnr,4),
                      "acc": round(acc,4), "prec": round(prec,4), "f1": round(f1,4)})
        if f1 > best_f1:
            best_f1, best_t = f1, t

    print(f"\n  → Optimal threshold: {best_t:.2f}  (F1={best_f1*100:.1f}%)")
    all_results[label] = {"pos_sims": pos_sims.tolist(), "neg_sims": neg_sims.tolist(), "sweep": sweep}

# ── BODHI pass2 detailed accuracy ─────────────────────────────────────────────
print(f"\n{'='*80}")
print("  BODHI PASS-2 (PubMedBERT) — PREDICTION ACCURACY DETAIL")
print(f"{'='*80}")
bodhi_key = "PubMedBERT BODHI (bf16)"
if bodhi_key in all_results:
    r = all_results[bodhi_key]
    pos_sims = np.array(r["pos_sims"])
    neg_sims = np.array(r["neg_sims"])
    print(f"\n  Positive pairs (should be > threshold):")
    for i, (p, s) in enumerate(zip(CROSS_CONNECTED, pos_sims)):
        flag = "✓" if s >= 0.40 else "✗"
        print(f"  {flag} {s:.3f}  {p[0][:45]:<45} | {p[1][:35]}")
    print(f"\n  Hard-negative pairs (should be < threshold):")
    for i, (p, s) in enumerate(zip(HARD_NEGATIVES, neg_sims)):
        flag = "✓" if s < 0.40 else "✗"
        print(f"  {flag} {s:.3f}  {p[0][:45]:<45} | {p[1][:35]}")

    for t in [0.35, 0.40, 0.45, 0.50]:
        tp = (pos_sims >= t).sum(); tn = (neg_sims < t).sum()
        n = len(pos_sims) + len(neg_sims)
        print(f"\n  Threshold {t:.2f}: accuracy={(tp+tn)/n*100:.1f}%  "
              f"({tp}/{len(pos_sims)} pos correct, {tn}/{len(neg_sims)} neg correct)")

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out = OUT_DIR / f"threshold_sweep_{ts}.json"
json.dump(all_results, out.open("w"), indent=2)
print(f"\n[saved] {out}")