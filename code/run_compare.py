"""
run_compare.py  --base <ov_dir>  --tuned <ov_dir>  --label <name>
Thin wrapper that runs the same evaluation as compare_finetuned.py
for any base/finetuned OV model pair.
"""
import argparse, json, os, time, warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import openvino as ov
import torch
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--base",  required=True)
parser.add_argument("--tuned", required=True)
parser.add_argument("--label", default="model")
args = parser.parse_args()

BASE_OV_DIR  = Path(args.base)
TUNED_OV_DIR = Path(args.tuned)
LABEL        = args.label

N_CORES = os.cpu_count()
torch.set_num_threads(N_CORES)
os.environ.update({
    "OMP_NUM_THREADS": str(N_CORES), "MKL_NUM_THREADS": str(N_CORES),
    "KMP_BLOCKTIME": "1", "KMP_AFFINITY": "granularity=fine,compact,1,0",
    "ONEDNN_VERBOSE": "0",
})

BENCHMARKS_DIR = Path(__file__).parent / "benchmarks"
BENCHMARKS_DIR.mkdir(exist_ok=True)
BATCH_SIZES = [128, 256, 384, 512]
N_THROUGHPUT = 512

OV_CORE = ov.Core()

CONNECTED_PAIRS = [
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

DISCONNECTED_PAIRS = [
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

_BASE_TEXTS = [
    "BRCA1 mutation linked to hereditary breast and ovarian cancer",
    "Patient exhibits elevated cortisol and reports persistent anxiety",
    "DNA methylation changes observed in promoter regions",
    "Feeling very low energy, mood score 2/10 today",
    "HbA1c 8.5%, fasting glucose 180 mg/dL, insulin resistance suspected",
    "Hallucinations and paranoid ideation consistent with psychosis",
    "EGFR amplification detected in tumor biopsy sample",
    "Journaled: slept poorly, high stress at work, mood deteriorating",
    "TP53 R175H missense mutation found in Li-Fraumeni syndrome patient",
    "TSH 0.02 suppressed, free T4 elevated, clinical hyperthyroidism",
    "Serum cortisol 28.4 µg/dL above reference range, patient fatigued",
    "Troponin I 2.1 ng/mL markedly elevated, consistent with MI",
    "KRAS G12D mutation in pancreatic ductal adenocarcinoma specimen",
    "Patient scored 6/10 on PHQ-9, sleep disrupted, anhedonia present",
    "LDL 185 mg/dL, HDL 38 mg/dL, triglycerides 290 mg/dL, statin review",
    "CDKN2A deletion identified in melanoma via NGS panel",
]
THROUGHPUT_TEXTS = (_BASE_TEXTS * ((N_THROUGHPUT // len(_BASE_TEXTS)) + 1))[:N_THROUGHPUT]

_ov_cache: dict = {}
_tok_cache: dict = {}

def _compile(model_dir: Path) -> ov.CompiledModel:
    k = str(model_dir)
    if k not in _ov_cache:
        xml = model_dir / "openvino_model.xml"
        cfg = {"PERFORMANCE_HINT": "THROUGHPUT",
               "INFERENCE_NUM_THREADS": str(N_CORES),
               "INFERENCE_PRECISION_HINT": "bf16"}
        print(f"  [compile] {model_dir.name} ...", flush=True)
        _ov_cache[k] = OV_CORE.compile_model(str(xml), "CPU", cfg)
    return _ov_cache[k]

def _tok(model_dir: Path) -> AutoTokenizer:
    k = str(model_dir)
    if k not in _tok_cache:
        _tok_cache[k] = AutoTokenizer.from_pretrained(k)
    return _tok_cache[k]

def _embed(model_dir: Path, texts: list) -> np.ndarray:
    compiled = _compile(model_dir)
    tok      = _tok(model_dir)
    valid    = [i.get_any_name() for i in compiled.inputs]
    enc      = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="np")
    inputs   = {k: v for k, v in enc.items() if k in valid}
    out      = compiled(inputs)
    lhs      = list(out.values())[0]
    mask     = enc["attention_mask"][..., np.newaxis].astype(np.float32)
    pooled   = (lhs * mask).sum(1) / mask.sum(1)
    norms    = np.linalg.norm(pooled, axis=1, keepdims=True)
    return pooled / np.clip(norms, 1e-8, None)

def eval_pairs(model_dir, pairs):
    emb_a = _embed(model_dir, [p[0] for p in pairs])
    emb_p = _embed(model_dir, [p[1] for p in pairs])
    return [float(np.dot(emb_a[i], emb_p[i])) for i in range(len(pairs))]

def bench_throughput(model_dir, texts, bs):
    compiled = _compile(model_dir)
    tok      = _tok(model_dir)
    valid    = [i.get_any_name() for i in compiled.inputs]
    enc = tok(texts[:bs], padding=True, truncation=True, max_length=512, return_tensors="np")
    compiled({k: v for k, v in enc.items() if k in valid})
    t0 = time.perf_counter()
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i+bs], padding=True, truncation=True, max_length=512, return_tensors="np")
        compiled({k: v for k, v in enc.items() if k in valid})
    return round(len(texts) / (time.perf_counter() - t0), 1)

# ── Run ───────────────────────────────────────────────────────────────────────
print(f"\n[platform] {N_CORES} cores | OV {ov.__version__.split('-')[0]}")
print(f"  Label    : {LABEL}")
print(f"  Base     : {BASE_OV_DIR}")
print(f"  Finetuned: {TUNED_OV_DIR}\n")

models = {"base_bf16": BASE_OV_DIR, "finetuned_bf16": TUNED_OV_DIR}
sim_data: dict = {}
results = {"run_at": datetime.now().isoformat(), "label": LABEL,
           "platform": f"{N_CORES} cores, OV {ov.__version__.split('-')[0]}", "models": {}}

print("=" * 70)
print("  SIMILARITY EVALUATION")
print("=" * 70)

for mkey, mdir in models.items():
    print(f"\n  [{mkey}]")
    conn  = eval_pairs(mdir, CONNECTED_PAIRS)
    disc  = eval_pairs(mdir, DISCONNECTED_PAIRS)
    sim_data[mkey] = {"connected": conn, "disconnected": disc}
    print(f"  Connected    (n={len(conn)})  mean={np.mean(conn):.3f}  "
          f"min={min(conn):.3f}  max={max(conn):.3f}")
    print(f"  Disconnected (n={len(disc)})  mean={np.mean(disc):.3f}  "
          f"min={min(disc):.3f}  max={max(disc):.3f}")

print(f"\n{'─'*70}")
print(f"  CONNECTED PAIRS — per-pair similarity")
print(f"  {'Pair':<50}  {'Base':>6}  {'Tuned':>6}  {'Δ':>7}")
print(f"  {'─'*65}")
for i, (a, b) in enumerate(CONNECTED_PAIRS):
    label = f"{a[:28]}… / {b[:18]}…"
    bs, ts = sim_data["base_bf16"]["connected"][i], sim_data["finetuned_bf16"]["connected"][i]
    d = ts - bs
    print(f"  {label:<50}  {bs:>6.3f}  {ts:>6.3f}  {d:>+7.3f}{'  ▲' if d>0.05 else ('  ▼' if d<-0.05 else '')}")

print(f"\n{'─'*70}")
print(f"  DISCONNECTED PAIRS — per-pair similarity")
print(f"  {'Pair':<50}  {'Base':>6}  {'Tuned':>6}  {'Δ':>7}")
print(f"  {'─'*65}")
for i, (a, b) in enumerate(DISCONNECTED_PAIRS):
    label = f"{a[:28]}… / {b[:18]}…"
    bs, ts = sim_data["base_bf16"]["disconnected"][i], sim_data["finetuned_bf16"]["disconnected"][i]
    d = ts - bs
    print(f"  {label:<50}  {bs:>6.3f}  {ts:>6.3f}  {d:>+7.3f}{'  ▼' if d<-0.05 else ('  ▲' if d>0.05 else '')}")

base_gap  = np.mean(sim_data["base_bf16"]["connected"])  - np.mean(sim_data["base_bf16"]["disconnected"])
tuned_gap = np.mean(sim_data["finetuned_bf16"]["connected"]) - np.mean(sim_data["finetuned_bf16"]["disconnected"])
print(f"\n  Discrimination gap (connected_mean − disconnected_mean):")
print(f"    Base     : {base_gap:+.3f}")
print(f"    Finetuned: {tuned_gap:+.3f}  (Δ = {tuned_gap - base_gap:+.3f})")

for mkey in models:
    ca = np.mean([s > 0.50 for s in sim_data[mkey]["connected"]])  * 100
    da = np.mean([s < 0.35 for s in sim_data[mkey]["disconnected"]]) * 100
    print(f"  {mkey:<20}  connected>0.50: {ca:.0f}%   disconnected<0.35: {da:.0f}%")

print(f"\n{'='*70}")
print(f"  THROUGHPUT  (sent/sec)  n={N_THROUGHPUT}")
print(f"  {'Model':<22}  " + "  ".join(f"bs={b:>3}" for b in BATCH_SIZES))
print(f"  {'─'*66}")

throughput_data: dict = {m: {} for m in models}
for mkey, mdir in models.items():
    row = []
    for bs in BATCH_SIZES:
        sps = bench_throughput(mdir, THROUGHPUT_TEXTS, bs)
        throughput_data[mkey][bs] = sps
        row.append(f"{sps:>8.1f}")
    print(f"  {mkey:<22}  " + "  ".join(row))

for mkey in models:
    results["models"][mkey] = {
        "connected_scores":    sim_data[mkey]["connected"],
        "disconnected_scores": sim_data[mkey]["disconnected"],
        "connected_mean":      float(np.mean(sim_data[mkey]["connected"])),
        "disconnected_mean":   float(np.mean(sim_data[mkey]["disconnected"])),
        "discrimination_gap":  float(np.mean(sim_data[mkey]["connected"]) -
                                     np.mean(sim_data[mkey]["disconnected"])),
        "connected_acc_50":    float(np.mean([s > 0.50 for s in sim_data[mkey]["connected"]])),
        "disconnected_acc_35": float(np.mean([s < 0.35 for s in sim_data[mkey]["disconnected"]])),
        "throughput":          throughput_data[mkey],
    }

ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
out = BENCHMARKS_DIR / f"compare_{LABEL}_{ts}.json"
json.dump(results, open(out, "w"), indent=2)
print(f"\n[saved] {out}")