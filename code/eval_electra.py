"""
eval_electra.py — ELECTRA evaluation: base via PyTorch, finetuned via OV BF16.
Usage: python eval_electra.py
"""
import os, json, warnings
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import openvino as ov

warnings.filterwarnings("ignore")

N_CORES = os.cpu_count()
torch.set_num_threads(N_CORES)
os.environ.update({
    "OMP_NUM_THREADS": str(N_CORES), "MKL_NUM_THREADS": str(N_CORES),
    "KMP_BLOCKTIME": "1", "KMP_AFFINITY": "granularity=fine,compact,1,0",
    "ONEDNN_VERBOSE": "0",
})

BASE_HF   = "sultan/BioM-ELECTRA-Large-SQuAD2"
TUNED_OV  = Path(__file__).resolve().parent.parent / "models" / "openvino" / "finetuned-electra-bf16"
BENCHMARKS = Path(__file__).parent / "benchmarks"
BENCHMARKS.mkdir(exist_ok=True)

# ── data ─────────────────────────────────────────────────────────────────────
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
INTRA_DOMAIN = {
    "genomics": [
        ("BRCA1 pathogenic variant hereditary breast cancer",
         "BRCA2 deleterious mutation ovarian cancer risk"),
        ("TP53 gain-of-function mutation Li-Fraumeni syndrome",
         "CDKN2A deletion melanoma familial cancer predisposition"),
        ("KRAS G12D oncogenic mutation pancreatic adenocarcinoma",
         "NRAS Q61K activating mutation melanoma RAS pathway"),
    ],
    "psychology": [
        ("PHQ-9 score 18 severe depression suicidal ideation",
         "GAD-7 score 15 generalised anxiety disorder panic attacks"),
        ("First-episode psychosis hallucinations paranoid ideation",
         "Bipolar disorder manic episode grandiosity decreased sleep"),
        ("PTSD hypervigilance intrusive memories trauma exposure",
         "OCD obsessional thoughts compulsive checking behaviour"),
    ],
    "biomarkers": [
        ("HbA1c 9.2% fasting glucose 180 mg/dL insulin resistance",
         "Fasting insulin elevated HOMA-IR 4.8 metabolic syndrome"),
        ("Troponin I 3.8 ng/mL myocardial injury acute coronary",
         "CK-MB elevated creatine kinase myocardial band infarction"),
        ("BNP 820 pg/mL congestive heart failure ventricular dysfunction",
         "NT-proBNP 3200 pg/mL decompensated cardiac failure prognosis"),
    ],
    "clinical": [
        ("Chest pain diaphoresis dyspnoea acute coronary syndrome",
         "Palpitations syncope exertional angina unstable cardiac"),
        ("Fatigue weight gain cold intolerance hypothyroid symptoms",
         "Heat intolerance weight loss tremor hyperthyroid clinical"),
        ("Memory loss word-finding difficulty early dementia presentation",
         "Executive dysfunction personality change frontal lobe cognitive"),
    ],
}
INTER_DOMAIN = [
    ("BRCA1 pathogenic germline mutation", "PHQ-9 score 18 severe depression"),
    ("Troponin I 3.8 myocardial infarction", "KRAS G12C colorectal cancer mutation"),
    ("HbA1c 9.2% glycaemic control", "First-episode psychosis hallucinations"),
    ("BNP elevated heart failure", "TP53 Li-Fraumeni syndrome mutation"),
    ("FEV1/FVC COPD obstructive airways", "Serum cortisol HPA axis adrenal"),
    ("Lumbar disc herniation L4-L5", "TSH suppressed hyperthyroidism Graves"),
    ("Spirometry restrictive pattern fibrosis", "PHQ-9 depression suicidal ideation"),
    ("Creatinine elevated acute kidney injury", "EGFR mutation lung adenocarcinoma"),
    ("LDL 185 dyslipidaemia statin therapy", "FMR1 fragile X intellectual disability"),
    ("APOE e4 Alzheimer amyloid risk", "HbA1c diabetes glycaemic control"),
    ("DMD dystrophin absent muscular dystrophy", "BNP heart failure decompensated"),
    ("Cortisol Cushing central obesity striae", "KRAS colorectal chemotherapy resistance"),
]

# ── encoders ─────────────────────────────────────────────────────────────────
print(f"\n[platform] {N_CORES} cores | OV {ov.__version__.split('-')[0]}")

# Base: PyTorch mean-pool
print("  [load] base PyTorch ...", flush=True)
base_tok = AutoTokenizer.from_pretrained(BASE_HF)
base_mdl = AutoModel.from_pretrained(BASE_HF)
base_mdl.eval()

def embed_pt(texts, bs=64):
    all_emb = []
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        for i in range(0, len(texts), bs):
            enc = base_tok(texts[i:i+bs], padding=True, truncation=True,
                           max_length=512, return_tensors="pt")
            out = base_mdl(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out * mask).sum(1) / mask.sum(1)
            all_emb.append(pooled.float().numpy())
    emb = np.vstack(all_emb)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.clip(norms, 1e-8, None)

# Finetuned: OV BF16
print("  [compile] finetuned OV ...", flush=True)
OV_CORE = ov.Core()
cfg = {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": str(N_CORES), "INFERENCE_PRECISION_HINT": "bf16"}
ov_compiled = OV_CORE.compile_model(str(TUNED_OV / "openvino_model.xml"), "CPU", cfg)
ov_tok = AutoTokenizer.from_pretrained(str(TUNED_OV))
ov_valid = [i.get_any_name() for i in ov_compiled.inputs]

def embed_ov(texts, bs=64):
    all_emb = []
    for i in range(0, len(texts), bs):
        enc = ov_tok(texts[i:i+bs], padding=True, truncation=True,
                     max_length=512, return_tensors="np")
        inp = {k: v for k, v in enc.items() if k in ov_valid}
        lhs = list(ov_compiled(inp).values())[0]
        mask = enc["attention_mask"][..., np.newaxis].astype(np.float32)
        pooled = (lhs * mask).sum(1) / mask.sum(1)
        all_emb.append(pooled)
    emb = np.vstack(all_emb)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.clip(norms, 1e-8, None)

def cosines(emb_a, emb_b):
    return [float(np.dot(emb_a[i], emb_b[i])) for i in range(len(emb_a))]

# ── BIOSSES ───────────────────────────────────────────────────────────────────
print("\n  Loading BIOSSES ...", flush=True)
try:
    ds = load_dataset("mteb/biosses-sts", split="test")
    biosses_pairs = list(zip(ds["sentence1"], ds["sentence2"]))
    biosses_gold  = [float(s) / 5.0 for s in ds["score"]]
    print(f"  BIOSSES n={len(biosses_pairs)}")
except Exception as e:
    print(f"  BIOSSES load failed: {e} — using 0 pairs")
    biosses_pairs, biosses_gold = [], []

results = {"run_at": datetime.now().isoformat(), "label": "bioelectra", "models": {}}

for mkey, embed_fn in [("base_pt", embed_pt), ("finetuned_ov_bf16", embed_ov)]:
    print(f"\n  [{mkey}]")
    r = {}

    # BIOSSES
    if biosses_pairs:
        ea = embed_fn([p[0] for p in biosses_pairs])
        eb = embed_fn([p[1] for p in biosses_pairs])
        sims = cosines(ea, eb)
        r["biosses_spearman"], _ = spearmanr(sims, biosses_gold)
        print(f"  BIOSSES Spearman: {r['biosses_spearman']:.4f}")

    # Cross-domain
    ea = embed_fn([p[0] for p in CROSS_CONNECTED])
    eb = embed_fn([p[1] for p in CROSS_CONNECTED])
    cc_sims = cosines(ea, eb)
    r["cross_domain_acc"] = float(np.mean([s > 0.50 for s in cc_sims]))
    print(f"  Cross-domain acc (>0.50): {r['cross_domain_acc']*100:.1f}%")

    # Hard negatives
    ea = embed_fn([p[0] for p in HARD_NEGATIVES])
    eb = embed_fn([p[1] for p in HARD_NEGATIVES])
    hn_sims = cosines(ea, eb)
    r["hard_neg_detect"] = float(np.mean([s < 0.40 for s in hn_sims]))
    all_sims = cc_sims + hn_sims
    r["pairwise_acc"] = float(np.mean(
        [s > 0.50 for s in cc_sims] + [s < 0.50 for s in hn_sims]))
    print(f"  Hard-neg detect (<0.40): {r['hard_neg_detect']*100:.1f}%")
    print(f"  Pairwise acc (t=0.50):   {r['pairwise_acc']*100:.1f}%")

    # Intra/inter domain
    intra_sims = []
    for dom, pairs in INTRA_DOMAIN.items():
        ea = embed_fn([p[0] for p in pairs])
        eb = embed_fn([p[1] for p in pairs])
        intra_sims.extend(cosines(ea, eb))
    ea = embed_fn([p[0] for p in INTER_DOMAIN])
    eb = embed_fn([p[1] for p in INTER_DOMAIN])
    inter_sims = cosines(ea, eb)
    r["intra_mean"] = float(np.mean(intra_sims))
    r["inter_mean"] = float(np.mean(inter_sims))
    r["intra_inter_ratio"] = r["intra_mean"] / max(r["inter_mean"], 1e-8)
    print(f"  Intra/inter ratio: {r['intra_inter_ratio']:.4f}  "
          f"(intra={r['intra_mean']:.3f}, inter={r['inter_mean']:.3f})")

    results["models"][mkey] = r

# ── Summary ───────────────────────────────────────────────────────────────────
base_r  = results["models"]["base_pt"]
tuned_r = results["models"]["finetuned_ov_bf16"]
print(f"\n{'='*72}")
print(f"  FULL EVALUATION — BIOELECTRA")
print(f"{'='*72}")
print(f"  {'Metric':<35}  {'Base PT':>10}  {'Tuned OV':>10}  {'Δ':>8}")
print(f"  {'─'*68}")

def row(label, key, fmt=".4f", pct=False):
    b, t = base_r.get(key, float('nan')), tuned_r.get(key, float('nan'))
    d = t - b
    if pct:
        print(f"  {label:<35}  {b*100:>9.1f}%  {t*100:>9.1f}%  {d*100:>+7.1f}pp")
    else:
        print(f"  {label:<35}  {b:>10{fmt}}  {t:>10{fmt}}  {d:>+8{fmt}}")

row("BIOSSES Spearman",           "biosses_spearman")
row("Cross-domain acc (>0.50)",   "cross_domain_acc",  pct=True)
row("Pairwise acc (t=0.50)",      "pairwise_acc",      pct=True)
row("Hard-neg detect (<0.40)",    "hard_neg_detect",   pct=True)
row("Intra/inter ratio",          "intra_inter_ratio", fmt=".4f")

ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
out = BENCHMARKS / f"eval_full_bioelectra_{ts}.json"
json.dump(results, out.open("w"), indent=2)
print(f"\n[saved] {out}")