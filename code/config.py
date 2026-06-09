"""
Self-contained config for aws_benchmark — all paths relative to the
aws_benchmark/ root so this folder can be shipped (zipped) and run anywhere.

Expected layout (when unzipped on a target host):
  aws_benchmark/
    code/         ← this file lives here
    models/
      pytorch/{pubmedbert_pass1, pubmedbert_bodhi, biobert_finetuned}/
      openvino/{finetuned-pubmed-bf16, finetuned-pubmed-bodhi-bf16, finetuned-biobert-bf16}/
      openvino_quantized/{pubmed,bodhi,biobert}-{int8,int4}/   ← created by quantize.py
    results/      ← runs land here
"""
from pathlib import Path
import os

ROOT          = Path(__file__).resolve().parent.parent      # → aws_benchmark/
MODELS_BASE   = Path(os.environ.get("AWS_BENCH_MODELS", ROOT / "models"))
PT_BASE       = MODELS_BASE / "pytorch"
OV_BASE       = MODELS_BASE / "openvino"
QUANT_BASE    = MODELS_BASE / "openvino_quantized"
RESULTS_DIR   = Path(os.environ.get("AWS_BENCH_RESULTS", ROOT / "results"))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
QUANT_BASE.mkdir(parents=True, exist_ok=True)

# ── Model registry ────────────────────────────────────────────────────────────
MODELS = {
    "pubmed":  "PubMedBERT Pass1",
    "bodhi":   "PubMedBERT BODHI",
    "biobert": "BioBERT Fine-Tuned",
}

PT_PATHS = {
    "pubmed":  str(PT_BASE / "pubmedbert_pass1"),
    "bodhi":   str(PT_BASE / "pubmedbert_bodhi"),
    "biobert": str(PT_BASE / "biobert_finetuned"),
}

OV_PATHS = {
    "pubmed_bf16":  str(OV_BASE / "finetuned-pubmed-bf16"),
    "bodhi_bf16":   str(OV_BASE / "finetuned-pubmed-bodhi-bf16"),
    "biobert_bf16": str(OV_BASE / "finetuned-biobert-bf16"),
    "pubmed_int8":  str(QUANT_BASE / "pubmed-int8"),
    "bodhi_int8":   str(QUANT_BASE / "bodhi-int8"),
    "biobert_int8": str(QUANT_BASE / "biobert-int8"),
    "pubmed_int4":  str(QUANT_BASE / "pubmed-int4"),
    "bodhi_int4":   str(QUANT_BASE / "bodhi-int4"),
    "biobert_int4": str(QUANT_BASE / "biobert-int4"),
}

VARIANTS = [
    ("pytorch", "fp32"),
    ("pytorch", "fp16"),
    ("pytorch", "bf16"),
    ("openvino", "bf16"),
    ("openvino", "int8"),
    ("openvino", "int4"),
]

BATCH_SIZES   = [64, 128, 256, 512]
WORKER_COUNTS = [1, 2, 4, 8]
WARMUP        = 5
BENCH_SECS    = 30

# ── NUMA topology (Intel Xeon 6737P) ─────────────────────────────────────────
NUMA_NODES    = 2
PHYS_PER_NODE = 32

SERVER_PLAN_PHYS = [
    (0, list(range(0,  8))),  (0, list(range(8,  16))),
    (0, list(range(16, 24))), (0, list(range(24, 32))),
    (1, list(range(32, 40))), (1, list(range(40, 48))),
    (1, list(range(48, 56))), (1, list(range(56, 64))),
]

SERVER_PLAN_HT_16 = [
    (0, [0,1,2,3,    64,65,66,67]),
    (0, [4,5,6,7,    68,69,70,71]),
    (0, [8,9,10,11,  72,73,74,75]),
    (0, [12,13,14,15,76,77,78,79]),
    (0, [16,17,18,19,80,81,82,83]),
    (0, [20,21,22,23,84,85,86,87]),
    (0, [24,25,26,27,88,89,90,91]),
    (0, [28,29,30,31,92,93,94,95]),
    (1, [32,33,34,35, 96,97,98,99]),
    (1, [36,37,38,39, 100,101,102,103]),
    (1, [40,41,42,43, 104,105,106,107]),
    (1, [44,45,46,47, 108,109,110,111]),
    (1, [48,49,50,51, 112,113,114,115]),
    (1, [52,53,54,55, 116,117,118,119]),
    (1, [56,57,58,59, 120,121,122,123]),
    (1, [60,61,62,63, 124,125,126,127]),
]

SERVER_PLAN_HT_32 = [
    (0, [0,1, 64,65]),   (0, [2,3, 66,67]),   (0, [4,5, 68,69]),   (0, [6,7, 70,71]),
    (0, [8,9, 72,73]),   (0, [10,11,74,75]),  (0, [12,13,76,77]),  (0, [14,15,78,79]),
    (0, [16,17,80,81]),  (0, [18,19,82,83]),  (0, [20,21,84,85]),  (0, [22,23,86,87]),
    (0, [24,25,88,89]),  (0, [26,27,90,91]),  (0, [28,29,92,93]),  (0, [30,31,94,95]),
    (1, [32,33, 96,97]), (1, [34,35, 98,99]), (1, [36,37,100,101]),(1, [38,39,102,103]),
    (1, [40,41,104,105]),(1, [42,43,106,107]),(1, [44,45,108,109]),(1, [46,47,110,111]),
    (1, [48,49,112,113]),(1, [50,51,114,115]),(1, [52,53,116,117]),(1, [54,55,118,119]),
    (1, [56,57,120,121]),(1, [58,59,122,123]),(1, [60,61,124,125]),(1, [62,63,126,127]),
]

SENTENCES = [
    "Atrial fibrillation increases the risk of stroke and systemic embolism.",
    "BRCA1 mutations are associated with hereditary breast and ovarian cancer syndrome.",
    "Metformin inhibits hepatic gluconeogenesis via activation of AMPK pathway.",
    "The blood-brain barrier restricts passive diffusion of large polar molecules.",
    "Septic shock is defined by vasopressor requirement and elevated lactate levels.",
    "Checkpoint inhibitor therapy targeting PD-1 improves survival in melanoma.",
    "Amyloid-beta plaques and tau tangles are hallmarks of Alzheimer's disease.",
    "CRISPR-Cas9 enables precise genome editing at targeted chromosomal loci.",
    "IL-6 is a pleiotropic cytokine involved in acute phase inflammatory response.",
    "Coronary artery disease results from atherosclerotic plaque accumulation.",
    "T-cell exhaustion limits antitumor immunity in the tumor microenvironment.",
    "Opioid receptors modulate pain transmission in the dorsal horn of spinal cord.",
    "Ventricular hypertrophy is a compensatory response to chronic pressure overload.",
    "mRNA stability is regulated by 3-prime UTR binding proteins and miRNA.",
    "Neutrophil extracellular traps contribute to thrombosis in COVID-19 patients.",
    "Insulin resistance in skeletal muscle impairs glucose uptake and metabolism.",
    "The complement cascade amplifies innate immune responses to pathogens.",
    "EGFR tyrosine kinase inhibitors are first-line therapy in NSCLC with mutations.",
    "Gut microbiome dysbiosis is associated with inflammatory bowel disease.",
    "Hypoxia-inducible factor 1-alpha regulates angiogenesis in solid tumors.",
    "Myocardial infarction triggers inflammatory cascade leading to ventricular remodeling.",
    "Glucocorticoid receptor activation suppresses NF-kB mediated gene transcription.",
    "Heparin anticoagulation prevents thrombus extension in deep vein thrombosis.",
    "Mitochondrial dysfunction leads to reactive oxygen species overproduction.",
    "Renal tubular acidosis impairs hydrogen ion secretion in the collecting duct.",
    "CD19 CAR-T cell therapy achieves remission in relapsed B-cell lymphoma.",
    "Spinal muscular atrophy results from SMN1 gene deletion or point mutation.",
    "Cholesterol biosynthesis is regulated by SREBP transcription factor pathway.",
    "Macrophage polarization to M1 phenotype enhances proinflammatory cytokine output.",
    "Platelet aggregation is initiated by collagen binding to GPVI receptor complex.",
]

TOKENS_PER_SENTENCE = {"pubmed": 13.41, "bodhi": 13.41, "biobert": 21.0}
