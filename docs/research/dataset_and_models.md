# Dataset lock-in and expert pool (Prompts 2 and 3, narrowed)

Scope: only MIMIC (train) → eICU (external test), and TCGA / MLOmics, as agreed with the TA.
Written 2026-09-29 as a critical advisor note, not a sales pitch. Numbers marked "approx." are from memory of the
dataset docs and should be checked against the official pages before they go into the thesis.

---

## 0. The short answer

**Lock in MIMIC-IV → eICU-CRD (with MIMIC-III as a second training source for ablation, §1.1b), on harmonised structured ICU time series (vitals, labs, treatments, static
demographics), with 24h in-hospital mortality as the primary task and AKI as the secondary task.**
Use TCGA/MLOmics only as a fallback or as a small "second domain" sanity check, not as the main setup.

Three reasons, in order of importance:

1. **It is the only option that gives you a real, established external test.** YAIB (ICLR 2024) and
   Burger et al. 2024 already harmonise MIMIC-IV, eICU, HiRID and AUMCdb with `ricu` and report
   train-on-one / test-on-another results. You inherit a defensible distribution-shift protocol instead of
   inventing one. MLOmics has no external cohort at all.
2. **Architecture synthesis needs cheap training runs, and ICU time series are cheap.** 52 features × 24 hourly
   steps × ~50–70k stays trains a GRU in minutes on one GPU. You can afford hundreds of candidate
   architectures. On TCGA, the per-task sample size (hundreds) makes validation scores so noisy that any
   search will overfit the validation set; that would sink the thesis's central comparison.
3. **The input admits genuinely different inductive biases.** Irregular multivariate time series are
   naturally handled by RNNs, dilated CNNs, attention, SSMs, graph-over-sensors models, set functions and
   GBDTs. That is exactly the architecture diversity your AES and synthesis stages need. Gene-level omics
   vectors have no natural order, so sequence models (CNN/SSM) have no honest inductive bias there.

### The one thing you must settle with your TA

**eICU has no free-text notes and no images.** MIMIC-IV-Note (discharge summaries, radiology reports) and
MIMIC-CXR exist only on the MIMIC side. So "multimodal MIMIC → eICU" can only mean one of:

- **(A) Heterogeneous structured modalities** (static tabular + dense vitals + sparse labs + treatment events).
  Honest, transferable, but a weak use of the word "multimodal". **My recommendation.**
- **(B) A missing-modality design:** train experts per modality on MIMIC (structured, notes, CXR), then at
  eICU test time only the structured experts are available. This turns into a neat AES question
  ("does the router degrade gracefully when experts disappear?"), but it roughly doubles the engineering
  and the notes/CXR experts cannot be externally validated. Treat as an optional extension only.

Ask your TA directly whether (A) satisfies "multimodal". If they insist on images/text being evaluated
externally, MIMIC → eICU is structurally the wrong pair and TCGA (omics + WSI) becomes the only option.

---

## 1. Dataset cards

### 1.1 MIMIC-IV (training / development)

| Field | Value |
|---|---|
| Source | Beth Israel Deaconess Medical Center, Boston (single academic centre), 2008–2019 |
| URL | https://physionet.org/content/mimiciv/ (v3.1) |
| Paper | Johnson et al., *Scientific Data* 2023, "MIMIC-IV, a freely accessible EHR dataset" |
| Samples | approx. 364k patients, 546k admissions, 94k ICU stays (v3.1); after YAIB-style cohort filters (adult, ≥ 24h stay, first stay) expect roughly 50–70k stays |
| Input type | Hourly/irregular vitals, labs, fluids, vasopressors, ventilation, static demographics. Optional: MIMIC-IV-Note (approx. 330k discharge notes, 2.3M radiology reports), MIMIC-CXR (approx. 377k images), MIMIC-IV-ECG |
| Label type | Binary (mortality, AKI, sepsis), regression (LoS), multiclass (KDIGO stage) |
| Classes | 2 for primary tasks; mortality prevalence roughly 8–12% depending on cohort |
| Size | ~10 GB compressed core; the harmonised 52-feature hourly cohort is < 1 GB |
| Predefined splits | None official. YAIB ships its own cohort + CV splits |
| Recommended split | Patient-level (subject_id) splits; plus a **temporal split** using `anchor_year_group` (e.g. train 2008–2016, test 2017–2019) as a second, cheaper shift |
| Leakage risks | (1) ICD codes are assigned at discharge, never use them as inputs for in-stay prediction. (2) Multiple stays per patient across splits. (3) Features recorded after the prediction time (DNR/comfort-care orders, discharge location). (4) AKI labels are computed from creatinine/urine output, which are also inputs, so the prediction horizon must exclude the label window |
| Metrics | AUROC, AUPRC (important with ~10% prevalence), Brier score / ECE for calibration |
| Benchmark results | YAIB, MIMIC-IV 24h mortality: LightGBM 87.7 AUROC / 44.2 AUPRC, GRU 87.6 / 42.8. AKI: GRU 90.7 / 69.6, LightGBM 83.8 / 53.3 |
| Baselines | LR, LightGBM, GRU, LSTM, TCN, Transformer (all in YAIB) |
| Pretrained models | Few usable ones (see §3). Most ICU work trains from scratch, which is fine for you |
| Compute | Single consumer GPU is enough; a GRU run is minutes; data extraction via `ricu` (R) or YAIB-cohorts |
| Access | PhysioNet credentialed: CITI "Data or Specimens Only Research" course + DUA; typically 1–2 weeks. **Free demo subsets of MIMIC-IV and eICU are open**, so you can build the pipeline before access clears |
| Realistic for a bachelor's? | Yes. This is the main recommendation |

### 1.1b MIMIC-III vs MIMIC-IV as the training source

You said you mean both. They are the **same hospital (BIDMC)**, so treat them as two training-source options,
never as train/test for each other.

| | MIMIC-III (v1.4) | MIMIC-IV (v3.1) |
|---|---|---|
| Years | 2001–2012 | 2008–2019 |
| Size (approx.) | 46.5k patients, 59k admissions, 61.5k ICU stays | 364k patients, 546k admissions, 94k ICU stays |
| EHR systems | Two: CareVue (2001–08) and MetaVision (2008–12), with different item IDs and charting | MetaVision only for ICU charting |
| Free text | NOTEEVENTS, approx. 2M notes incl. **nursing and physician notes with chart times during the stay** | MIMIC-IV-Note: discharge summaries + radiology reports only |
| Images | None linked | MIMIC-CXR linkable by subject_id |
| Standard benchmarks | Harutyunyan et al. 2019 (48h in-hospital mortality, decompensation, LoS, phenotyping; 17 variables), MIMIC-Extract; most irregular-time-series papers (mTAND, STraTS, etc.) report on it | YAIB, Burger et al. 2024, PULSE-ICU |
| In YAIB / ricu | Yes | Yes |
| Gap to eICU (2014–15) | Larger (ends 2 years before eICU starts) | Overlaps eICU's years |

**Critical points:**

- **Do not pool them and do not test one on the other.** MIMIC-IV includes 2008–2012 MetaVision stays that are
  also in MIMIC-III, under different de-identified IDs and with no official linkage table. Pooling duplicates
  patients across splits; "MIMIC-III → MIMIC-IV" is not an external test.
- **MIMIC-IV should stay the primary source.** It is larger, uses one charting system, is closer in time to
  eICU, is still maintained, and is what recent ICU transfer work uses.
- **MIMIC-III is still worth using in two cheap ways,** since `ricu` gives you both with the same code:
  1. **Training-source ablation:** train the same experts on MIMIC-III and on MIMIC-IV, test both on eICU.
     Question: does source recency/size change which expert (or which synthesised architecture) transfers best?
     Restrict MIMIC-IV to its non-overlapping years (2013–2019) if you want the two sources to be disjoint.
  2. **A fourth shift, CareVue → MetaVision inside MIMIC-III:** same hospital, different EHR software. This is
     a known source of performance drop (Nestor et al., MLHC 2019, "Feature robustness in non-stationary health
     records") and isolates documentation shift from population shift.
- **Text is where MIMIC-III is genuinely better.** MIMIC-IV discharge summaries are written at discharge and
  often state the outcome, so they **leak the label** for mortality prediction; only radiology reports are
  safely usable during the stay. MIMIC-III nursing/physician notes with chart times are usable for in-stay
  prediction. If your TA wants a text modality (option B in §0), take text from MIMIC-III; if they want
  images, CXR only links to MIMIC-IV. You cannot get both notes-during-stay and CXR from one database, and
  neither is available in eICU.
- **Reporting:** results on MIMIC-III are more comparable to the older irregular-time-series literature;
  results on MIMIC-IV are more comparable to YAIB and current ICU transfer work. Use YAIB's task definitions
  on both, not the Harutyunyan ones, so the eICU comparison is like-for-like.

### 1.2 eICU-CRD (external test)

| Field | Value |
|---|---|
| Source | Philips eICU telehealth programme, 208 US hospitals, 2014–2015 |
| URL | https://physionet.org/content/eicu-crd/ (v2.0) |
| Paper | Pollard et al., *Scientific Data* 2018 |
| Samples | approx. 200k ICU unit stays, 139k patients |
| Input type | Structured only: vitals (periodic + aperiodic), labs, meds, infusions, diagnoses (entered during stay with offsets), APACHE variables. **No free text, no images** |
| Labels | Same task definitions as MIMIC via `ricu`/YAIB |
| Predefined splits | None; hospital IDs and US region are available |
| Recommended use | Pure external test for models trained on MIMIC. Bonus: **leave-region-out or leave-hospital-out** within eICU gives a third kind of shift (multi-site) |
| Leakage risks | APACHE predicted mortality columns (never use as input); uniquepid across splits; hospitals with very few stays inflate variance, so report per-hospital results with caution |
| Metrics | Same as MIMIC, plus per-hospital AUROC spread |
| Known shift | Lower data density than MIMIC (many small community hospitals), different charting practices, different case mix. YAIB and Burger et al. report several AUROC points of drop on cross-database transfer for mortality; PULSE-ICU (2025) reports 0.864 AUROC ICU mortality on eICU, but only after fine-tuning on eICU (zero-shot ≈ 0.65; see mimic_to_eicu_literature.md §6) |
| Access | Same PhysioNet credentialing as MIMIC |
| Realistic? | Yes |

**Four shifts from one credentialing process:** temporal (MIMIC-IV 2008–16 → 2017–19), EHR-system (MIMIC-III
CareVue → MetaVision), cross-database (MIMIC → eICU), and cross-hospital (eICU region A → region B). That is a stronger generalization section than
most published papers have, and it directly serves your Section 4 goals. HiRID (Swiss) and AUMCdb (Dutch) are
drop-in extra external tests through the same `ricu` concepts if you ever want a non-US shift.

### 1.3 TCGA and MLOmics

| Field | Value |
|---|---|
| Source | The Cancer Genome Atlas (NCI/NHGRI), processed into MLOmics |
| URL | MLOmics: Hugging Face `AIBIC/MLOmics`, Figshare; paper https://www.nature.com/articles/s41597-025-05235-x |
| Paper | Yang et al., *Scientific Data* 2025, "MLOmics: Cancer Multi-Omics Database for Machine Learning" |
| Samples | 8,314 patients with all 4 omics, 32 cancer types |
| Input type | mRNA, miRNA, DNA methylation, CNV (Original / Aligned / Top feature versions). No images; TCGA WSIs exist separately |
| Labels / tasks | 1 pan-cancer classification set, 5 gold-standard subtype sets (BRCA, COAD, GBM, LGG, OV), 9 unlabelled clustering sets, 5 imputation sets |
| Predefined splits | Provided in the benchmark code |
| Benchmark results | XGBoost, SVM, RF, LR, Subtype-GAN, DCAP, XOmiVAE, CustOmics, DeepCC, MAUI; the paper reports deep models generally ahead on classification |
| License | CC-BY-4.0, open, no credentialing. **This is its main practical advantage** |
| Compute | Tiny tabular data; CPU is enough for most baselines |

**Why TCGA/MLOmics is the weaker choice for this thesis:**

- **No external test.** MLOmics is TCGA only. Real external cohorts (METABRIC for breast, CPTAC, ICGC,
  GEO) use different platforms (microarray vs RNA-seq) and usually have fewer omics layers (METABRIC has
  expression + CNA, no miRNA or methylation). The shift is dominated by batch/platform effects, so a
  generalization result becomes a batch-correction study, not an architecture study.
- **Circular labels.** BRCA PAM50 subtypes are *defined* from a 50-gene mRNA signature; GBM subtypes are
  expression-derived too. Predicting them from mRNA is partly reproducing the labelling rule. It will look
  easy and tell you little.
- **Pan-cancer tissue-of-origin is too easy** (expect very high accuracy from linear models), leaving no room to
  separate AES from synthesis.
- **Small n, huge p.** Subtype tasks have hundreds of samples and thousands of features. Validation noise will
  swamp architecture differences; NAS on this is a textbook way to overfit the validation set.
- **Weak architecture diversity.** Genes have no order, so CNN/SSM/sequence experts are only justifiable
  with artificial orderings (chromosome position). Honest experts are MLP, VAE (CustOmics), GNN on patient
  similarity (MOGONET) and GBDT, mostly MLP-family.
- **Site leakage** if you ever add WSIs: tissue-source-site signatures are learnable (Howard et al.,
  *Nat. Commun.* 2021), so splits must be by site.

Where TCGA *would* win: if the TA insists that images/text must be evaluated externally. Then the right TCGA
setup is survival prediction with WSI + omics (MCAT, SurvPath, PORPOISE line of work) with precomputed
pathology features (UNI/CONCH), split by tissue-source site. That is a considerably harder and more
expensive project.

### 1.4 Datasets / setups to avoid

| Setup | Why not |
|---|---|
| Sepsis onset as primary label on MIMIC → eICU | Sepsis-3 needs cultures + antibiotics; eICU microbiology is sparse and definitions differ across DBs, so label shift dominates. Noisy |
| MIMIC notes / CXR as the evaluated modality | Cannot be tested on eICU (no such data) |
| Pooling MIMIC-III + MIMIC-IV, or MIMIC-III → MIMIC-IV as "external" | Same hospital, overlapping 2008–2012 patients with no linkage table; duplicates patients and is not an external test (see §1.1b) |
| MIMIC-IV discharge summaries as an input | Written at discharge, often state the outcome: label leakage |
| MLOmics pan-cancer classification | Too easy |
| MLOmics GS-BRCA/GS-GBM as the headline task | Circular labels, small n |
| TCGA WSI end-to-end | Too expensive (TBs of slides) for a bachelor's thesis |
| eICU → MIMIC as the main direction | Legitimate, but "train MIMIC, test eICU" is the convention your TA mentioned; keep reverse direction as an ablation |

### 1.5 Ranking (within the allowed options)

1. **Best overall:** MIMIC-IV → eICU, structured, 24h mortality (primary) + AKI (secondary).
2. **Best for generalization:** same, with the three-shift protocol (temporal / cross-DB / cross-hospital).
3. **Best for architecture diversity:** MIMIC-IV → eICU (RNN, CNN, attention, SSM, GNN, set functions, GBDT all fit naturally).
4. **Best for computational feasibility:** MLOmics (CPU-scale), with MIMIC → eICU close behind (single GPU).
5. **Best fallback:** MLOmics subtype classification with cross-cancer-type or reverse-direction ablations, if PhysioNet access is blocked. Start on the MIMIC/eICU demo data meanwhile.

### 1.6 Compliance note you need to know now

PhysioNet's credentialed data use agreement prohibits sharing the data with third parties, and PhysioNet has
issued guidance that this includes sending records to online LLM services. Do not paste MIMIC/eICU rows,
notes or patient-level outputs into any chatbot, including me. Aggregate metrics and code are fine.

---

## 2. Linked papers: how they position the thesis

| arXiv | Paper (from abstract) | Relevance |
|---|---|---|
| 1701.06538 | Shazeer et al., Sparsely-Gated Mixture-of-Experts | Canonical learned routing; your AES is a *post-hoc*, heterogeneous-expert version. Cite as the contrast: MoE experts share an architecture and are trained jointly; yours are architecturally different and pre-trained |
| 1802.03268 | ENAS (efficient NAS via weight sharing) | The cost baseline for "search is expensive". Weight sharing is the natural way to make your synthesis affordable |
| 2509.26037 | CoLLM-NAS, LLM-guided NAS | Shows "knowledge-guided NAS" with LLMs is active in 2025; weakens novelty of "using prior architectural knowledge to guide search" in general, so your novelty must be the *model-zoo-derived component library* and the *AES vs synthesis cost trade-off*, not knowledge-guided search per se |
| 2403.19887 | Jamba (hybrid Transformer–Mamba) | A hand-designed version of exactly what your synthesis would produce. HyMaTE (hybrid Mamba+Transformer for EHR) exists too, so "attention + SSM for EHR" alone is not novel |
| 2101.11605 | BoTNet (attention inside a CNN backbone) | Another hand-made hybrid; evidence that block-level component swaps work |
| 2112.13208 | DNN modularization and recomposition via weight masks | Directly relevant to "can components be reused independently": this is *weight-level* module reuse within one architecture, not cross-architecture. Your thesis operates at the block/architecture level; make that distinction explicit |
| 2310.01154 | Modularity in deep learning: a review | Terminology source ("model modularity", combinability, reusability) |
| 2412.11211 | State-space models in deep learning: a review | Background for the SSM expert |
| 2510.27039 | Hybrid ST-GNN + Transformer for traffic forecasting | Weak relevance; an applied hybrid. I would not cite it prominently |

Critical takeaway: hybrids of existing blocks are common and usually hand-designed. The defensible
contribution is **(i)** a search space built from the components of an actually-trained, diverse expert zoo,
**(ii)** evidence on whether per-sample expert specialization (from AES) predicts which components are worth
composing, and **(iii)** a cost-aware comparison AES vs synthesis under real distribution shift.
Burger et al. 2024 already compared GRU/Transformer/Mamba/xLSTM across ICU databases, so "which single
architecture transfers best" is *not* your question; that paper is a baseline for you.

---

## 3. Candidate models for MIMIC-IV → eICU (structured)

All candidates below consume the same harmonised input (YAIB/`ricu`: 4 static + 48 time-series features,
hourly grid, missingness indicators), except where noted. That shared input is what makes AES,
ensembling and component reuse fair.

### 3.1 LightGBM on engineered features (GBDT, non-neural)
- Paper: Ke et al., NeurIPS 2017. In YAIB. Params: n/a (trees). Weights: train yourself (minutes, CPU).
- Input: flattened window statistics (min/max/mean/count per feature).
- Strength: matches or beats deep models on mortality (87.7 AUROC in YAIB); strong, robust baseline.
- Weakness: no temporal representation; nothing to decompose.
- Captures: marginal severity signals and thresholds.
- Reusable components: none for synthesis. Use in **AES and ensembles only**.

### 3.2 GRU-D (RNN with missingness-aware decay)
- Paper: Che et al., *Sci. Rep.* 2018. Code: several PyTorch ports; plain GRU/LSTM are in YAIB.
- Input: values + mask + time-since-last-observation.
- Params: ~0.1–1M. GPU: any.
- Strength: explicitly models informative missingness, which differs a lot between MIMIC and eICU (so it is
  an interesting generalization probe: it may overfit MIMIC charting patterns).
- Captures: sequential dynamics, missingness patterns.
- Reusable components: **decay input layer**, GRU cell. Good synthesis candidates.

### 3.3 TCN (dilated causal CNN)
- Paper: Bai et al. 2018. In YAIB.
- Params: ~0.1–1M. GPU: any.
- Strength: local temporal patterns, stable training, fast.
- Reusable components: **dilated residual conv block**. Clean, shape-compatible block.

### 3.4 Transformer encoder (YAIB) or STraTS
- YAIB Transformer (standard encoder on the hourly grid). STraTS (Tipirneni & Reddy, *TKDD* 2022) embeds
  (time, variable, value) triplets and handles irregularity natively.
- Params: ~0.5–5M. GPU: any.
- Strength: long-range interactions across time.
- Weakness: data-hungry; can overfit single-centre patterns.
- Reusable components: **self-attention block, FFN, triplet/value embedding**.
- Pretrained option: PULSE-ICU (2025, Longformer encoder pretrained on MIMIC-IV, 0.864 AUROC on eICU ICU
  mortality only after fine-tuning on eICU). Code repo exists; I could not confirm released weights, so treat as "maybe".

### 3.5 Mamba (selective SSM)
- Paper: Gu & Dao 2023 (`mamba-ssm`, needs CUDA). Used for ICU time series in Burger et al. 2024;
  EHRMamba (Fallahpour et al., ML4H 2024) is an EHR-token variant.
- Params: ~0.5–5M at this scale.
- Strength: linear-time long context; different inductive bias from attention; directly connects to your
  Jamba reading.
- Weakness: with 24 hourly steps its efficiency advantage is irrelevant; say that honestly. Use longer
  windows (e.g. 48–72h, or minute-level vitals) if you want SSM strengths to matter.
- Reusable components: **Mamba block**. Drop-in replacement for attention/GRU blocks in a residual stack.
- EHRMamba itself uses coded event sequences (not the ricu grid) and is harder to harmonise with eICU;
  do not make it a core expert.

### 3.6 Raindrop (GNN over sensors)
- Paper: Zhang, Zeman, Tsiligkaridis, Zitnik, ICLR 2022. Code: `mims-harvard/Raindrop`.
- Input: irregular observations; learns a graph between variables and passes messages when a sensor fires.
- Params: ~0.1–1M.
- Strength: models inter-variable dependencies explicitly; the only genuinely relational expert.
- Weakness: fiddlier code; originally evaluated on PhysioNet 2012/2019, needs porting to the ricu cohort.
- Reusable components: **sensor-graph message-passing layer**. Interesting for synthesis because it adds a
  "cross-variable" operator that none of the sequence blocks have.

### 3.7 Optional / not recommended as core
| Model | Verdict |
|---|---|
| SeFT (set functions, Horn et al. 2020), mTAND (Shukla & Marlin 2021) | Good extra diversity for irregular data; add only if time permits |
| xLSTM | Covered by Burger et al.; redundant with GRU-D for your purposes |
| MOMENT (T5-based TS foundation model, 40M–385M, HF `AutonLab/MOMENT-1-large`) | Channel-independent, not trained on ICU data, weak on 48 sparse channels. Only as a "generic pretrained" contrast |
| Foundation models pretrained on multi-DB ICU data incl. eICU (e.g. Burger et al.'s multi-centre setting) | **Contaminates the external test.** Never use a model that saw eICU in pretraining |
| ClinicalBERT / Clinical-Longformer / CXR encoders (CheXzero, BiomedCLIP, RAD-DINO) | Only for the optional missing-modality extension (§0 option B). Not externally testable |

### 3.8 Recommended expert pool (5 + 1)

| Expert | Family | Why it is in the pool |
|---|---|---|
| LightGBM | GBDT | The honest strong baseline; if AES cannot beat it, that is a finding |
| GRU-D | RNN | Missingness-aware sequential bias; likely to be charting-sensitive, a good shift probe |
| TCN | CNN | Local temporal patterns; clean reusable block |
| Transformer (YAIB) | Attention | Global temporal interactions |
| Mamba | SSM | Distinct sequence operator; links to Jamba / hybrid literature |
| Raindrop *(+1, if porting is manageable)* | GNN | Only cross-variable relational operator |

They are architecturally different, public, cheap (all trainable on one GPU within hours for the full
set), and plausibly complementary: GBDT for thresholds, RNN for missingness, CNN for local shape,
attention/SSM for long-range, GNN for cross-variable structure.

### 3.9 Which models suit which experiment

| Experiment | Models |
|---|---|
| **AES** | All of them. The router should see only MIMIC data (train/val); eICU is strictly held out, including for router calibration |
| **Standard ensembles** | All of them (mean of probabilities, stacking with a MIMIC-only meta-learner) |
| **Model Soup** | **Only within one architecture, fine-tuned from one shared initialisation** (see below) |
| **Architecture decomposition** | GRU-D, TCN, Transformer, Mamba, Raindrop (their blocks map to a common `[B, T, d] → [B, T, d]` interface). Not LightGBM |
| **Architecture synthesis** | Blocks: decay-input layer, GRU layer, dilated conv block, attention block, Mamba block, sensor-graph layer, FFN, static-feature fusion (FiLM/concat), pooling head |

### 3.10 Model Soup: be careful

- Wortsman et al. 2022 soups work because the ingredients are **fine-tuned from the same pretrained
  checkpoint**, so they lie in one linearly connected low-loss basin. Averaging weights of different
  architectures is undefined, and averaging same-architecture models trained **from different random
  inits** generally fails (permutation symmetry; Git Re-Basin style alignment only partly fixes it).
- So a fair soup here is: **pick one architecture (Transformer or Mamba), pretrain it once on the MIMIC
  training split** (e.g. masked-value reconstruction or next-step forecasting), then fine-tune 10–20 copies
  with different hyperparameters/seeds/augmentations, and compare uniform soup vs greedy soup vs the best
  single fine-tune vs an ensemble of the same fine-tunes. Greedy soup selection must use MIMIC validation
  only.
- Without a shared pretrained start, a weaker but valid alternative is "shared early training": train one
  model for a few epochs, then branch (Frankle et al. 2020 linear mode connectivity). State clearly which
  you use.
- No soup across experts, and no soup for LightGBM. The fair comparison for the soup is against an
  ensemble of the **same** fine-tunes, not against the heterogeneous AES.

---

## 4. If you do end up on TCGA/MLOmics (fallback pool)

| Expert | Family | Notes |
|---|---|---|
| XGBoost / LR (elastic net) | GBDT / linear | Very strong on p>>n |
| MLP with per-omics encoders + late fusion | MLP | Simple, soupable if pretrained once |
| CustOmics (hierarchical VAE) | VAE | In MLOmics baselines |
| MOGONET (GCN on patient-similarity graphs + VCDN fusion) | GNN | Nature Commun. 2021; the only structurally different deep expert |
| Gene-token Transformer (e.g. BulkRNABert-style, bulk RNA-seq pretraining) | Attention | Only defensible sequence-ish model for bulk omics; check weight availability before committing |

Architecture diversity is clearly narrower here, which is the main reason it is the fallback.

---

## 5. Concrete next steps

1. Start PhysioNet credentialing today (CITI course + MIMIC-IV and eICU DUAs).
2. Ask your TA the §0 question: is structured multi-source EHR "multimodal enough"?
3. Meanwhile, install YAIB + YAIB-cohorts and run the full pipeline on the open MIMIC-IV and eICU **demo**
   datasets, so day one after access is just re-running on the full data.
4. Reproduce YAIB's LightGBM and GRU mortality numbers on MIMIC-IV before building anything new. If you
   cannot reproduce ~87–88 AUROC, fix that first.
5. Then add TCN, Transformer, Mamba (and Raindrop) on the same cohort, and log per-sample predictions on
   MIMIC val/test and eICU. Those per-sample logs are the raw material for AES and for Stage 7
   (expert specialization analysis).

---

## Sources

- YAIB: van de Water et al., ICLR 2024, https://openreview.net/forum?id=ox2ATRM90I ; code https://github.com/rvandewater/YAIB
- Burger et al., "Towards Foundation Models for Critical Care Time Series", arXiv 2411.16346
- PULSE-ICU, arXiv 2511.22199 ; https://github.com/sejeongak/PULSE-ICU
- EHRMamba, Fallahpour et al., ML4H 2024, arXiv 2405.14567
- MLOmics, Yang et al., Scientific Data 2025, https://www.nature.com/articles/s41597-025-05235-x ; https://huggingface.co/datasets/AIBIC/MLOmics
- MIMIC-III: https://physionet.org/content/mimiciii/ ; Harutyunyan et al., Sci. Data 2019 (MIMIC-III benchmark) ; Nestor et al., MLHC 2019
- MIMIC-IV: https://physionet.org/content/mimiciv/ ; eICU-CRD: https://physionet.org/content/eicu-crd/
- Raindrop: Zhang et al., ICLR 2022, https://github.com/mims-harvard/Raindrop
- Model soups: Wortsman et al., ICML 2022
- Linked papers: arXiv 1701.06538, 1802.03268, 2509.26037, 2403.19887, 2101.11605, 2112.13208, 2310.01154, 2412.11211, 2510.27039
