# CLAUDE.md — Project handoff: ICU AES vs cross-family NAS (bachelor thesis)

This file gives Claude Code the full context of the project. It summarises about six weeks of planning, research
and coding done in a separate Claude project (Sept–Nov 2026). Keep it in the repo root so Claude Code loads it
automatically. Update the "Status" and "Next steps" sections as work progresses.

---

## 0. Rules for Claude working in this repo (read first)

1. **Never read, print, open or summarise real patient data.** MIMIC-IV, eICU and HiRID are credentialed PhysioNet
   datasets. Their data use agreement forbids sending records to online services, and that includes Claude.
   Do not open anything under `data/`, `artifacts/predictions/`, `*.parquet`, `*.npy`, `*.csv.gz`, checkpoints or
   per-patient outputs. **`logs/ian-extract-*.out` also contain patient rows** (YAIB's R scripts print table heads):
   only `grep` them for error lines, never `tail`/`cat` them. EDA outputs (`outputs/eda/`) are aggregates. Only aggregate numbers (AUROC tables, attrition counts, metric summaries) may be shared.
   For debugging, use the fabricated data from `src/icu_aes_nas/data/synthetic.py` (ids start at 9,000,000).
2. **Never commit data.** `data/`, `artifacts/`, `outputs/`, `*.parquet`, `*.npy` and checkpoints are gitignored.
3. **Ask Kiril before making new methodological decisions.** Kiril explicitly asked to be consulted before any
   extra decision is made. Section 3 lists what is already locked; anything not there needs a question first.
4. **Do not change Pipeline B to add Pipeline A.** Pipeline A must only *add* code (see §6).
5. Kiril is a bachelor student and has said they get overwhelmed by many decisions at once. Prefer short answers
   with one clear recommendation, and group questions.
6. The thesis is being written in "we", with numbered [1]-style citations, in chapter style.

---

## 1. The thesis in one paragraph

**Working title:** *Combining expert models for ICU mortality prediction across hospitals: adaptive expert selection
versus architecture synthesis.*

The project started as an Intelligent Information Systems course project on **Adaptive Expert Selection (AES)** and
grew into a bachelor thesis on **architecture synthesis**. The research question: *do new architectures built by
connecting blocks from different trained expert models (cross-family NAS, e.g. a TCN block feeding an attention block,
with inherited weights) generalise to unseen hospitals better than choosing among / combining the experts (AES,
ensembles, model soup), and at what compute cost?* Every method is judged on **generalisation** (external test on a
different database, eICU) and **compute cost** (logged from day one). The user's original idea was phrased as
"heterogeneous graphs": this means a NAS whose candidate architectures are typed graphs connecting blocks from
*different* model families.

**Novelty positioning (important for the writing):** DeRy (NeurIPS 2022) already reassembles CNN/Transformer blocks
from different pretrained models, model stitching across families works (2025–26 papers), and hybrid attention+Mamba
EHR models already exist. So the contribution must be: (a) a *clinical* expert zoo as the block library, (b) AES as
the competing method, (c) evaluation under real distribution shift (MIMIC → eICU, temporal), with (d) costs counted.

---

## 2. Two pipeline versions

| | Pipeline A | Pipeline B (**coded now**) |
|---|---|---|
| Data | MIMIC-IV structured + MIMIC-IV-ECG + MIMIC-CXR | Structured hourly ICU time series from several databases |
| Training | MIMIC-IV | MIMIC-IV v2.2 (+ HiRID v1.1.1 when enabled; AUMCdb later) |
| Tests | internal test, temporal 2017–19; eICU only as structured-only test (ECG/CXR missing) | internal, temporal, **eICU v2.0 external** (headline) |
| Strength | Best fit for CNN↔Transformer block mixing (images/signals) | Every architecture gets a real external test |

Kiril wants to try **both** and decide later whether to lock in one or do both. Original advice: A on a ~3-month plan,
B on a ~1-month plan. B was coded first; A plugs into the same code later. **D19 = b:** the course project includes
Version A and B extras, not only the shared core.

Version A risks found during research:
* ECG machine clocks are not synchronised with MIMIC-IV → use a safety margin before hour 24 (D21: 1 h, 3 h sensitivity).
* RAD-DINO and ECG-FM were pretrained on MIMIC-CXR / MIMIC-IV-ECG → they've seen the test patients. Use non-MIMIC
  encoders (ECGFounder, BiomedCLIP), frozen, plus one CNN per modality trained by us (D23).
* **Unknown coverage**: how many ICU stays have an ECG or CXR in the first 24 h. Measure this first; it decides if A
  is viable.
* D22 = c: all ECGs/CXRs before the cut-off become tokens, capped at the latest 4 of each per stay.
* No radiology reports or ECG notes (outcome leakage). Frontal CXR only.

---

## 3. Locked decisions (do not re-litigate without asking Kiril)

### Data and protocol
* **MIMIC-IV v2.2, not v3.1** — ricu (and therefore YAIB) only supports v2.2 (2008–2019); also keeps COVID out.
* MIMIC-III dropped entirely (same hospital as MIMIC-IV, overlapping unlinkable patients). Four DBs in scope:
  MIMIC-IV, eICU, HiRID, AUMCdb.
* Harmonisation via **ricu + YAIB-cohorts** (R), pinned to commit `74ac699`; ricu is the `prockenschaub/ricu-package`
  fork at `0969a52` (installed by `r/install_r_packages.R`), not CRAN ricu 0.6.0: 48 hourly dynamic variables + 4 statics
  (age, sex, height, weight), hourly grid from ICU admission.
* **Primary task:** `mortality24` = **ICU mortality** (not in-hospital — YAIB's label is ICU mortality) predicted from
  the first 24 h, one label per stay. **Secondary:** `aki` (KDIGO), hourly onset, one label per hour, needs causal
  blocks. AKI is supported from day one but run after mortality works; not named in the abstract until it's run.
* **First ICU stay per patient only** (YAIB splits by stay, which leaks patients; we fix this).
* **Splits:** patient-level, label-stratified **60/15/10/15 = train/val/meta/test** on MIMIC-IV stays up to 2016.
  Stays from **2017 on → temporal holdout** (year reconstructed from `anchor_year_group`, ≈±1.5 y; a patient's earliest
  stay decides their split). eICU = external, **never used for any decision**.
  * `train` fits weights; `val` early stopping, model selection, NAS fitness; `meta` fits routers, stacking, KNORA-E,
    temperature scaling, greedy soup (so combiners never see data the experts were tuned on); `test` in-distribution.
* **Repeats:** 3 split seeds × 3 training seeds for everything. **Changed 2026-11-23 (Kiril):** NAS (all 4 variants) and the soup now also run on split seeds 1 and 2 (before: split 0 only), so NAS vs AES is matched run for run. Decided after the first results; all splits are reported whatever they show, and the change is stated in the methods.
* **Missingness masks** on, with an on/off ablation per expert (tests the "charting habits" hypothesis: count /
  missingness features help on MIMIC but hurt on eICU — a 2026 study found this).
* **Selection metric:** validation AUROC. **Calibration:** temperature scaling on meta, never refit on test sets.
* **Pre-registered primary comparisons (Holm-corrected):** NAS vs soft AES, soft AES vs best expert, NAS vs best
  expert — on mortality24, internal + external test.
* **Version B training (D25–28):** pooled **MIMIC-IV + HiRID → eICU**, plus leave-one-database-out over the three
  available DBs. AUMCdb added as a third training source / fourth LODO fold when access arrives, with no code change.
  Statics in **5-unit bins in all sources** (stops models recognising the source; HiRID only has binned age).
  Source experts (one per training source) for MIMIC-IV and HiRID now.

### Models and methods
* **Experts:** LightGBM, GRU-D, TCN, Transformer, Mamba (+ GRU optional). Raindrop only if porting works in ~1 week.
* **AES / ensembles:** static best, mean ensemble, stacking, KNORA-E, soft router, hard router, oracle (upper bound only).
* **Model soup:** Transformer only, pretrained once by masked-value reconstruction, 10 fine-tuned ingredients,
  uniform + greedy soup. Soups across different architectures don't work. **Head init = LP-FT (Kiril, 2026-11-23):**
  a linear probe (head only, frozen encoder in eval mode, train split, early stopping on val AUROC) is fitted once;
  all 10 ingredients start from its weights, head included [Kumar et al. 2022, LP-FT; Wortsman et al. 2022] (verify).
  Reason: the first run gave every ingredient a fresh random head, so the uniform soup averaged unrelated heads
  (eICU ECE 0.239, below every ingredient; greedy kept 1 of 10). That run is archived in
  `artifacts/soup_archive_randomhead_2026-11-22/`; its cost-ledger rows (`soup_ingredient*` from SLURM jobs 2926 and 2982–2990) must
  be excluded from the soup's cost. Valid soup = jobs 2925 (pretrain), 3004 (LP), 3005 + 3008–3016 (ingredients).
* **NAS:** regularised evolution + random search with the same budget. 300 low-fidelity candidates, top-5 retrained
  with 3 seeds. Built-in ablations: inherit vs scratch weights, uniform vs AES-guided family prior.
* **Cost logging** (GPU-h, params, FLOPs, memory, wall time) for every run from day one.

### Infrastructure
* Own Python package (not wrapping YAIB's training code): reads YAIB-cohorts output and reimplements YAIB's
  preprocessing exactly; validated by reproducing YAIB's LightGBM/GRU numbers. (Decided via decision card; recommended
  option taken.)
* PyTorch + Lightning, Hydra configs, MLflow off by default (local only if on), private shared GitHub repo.
* One codebase for both versions (`pipeline: A | B` switch).

### Constraints (facts about Kiril's setup)
* University cluster with **SLURM**. GPU nodes (`gpgpu01`) have an **NVIDIA A100-SXM4-80GB** (MIG 7g.80gb = whole card).
  Jobs run up to **8 h**; GPU jobs request 32 GB RAM, CPU jobs 4–8 GB. GPUs are often busy: CPU-only smoke tests can
  be run with plain `bash` on the login node. Internet available on login and compute nodes. Login node has no
  `nvcc`. R is not installed system-wide but runs from `.r-env` (micromamba).
* Storage: the shared `/home/hpc` filesystem has plenty of free space (raw MIMIC-IV + eICU now sit in `dip/data/raw`);
  the original cloud plan below is a fallback only. Original plan: read raw data from PhysioNet's Google Cloud copies, run
  ricu/YAIB extraction once on a spot GCE VM (e2-highmem-8, ~$0.11–0.22/h), store processed data + checkpoints in
  Cloudflare R2 (no egress fees). Expected total well under $50. Still to confirm: node-local scratch / project
  space from cluster admins, and supervisor OK on a private cloud bucket.
* Access: MIMIC-IV, eICU, MIMIC-CXR, HiRID all OK (HiRID application text in `applications/`); **AUMCdb pending**.
  MIMIC-IV v2.2 and eICU v2.0 are in `dip/data/raw/` on the cluster (downloaded 2026-11-12).
* Course colleague is credentialed and shares the repo.

---

## 4. Literature anchors (from the research files)

* MIMIC → eICU zero-shot reference: **YAIB GRU 87.6 → 81.7 AUROC, 42.8 → 28.0 AUPRC** (ICU mortality). METRE LSTM
  0.869 → 0.824. PULSE-ICU's 0.864 on eICU is *after fine-tuning* on eICU, not zero-shot.
* Sepsis/AKI/resp-failure transfer much worse than mortality (labels depend on documentation).
* Pooling more databases beats any single source in recent work, but Londschien et al.: extra DBs barely help once you
  have 10–50k own patients; transfer is better within a continent (HiRID/AUMCdb are European).
* Plain fine-tuning was the weakest adaptation strategy in a 2026 sepsis study.
* ~450 deaths in the test set → one AUROC is only ±~2 points; paired comparisons narrow this to ±~1.2.
* Kiril's seed papers (arXiv): 2112.13208, 2310.01154, 1701.06538 (sparse MoE), 2509.26037, 1802.03268 (ENAS),
  2403.19887, 2101.11605, 2412.11211, 2510.27039.
* Many facts in the research files are marked **(verify)** — they were written from memory or abstracts. Check them
  against the PDFs before citing.

---

## 5. Code: `icu-aes-nas` (Pipeline B) — layout and status

### Status (as of 2026-11-23)
* **Full pipeline (mortality24, HiRID on) ran 2026-11-22/23:** prepare, 45 experts, AES, 4 NAS variants (300
  candidates each, top-5 × 3 seeds), soup — no errors. Split-0 val AUROC: GRU-D 85.3 best expert; NAS finals
  85.3–85.7. Seed-0 eICU AUROC (preliminary): stacking 85.4, aes_soft 85.0, mean ensemble 84.8, lgbm 84.1, grud 83.7,
  NAS 83.0–83.6.
* **Soup re-run with LP-FT done + all 9 evaluation seeds (2026-11-23).** Results, mortality24, AUROC mean over runs
  (experts/AES: 3 splits × 3 seeds; NAS: split 0 × 3 seeds; soup: 1 run), internal test / temporal / HiRID test / eICU:
  stacking 88.3/89.9/88.1/**85.0**; aes_soft 88.0/89.7/88.0/84.7 (best eICU ECE 0.006); mean ens 87.8/89.6/87.5/84.5;
  lgbm (= best_expert on val) 87.2/88.9/86.8/84.1; grud 87.2/88.4/86.8/83.3; soup_uniform 87.8/88.7/87.3/84.1 (ECE
  0.012, was 0.239 with random heads; greedy kept 2/10 and generalised worse than uniform: 83.2 on eICU); NAS main
  (evolution_uniform_inheritTrue) 87.6/88.4/86.2/83.1; NAS scratch (inheritFalse) 83.4, random 83.2, AES prior 83.1.
  Primary comparisons (Holm): aes_soft > lgbm on eICU in 8/9 runs (+0.65); NAS < aes_soft on eICU 3/3 (−1.9) and
  temporal 3/3 (−1.1); NAS < lgbm on eICU 3/3 (−1.0); internal-test differences not significant.
  GPU-h: NAS 3.2 (4 variants), experts 0.30 (45 runs), soup 0.09, AES 0.01.
* `train_deep(..., trainable_prefixes=("head.",))` freezes everything else and keeps frozen children in eval mode
  (used by the soup linear probe); smoke test checks encoder unchanged + identical ingredient init.
* Complete Pipeline B code. Unit tests (22 + 1 GPU-only) and the three smoke suites pass on CPU with PyTorch 2.4.1,
  Lightning 2.4.0 and the Mamba kernels installed. On the cluster, `smoke_base_models` and `smoke_aes` passed as
  GPU jobs (A100 80 GB).
* **Mamba:** `mamba-ssm` 2.2.2 + `causal-conv1d` 1.4.0 installed from prebuilt wheels (no `nvcc` on the login node).
  `core/blocks/ssm.py` now has one parameter layout and picks the CUDA `selective_scan_fn` per forward call only for
  CUDA tensors (fixed the `x.is_cuda()` crash in `smoke_nas`); checkpoints are portable between CPU and GPU.
  `tests/unit/test_ssm_paths.py` checks kernel = reference scan when a GPU is present — not yet run on a GPU.
* **Data:** MIMIC-IV v2.2 and eICU v2.0 downloaded with `scripts/download_physionet.sh` into `data/raw/` and imported to
  fst. YAIB-cohorts extraction **done for eICU** (`data/cohorts/{base,mortality24,aki}/eicu`). **MIMIC-IV extraction
  failed** in job 2699 because ricu saw only 30/31 miiv tables: the fork's config wants `hosp/caregiver.csv.gz`,
  v2.2 ships it in `icu/`. Fixed (2026-11-18): `extract_cohorts.sh` symlinks it; caregiver imported, ricu banner
  now shows `miiv: 31 of 31`. **miiv cohorts extracted** (job 2709: base, mortality24, aki). HiRID download in progress (Kiril).
* **Kiril's changes (Nov 17):** `extract_cohorts.sh` uses `.r-env` via `PATH` and skips renv (`--no-init-file`,
  `RENV_ACTIVATE_PROJECT=FALSE`); `install_r_packages.R` pins the ricu fork; `extract_cohorts.sbatch` back to CPU,
  8 GB, relative `logs/`; new `notebooks/eda.ipynb` + `slurm/eda.sbatch` (CPU, 3 h; aggregates to `outputs/eda/`).
  EDA has not completed yet (job 2697 ran before any cohort existed).
* **SLURM settings (Nov 12):** GPU jobs (train_experts, soup, nas, smoke_*) and also prepare/aes/evaluate request 8 h,
  32 GB, 1 GPU, with absolute log paths under `dip/logs/` (`.out` + `.err`) and absolute `IAN_ROOT`, like Kiril's
  fraud-detection project. prepare/aes/evaluate do no GPU work: their GPU line can be removed if the queue is slow.
  `_common.sh` prints GPU name + VRAM. New: `slurm/download_physionet.sbatch` (CPU). `eda` extra added to
  `pyproject.toml`, installed by `setup_env.sh`.
* **Not yet verified:** MIMIC-IV cohort extraction; Mamba kernel + mixed precision inside a GPU job; SLURM requeue;
  runtime/memory at full size.
* Two choices made without asking (flag them if relevant): the four NAS variants are saved in separate folders
  (`nas.tag`); for the temporal holdout a patient's earliest stay decides their split.
* HiRID is **on** in `conf/data/pipeline_b.yaml` since 2026-11-22 (pooled MIMIC-IV + HiRID → eICU), consistent with
  the thesis text. All three cohorts extracted; EDA done (`outputs/eda/`: databases nearly perfectly separable by a
  domain classifier, AUROC ≈ 0.999, driven by charting habits, e.g. methb measured in 91 % of HiRID stays vs <1 % in
  MIMIC-IV; HiRID AKI prevalence 23 % vs ~40 %).
* **YAIB reproduction check passed (2026-11-22)**, MIMIC-IV only → eICU, split 0, 3 seeds, separate folders
  `data/processed_yaibcheck`, `artifacts_yaibcheck`: GRU AUROC 86.6 internal / 81.5 eICU (YAIB 87.6 / 81.7), AUPRC
  39.6 / 28.8 (YAIB 42.8 / 28.0); LightGBM 87.1 / 83.3; temporal 87.9 (GRU) / 88.4 (LGBM). GRU trains in ~17 s on the
  A100. Note: `ian-evaluate` bootstrap (1000, eICU clustered by hospital) takes ~10 min per seed on CPU.

### Commands
```bash
./setup_env.sh [--cuda cpu|cu118] [--with-r] [--with-mamba]; source .venv/bin/activate
bash scripts/download_physionet.sh [miiv] [eicu]                # PhysioNet download (asks login or ~/.netrc)
bash scripts/extract_cohorts.sh "miiv eicu" "mortality24 aki"   # R / YAIB-cohorts, step 1
sbatch slurm/eda.sbatch                                          # optional EDA notebook -> outputs/eda/
ian-prepare task=mortality24                                     # splits + preprocessing, step 2
ian-train model={lgbm,grud,tcn,transformer,mamba} seed=0 split_seed=0
ian-pretrain-soup cli.soup_stage={pretrain|linear_probe|ingredient|combine} [cli.ingredient=i]
ian-aes seed=0 split_seed=0
ian-nas nas.strategy=evolution nas.prior=uniform nas.inherit=true cli.nas_mode={search|finalize|summary}
ian-evaluate seed=0 split_seed=0
bash slurm/submit_pipeline_b.sh mortality24                      # whole chain with afterok dependencies (NAS + soup on NAS_SPLITS, default "0 1 2")
bash slurm/submit_followup.sh [splits] [nomask] [lodo] [aki]      # follow-up runs after the first mortality run; DRY_RUN=1 prints only
pytest tests/unit; pytest -m smoke_base; pytest -m smoke_aes; pytest -m smoke_nas
```
Fake-data demo: `export IAN_ROOT=$PWD/tmp_demo; ian-make-synthetic cli.synthetic_patients=150; ian-prepare data.n_split_seeds=1; ian-train trainer=smoke model=tcn model.spec.d_model=16 …`

### Architecture
* All deep experts are one class, `core/network.py::BlockNetwork`:
  `batch → stem → [static conditioning] → block 1 → … → LayerNorm → head → logit(s)`. This is what makes cross-family
  NAS possible.
* **Token bus** (`core/bus.py`): every block reads/writes `tokens [B,L,d]`, `mask [B,L]`, `time [B,L]` (hours),
  `type [B,L]` (0 static, 1 EHR hour, 2 ECG, 3 CXR). Blocks convert to their native format internally (TCN works
  channels-first `[B,d,L]`).
* **Stems:** `linear` (projects `[x, m]`), `grud` (GRU-D decay). **Blocks** (`core/blocks/`), pre-LN residual, all
  causal (unit-tested): `gru`, `tcn`, `attn`, `mamba` (CUDA kernel or pure-PyTorch fallback), `ffn`, `identity`.
  **Heads:** `last`, `mean`, `attnpool` (cumulative pooling for hourly tasks). **Static conditioning:** `add`/`token`/`none`.
* **Preprocessing** (`data/preprocess.py`, mirrors YAIB): z-score with pooled-train stats → missing indicator → forward
  fill → zero fill; `delta` (hours since last measurement, capped 48 h); statics binned (5 units), z-scored, plus
  missing indicators. Memory-mapped arrays `X, M, D, S, valid, y, ymask` + `index.parquet`. T = 24 (mortality24),
  168 (aki).
* **LightGBM** on tabular features (`data/features.py`): last/min/max/mean/count per variable + statics.
  Ablations: `trainer.use_missingness_mask=false` (`<model>_nomask`), `model.use_count_features=false` (`lgbm_nocounts`).
* **AES** (`aes/run.py`): all combiners fit on meta. Router input = meta-features (missingness rate, measurements/hour,
  hours observed, age, hour) + expert logits (+ source one-hot, all zeros on eICU). Writes `router_weights_seed*.parquet`
  and `nas_prior_seed*.json` (hard-router family frequencies → AES-guided NAS prior, mixed 50/50 with uniform).
* **NAS** (`nas/`): genome = stem + 2–4 block slots (family, options, source = `scratch` or an expert name for inherited
  weights) + head + d_model ∈ {64,128}. Identity-initialised linear **stitch** (ridge-fitted) between blocks from
  different experts (`nas/inherit.py`). `is_legal` rejects illegal genomes. Zero-cost proxy (SNIP/grad-norm) logged
  and optionally used to pre-rank offspring. State saved after every candidate; exits 75 near time budget for SLURM
  requeue. Finalize = top 5 × 3 seeds at full fidelity. Variants tagged `<strategy>_<prior>_inherit<bool>`.
* **Evaluation** (`eval/`): AUROC, AUPRC, Brier + scaled Brier, ECE (15 bins), calibration slope/intercept; 95 %
  bootstrap CIs clustered by **hospital on eICU**; DeLong + paired clustered bootstrap; Holm. `best_expert` chosen on
  **validation** AUROC. Cost ledger: `artifacts/cost_ledger.jsonl`.
* **SLURM resumability:** `done.json` skip markers, `last.ckpt` every epoch, `--signal=USR1@300`, training self-stops
  at 7.5 h and exits 75, `slurm/_common.sh::run_resumable` requeues on 75 (fallback: resubmit, `MAX_RESUBMITS=20`).
* Env vars: `IAN_ROOT`, `IAN_DATA`, `IAN_ARTIFACTS`, `IAN_OVERRIDES` (extra Hydra overrides for every SLURM job).

### Code map
```
src/icu_aes_nas/
├── constants.py   48 dynamic vars, statics, split names, token types
├── data/          sources, cohort, splits, preprocess, features, dataset, prepare, synthetic
├── core/          bus, blocks/, stems, heads, network
├── models/        Lightning module (BCE + masked pretraining), expert specs, LightGBM
├── training/      resumable training, prediction writer, artifact layout
├── aes/           predictions loader, meta-features, calibration, selectors, router, soup, run
├── nas/           genome, space, inherit (+ stitching), proxies, search, finalize
├── eval/          metrics, stats (bootstrap, DeLong, Holm), report
├── pipelines/     registry (B now, A later)
├── utils/         seed, atomic I/O, logging, SLURM helpers, cost tracking
├── cli/           ian-* entry points
└── conf/          Hydra configs (paths, data, task, model, trainer, aes, nas, eval)
r/ scripts/ slurm/ tests/{unit,smoke}/ setup_env.sh
```
Full details: `README.md` (18 sections, including troubleshooting and outputs reference).

---

## 6. Adding Pipeline A later (additions only)
1. **Data:** link MIMIC-IV-ECG and MIMIC-CXR by `subject_id` + time window; write `ecg.npy`, `cxr.npy`, their times and
   an availability mask next to `X.npy`.
2. **Stems** (`core/stems.py`): ECG/CXR encoders emitting tokens with `type = TOKEN_ECG / TOKEN_CXR` and own `time`.
3. **Fusion** (`core/network.py`): concatenate modality tokens into the bus; only blocks with
   `can_fuse_modalities = True` (attention) mix token types; TCN/GRU act on EHR tokens only.
4. **AES:** pass per-patient modality availability to `aes/router.py` (`avail`).
5. **Registry:** add `"A"` to `pipelines/registry.py` + `conf/data/pipeline_a.yaml`; run with `pipeline=A data=pipeline_a`.

---

## 7. Thesis writing status
* A Claude Doc "Thesis draft: structure, abstract and introduction" exists on claude.ai
  (https://claude.ai/artifact/CQAMjeiYpTZey8Q1W5mxH5). It holds: chapter plan with write-now/write-later marks,
  abstract with **[R: …]** result placeholders, Chapter 1 Introduction, Chapter 2 Background & related work (ends with
  a prior-work comparison table), Chapter 3 Data (databases, cohort/label, preprocessing, splits, cohort description
  3.5, data access), references [1]–[113] (some marked [authors]/[exact title]/(verify)), and "Guide" quote blocks to
  delete before submission.
* Kiril's style answers: chapter style; [1] citations; "we"; the ~25 ECG/CXR/TCGA papers cited in **one sentence** near
  the end of §1.6 as [84]–[108]; AKI not named in the abstract until it's run.
* EDA: allowed, but only aggregate statistics, framed as "cohort characteristics and differences between databases"
  in the Data chapter. No patient rows, no single-patient time series, percentiles instead of raw min/max. Groups
  <10 patients merged or reported as "<10" (an extra precaution, not a licence requirement).
* Suggested next writing step: the evaluation protocol (Methods), fixed before the first eICU test run.

---

## 8. Open questions and next steps

**Immediate (Kiril):**
1. `bash slurm/submit_followup.sh` (queued 2026-11-23?): NAS + soup on splits 1–2, missingness ablation (`<model>_nomask`, `lgbm_nocounts`), mortality re-evaluation; LODO folds (HiRID / MIMIC-IV held out → `artifacts_lodo_<db>`, experts + AES + main NAS variant on split 0 (Kiril, 2026-11-23)); full AKI pipeline. Then check results.
1b. Re-run `sbatch slurm/smoke_nas.sbatch` and `smoke_base_models` on a GPU to confirm the Mamba fix with the kernel.
2. Ask cluster admins about node-local scratch (`$TMPDIR`) / 20–50 GB project space, and whether 8 GB is RAM or GPU memory.
3. Confirm with supervisor that a private cloud bucket is acceptable under the DUA.
(Done: cohorts for all three DBs, `ian-prepare`, YAIB reproduction check, HiRID download + toggle.)

**Waiting on results (Part C of decisions file — don't decide early):** lock A, B or both (needs ECG/CXR coverage
audit + access timing); keep Raindrop; keep AKI; Version A fusion points/windows; fine-tune vs frozen encoders;
NAS search-space size and budget (after timing one candidate); λ in the Version B selection rule (preset 0.5); GNN
surrogate; whether the AES-guided prior becomes the main method.

**Other open threads:** whether "multimodal" (TA's requirement) is satisfied by structured EHR (Version B) or needs
images/signals (Version A) — ask the TA; AUMCdb access (separate application via Amsterdam Medical Data Science; ricu
reads only v1.0.2).

---

## 9. Research file map (copies are in `docs/research/`)
| File | What it holds |
|---|---|
| `research/dataset_and_models.md` | Dataset cards (MIMIC-IV, MIMIC-III vs IV, eICU, TCGA/MLOmics), why MIMIC→eICU won, linked papers' bearing on novelty, expert pool reasoning, soup caveats |
| `research/mimic_to_eicu_literature.md` | ~12 papers (2023+) on MIMIC-IV→eICU: harmonisation methods, performance drops |
| `research/mimic_iv_splits_and_dataset_lockin.md` | Locked split spec, leakage rules, model-selection rules, reporting checklist, multi-source (HiRID/AUMCdb) analysis |
| `research/fake_datapoint_examples.md` | One fabricated patient shown in each DB's raw format and after ricu/YAIB, plus model-input shapes |
| `research/pipeline_proposal_two_versions.md` | Main design doc (74 KB): both versions, token bus and adapters (§3, with a worked TCN→Transformer example), NAS as typed graph (§4), costs, risks, timelines (§9), how each model works (§11), statistics (§12), 26-decision log (§13) |
| `research/decisions_to_lock_before_coding.md` | Parts A–D: setup facts, 32 decisions with Kiril's locked answers, decisions that wait for results, cloud storage/compute plan |
| `applications/hirid_access_application.md` | HiRID access application: title, scientific abstract (259 words), lay summary (162 words) |
