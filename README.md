# icu-aes-nas

**Adaptive Expert Selection (AES) versus cross-family block NAS for ICU prediction.**

This repository holds the code for the bachelor thesis and the Intelligent Information Systems course project.
It compares five individual "expert" models, classic ensembles, a model soup, and Adaptive Expert Selection
(routers that pick or weight experts per patient). These are compared against a Neural Architecture Search
that builds new networks by connecting blocks from *different* model families. One example is a TCN block
feeding an attention block, with weights inherited from the trained experts. Every method is costed on
generalisation (external test on another database) and on compute (logged from day one).

This is **Pipeline B**: structured hourly ICU time series from several databases. Pipeline A (MIMIC-IV plus
ECG and chest X-ray) plugs into the same code later (see [Adding Pipeline A](#13-adding-pipeline-a)).

---

## Contents

1. [Data use rules](#1-data-use-rules)
2. [Study design at a glance](#2-study-design-at-a-glance)
3. [Installation](#3-installation)
4. [Data: download and folder layout](#4-data-download-and-folder-layout)
5. [Step 1: cohort extraction (R / YAIB-cohorts)](#5-step-1-cohort-extraction-r--yaib-cohorts)
6. [Step 2: splits and preprocessing](#6-step-2-splits-and-preprocessing)
7. [Step 3: base models (experts)](#7-step-3-base-models-experts)
8. [Step 4: ensembles, soup and AES](#8-step-4-ensembles-soup-and-aes)
9. [Step 5: cross-family NAS](#9-step-5-cross-family-nas)
10. [Step 6: evaluation and statistics](#10-step-6-evaluation-and-statistics)
11. [Running on SLURM (8 h / 32 GB / 1 GPU jobs)](#11-running-on-slurm-8-h--32-gb--1-gpu-jobs)
12. [Configuration (Hydra)](#12-configuration-hydra)
13. [Adding Pipeline A](#13-adding-pipeline-a)
14. [Testing](#14-testing)
15. [Outputs reference](#15-outputs-reference)
16. [Code map](#16-code-map)
17. [Troubleshooting](#17-troubleshooting)
18. [What has and has not been verified](#18-what-has-and-has-not-been-verified)

---

## 1. Data use rules

MIMIC-IV, eICU and HiRID are credentialed PhysioNet datasets under data use agreements. Everyone who runs this
code must hold their own credentials.

* **Never commit data.** `data/`, `artifacts/`, `outputs/`, `*.parquet`, `*.npy` and checkpoints are in `.gitignore`.
  Keep it that way.
* **Never send records or per-patient outputs to online services**, LLM chat tools included. Only aggregate
  numbers (AUROC tables, plots of aggregate metrics) leave the secure environment.
* MLflow tracking is off by default. If you turn it on (`tracking.mlflow=true`), it writes to local disk only.
* Tests and smoke runs use **fabricated** data (`icu_aes_nas/data/synthetic.py`). Every value is random, and
  ids start at 9,000,000 so they can never be mistaken for real ones.

## 2. Study design at a glance

| Item | Choice |
|---|---|
| Training data | MIMIC-IV v2.2 (pooled with HiRID v1.1.1 when enabled; AUMCdb pending) |
| External test | eICU v2.0, all stays, never used for any decision |
| Tasks | `mortality24`: ICU mortality from the first 24 h (primary, one label per stay). `aki`: hourly AKI onset (secondary, one label per hour) |
| Cohort | YAIB-cohorts definitions; **first ICU stay per patient** |
| Splits | Patient-level, label-stratified 60/15/10/15 = train / val / meta / test on MIMIC-IV stays up to 2016. Stays from 2017 onward form a **temporal** holdout |
| Repeats | 3 split seeds × 3 training seeds. NAS runs on split seed 0 |
| Experts | LightGBM, GRU-D, TCN, Transformer, Mamba (plus GRU as an optional extra) |
| Ensembles / AES | static best, mean ensemble, stacking, KNORA-E, soft router, hard router, oracle (upper bound) |
| Soup | Transformer pretrained by masked reconstruction, 10 fine-tuned ingredients, uniform and greedy soup |
| NAS | Regularised evolution and random search (same budget). Inherit vs scratch genes. Uniform vs AES-guided family prior |
| Selection | Validation AUROC. Temperature scaling fitted on the meta split |
| Statistics | Bootstrap CIs (clustered by hospital on eICU), DeLong and paired bootstrap, Holm correction over the pre-registered comparisons |

**What each split is for:**

* `train` fits model weights.
* `val` handles early stopping, model selection and NAS fitness.
* `meta` fits routers, stacking, KNORA-E, temperature scaling and the greedy soup. It is kept separate so the
  combiners never see data the experts were tuned on.
* `test` gives the in-distribution estimate.
* `temporal` checks drift over time.
* `external` (eICU) checks transfer to another database and is the headline result.

## 3. Installation

Requirements: Linux, Python 3.10–3.12, git, and internet access on the node that runs the setup. A GPU is
optional; everything also runs on CPU, just more slowly.

```bash
git clone <this repo> icu-aes-nas && cd icu-aes-nas
./setup_env.sh                 # .venv with PyTorch 2.4.1 (CUDA 12.1) + all pinned packages, clones YAIB-cohorts
source .venv/bin/activate
```

| Option | Effect |
|---|---|
| `--cuda cpu` | CPU-only PyTorch (laptop, CI) |
| `--cuda cu118` | Older CUDA drivers |
| `--with-r` | Also installs R 4.2 + ricu 0.6.0 into `.r-env/` with micromamba (no root or admin needed). Only needed for cohort extraction |
| `--with-mamba` | Tries to build the fast `mamba-ssm` CUDA kernels. If that fails, the Mamba expert falls back to a pure-PyTorch scan (same maths, slower) |

On a cluster, load a Python module first if the system Python is too old (`module load python/3.11`). Set
`PYTHON=/path/to/python3.11 ./setup_env.sh` to choose a specific interpreter. Running the script again is safe;
existing environments are reused.

The script ends with a check that prints the torch version, whether CUDA is visible, and whether the Mamba
kernels are in use.

## 4. Data: download and folder layout

Download from PhysioNet into `data/raw/` using the layout ricu expects. Set `IAN_DATA=/some/other/path` if the
data lives elsewhere, for example on a mounted cloud bucket.

```
data/
├── raw/
│   ├── miiv/hosp/*.csv.gz          MIMIC-IV v2.2  (hosp module)
│   ├── miiv/icu/*.csv.gz           MIMIC-IV v2.2  (icu module)
│   ├── eicu/*.csv.gz               eICU-CRD v2.0
│   └── hirid/                      HiRID v1.1.1 (optional)
│       ├── general_table.csv
│       └── observation_tables/csv/part-*.csv  (+ pharma_records, reference_data)
├── cohorts/<task>/<src>/{dyn,sta,outc}.parquet   ← written by step 1
└── processed/<task>/split<k>/...                  ← written by step 2
```

Approximate sizes: MIMIC-IV hosp+icu ≈ 10 GB compressed, eICU ≈ 5 GB, HiRID ≈ 20 GB. ricu's fst copies
roughly double that. The cohorts and processed arrays are much smaller (a few GB).

### HiRID and AUMCdb toggles

Sources are listed in `conf/data/pipeline_b.yaml`:

```yaml
sources:
  miiv:  {enabled: true,  role: train}
  eicu:  {enabled: true,  role: test}
  hirid: {enabled: false, role: train}   # flip to true once downloaded and extracted
  aumc:  {enabled: false, role: train}   # access pending; no code change needed later
```

To enable HiRID for one run, add the override `data.sources.hirid.enabled=true`. To make it permanent, edit the
YAML. With two or more training sources:

* the preprocessing statistics are pooled over all training sources,
* the router also receives a one-hot of the training source (all zeros on eICU, so it cannot learn "this is the
  test database"),
* leave-one-database-out runs become possible by flipping a role, for example
  `data.sources.hirid.role=test`.

## 5. Step 1: cohort extraction (R / YAIB-cohorts)

Cohorts come from the [YAIB-cohorts](https://github.com/rvandewater/YAIB-cohorts) R scripts, built on ricu and
pinned to commit `74ac699`. They harmonise 48 hourly variables (vitals and labs) and 4 statics (age, sex,
height, weight) across databases, and define the task labels.

```bash
./setup_env.sh --with-r                         # once
bash scripts/extract_cohorts.sh "miiv eicu" "mortality24 aki"
# or on SLURM, one source per job (MIMIC-IV is the heaviest):
sbatch slurm/extract_cohorts.sbatch miiv
sbatch slurm/extract_cohorts.sbatch eicu
```

What it does:

1. `r/import_ricu.R` converts the raw CSVs to ricu's fst format. This happens once per source; tables that are
   already imported are skipped.
2. It writes `third_party/YAIB-cohorts/config.yaml` with absolute paths to `data/raw` and `data/cohorts`.
3. It runs `base_cohort.R`, then `mortality.R` and/or `aki.R` for each source.

Output per task and source: `dyn.parquet` (stay_id, time in hours, 48 variables), `sta.parquet` (statics),
`outc.parquet` (labels; with a `time` column for AKI), plus an attrition table.

If R cannot be installed on the cluster, run this step anywhere with R (a laptop, or a cloud VM) and copy
`data/cohorts/` across. Nothing downstream needs R.

## 6. Step 2: splits and preprocessing

```bash
ian-prepare task=mortality24          # all 3 split seeds
ian-prepare task=aki
```

For each enabled source this:

1. **Builds the cohort** (`data/cohort.py`). It reads the YAIB output and joins stay metadata from the raw
   files (`data/sources.py`: patient id, stay order, estimated year, hospital id). It keeps only each patient's
   **first ICU stay**, decided over all of the patient's stays and not just the ones YAIB kept, and records
   attrition in `processed/<task>/attrition.json`.
2. **Splits** (`data/splits.py`). Training sources are split per patient, stratified by label, with a seeded
   RNG that is stable across processes. A patient never appears in two splits, and this is checked every time.
   MIMIC-IV stays with an estimated year ≥ 2017 go to `temporal`. The year is reconstructed from
   `anchor_year_group`, so it is approximate (about ±1.5 years). Test-role sources are entirely `external`.
3. **Fits the preprocessor on the pooled `train` split only** (`data/preprocess.py`). This mirrors YAIB's
   recipe:
   * dynamics: z-score with train statistics → missing indicator → forward fill → zero fill (zero is the train
     mean after scaling);
   * time since last measurement (`delta`, capped at 48 h and scaled), which GRU-D needs;
   * statics: age, height and weight binned in identical 5-unit bins in every source, z-scored, sex encoded as
     0/1, plus a missing indicator for each.
4. **Writes model-ready arrays** in chunks as memory-mapped `.npy` files, so no job needs more than 8 GB RAM:

| File | Shape / type | Meaning |
|---|---|---|
| `X.npy` | [N, T, 48] float32 | scaled, forward-filled values |
| `M.npy` | [N, T, 48] uint8 | 1 = measured in this hour |
| `D.npy` | [N, T, 48] float16 | hours since last measurement / cap |
| `S.npy` | [N, 8] float32 | statics + static missing indicators |
| `valid.npy` | [N, T] uint8 | 1 = this hour exists for the stay (shorter stays are right-padded) |
| `y.npy` | [N] or [N, T] int8 | labels |
| `ymask.npy` | [N, T] uint8 | AKI only: hours that carry a label |
| `index.parquet` | | stay_id, patient_id, hospital_id, source, split, label |

T is 24 for `mortality24` and 168 for `aki`.

## 7. Step 3: base models (experts)

```bash
ian-train model=lgbm   seed=0 split_seed=0
ian-train model=grud   seed=0 split_seed=0
ian-train model=tcn    seed=0 split_seed=0
ian-train model=transformer seed=0 split_seed=0
ian-train model=mamba  seed=0 split_seed=0
```

All deep experts are the *same* network class (`core/network.py::BlockNetwork`), which is what makes
cross-family NAS possible:

```
batch → stem → [static conditioning] → block 1 → block 2 → … → LayerNorm → head → logit(s)
```

* **Token bus** (`core/bus.py`). Every block reads and writes the same format: `tokens [B, L, d]`,
  `mask [B, L]`, `time [B, L]` (hours) and `type [B, L]` (0 static, 1 EHR hour, 2 ECG, 3 CXR). Each block
  converts the bus to its native format and back. A TCN works channels-first `[B, d, L]`; attention and the
  recurrent and SSM blocks work on `[B, L, d]`. Blocks therefore never need to know what came before them.
* **Stems** turn the arrays into tokens. `linear` projects `[x, m]` per hour. `grud` implements GRU-D (input
  decay toward the train mean, hidden-state decay from `delta`).
* **Blocks** (`core/blocks/`) are pre-LayerNorm residual blocks and all **causal** (unit-tested):
  * `gru`
  * `tcn` (dilated causal convolution)
  * `attn` (causal self-attention with time and type embeddings)
  * `mamba` (selective SSM; CUDA kernel or pure PyTorch)
  * `ffn`
  * `identity`
* **Heads**: `last`, `mean` and `attnpool`. For hourly tasks they pool cumulatively, so hour t only sees hours
  ≤ t.
* **Static conditioning**: `add` (projected statics added to every token), `token` (a prefix token) or `none`.

Expert definitions live in `conf/model/*.yaml`. For example, the Transformer is `stem: linear`,
`blocks: [attn, attn]`, `d_model: 64`.

Training uses PyTorch Lightning:

* AdamW, BCE loss.
* Early stopping on validation AUROC (validation loss for AKI).
* Mixed precision on GPU.
* Checkpoints every epoch, so a run can resume.

LightGBM is trained on tabular features (`data/features.py`): last, min, max, mean and measurement count per
variable over the observed window, plus statics.

**Ablations** are saved under their own names:
* `trainer.use_missingness_mask=false` trains without the measurement mask (saved as `<model>_nomask`).
* `model.use_count_features=false` trains LightGBM without count features (saved as `lgbm_nocounts`).

Every finished run writes `done.json`. Running the same command again is a no-op, which is what makes SLURM
requeueing safe.

## 8. Step 4: ensembles, soup and AES

```bash
ian-aes seed=0 split_seed=0
```

`aes/run.py` loads the experts' predictions on every split and fits every combiner **on the meta split**:

| Method | Idea |
|---|---|
| `static_best` | The single expert with the best meta AUROC |
| `mean_ensemble` | Average of the calibrated expert logits |
| `stacking` | Logistic regression on the expert logits |
| `knora_e` | Dynamic ensemble selection: for each patient, use the experts that are correct on all k nearest meta neighbours (KNORA-Eliminate) |
| `aes_soft` | Small MLP router. Input: meta-features (missingness rate, measurements per hour, hours observed, age, hour) + expert logits (+ source one-hot). Output: softmax weights over experts |
| `aes_hard` | Same router, but picks the arg-max expert |
| `oracle` | Picks the best expert per patient using the label. An upper bound only, never a real method |

All outputs are temperature-scaled on meta. The run also writes:

* `router_weights_seed*.parquet` (which expert each patient was routed to; used for the thesis analysis),
* `nas_prior_seed*.json` (how often the hard router chose each block family; used by the AES-guided NAS
  prior),
* `report_seed*.json`.

**Model soup** (`aes/soup.py`) has three stages:

```bash
ian-pretrain-soup cli.soup_stage=pretrain                       # masked-reconstruction pretraining
ian-pretrain-soup cli.soup_stage=ingredient cli.ingredient=0    # … up to 9: fine-tunes with varied lr/seed
ian-pretrain-soup cli.soup_stage=combine                        # uniform soup, greedy soup (on meta), baselines
```

## 9. Step 5: cross-family NAS

```bash
ian-nas nas.strategy=evolution nas.prior=uniform nas.inherit=true cli.nas_mode=search
ian-nas nas.strategy=evolution nas.prior=uniform nas.inherit=true cli.nas_mode=finalize
ian-nas ... cli.nas_mode=summary          # proxy-vs-score Kendall τ and search statistics
```

**Search space** (`nas/genome.py`, `nas/space.py`). A genome is:
* a stem (`linear` | `grud`),
* 2–4 block slots, each a family + its options + a *source*,
* a head (`last` | `mean` | `attnpool`),
* a width d_model ∈ {64, 128}.

A slot's source is either `scratch` (random init) or the name of a trained expert. In the second case the
block's weights are **inherited** from that expert.

* When two neighbouring blocks come from different experts, an identity-initialised linear **stitch** is
  inserted between them. It is fitted by ridge regression so the receiving block sees inputs on the scale it
  was trained on (`nas/inherit.py`).
* Illegal genomes are rejected (`is_legal`). For example, inheritance requires a matching width, and blocks
  must be causal for hourly tasks.

**Search loop** (`nas/search.py`):

1. Regularised evolution: tournament selection, one mutation (swap family, change source, add or remove a slot,
   change stem, head or width), and the oldest member is removed.
2. Random search with the **same budget** is the baseline.
3. Each candidate is trained for a few epochs (low fidelity) and scored on validation AUROC.
4. A zero-cost proxy (SNIP / grad-norm) is logged for every candidate. When `nas.proxy.enabled`, it pre-ranks
   offspring, so the search tries the most promising child first.
5. State (population, history, RNG state) is saved after **every** candidate. The controller stops before
   `nas.time_budget_hours` and exits with code 75 so SLURM requeues it.

**Finalize** retrains the top 5 candidates with 3 seeds each at full fidelity, picks the winner on mean
validation AUROC, and writes its predictions as `nas_<tag>_best`. This reduces the optimism of picking the
winner from noisy low-fidelity scores.

**Variants** are kept apart on disk by `nas.tag = <strategy>_<prior>_inherit<true|false>`:

| Variant | Command overrides |
|---|---|
| Main | `nas.strategy=evolution nas.prior=uniform nas.inherit=true` |
| Random baseline | `nas.strategy=random` |
| No inheritance | `nas.inherit=false` |
| AES-guided prior | `nas.prior=aes` (reads `nas_prior_seed0.json`, so run `ian-aes` first) |

The AES prior is mixed 50/50 with uniform, so no family becomes unreachable.

## 10. Step 6: evaluation and statistics

```bash
ian-evaluate seed=0 split_seed=0
```

For every method's predictions on `test`, `temporal` and `external`, `eval/report.py` computes:

* AUROC, AUPRC, Brier and scaled Brier, ECE (15 bins), calibration slope and intercept,
* 95 % bootstrap CIs. The bootstrap resamples patients, or **hospitals on eICU** (`eval.cluster_column`),
  because patients in one hospital are not independent.

`best_expert` is defined as the expert with the best **validation** AUROC (never chosen on test).
`nas_best` is the main NAS variant.

The pre-registered comparisons in `conf/eval/default.yaml` are:

* NAS vs soft AES,
* soft AES vs best expert,
* NAS vs best expert.

Each one gets a DeLong test and a paired clustered bootstrap of the AUROC difference. P-values are
Holm-corrected across the comparisons.

Outputs go to `artifacts/reports/<task>/split<k>/`: `metrics_seed*.csv`, `comparisons_seed*.csv`,
`summary_seed*.json`, and a running `evaluation_log.jsonl`.

**Cost.** Every training run, AES fit and NAS candidate appends a line to `artifacts/cost_ledger.jsonl`. Each
line holds method, run id, wall seconds, GPU hours, peak GPU memory, parameters, FLOPs per batch, epochs, host
and SLURM job id. The "NAS vs AES at what compute cost" comparison is computed from this file.

## 11. Running on SLURM (8 h / 32 GB / 1 GPU jobs)

Every job script asks for 8 h, 32 GB RAM and one GPU (`--gres=gpu:1`), and every step can resume. Log paths and `IAN_ROOT` are absolute (`/home/hpc/users/ml_models/konstantin.lozhankoski/dip`), as in the other projects on this cluster; stdout goes to `logs/*.out`, stderr to `logs/*.err`:

| Mechanism | Where |
|---|---|
| `done.json` marks a finished run; reruns skip it | `training/train.py` |
| Lightning saves `last.ckpt` every epoch and resumes from it | `training/train.py` |
| `#SBATCH --signal=USR1@300` makes Lightning save an emergency checkpoint 5 min before the limit and requeue the job | SLURM plugin |
| Training stops itself at `trainer.max_time_hours` (7.5 h) and the entry point exits with **75** | `cli/_hydra.py` |
| NAS saves state after every candidate and exits 75 near `nas.time_budget_hours` | `nas/search.py` |
| On exit 75, `run_resumable` calls `scontrol requeue` on the same job id, so dependent jobs keep waiting | `slurm/_common.sh` |

If your cluster forbids user requeues, `_common.sh` falls back to a new `sbatch` and prints a warning. In that
case, resubmit any dependent jobs by hand. `MAX_RESUBMITS` (default 20) stops endless loops.

### Full chain

```bash
export SBATCH_PARTITION=<partition>     # if your cluster needs it (sbatch reads these variables)
export SBATCH_ACCOUNT=<account>
sbatch slurm/extract_cohorts.sbatch miiv && sbatch slurm/extract_cohorts.sbatch eicu   # once
bash slurm/submit_pipeline_b.sh mortality24
bash slurm/submit_pipeline_b.sh aki
squeue -u $USER          # logs in logs/
```

`submit_pipeline_b.sh` submits these jobs with `afterok` dependencies:

```
prepare ─┬─ train_experts (array: 3 splits × 5 models × 3 seeds = 45)
         │      ├── aes (array: 3 splits × 3 seeds) ─────────────┐
         │      └── nas search ×4 variants → nas finalize ×4 ──────┤
         └─ soup pretrain → soup ingredients (array 0-9) → combine ┴── evaluate (array 9)
```

To add Hydra overrides to every job: `IAN_OVERRIDES="data.sources.hirid.enabled=true" bash slurm/submit_pipeline_b.sh`.

### Individual jobs

| Script | Resources | Notes |
|---|---|---|
| `extract_cohorts.sbatch <src>` | 1 GPU (CPU work) | R/ricu, one source per job |
| `prepare.sbatch [task]` | 1 GPU (CPU work) | |
| `train_experts.sbatch [task]` | 1 GPU, array 0-44 | `MODELS="gru tcn" sbatch --array=0-17 ...` for a subset |
| `soup.sbatch <stage> [task] [split]` | 1 GPU | `--array=0-9` for ingredients |
| `aes.sbatch [task]` | 1 GPU (CPU work), array 0-8 | |
| `nas.sbatch [task]` | 1 GPU | env vars `STRATEGY`, `PRIOR`, `INHERIT`, `NAS_MODE` |
| `evaluate.sbatch [task]` | 1 GPU (CPU work), array 0-8 | |
| `smoke_{base_models,aes,nas}.sbatch` | 1 GPU | fabricated data only |

Since the GPU type is unknown, the code checks at run time: mixed precision only when CUDA is present, and the
Mamba CUDA kernel only when it imports. Run `sbatch slurm/smoke_base_models.sbatch` first; the log
prints the GPU name.

### Storage

There is no room on the cluster or on a laptop, so the plan in the decision log (Part D) applies. Keep raw data
and cohorts in a cloud bucket or on a cloud VM and point `IAN_DATA` / `IAN_ARTIFACTS` at the mounted path. Only
`data/processed` (memmaps) needs fast local disk while jobs run.

## 12. Configuration (Hydra)

All settings live in `src/icu_aes_nas/conf/`. Any key can be overridden on the command line.

| Group | File | Main keys |
|---|---|---|
| `paths` | `paths/default.yaml` | `root` (`$IAN_ROOT`), `data` (`$IAN_DATA`), `artifacts` (`$IAN_ARTIFACTS`) |
| `data` | `data/pipeline_b.yaml` | sources, `first_stay_only`, `split_fractions`, `temporal_holdout`, `n_split_seeds`, `static_bin_width`, `delta_cap_hours` |
| `task` | `task/mortality24.yaml`, `task/aki.yaml` | `type` (static/dynamic), `seq_len` |
| `model` | `model/*.yaml` | `spec` (d_model, stem, blocks, head, static_mode, dropout) or LightGBM `params` |
| `trainer` | `trainer/default.yaml`, `trainer/smoke.yaml` | epochs, batch size, lr, patience, precision, `max_time_hours`, `use_missingness_mask` |
| `aes` | `aes/default.yaml` | `experts`, router hyper-parameters, `knora_k`, soup settings |
| `nas` | `nas/default.yaml` | `strategy`, `budget`, `population`, `families`, `inherit`, `prior`, `proxy`, `top_k`, `final_seeds`, `time_budget_hours` |
| `eval` | `eval/default.yaml` | `n_bootstrap`, `ece_bins`, `primary_comparisons`, `cluster_column` |

Examples:

```bash
ian-train model=transformer model.spec.d_model=128 trainer.lr=3e-4
ian-train model=tcn 'model.spec.blocks=[{family: tcn, dilation: 1},{family: tcn, dilation: 4}]'
ian-nas nas.budget=100 nas.families='[gru,tcn,attn]'
```

Each run's resolved config is saved under `outputs/hydra/<date>/<time>_<job>/`.

## 13. Adding Pipeline A

Pipeline A (MIMIC-IV structured data + ECG + chest X-ray, all as tokens with the latest 4 per stay, D22 c)
reuses everything above. Only additions are needed; nothing in Pipeline B changes:

1. **Data.** Link MIMIC-IV-ECG and MIMIC-CXR by `subject_id` and time window, and write extra arrays next to
   `X.npy` (e.g. `ecg.npy`, `cxr.npy`, plus their times and an availability mask).
2. **Stems** in `core/stems.py`. Add ECG and CXR encoders that emit tokens with `type = TOKEN_ECG / TOKEN_CXR`
   and their own `time`.
3. **Fusion** in `core/network.py`. Concatenate modality tokens into the bus. Only blocks with
   `can_fuse_modalities = True` (attention) mix token types; TCN and GRU act on EHR tokens only.
4. **AES.** Pass the per-patient modality availability mask to `aes/router.py` (`avail`), so experts that need
   a missing modality are not selected.
5. **Registry.** Add `"A"` to `pipelines/registry.py` and a `conf/data/pipeline_a.yaml`, then run with
   `pipeline=A data=pipeline_a`.

## 14. Testing

```bash
pytest tests/unit             # splits, preprocessing, block causality, statistics, NAS genomes (seconds)
pytest -m smoke_base          # synthetic data → prepare → all 6 deep/GBDT experts → predictions; AKI; ablation
pytest -m smoke_aes           # experts → all AES methods → soup (3 stages) → evaluation report
pytest -m smoke_nas           # evolution, random, scratch, AES prior, resume after interruption, finalize
```

The three smoke suites are independent. Each one builds its own synthetic workspace with MIMIC-IV, HiRID and
eICU-shaped data, so the HiRID toggle and pooled training are exercised too. On the cluster, use
`sbatch slurm/smoke_base_models.sbatch` (and `smoke_aes`, `smoke_nas`).

To try the CLI by hand on fake data:

```bash
export IAN_ROOT=$PWD/tmp_demo && mkdir -p $IAN_ROOT
ian-make-synthetic cli.synthetic_patients=150
ian-prepare data.n_split_seeds=1
ian-train trainer=smoke model=tcn model.spec.d_model=16
ian-train trainer=smoke model=lgbm model.params.n_estimators=10
ian-aes trainer=smoke 'aes.experts=[lgbm,tcn]'
ian-evaluate trainer=smoke
```

## 15. Outputs reference

```
artifacts/
├── models/<task>/split<k>/<model>/seed<s>/      spec.json, best.ckpt, last.ckpt, net.pt, done.json, logs/
├── predictions/<task>/split<k>/<method>/seed<s>.parquet
│       columns: source, split, row, time, stay_id, hospital_id, y, logit, p
├── aes/<task>/split<k>/                         report, router weights, nas prior
├── soup/<task>/split<k>/transformer/            pretrain/, ingredient<i>/, report.json
├── nas/<task>/split<k>/<tag>/                   state.json, history.jsonl, candidates/, final/, final.json
├── reports/<task>/split<k>/                     metrics, comparisons, summary, evaluation_log.jsonl
└── cost_ledger.jsonl
```

Every prediction file has the same columns, so any new method only has to write a file in this format to be
included in the evaluation.

## 16. Code map

```
src/icu_aes_nas/
├── constants.py        48 dynamic variables, statics, split names, token types
├── data/               sources (metadata per DB), cohort, splits, preprocess, features, dataset, prepare, synthetic
├── core/               bus, blocks/ (gru, tcn, attn, mamba, ffn, identity), stems, heads, network
├── models/             Lightning module (BCE + masked pretraining), expert specs, LightGBM
├── training/           resumable training, prediction writer, artifact layout
├── aes/                predictions loader, meta-features, calibration, selectors, router, soup, run
├── nas/                genome, search space, inheritance + stitching, proxies, search, finalize
├── eval/               metrics, statistics (bootstrap, DeLong, Holm), report
├── pipelines/          registry (B now, A later)
├── utils/              seeding, atomic I/O, logging, SLURM helpers, cost tracking
├── cli/                entry points (ian-*)
└── conf/               Hydra configs
r/                      R package pins and ricu import
scripts/                cohort extraction wrapper
slurm/                  job scripts, shared helper, pipeline submitter
tests/                  unit/ and smoke/
setup_env.sh            environment setup
```

## 17. Troubleshooting

| Symptom | Fix |
|---|---|
| `Rscript not found` | `./setup_env.sh --with-r`, or extract cohorts on another machine and copy `data/cohorts/` |
| PyTorch download fails | The script falls back to the PyPI wheel. Or set `--cuda` to match your driver (`nvidia-smi`) |
| `mamba-ssm` won't build | Ignore it; the pure-PyTorch Mamba block is used automatically |
| Out of memory (8 GB) | Lower `trainer.batch_size`, keep `trainer.num_workers` at 2, and use `model.spec.d_model=64` |
| A job keeps requeueing | Check `logs/`. `MAX_RESUBMITS` caps it. Delete a run's `last.ckpt` to restart it from scratch |
| `nas.prior=aes needs ... nas_prior_seed0.json` | Run `ian-aes seed=0 split_seed=0` first |
| `Expert predictions missing` warning in AES | That expert has not finished; AES runs on the experts it finds |
| Want to redo a finished run | Delete its `done.json` (or the whole run folder) |

## 18. What has and has not been verified

**Verified here:** the unit tests and the three smoke suites pass on CPU with PyTorch 2.4.1 and Lightning 2.4.0,
using fabricated data. The CLI and SLURM scripts also ran end to end locally, outside SLURM.

**Not yet verified:**
* the R / micromamba install and the YAIB-cohorts extraction (needs the real data);
* the Mamba CUDA kernels and mixed precision on a GPU;
* SLURM requeue behaviour on your cluster;
* runtime and memory on full-size MIMIC-IV and eICU.

Run the smoke jobs on the cluster first, then `ian-prepare` on the real cohorts. Check `attrition.json` and the
split counts it prints against the YAIB paper before training.
