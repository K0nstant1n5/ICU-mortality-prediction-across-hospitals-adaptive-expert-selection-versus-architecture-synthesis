# Decisions to lock before the code is written

Written 2026-11-02. This file lists only decisions that **do not depend on results**, so you can make them now and
they will not change after the first experiments. Decisions that need results are in Part C. None of them are asked
about now.

**How to answer:** reply "defaults" to accept every ★ recommendation. Otherwise, name only the ones you want to
change, e.g. "D7 b, D15 a". Part A is the exception: those are facts about your setup, not choices, so they need
real answers.

Background for every recommendation: `pipeline_proposal_two_versions.md` (§13 decision log) and
`mimic_iv_splits_and_dataset_lockin.md`.

## Status, 2026-11-02: locked

Kiril's answers: **all ★ defaults accepted except the changes below.**

| Item | Locked value |
|---|---|
| A1 Scheduler | SLURM. GPU type unknown (no permission to query); the code must not assume a specific GPU, and Mamba gets a fallback (pure-PyTorch implementation) if its CUDA kernels fail to build |
| A2 Storage | No space locally or on the cluster: data is kept in the cloud (plan in Part D) |
| A3 Jobs | Batch jobs only, **8 h and 8 GB per job**, internet available, R unknown. Every training job, AES fit and the NAS loop must checkpoint and resume (Part D) |
| A4 Access | Everything except AUMCdb |
| A5 Colleague | Credentialed; one shared repository |
| D19 | **b) the course project also includes Version A and B extras**, not just the shared structured core |
| D22 | **c) all ECGs and CXRs before the cut-off become tokens.** Default cap: the latest 4 of each per stay, so the sequence length stays bounded. Revisit the cap after the coverage audit |
| D25 | Pooled **MIMIC-IV + HiRID → eICU**, plus leave-one-database-out over the three available databases. AUMCdb is added as a third training source (and a fourth LODO fold) when access arrives, with no code change |
| D26 | Start without AUMCdb (as recommended) |
| D27 | 5-unit bins for statics in MIMIC-IV, HiRID, eICU (and AUMCdb later) |
| D28 | Source experts for MIMIC-IV and HiRID now; an AUMCdb expert later |

---

## Part A: facts I need from you (no defaults possible)

| # | Question | Why it matters for the code |
|---|---|---|
| A1 | Which **job scheduler** does the cluster use (SLURM, PBS, something else), and which **GPU type** (e.g. A100, V100, RTX)? | Job scripts, and whether Mamba's CUDA kernels will build |
| A2 | What is your **storage quota** on the cluster? | Version A needs roughly 90 GB for MIMIC-IV-ECG and several hundred GB for MIMIC-CXR-JPG if you download all of it (about 570 GB, **(verify)**). With a small quota, the code downloads only the images your cohort needs |
| A3 | Can you run **R** on the cluster, and do compute nodes have **internet access**? | ricu/YAIB-cohorts are R; downloads from PhysioNet need internet on some node |
| A4 | Where is your **access** now: PhysioNet credentialing done? MIMIC-IV, eICU, MIMIC-CXR signed? HiRID requested? AUMCdb applied? | Decides what can run first. The demo datasets cover pipeline building until access arrives |
| A5 | Is your **course colleague** credentialed too, and will you share one code repository? | Decides whether the repo is set up for two people and who runs what on the data |

---

## Part B: decisions you can lock now (★ = my recommendation)

### Code and infrastructure

| # | Decision | Options | Why ★ |
|---|---|---|---|
| D1 | Codebase base | ★ a) YAIB for data loading and the baseline experts, plus our own package for the token bus, NAS and AES · b) everything from scratch | YAIB already reproduces the benchmark numbers you will cite; you only write what is new |
| D2 | Framework | ★ a) PyTorch + PyTorch Lightning · b) plain PyTorch | YAIB uses Lightning; one framework for everything |
| D3 | Config system for our code | ★ a) Hydra (YAML) · b) gin (what YAIB uses) · c) plain argparse | Hydra makes sweeps over seeds and NAS candidates easy on SLURM |
| D4 | Experiment tracking | ★ a) MLflow on the cluster (stays local) · b) Weights & Biases with aggregate metrics only · c) CSV files | The data use agreement forbids sending records to outside services. MLflow keeps everything on the cluster. W&B is acceptable only for aggregate metrics, never per-sample predictions |
| D5 | Code repository | ★ a) private GitHub repo, data and outputs outside the repo, `.gitignore` for all data paths · b) cluster-only git | Backup and history; nothing patient-level is ever committed |
| D6 | One codebase for both versions? | ★ a) yes, with a `version: A / B` config switch · b) two separate repos | 80% of the pipeline is shared; one switch keeps them comparable |

### Data and protocol (shared by A and B)

| # | Decision | Options | Why ★ |
|---|---|---|---|
| D7 | Primary label | ★ a) ICU mortality at 24 h (YAIB) · b) in-hospital mortality | Matches YAIB numbers; most stable across databases |
| D8 | AKI | ★ a) supported by the code from day one, run after mortality works · b) mortality only | AKI needs causal blocks; designing for it later means rewriting adapters |
| D9 | Sample unit | ★ a) first ICU stay per patient · b) all stays | Removes patient overlap between train and test |
| D10 | Splits | ★ a) dev 2008–2016 → 60/15/10/15 (train/val/meta/test) + temporal 2017–2019 · b) random split over all years | Meta split keeps routers honest; temporal holdout is the only shift test for Version A's images |
| D11 | Repeats | ★ a) 3 split seeds × 3 training seeds; NAS on split seed 0 only · b) 5 × 5 | Affordable and enough to spot ties |
| D12 | Missingness masks | ★ a) on, with an on/off ablation for every expert · b) on only | The ablation tests the "charting habits" hypothesis |
| D13 | Metric used to select models (val) | ★ a) AUROC · b) AUPRC | AUROC is less noisy for selection; all metrics are still reported |
| D14 | Calibration | ★ a) temperature scaling fitted on meta, never refitted on test sets · b) none | Keeps the external test honest |
| D15 | Primary comparisons (pre-registered, Holm-corrected) | ★ a) NAS vs AES, AES vs best expert, NAS vs best expert, on mortality24, internal + external test · b) choose later | Must be fixed before the first test run, or the statistics lose meaning |

### Experts and baselines

| # | Decision | Options | Why ★ |
|---|---|---|---|
| D16 | Expert pool | ★ a) LightGBM, GRU-D, TCN, Transformer, Mamba now; Raindrop added once the others work · b) all six from the start | Raindrop is the slowest to port; the others come from YAIB or standard code |
| D17 | AES variants | ★ a) static best, KNORA-E, soft router, hard router (+ oracle as upper bound) · b) soft router only | Cheap to run all; the comparison is part of the course project |
| D18 | Model soup architecture | ★ a) Transformer, pretrained once on MIMIC train with masked-value reconstruction · b) Mamba · c) GRU | Transformers have the cleanest pretraining recipe; soups need a shared starting checkpoint |
| D19 | Course project scope | ★ a) the shared structured core only (experts, ensembles, soup, AES on MIMIC-IV) · b) include Version A or B extras | Finishes the course on work both versions reuse |

### Version A specifics

| # | Decision | Options | Why ★ |
|---|---|---|---|
| D20 | Image/signal content | ★ a) CXR images (frontal only) + ECG waveforms; no reports, no ECG notes · b) add radiology reports | Reports are written after the image and can leak the outcome |
| D21 | ECG time margin before hour 24 | ★ a) 1 h, with a 3 h sensitivity run · b) none · c) 3 h | ECG clocks are not synchronised with MIMIC-IV |
| D22 | Which ECG/CXR per stay | ★ a) the latest one before the cut-off · b) earliest · c) all of them as tokens | Latest is closest to the prediction time; "all" can come later |
| D23 | Pretrained encoders | ★ a) only encoders not trained on MIMIC (ECGFounder, BiomedCLIP), frozen, plus one CNN per modality trained by us · b) also RAD-DINO / ECG-FM, with disclosure | MIMIC-pretrained encoders have seen your test patients |
| D24 | eICU in Version A | ★ a) yes, as a structured-only test with ECG/CXR missing · b) no eICU | Free extra test of graceful degradation |

### Version B specifics

| # | Decision | Options | Why ★ |
|---|---|---|---|
| D25 | Training setup | ★ a) pooled MIMIC-IV + HiRID + AUMCdb → eICU, plus leave-one-database-out · b) MIMIC-IV only → test on the other three | Pooled training is what gives the router real domains to specialise on |
| D26 | If AUMCdb access is late | ★ a) start with MIMIC-IV + HiRID, add AUMCdb when it arrives · b) wait | Don't block on a separate application |
| D27 | Static features | ★ a) 5-unit bins in all sources · b) exact values where available | Stops models from recognising the source by its static format |
| D28 | Source experts | ★ a) yes, one per training source using the best pooled architecture · b) architecture experts only | Gives AES its "most similar hospital" choice |

### NAS (both versions)

| # | Decision | Options | Why ★ |
|---|---|---|---|
| D29 | Search method | ★ a) regularised evolution + random-search baseline with the same budget · b) DARTS / one-shot | Candidates are cheap; the random baseline is what makes the result defensible |
| D30 | Budget per version | ★ a) 300 low-fidelity candidates + top-5 retrained with 3 seeds · b) 500 + top-10 | Fits a few GPU-days; can scale up later (Part C) |
| D31 | Search ablations built in from the start | ★ a) inherit vs scratch weights, uniform vs AES-guided prior · b) add later | Both are core thesis questions and need logging from the first candidate |
| D32 | Cost logging | ★ a) GPU-hours, models trained, params, FLOPs, latency, memory for every run from day one · b) only for final models | Cost is half of the thesis question; it can't be reconstructed afterwards |

---

## Part C: decisions that wait for results (not asked now)

| Decision | What it waits for |
|---|---|
| Lock in A, B or both | Coverage audit (how many stays have ECG/CXR in the first 24 h), access timing, how long the shared core takes |
| Keep Raindrop | Whether porting works within ~1 week |
| Keep AKI | Whether mortality24 is done on time |
| Fusion-point options and modality windows in Version A | Coverage audit |
| Fine-tune encoders vs frozen only | Whether frozen embeddings already help |
| Search-space size, budget increase | How long one candidate trains on your GPUs |
| λ in the Version B selection rule | Pre-set at 0.5; only revisited if the ablation shows it matters |
| GNN surrogate | Whether evolution alone fits the budget |
| AES-guided prior as the main method | Result of the D31 ablation |

**Suggested first milestone, independent of the A/B choice:** build the shared structured core on the demo data,
reproduce YAIB's LightGBM and GRU numbers, and run the ECG/CXR coverage audit. Those three results answer most of
Part C.

---

## Part D: storage and compute plan for your constraints

Checked: PhysioNet's MIMIC cloud docs (mimic.mit.edu), the PhysioNet DUA v1.5.0, GCS vs R2 pricing (vantage.sh),
GCE VM prices (gcloud-compute.com). Dataset sizes are from memory and marked **(verify)**.

### D.1 The idea

Keep **raw data in PhysioNet's own cloud copies**, do the one-time heavy preprocessing on a **short-lived cloud
VM**, and keep only **small processed files** in cheap object storage. The cluster's free GPUs do all training and
pull the small files into each job.

```
PhysioNet cloud copies (MIMIC-IV, eICU, MIMIC-CXR-JPG, ...)  ── free to read inside Google Cloud
        │
        ▼
Cloud VM, a few hours, once per dataset version (R + ricu + YAIB-cohorts, ECG/CXR selection and resizing)
        │   writes only compact outputs
        ▼
Object storage bucket  (processed data ≈ 10–30 GB, plus checkpoints and logs)
        │   each SLURM job downloads what it needs to node-local scratch
        ▼
University cluster GPUs (8 h / 8 GB jobs, checkpoint → bucket → resume)
```

### D.2 Recommended services (cheapest that works)

| Need | Recommendation | Price (checked) | Why |
|---|---|---|---|
| Raw data | **PhysioNet's Google Cloud copies.** Link your Google account in your PhysioNet profile, then request cloud access on each dataset page | Free to read; PhysioNet pays for downloads from its buckets (mimic.mit.edu) | Nothing raw needs to be stored by you. MIMIC-IV and MIMIC-IV-CXR are on GCS/BigQuery per the docs. **Check on each dataset page (logged in) whether eICU, MIMIC-CXR-JPG, MIMIC-IV-ECG and HiRID have cloud options;** if not, the VM downloads them from PhysioNet over HTTPS (MIMIC-IV-ECG is open access, 34 GB zip) |
| Preprocessing machine | **Google Compute Engine e2-highmem-8 (8 vCPU, 64 GB RAM), Spot**, in the same region as the data | $0.22/h in us-central1 spot ($0.36 on demand); $0.11/h spot in europe-north2 | ricu needs far more than 8 GB of RAM **(verify how much)**, and R is guaranteed there. A full extraction of all databases should take hours, so roughly $5–20 in total |
| Processed data, checkpoints, logs | **Cloudflare R2** | $0.015/GB-month, first 10 GB free, **no egress fees** | Every SLURM job downloads its data again. With Google Cloud Storage that costs $0.12/GB of egress; 300 NAS jobs × 2 GB would be ≈ $70. With R2 it is free |
| Free credits | Google Cloud's free trial credit for new accounts, and ask your university about cloud research credits **(verify current offers)** | — | Likely covers the whole VM cost |

Expected total for the project: **well under $50**, almost all of it VM hours. This is an estimate; set a billing
alert at $20 on both accounts.

**First, ask your cluster admins** for a small project storage allocation (20–50 GB) or confirm that jobs get
node-local scratch (`$TMPDIR`). Free cluster storage beats any cloud option, and local scratch is needed anyway for
the per-job download.

### D.3 What gets stored (processed artefacts, rough sizes)

| Artefact | Approx. size | Used by |
|---|---|---|
| Harmonised structured cohorts (MIMIC-IV, eICU, HiRID), Parquet | < 1 GB each | everything |
| Split files, normalisation statistics | KB | everything |
| ECG waveforms for cohort stays, float16, up to 4 per stay | ~5–20 GB at 500 Hz, 4× less at 100 Hz **(verify after the audit)** | ECG CNN training |
| CXR frontal images for cohort stays, resized to 224 and 518 px | a few GB | CXR CNN fine-tuning, embeddings |
| Frozen-encoder embeddings (ECG, CXR) | < 1 GB | AES, fusion, NAS (most jobs only need these) |
| Checkpoints, MLflow logs, per-sample predictions | grows; prune to the best checkpoints | resuming, AES, analysis |

Raw MIMIC-CXR-JPG is several hundred GB **(verify)**; the VM downloads only the frontal images of cohort stays, which
is a small fraction.

### D.4 Working within 8 h / 8 GB jobs

- **Checkpoint and resume everything.** Lightning checkpoints every N minutes; the job script catches SLURM's
  pre-timeout signal (`#SBATCH --signal=USR1@300`), saves, uploads to the bucket and resubmits itself. Resuming
  restores model, optimiser, scheduler, random state and the epoch.
- **NAS loop state** (population, scores, the random-search log) is written to the bucket after **every** candidate, so
  a killed job loses at most one candidate. One candidate = one short job; the controller is a tiny job that only
  submits and reads results.
- **8 GB RAM:** data is memory-mapped (NumPy memmap or Zarr), never loaded whole; ECG/CXR are read per batch.
  LightGBM on ~50k stays × a few hundred summary features fits comfortably.
- **Unknown GPU:** mixed precision on, batch size set from available GPU memory at start-up, Mamba with a fallback.
- **No R on the cluster is fine:** R only runs on the cloud VM.

### D.5 Data-use agreement check

The DUA (v1.5.0) says not to share access with anyone else and to keep the data physically and electronically secure;
it says nothing specific about cloud storage. PhysioNet itself distributes MIMIC through Google Cloud. My reading is
that a **private, access-controlled bucket in your own account**, used only by you and your credentialed colleague,
fits the DUA. That is an interpretation, not legal advice: confirm with your supervisor, and check whether your
university has a policy on health data in commercial clouds. Never make a bucket public, never share keys, and never
paste records into an LLM service.
