# Pipeline proposal: Version A (MIMIC-IV + ECG + CXR) vs Version B (four ICU databases)

Written 2026-10-09, expanded 2026-10-27 with §11 (how each model and component works), §12 (statistics) and
§13 (decision log with reasoning). Builds on `dataset_and_models.md`, `mimic_to_eicu_literature.md`,
`mimic_iv_splits_and_dataset_lockin.md` and `fake_datapoint_examples.md`; it does not repeat their content.

Kiril's answers that shape this document:
- **"Heterogeneous"** means a NAS system that connects blocks taken from *different* base models (for example a CNN
  block from the TCN feeding a Transformer block), not heterogeneous graph neural networks.
- **Compute:** university cluster.
- **Version A modalities:** ECG and chest X-ray (CXR), on top of the structured ICU data.
- **Time:** about 3 months, with a compressed plan of a little over 1 month (§9).

What was checked this time: the PhysioNet pages for MIMIC-IV-ECG and Symile-MIMIC, the RAD-DINO model card, the
ECG-FM paper (PMC), and the abstracts of FuseMoE, DeRy, STAR, ReStNet, "Revisiting Model Stitching in the
Foundation Model Era", Traft 2026, "Fusion or Confusion?", MedFuse and AutoFM. Everything else is from memory and
marked **(verify)** where a number or detail matters.

---

## 0. The short answer

1. **Both versions share 80% of the pipeline.** Same structured core (ricu/YAIB, 52 features, first stay,
   train/val/meta/test + temporal holdout), same six structured experts, same block library, same NAS loop.
   Version A adds two encoders (ECG, CXR) and a fusion stage. Version B adds three databases and source experts.
2. **The key engineering object is one shared tensor format (the "token bus", §3)** that every block reads and
   writes: `tokens [B, L, d]` + `mask [B, L]` + `time [B, L]` + `type [B, L]`. Each block from a base model is
   wrapped with a small adapter that converts to and from its native format (CNN: channels-first, Transformer/
   Mamba/GRU: sequence-first, image CNN: 2-D grid, graph: per-variable nodes). With that, "CNN block → Transformer
   block" becomes a transpose, a norm swap and a positional encoding, and the NAS can treat blocks as swappable.
3. **The idea is not new in general, so position it carefully.** Deep Model Reassembly (DeRy, NeurIPS 2022) already
   cuts heterogeneous pretrained models (CNNs and Transformers) into blocks and reassembles them under a budget.
   Model stitching across CNN and Transformer families works (ReStNet 2025; Traft 2026; Mai et al. 2026). STAR (2024)
   evolves hybrid attention/conv/recurrent architectures from a genome. What none of them does: build the library
   from a **zoo trained on the same clinical task**, use **AES specialisation as the search prior**, and measure
   **synthesis vs AES under real distribution shift with cost accounting**. That is the thesis contribution.
4. **Version A is the better fit for "heterogeneous" blocks; Version B is the better fit for "generalisation".**
   - In A, families really differ by input (1-D ECG CNN, 2-D image CNN/ViT, sequence models, fusion Transformer),
     so cross-family connection is the interesting part. But the multimodal result can only be tested inside
     MIMIC-IV (internal + 2017–2019 temporal), never on another hospital.
   - In B, every block reads the same hourly sequence, so adapters are trivial, but every architecture the NAS
     produces gets a real external test on eICU and leave-one-database-out.
5. **My recommendation:** if you have ~3 months and want multimodality, do **A as the main study with eICU as a
   structured-only test of the EHR branch** (missing-modality evaluation). If you are on the 1-month plan, do **B**:
   less new engineering (ricu does the harmonisation), cheaper candidates, and a stronger generalisation claim.
   Do not try to do both fully.

---

## 1. Shared core (both versions)

Unchanged from `mimic_iv_splits_and_dataset_lockin.md` §7:

| Item | Value |
|---|---|
| Structured source | MIMIC-IV v2.2 via ricu 0.6.x + YAIB-cohorts + YAIB preprocessing |
| Cohort | YAIB base filters + task filters + first ICU stay per patient |
| Features | 4 static + 48 hourly dynamic, + 48 missingness masks (96 channels) |
| Tasks | mortality24 (primary, one prediction at hour 24), AKI (secondary, hourly, causal) |
| Splits | development 2008–2016 → train 60 / val 15 / meta 10 / test 15; temporal holdout 2017–2019 |
| Repeats | 3 split seeds; ≥ 3 training seeds per model; NAS on split seed 0 |
| Metrics | AUROC, AUPRC, Brier, ECE; paired patient-level bootstrap; GPU-hours and models trained per method |

**Structured expert pool (both versions):** LightGBM, GRU-D, TCN, Transformer, Mamba, Raindrop.

### 1.1 Preprocessing, step by step (structured part)

Why each step exists is in the decision log (§13, rows 1–9). The short version: every statistic is learned on
MIMIC train only, so anything evaluated later (temporal, eICU, HiRID, AUMCdb) is a true unseen test.

| # | Step | Output |
|---|---|---|
| 1 | ricu concept extraction per database (item IDs → concepts, unit conversion, plausibility limits) | long table: stay, time, concept, value |
| 2 | YAIB cohort scripts: filters, labels (`death_icu`, KDIGO AKI) | cohort + `outc` |
| 3 | First stay per patient; attach `subject_id`, estimated stay year; write split file | `subject_id → split` |
| 4 | Hourly grid, hours 0–24 (mortality) or 0–168 (AKI); aggregate within the hour | `[T, 48]` with NaN |
| 5 | Missingness mask `m` and time-since-last-observation `δ` (needed by GRU-D) | `[T, 48]` each |
| 6 | Forward fill, then train-mean fill for never observed | dense `[T, 48]` |
| 7 | z-score with train-split statistics (pooled train splits in Version B) | normalised `[T, 48]` |
| 8 | Static: age, sex, height, weight, z-scored (binned identically in Version B, §5.1) | `[4]` |
| 9 | Per-model views (§3.2): flattened statistics for LightGBM, `(x, m, δ)` for GRU-D, `[T, 96]` for others | model-ready tensors |

Every fitted statistic in steps 6–8 comes from train only and is frozen for val, meta, test, temporal and any
external database.

---

## 2. The two versions at a glance

| | **Version A: MIMIC-IV multimodal** | **Version B: four ICU databases** |
|---|---|---|
| Training data | MIMIC-IV v2.2 structured + MIMIC-IV-ECG v1.0 + MIMIC-CXR-JPG | MIMIC-IV v2.2 + HiRID v1.1.1 + AUMCdb v1.0.2 (structured) |
| Test data | MIMIC internal test + temporal holdout; eICU for the EHR branch only (ECG/CXR missing) | eICU (external); leave-one-database-out over all four |
| Experts | 6 structured + 2 ECG + 2 CXR + 2–3 fusion models | 6 architecture experts (pooled) + source experts (one architecture per source) |
| AES type | Modality-availability-aware router (masks experts whose modality is missing) | Domain-aware router (uses "which source does this stay resemble") |
| NAS type | Cross-family block NAS with a fusion stage (where to fuse, which encoder, which post-fusion blocks) | Cross-family block NAS over sequence blocks with a domain-generalisation objective |
| Why "heterogeneous" is interesting | Blocks differ in input format (1-D signal, 2-D image, sequence, graph) | Blocks share one format; heterogeneity is mostly in the operator |
| Generalisation claim | Temporal only for the full multimodal model | Cross-hospital, cross-country |
| New engineering vs the locked spec | ECG/CXR linking and timing, two encoders, fusion | Two more access requests (AUMCdb is separate), pooled preprocessing |
| Risk | Low modality coverage in the first 24 h; ECG clock offsets; contamination of pretrained encoders | Source shortcuts (binned statics, sampling density); small eICU gain from pooling |

---

## 3. How data changes between blocks (the full process)

This is the part that makes cross-family NAS possible. The rule is: **blocks never talk to each other in their
native formats. They talk through one shared format, and each block has an adapter in and an adapter out.**

### 3.1 The token bus

```
tokens : [B, L, d]   float   one row per token (an hour, an ECG segment, an image patch, a static vector)
mask   : [B, L]      bool    True = real token, False = padding or missing modality
time   : [B, L]      float   hours since ICU admission (ECG/CXR get their own acquisition time)
type   : [B, L]      int     0 = static, 1 = EHR hour, 2 = ECG, 3 = CXR  (Version B: type also = source id, optional)
```

`d` is fixed per candidate architecture (e.g. 64 or 128). Any block whose native width differs gets a linear
projection in its adapter.

**Why one shared format.** Without it, every pair of families needs its own converter (6 families → 30 directed
pairs). With a bus, each family needs one adapter in and one out (12 adapters), and the NAS can place any block in any
legal slot. The `time` field replaces position indices, so tokens that are not hourly (an ECG at 3.5 h) still have a
meaningful place in the sequence. The `mask` field carries missingness and missing modalities through every block, so
the same network runs on MIMIC with images and on eICU without them. `type` lets attention tell modalities apart.

### 3.2 Native format of each block family

| Family (source expert) | Native input | Native output | What the block assumes |
|---|---|---|---|
| GBDT (LightGBM) | `[B, F]` flat statistics (min/max/mean/last/count per feature) | `[B]` probability | No time axis. **Not a block**: AES/ensembles only |
| GRU-D decay stem | `x, m, δ : [B, T, V]` raw values, mask, time gaps | `[B, T, h]` | Needs raw mask and δ, so it is only valid as the **first** block |
| GRU / LSTM layer | `[B, T, d]` time-major sequence + lengths | `[B, T, d]` | Causal by construction |
| TCN / Conv1d block | `[B, C, T]` **channels-first** | `[B, C, T]` | Local receptive field; position is implicit in convolution; BatchNorm or weight norm; causal padding for AKI |
| Transformer encoder block | `[B, T, d]` + key-padding mask (+ causal mask for AKI) | `[B, T, d]` | Needs **explicit** position/time encoding; LayerNorm; permutation-invariant otherwise |
| Mamba block | `[B, T, d]` | `[B, T, d]` | Causal scan; no padding mask, so padding must be at the end and zeroed |
| Raindrop sensor-graph layer | Per-variable nodes `[B, T, V, d_v]` + learned graph over V | `[B, T, V, d_v]` | Needs a variable axis; messages only when a sensor is observed |
| ECG 1-D CNN (xresnet1d / Inception1D) | `[B, 12, 5000]` (12 leads, 10 s at 500 Hz) | `[B, C, T']` with T' ≈ 5000 / stride | Channels-first, very high frequency |
| ECG Transformer (ECG-FM style, wav2vec 2.0) | 5-s segments `[B, 12, 2500]` → conv feature extractor → `[B, T', 768]` | `[B, T', 768]` | Already a CNN stem + Transformer: a natural cross-family example |
| CXR CNN (DenseNet-121 / ResNet) | `[B, 3, H, W]` (e.g. 224×224) | feature map `[B, C, h, w]` (e.g. 1024 × 7 × 7) | 2-D grid, BatchNorm |
| CXR ViT (RAD-DINO, BiomedCLIP) | `[B, 3, H, W]` → patches → `[B, N, 768]` (RAD-DINO: 37 × 37 patches at 518 px) | `[B, N, 768]` + CLS | Patch tokens + 2-D position embedding |
| Fusion Transformer / MoE | token bus | token bus | Modality-type embeddings, missing-modality mask |
| Head | `[B, d]` pooled (last token, mean, CLS or attention pooling) | `[B]` or `[B, T]` | — |

### 3.3 Adapters: what changes when data moves between families

| From → To | What the adapter does | Why |
|---|---|---|
| **CNN (TCN) → Transformer** | `x.transpose(1, 2)` (`[B, C, T] → [B, T, C]`); linear `C → d`; swap BatchNorm statistics for LayerNorm; **add time encoding** (sinusoidal over `time`, or learned); pass key-padding mask; add causal mask if the task is AKI | Conv encodes position implicitly through its kernel, attention does not. Without a time encoding the Transformer sees a bag of hours |
| **Transformer → CNN (TCN)** | `transpose(1, 2)`; zero out padded positions (conv has no mask); left-pad for causality | Conv mixes neighbours, so padding values leak into real tokens unless zeroed |
| **GRU-D stem → anything** | The stem consumes `(x, m, δ)` and emits a dense `[B, T, h]`; after it the mask is all True for observed hours | Missingness is now inside the hidden state. Ablate this: it is the expert most likely to learn MIMIC charting habits |
| **Sequence → Raindrop graph layer** | Split `d` into `V` variable slots: `[B, T, d] → [B, T, V, d_v]` via a per-variable linear (needs the raw observation mask `m` to know which nodes fired); after the layer, pool over V and project back to `d` | Graph layers need a variable axis that sequence blocks have mixed away. In practice put the graph layer **first** (right after embedding, where variables are still separable) or keep a per-variable residual stream |
| **Sequence → Mamba** | Ensure padding is right-aligned and zeroed; no attention mask exists | Mamba scans everything; masked tokens must not carry signal |
| **Bidirectional → causal** (AKI) | Every block in an AKI candidate must be causal: causal attention mask, causal conv padding, unidirectional GRU, Mamba as-is. The search space simply forbids bidirectional variants for AKI | Otherwise hour t sees hour t+1 and the label leaks |
| **ECG 1-D CNN → token bus** | `[B, 12, 5000]` → stem (stride ~32) → `[B, C, ~157]` → transpose → `[B, 157, C]` → either keep k pooled tokens (e.g. adaptive avg pool to 8) or one CLS-like token → linear to d; all ECG tokens get `time = t_ecg` (hours since ICU admission) and `type = 2` | ECG runs at 500 Hz, EHR at 1/hour. Never align at sample level; an ECG is an *event* in the EHR timeline |
| **CXR CNN → token bus** | Feature map `[B, C, h, w]` → `flatten(2).transpose(1, 2)` → `[B, h·w, C]` (49 tokens for 7×7) → linear to d → add 2-D position embedding (only needed if later blocks are attention) → optionally pool to 1–4 tokens; `time = t_cxr`, `type = 3` | This is the BoTNet / hybrid-ViT conversion: a CNN grid becomes a token sequence |
| **CXR ViT → token bus** | Use CLS (`[B, 768]`) or a few pooled patch tokens; linear to d | Keeps the bus short; 1,369 patch tokens would dominate attention cost |
| **Missing modality** | No ECG/CXR in the window → the modality's tokens are absent and `mask = False`; optionally insert one learnable "missing" token (FuseMoE style) | Lets the same network run on MIMIC with and without images, and on eICU (structured only) |
| **Static features** | `[B, 4]` → linear → one token with `type = 0`, `time = 0`; or FiLM conditioning of each block | Same bus, no special path |
| **Token bus → head** | mortality24: pool (last EHR token at hour 24, or attention pooling over all real tokens); AKI: per-hour EHR tokens → `[B, T]` | Head never sees padding |
| **Width mismatch with inherited weights** | Stitching layer: linear `d_A → d_B`, initialised by least squares on activations of a training batch, then fine-tuned (Bansal et al. 2021; Csiszárik et al. 2021 **(verify)**) | Lets a block reuse its trained weights instead of training from scratch |

### 3.4 Worked example: one mortality24 stay through "TCN block → Transformer block → head"

```
x (normalised values) [1, 24, 48], m (masks) [1, 24, 48], static [1, 4]      # from preprocessing step 9

embed     : cat(x, m) [1, 24, 96] → Linear(96→64) → tokens [1, 24, 64], time = 0..23, type = 1
static    : Linear(4→64) → [1, 1, 64], prepend → tokens [1, 25, 64], mask all True

adapter_in(TCN) : tokens.transpose(1, 2) → [1, 64, 25]; zero padded positions
TCN block       : dilated causal Conv1d ×2 + residual → [1, 64, 25]          # inherited from the trained TCN expert
adapter_out     : transpose → [1, 25, 64]; LayerNorm

adapter_in(Transformer) : tokens + TimeEnc(time) + TypeEmb(type); key_padding_mask = ~mask
Transformer block       : MHSA + FFN → [1, 25, 64]                            # inherited from the trained Transformer expert
                          (stitching Linear(64→64) if the two experts' widths/spaces differ)

head : token at hour 23 → Linear(64→1) → sigmoid → p(death in ICU)
```

Version A adds, before the TCN block or before the Transformer block (that choice is a NAS decision, §4.3):

```
ECG [1, 12, 5000] → ECG-CNN stem → [1, 256, 157] → AdaptiveAvgPool1d(4) → transpose → [1, 4, 256] → Linear(256→64)
     time = t_ecg (e.g. 3.5 h), type = 2
CXR [1, 3, 224, 224] → DenseNet-121 → [1, 1024, 7, 7] → global pool → [1, 1, 1024] → Linear(1024→64)
     time = t_cxr (e.g. 11.2 h), type = 3
tokens = cat(static, EHR hours, ECG tokens, CXR token) → [1, 30, 64]
```

Note: concatenating ECG/CXR tokens into the sequence works for attention and Mamba (if ordered by time), but **not**
for TCN (a conv kernel would mix an ECG token with neighbouring hours as if they were adjacent time steps). So in
the search space, TCN/GRU blocks act on EHR tokens only, and fusion happens at an attention/Mamba/fusion block. That
is exactly the kind of compatibility rule the architecture graph (§4.2) encodes.

---

## 4. The cross-family block NAS (both versions)

### 4.1 Component library (built from the trained zoo)

After Stage 4 (individual experts trained), each expert is cut into blocks with their trained weights:

| Block | From | Format | Valid positions |
|---|---|---|---|
| `decay_stem` | GRU-D | `(x, m, δ) → SEQ` | first only |
| `linear_stem` | Transformer/Mamba/TCN | `[x; m] → SEQ` | first only |
| `gru_layer` | GRU-D | SEQ → SEQ | anywhere, EHR tokens only |
| `tcn_block` | TCN | CHF ↔ SEQ via adapter | anywhere, EHR tokens only |
| `attn_block` | Transformer | SEQ → SEQ | anywhere; can fuse modalities |
| `mamba_block` | Mamba | SEQ → SEQ | anywhere; can fuse if tokens are time-ordered |
| `sensor_graph` | Raindrop | VARGRAPH → VARGRAPH | first or second (needs variable axis) |
| `ffn` | Transformer | SEQ → SEQ | anywhere |
| `film_static` | — | static → modulation | any block |
| `ecg_encoder`, `cxr_encoder` | Version A experts | signal/image → tokens | modality stems |
| `fusion_attn`, `fusion_moe` | Version A fusion experts | bus → bus | after modality stems |
| `head_{last, mean, attnpool}` | — | SEQ → VEC | last |

Each block also has a **weight source** gene: `inherit` (from the expert) or `scratch`. Comparing the two answers
your Stage-7/8 question "can components actually be reused independently?" directly.

### 4.2 Architecture representation: typed graph (the "heterogeneous graph" of architectures)

Represent each candidate as a DAG whose **node types** are block families and whose **edge types** are tensor
formats (SEQ, CHF, VARGRAPH, GRID, VEC). An edge is legal only if an adapter exists for it (§3.3), and position
rules (§4.1) are type constraints. This gives you:
- **Validity for free**: mutation operators only produce legal graphs.
- **A natural surrogate model**: a GNN over typed nodes/edges that predicts validation AUROC from the graph. Use a
  relational GNN (R-GCN, Schlichtkrull et al. 2018) or a graph-based NAS encoder (GATES, Ning et al. ECCV 2020;
  BRP-NAS, Dudziak et al. NeurIPS 2020; arch2vec, Yan et al. NeurIPS 2020 **(verify venues)**). GHN-2 (Knyazev et al.,
  NeurIPS 2021, arXiv 2110.13100) shows such graph encoders work for arbitrary computational graphs.
- **A clean thesis figure**: the same graph representation describes every expert in the zoo and every synthesised
  candidate, so "which components came from where" is visible.

Terminology warning for your literature search: "heterogeneous graph NAS" in the literature usually means
*searching heterogeneous GNN architectures* (DiffMG, KDD 2021; HGNAS, 2021). That is a different topic. Use
"cross-architecture model stitching", "model reassembly", "hybrid architecture search" instead.

### 4.3 Search space (macro skeleton)

```
Version B / structured branch:
  stem ∈ {decay_stem, linear_stem, sensor_graph+linear}
  → slot_1 … slot_K   (K ∈ {2, 3, 4}), each ∈ {gru, tcn, attn, mamba, ffn, identity} × {inherit, scratch}
  → head ∈ {last, mean, attnpool}
  + global genes: d ∈ {64, 128}, FiLM static on/off, missingness mask on/off

Version A adds:
  ecg_encoder ∈ {none, xresnet1d (scratch), ECGFounder (frozen)}, ecg_tokens ∈ {1, 4, 8}
  cxr_encoder ∈ {none, DenseNet-121 (fine-tuned), BiomedCLIP (frozen)}, cxr_tokens ∈ {1, 4}
  fusion_point ∈ {early (after stem), mid (after slot_1), late (after slot_K), output-level (late fusion of heads)}
  fusion_block ∈ {attn, mamba, fusion_moe, concat+MLP}
```

Roughly 10^4–10^5 candidates in B and 10^6 in A. Far too many to train exhaustively, which is what the search is for.

### 4.4 Search strategy (recommended for a bachelor's scale)

How each piece works (evolution, proxies, surrogate, stitching) is explained in §11.12–§11.15; how candidates are
compared statistically is in §12.4–§12.6 and §12.9.

1. **Prefilter** with zero-cost proxies (Abdelfattah et al., ICLR 2021, arXiv 2101.08134; Mellor et al. NASWOT, ICML
   2021) to drop clearly bad candidates. Report how well the proxy ranks candidates; on ICU data it may fail, and that
   is a result.
2. **Regularised evolution** (Real et al., AAAI 2019) with mutations that respect the typed graph: swap a block's
   family, toggle inherit/scratch, insert/delete a slot, move the fusion point. Multi-objective selection (NSGA-II
   style) on validation AUROC and parameter count/latency, since cost is part of your question.
3. **Low-fidelity evaluation**: train each candidate for a few epochs from inherited weights; full training only
   for the top 5 (with 3 seeds, as in the lock-in doc §5).
4. **Surrogate**: after ~200 evaluated candidates, train the typed-graph GNN predictor (§4.2) and use it to rank
   offspring before training them.
5. **AES-guided prior (the novel ablation)**: bias mutation probabilities toward block families whose expert won
   most often in the AES router on the meta split, and toward families that win on *different* patient subgroups.
   Compare against a uniform prior with the **same budget**. This tests "does expert specialisation predict useful
   components?".
6. **Mandatory baseline: random search with the same number of trained candidates** (Li & Talwalkar, UAI 2019). If
   evolution does not beat it, say so.

Why not DARTS-style one-shot (Liu et al., ICLR 2019)? It needs all candidate operations on an edge to share one
format and to be trained together in a supernet. With inherited, heterogeneous blocks and adapters this is awkward,
and DARTS is known to be unstable. Single ICU candidates train in minutes, so evolution with low-fidelity training is
affordable on a cluster. Mention one-shot as an alternative, not the main method. BossNAS (ICCV 2021) and UniNet
(ECCV 2022, arXiv 2207.05420) are the closest one-shot/hybrid CNN–Transformer search references, and UniNet's
finding that the *transition modules* between conv and attention stages matter is directly relevant to §3.3.

### 4.5 Closest prior work (what to cite against)

| Paper | What it does | How your work differs |
|---|---|---|
| **DeRy**, Yang et al., NeurIPS 2022, [arXiv 2210.17409](https://arxiv.org/abs/2210.17409) | Partitions heterogeneous pretrained models into functionally equivalent blocks (cover-set optimisation), reassembles under a resource budget with an integer program and a training-free proxy; 78.6% ImageNet without fine-tuning, 83.2% with | General-purpose vision zoo; yours is a task-specific clinical zoo, with AES as the competing method and shift as the test |
| **Model stitching**, Bansal et al., NeurIPS 2021, [arXiv 2106.07682](https://arxiv.org/abs/2106.07682) | Linear stitching layer between two networks' layers | The stitching-layer tool you use for `inherit` |
| **ReStNet**, Wang et al. 2025, [arXiv 2506.09066](https://arxiv.org/abs/2506.09066) | Stitches two pretrained models (CNN–CNN, Transformer–Transformer, **CNN–Transformer**), stitch point chosen by CKA similarity | Evidence that cross-family stitching works; CKA is a cheap compatibility score you can reuse |
| **Traft 2026**, UniReps, [PMLR 322](https://proceedings.mlr.press/v322/traft26a.html) | Stitches ResNet-50 lower layers onto Swin-Tiny upper layers with nonlinear/bottleneck adapters | Same: adapters make large representational gaps bridgeable |
| **Mai et al. 2026**, [arXiv 2603.12433](https://arxiv.org/abs/2603.12433) | Stitching heterogeneous vision foundation models; plain stitch training often fails at shallow points, a feature-matching loss fixes it | Practical tip: if inherited blocks fail to stitch, add a feature-matching loss |
| **Stitchable Neural Networks**, Pan et al., CVPR 2023, [arXiv 2302.06586](https://arxiv.org/abs/2302.06586) | Stitches models of one family at many points for an accuracy/latency trade-off | Same-family only |
| **STAR**, Thomas et al. 2024, [arXiv 2411.17800](https://arxiv.org/abs/2411.17800) | Evolutionary synthesis of hybrid architectures from a hierarchical genome (LIV systems), multi-objective (quality, size, cache) | Searches operators, not reused trained blocks; language modelling |
| **MAD**, Poli et al., ICML 2024, [arXiv 2403.17844](https://arxiv.org/abs/2403.17844) | Cheap synthetic tasks to design hybrid architectures | Idea: cheap proxies for block quality |
| **Jamba**, BoTNet (already in your reading list) | Hand-designed hybrids | Your NAS should at least rediscover hybrids of this kind |

---

## 5. Version B in detail: MIMIC-IV + HiRID + AUMCdb → eICU

### 5.1 Data and preprocessing (additions to §1.1)

1. Extract all four databases with the same ricu concepts (HiRID v1.1.1, AUMCdb v1.0.2 legacy, eICU v2.0).
2. Resample HiRID (2-min) and AUMCdb (minute-level) vitals to the same hourly grid with the same aggregation.
3. **Bin age, height and weight in 5-unit bins in every source** (HiRID is already binned) so the static token cannot
   identify the source.
4. Split each training source separately (train/val/meta/test); MIMIC-IV keeps its temporal holdout.
5. Normalise and impute with **pooled training-split** statistics only.
6. eICU: first unit stay per `uniquepid`, test-only, never touched before final evaluation.
7. Leave-one-database-out (LODO): repeat with each of the four as the held-out one.

### 5.2 Base models

- **Architecture experts (pooled training):** LightGBM, GRU-D, TCN, Transformer, Mamba, Raindrop.
- **Source experts:** one architecture (pick the best pooled one, likely LightGBM or GRU) trained separately on
  each source. These give the router a "which hospital is this patient like" choice.

### 5.3 AES for Version B

How the router computes its output and how it is trained: §11.10. How its behaviour is analysed: §12.7.

| Option | Router input | Router | Reference |
|---|---|---|---|
| B-1 Static selection | — | best expert on pooled val | baseline |
| B-2 Dynamic classifier selection | input meta-features (missingness rate, sampling density, severity stats) | kNN competence region (KNORA) or META-DES | Ko et al. 2008; Cruz et al., *Pattern Recognition* 2015, *Information Fusion* 2018 **(verify)** |
| B-3 Learned gating (recommended) | meta-features + each expert's probability and entropy + a **source-classifier's probabilities** | small MLP, softmax over experts, top-1 or soft | Shazeer et al. 2017 (gating); GMoE, Li et al., ICLR 2023, [arXiv 2206.04046](https://arxiv.org/abs/2206.04046) (MoE for domain generalisation); DAEL, Zhou et al., IEEE TIP 2021 **(verify)** (domain-expert ensembles) |

Fit the router on the **meta** splits of the training sources; never on eICU. Report the router's choice distribution
on eICU vs the training sources (lock-in doc §6).

### 5.4 NAS for Version B

- Search space: §4.3 structured branch only. Adapters are mostly transposes because everything is SEQ.
- **Objective:** domain-generalisation aware. Score = mean val AUROC over the three training sources minus λ × their
  standard deviation (or worst-source AUROC). Cross-check with leave-one-training-source-out. Gulrajani & Lopez-Paz
  (ICLR 2021) show model selection decides DG results, so fix this rule before the search. NAS-OoD (Bai et al.,
  ICCV 2021 **(verify)**) is the reference for searching architectures for out-of-distribution performance.
- Final test: the top architecture on eICU and on LODO, compared against AES with the same expert zoo.

### 5.5 What Version B can conclude

"Does a synthesised architecture from reused blocks transfer to unseen hospitals better than routing among the
experts it was built from, and at what extra cost?" That is your original Section 4 question, answered with a real
external test.

---

## 6. Version A in detail: MIMIC-IV + MIMIC-IV-ECG + MIMIC-CXR

### 6.1 Extra data sources

| Source | What it is | Access | Linking |
|---|---|---|---|
| **MIMIC-IV-ECG v1.0** ([PhysioNet](https://physionet.org/content/mimic-iv-ecg/1.0/)) | ~800k diagnostic 12-lead ECGs, ~160k patients, 10 s at 500 Hz, WFDB; ~55% overlap a hospital admission | **Open access** (cardiologist notes removed), 90 GB uncompressed | `subject_id` + `ecg_time`. **ECG machine clocks were not synchronised; times can be significantly off from MIMIC-IV times** |
| **MIMIC-CXR-JPG v2.x** (PhysioNet, credentialed) **(verify version)** | ~377k CXR images as JPG, with view position and study date/time metadata | Credentialed (same PhysioNet account) | `subject_id` + `StudyDate`/`StudyTime` |
| Existing linked set: **Symile-MIMIC** ([PhysioNet](https://www.physionet.org/content/symile-mimic/)) | 11,622 admissions with ECG and labs within 24 h of admission and CXR 24–72 h after; only admissions with all three | Credentialed | Useful as a reference for the linking code, **not** as your cohort: CXR window is after 24 h, so it leaks for mortality24 |

### 6.2 Preprocessing additions

1. Start from the §1 cohort (mortality24 / AKI, first ICU stay).
2. **ECG window:** ECGs with `ecg_time` in [hospital admission, ICU admission + 24 h − margin]. Use a margin (e.g.
   1–2 h) because of unsynchronised clocks. Keep the latest ECG in the window (and optionally the earliest). For AKI,
   an ECG is only visible to hours after `ecg_time`.
3. **CXR window:** frontal views (PA/AP) with study time in the same window. Images only. **Do not use radiology
   reports** unless you can prove the report was signed before hour 24; the report time is after the image time.
4. Record coverage: fraction of stays with ECG, with CXR, with both, per split. **Coverage is unknown and must be
   measured first** (it decides whether A is viable). If fewer than roughly a third of stays have either modality
   in the window, the multimodal gain will be hard to see with your test-set size (lock-in doc §5).
5. **ECG signal:** remove baseline wander (high-pass ~0.5 Hz), optional notch filter, z-score per lead (train
   statistics) or per recording, keep 500 Hz or downsample to 100 Hz for 1-D CNNs (PTB-XL benchmark practice,
   Strodthoff et al., IEEE JBHI 2021 **(verify)**); 5-s segments for ECG-FM-style models.
6. **CXR image:** resize (224 for DenseNet; 518 shorter side for RAD-DINO), convert to 3 channels, normalise with
   the encoder's own statistics. No augmentation at test time.
7. **Precompute embeddings** for frozen encoders once and store them. This keeps each NAS candidate as cheap as a
   structured-only model.
8. Splits: same `subject_id → split` file. MIMIC-CXR and MIMIC-IV-ECG patients inherit their split from it.

### 6.3 Base models

How the ECG, CXR and fusion models work: §11.7–§11.9.

| Role | Model | Pretrained on MIMIC? | Use |
|---|---|---|---|
| Structured experts | LightGBM, GRU-D, TCN, Transformer, Mamba, Raindrop | — | as before |
| ECG expert 1 | xresnet1d101 / Inception1D (PTB-XL benchmark family), trained from scratch on your train split | No | clean expert, CNN block source |
| ECG expert 2 | ECGFounder (Li et al., NEJM AI 2025, [arXiv 2410.04133](https://arxiv.org/abs/2410.04133); Harvard-Emory ECG database, >10M ECGs) **(verify architecture and weights licence)** | No (verify) | frozen embedding |
| ECG (avoid or disclose) | ECG-FM (McKeen et al., [arXiv 2408.05178](https://arxiv.org/abs/2408.05178); wav2vec 2.0, 90.9M params, 5-s 500 Hz input) | **Yes: pretrained on MIMIC-IV-ECG + PhysioNet 2021** | Saw your test patients' ECGs (unlabelled) in pretraining. Use only with explicit disclosure |
| CXR expert 1 | DenseNet-121 from ImageNet weights, fine-tuned on your train split | No | clean expert, CNN block source |
| CXR expert 2 | BiomedCLIP vision encoder (PMC-15M figures) **(verify that MIMIC-CXR is not included)** | No (verify) | frozen embedding, ViT block source |
| CXR (avoid or disclose) | RAD-DINO (ViT-B, 86.6M; trained on 882k images **incl. 369k from MIMIC-CXR**), CheXzero, TorchXRayVision "mimic" weights | **Yes** | Same contamination problem |
| Fusion experts | Late fusion (average or stack the unimodal experts); MedFuse-style LSTM fusion (Hayat et al., MLHC 2022, [arXiv 2207.07027](https://arxiv.org/abs/2207.07027)); a small token-bus Transformer fusion | — | AES pool + fusion block source |

Fusion references for the related-work section: MedFuse (EHR + CXR on MIMIC-IV, robust to missing CXR); HAIM
(Soenksen et al., npj Digit. Med. 2022, [arXiv 2202.12998](https://arxiv.org/abs/2202.12998)); DrFuse (Yao et al.,
AAAI 2024, [arXiv 2403.06197](https://arxiv.org/abs/2403.06197), EHR + CXR with missing modality); MUSE (Wu et al.,
ICLR 2024, missing modalities and labels, bipartite patient–modality graph); HetMed (Kim et al., AAAI 2023,
[arXiv 2211.15158](https://arxiv.org/abs/2211.15158)); **FuseMoE** (Han et al., NeurIPS 2024,
[arXiv 2402.03226](https://arxiv.org/abs/2402.03226)): MIMIC-IV vitals/labs + notes + CXR + ECG, per-modality
routers, learnable missing-modality embedding, Laplace gating; 48-h in-hospital mortality AUROC 81.0 (vitals + notes)
rising to ~83.9 with CXR and ECG. That last paper is the closest existing system to Version A's AES and should be
your fusion baseline.

**Critical caution:** "Fusion or Confusion?" (Rheude, Eils, Wild, [arXiv 2512.22991](https://arxiv.org/abs/2512.22991),
preprint) re-implemented 19 multimodal methods on nine datasets and found complex fusion often did **not** beat
unimodal or simple baselines under fair tuning. Always report the structured-only model and plain late fusion.

### 6.4 AES for Version A

Router mechanics: §11.10 (the availability mask is the `s_k = −∞` line). Analysis: §12.7. Check the subgroup
sample size first (§12.11).

| Option | Description | Reference |
|---|---|---|
| A-1 Static | best expert / best fusion model on val | baseline |
| A-2 **Availability-masked router (recommended)** | Router input: modality-availability vector + structured meta-features + each available expert's probability/entropy. Softmax only over experts whose modality is present (mask the rest). Fitted on the meta split | Flex-MoE (Yun et al., NeurIPS 2024, [arXiv 2410.08245](https://arxiv.org/abs/2410.08245)) for arbitrary modality combinations; FuseMoE for per-modality routers |
| A-3 Two-level router | First choose which modalities to trust, then which architecture within the modality | Inference; ablation only |

The interesting AES question in A: does the router learn to use ECG/CXR experts only for the patients where they help
(e.g. cardiac or respiratory admissions)? Check this with the router's choices per admission type.

### 6.5 NAS for Version A

- Search space: §4.3 full (structured branch + modality encoders + fusion point + fusion block).
- Multimodal NAS references: MFAS (Pérez-Rúa et al., CVPR 2019, fusion-architecture search), BM-NAS (Yin et al.,
  AAAI 2022, [arXiv 2104.09379](https://arxiv.org/abs/2104.09379), bilevel search of features and fusion operations),
  AutoFM ([arXiv 2401.11252](https://arxiv.org/abs/2401.11252), NAS over encoders and fusion strategies for
  multimodal EHR; preprint, details to verify).
- Objective: val AUROC on all stays (including those with missing modalities), with a secondary report on the
  "has ECG/CXR" subgroup. Without that, the search can win on the minority with images and lose on everyone else.
- Cost control: frozen encoders with precomputed embeddings; fine-tuned image/ECG encoders only for the final top-5.
- Generalisation test: internal test, temporal holdout (multimodal), and **eICU with all ECG/CXR tokens absent**
  (does the synthesised network degrade gracefully when modalities vanish?).

---

## 7. Cost estimate (rough, inference, check after the first runs)

| Item | Version A | Version B |
|---|---|---|
| One structured candidate (mortality24, low fidelity) | minutes on one GPU | minutes, ~2–3× data |
| Encoder fine-tuning | DenseNet-121 on ~tens of thousands of CXRs: a few GPU-hours per run; 1-D ECG CNN: about an hour | none |
| Embedding precompute | one pass per frozen encoder | none |
| Expert zoo (all experts, 3 seeds) | ~1–2 GPU-days | ~1 GPU-day |
| NAS (300–500 low-fidelity candidates + top-5 × 3 seeds) | ~1–3 GPU-days with precomputed embeddings | ~1–2 GPU-days |
| Data engineering | ECG/CXR linking, timing, coverage audit: the largest risk | ricu extraction of three more databases; AUMCdb access delay |

These are inferences from typical model sizes, not measurements. Log GPU-hours from the first run (lock-in doc §5).

---

## 8. Things that could break each version

| Risk | Version | Mitigation |
|---|---|---|
| Few stays have ECG/CXR in the first 24 h | A | Measure coverage before anything else; fall back to "ECG within 24 h" only, or to B |
| ECG clock offsets put ECGs inside the window that were really after hour 24 | A | Margin before the prediction time; sensitivity analysis with a larger margin |
| Encoders pretrained on MIMIC-CXR / MIMIC-IV-ECG | A | Prefer non-MIMIC encoders; disclose otherwise |
| Radiology reports leak the outcome | A | Images only |
| Multimodal gain is within noise | A | Paired bootstrap; report the "has modality" subgroup separately |
| AUMCdb access does not arrive in time | B | Run MIMIC-IV + HiRID → eICU; add AUMCdb later |
| Source shortcuts | B | Identical binning and grid; "which source" probe on learned features |
| Inherited blocks fail to stitch (worse than scratch) | both | That is a finding for "can components be reused"; try feature-matching loss (Mai et al. 2026) before concluding |
| NAS winner is selection noise | both | Top-5 retrain with 3 seeds; random-search baseline with equal budget |

---

## 9. Plans for your timeline

### 9.1 ~3 months (recommended: Version A, with eICU as structured-only test)

| Weeks | Work |
|---|---|
| 1–2 | ricu/YAIB pipeline on demo data, then full MIMIC-IV; reproduce YAIB LightGBM/GRU; **ECG/CXR coverage audit** |
| 3–4 | Structured experts (6) with per-sample prediction logs; ECG and CXR linking, embeddings |
| 5–6 | ECG and CXR experts, fusion experts; ensembles, soups (one architecture), AES (course project done here) |
| 7–8 | Block library + adapters + typed graph; random-search baseline |
| 9–10 | Evolutionary search (uniform vs AES-guided prior); top-5 retrain |
| 11–12 | Final tests (internal, temporal, eICU structured-only), cost tables, ablations (inherit vs scratch, masks on/off), writing |

### 9.2 ~1 month compressed (recommended: Version B, or Version B without AUMCdb)

| Week | Work |
|---|---|
| 1 | Pipeline + MIMIC-IV, HiRID, eICU extraction; reproduce YAIB baselines; drop Raindrop if porting is slow |
| 2 | Five experts (LightGBM, GRU-D, TCN, Transformer, Mamba) + source experts; ensembles and AES (B-3) |
| 3 | Block library (SEQ bus only), evolution with ~200 candidates + random search with the same budget |
| 4 | eICU + LODO evaluation, cost table, writing |

What to cut first if time runs out: Raindrop, the GNN surrogate, AKI (keep mortality24 only), and the soup
experiment. Do not cut the random-search baseline or the paired statistics: they are what makes the result
defensible.

---

## 10. Questions still worth asking your supervisor/TA

1. Is multimodality required for the thesis, or was it a preference? If not required, B gives the stronger
   generalisation claim on the shorter timeline.
2. Is a temporal holdout acceptable as the only shift test for the multimodal model (Version A)?
3. Is using an encoder pretrained on MIMIC-CXR/MIMIC-IV-ECG acceptable if disclosed?

---

## 11. How each model and component works

Each entry says what the model computes, why it is in the pool (which inductive bias it adds), and what to watch
for on this data. Equations are the standard formulations from the original papers, written from memory; check
notation against the papers before copying into the thesis.

### 11.1 LightGBM (gradient-boosted decision trees)

- **What it computes.** An additive model `F(x) = Σ_m η · tree_m(x)`. Each new tree is fitted to the negative
  gradient of the log loss of the current model (for binary labels: `y − σ(F(x))`), so every tree corrects the
  remaining errors of the trees before it. `η` is the learning rate.
- **LightGBM specifics.** Trees grow leaf-wise (always split the leaf with the largest loss reduction), feature
  values are bucketed into histograms (fast), and missing values are handled natively: each split learns a default
  direction for NaN.
- **Input here.** No time axis. Per feature over hours 0–24: min, max, mean, last value, standard deviation and
  number of measurements (YAIB uses similar summaries **(verify exact list)**).
- **Why it is in the pool.** It is the strongest baseline on tabular ICU data (YAIB 87.7 AUROC on mortality24) and
  it is robust to scaling. If AES or NAS cannot beat it, that is a result.
- **Watch for.** The native NaN handling and the count features let it learn *how often* something was measured,
  which is a hospital habit, not physiology (Yamamoto et al. 2026). Run it with and without count features.

### 11.2 GRU-D (recurrent network with missingness-aware decay)

A plain GRU updates a hidden state `h_t` every hour:

```
z_t = σ(W_z x_t + U_z h_{t−1} + b_z)                 update gate: how much to overwrite
r_t = σ(W_r x_t + U_r h_{t−1} + b_r)                 reset gate: how much past to use for the candidate
h̃_t = tanh(W x_t + U (r_t ⊙ h_{t−1}) + b)            candidate state
h_t = (1 − z_t) ⊙ h_{t−1} + z_t ⊙ h̃_t
```

GRU-D (Che et al., Sci. Rep. 2018) adds two decays driven by `δ_t` (hours since each variable was last measured)
and feeds the mask `m_t` into the gates:

```
γ_x = exp(−max(0, W_γx δ_t + b_γx))                  per-variable input decay (W_γx diagonal)
x̂_t = m_t ⊙ x_t + (1 − m_t) ⊙ (γ_x ⊙ x_last + (1 − γ_x) ⊙ x̄)
γ_h = exp(−max(0, W_γh δ_t + b_γh))                  hidden-state decay
h_{t−1} ← γ_h ⊙ h_{t−1}
```

- **Intuition.** A missing value is first replaced by the last observed value, and that guess slowly fades towards
  the training mean `x̄` the longer the variable goes unmeasured. The network learns how fast each variable becomes
  stale (a heart rate goes stale in an hour, a creatinine in a day).
- **Why it is in the pool.** It is the only expert that models missingness explicitly. That makes it a good probe for
  the shift question: it should look strong on MIMIC and lose more on eICU if it is learning charting habits.
- **Block use.** The decay part needs raw `x, m, δ`, so it can only be the first block (§4.1).

### 11.3 TCN (temporal convolutional network)

- **What it computes.** Stacked residual blocks of **causal dilated 1-D convolutions**. One filter with kernel size
  `k` and dilation `d` computes `y_t = Σ_{i=0}^{k−1} w_i · x_{t − d·i}`: it only looks backwards, skipping `d − 1` hours
  between taps. Dilations double per level (1, 2, 4, 8), so the receptive field grows exponentially:
  `RF = 1 + 2(k − 1) Σ_l d_l` with two convolutions per block. With `k = 3` and dilations 1, 2, 4, 8, RF = 61 hours,
  which covers the 24-hour window.
- Each block: conv → weight norm → ReLU → dropout, twice, plus a residual connection (a 1×1 conv when channel counts
  differ).
- **Why it is in the pool.** Local pattern detector ("lactate rising over 3 hours while MAP falls"), fast, stable to
  train, and a clean block with a fixed input/output shape (Bai et al. 2018).
- **Watch for.** Convolution assumes equally spaced, adjacent inputs. That is why ECG/CXR tokens must not be inserted
  into a TCN's sequence (§3.4).

### 11.4 Transformer encoder

- **What it computes.** Every hour is a token. Self-attention lets each token build its new representation as a
  weighted mix of all other tokens:
  `Attention(Q, K, V) = softmax(Q Kᵀ / √d_k + M) V` with `Q = X W_Q, K = X W_K, V = X W_V`. `M` is `−∞` on padded
  tokens and, for AKI, on future tokens (causal mask). Multi-head attention runs `h` of these in parallel on smaller
  subspaces and concatenates them. Each layer is attention + feed-forward network (two linear layers with GELU), each
  wrapped in LayerNorm and a residual connection.
- **Position.** Attention is permutation-invariant, so time must be added explicitly, e.g. sinusoidal
  `PE(t, 2i) = sin(t / 10000^{2i/d})`, `PE(t, 2i+1) = cos(…)`, computed from real hours (so it also works for ECG/CXR
  tokens at fractional times).
- **Cost.** `O(T² d)` per layer; irrelevant at T = 24 to 168.
- **Why it is in the pool.** Direct long-range interactions between any two hours, and the natural place to fuse
  modalities (any token can attend to an ECG token).
- **Watch for.** Data-hungry; tends to overfit single-centre patterns.

### 11.5 Mamba (selective state-space model)

- **What it computes.** A state-space model keeps a hidden state `h` and updates it linearly:
  continuous form `h'(t) = A h(t) + B x(t)`, `y(t) = C h(t)`. Discretised with step size `Δ`:
  `Ā = exp(ΔA)`, `B̄ ≈ ΔB`, then `h_t = Ā h_{t−1} + B̄ x_t`, `y_t = C h_t`.
- **"Selective".** In Mamba (Gu & Dao 2023), `B`, `C` and `Δ` are computed from the current input `x_t`. A large `Δ`
  means "reset and focus on this input", a small `Δ` means "ignore this input and keep the memory". This gives
  content-dependent memory like a gated RNN, but training is parallel (a scan), so cost is linear in T.
- **Block.** Linear expand → short causal depthwise conv → SiLU → selective SSM, multiplied by a SiLU gate branch →
  linear projection back to `d`, with a residual connection.
- **Why it is in the pool.** A sequence operator that is neither attention nor a classic RNN; it links directly to
  the hybrid literature (Jamba, STAR) that your thesis builds on.
- **Watch for.** At 24 hourly steps its efficiency advantage is irrelevant; say so. It needs CUDA kernels
  (`mamba-ssm`), which the cluster should have.

### 11.6 Raindrop (graph neural network over sensors)

- **What it computes** (Zhang et al., ICLR 2022; details from memory, verify against the paper). Each variable
  (sensor) is a node. When sensor `i` is observed at time `t`, its value is embedded and **passed as a message** along
  learned directed edges to the other sensors, weighted by attention, so unobserved sensors get an estimate from the
  ones that fired. The graph has a structure shared across patients plus per-patient edge weights. Each sensor's
  sequence of embeddings is then summarised with temporal attention, and all sensors are pooled into one vector.
- **Why it is in the pool.** The only expert that models relations **between variables** explicitly (e.g. creatinine
  and urine output). None of the sequence blocks has that operator.
- **Watch for.** Its code was written for PhysioNet 2012/2019, so porting to the YAIB cohort takes time. It is the
  first thing to drop on the 1-month plan.

### 11.7 ECG encoders

- **xresnet1d / Inception1D (1-D CNNs).** The 12 leads are input channels; time is the length axis. Residual blocks of
  Conv1d → BatchNorm → ReLU, with strided convolutions that halve the length step by step, then average + max pooling
  over time. Inception1D runs parallel filters of different lengths (e.g. 39, 19, 9 samples) so that short and long
  waves are both caught. At 500 Hz a QRS complex lasts about 40–50 samples, and a full beat about 400–500. These are
  the strongest simple baselines on PTB-XL (Strodthoff et al., IEEE JBHI 2021 **(verify)**).
- **wav2vec 2.0-style Transformer (ECG-FM).** A CNN feature encoder turns the raw signal into frames
  (`[B, 12, 2500] → [B, T', 768]`), a Transformer contextualises them. Pretraining masks spans of frames and asks the
  model to pick the true frame representation among distractors (contrastive loss). ECG-FM adds a patient-level
  contrastive objective and random lead masking. It is itself a CNN → Transformer hybrid, which makes it a good
  illustration for the thesis.
- **ECGFounder.** Trained with labels on >10M ECGs from the Harvard-Emory database; verify its architecture before
  committing (§6.3).

### 11.8 CXR encoders

- **DenseNet-121.** Four dense blocks; inside a block every layer receives the concatenation of all earlier feature
  maps (growth rate 32), which reuses features and keeps parameters low (~8M). Transition layers (1×1 conv + 2×2
  average pooling) halve the resolution. At 224×224 input the output is `[1024, 7, 7]`. It is the standard CXR
  backbone since CheXNet.
- **ViT / DINOv2 (RAD-DINO).** The image is cut into 14×14-pixel patches, each patch becomes a token, a CLS token and
  2-D position embeddings are added, and a Transformer processes them. DINOv2 is self-supervised: a student network
  must match the output of a slowly updated teacher (exponential moving average of the student) on different crops of
  the same image, plus a masked-patch objective. No labels are used.
- **CLIP-style (BiomedCLIP).** An image encoder and a text encoder are trained so that matching image–caption pairs
  have high cosine similarity and non-matching pairs low (InfoNCE loss over a batch). The image encoder alone gives
  general-purpose embeddings.

### 11.9 Fusion models

- **Late fusion.** Each modality's expert outputs a probability; combine by averaging or by a logistic regression on
  the probabilities (stacking), fitted on the meta split. Missing modality → drop its term. Simple and strong.
- **MedFuse (LSTM fusion).** Each modality's encoder gives one vector; the vectors are fed as a short sequence
  `[EHR, CXR]` into an LSTM, whose last state is classified. A missing CXR simply makes the sequence shorter, which is
  why it handles partial pairing naturally.
- **Token-bus Transformer fusion.** All modality tokens share one sequence (§3.4); attention decides how much each
  hour attends to the ECG or CXR token. Missing modalities are masked.
- **FuseMoE.** Per-modality routers send each token to the top-k of several expert FFNs. Laplace gating scores experts
  by `−‖W_e − x‖₂` (distance) instead of a dot product, so experts are not favoured for having large weights. A
  learned "missing" embedding stands in for absent modalities.

### 11.10 The AES router

Given experts `f_1 … f_K` that each output a probability `p_k(x)`, and a router `g` that sees features `φ(x)`:

```
s = MLP(φ(x))                         one score per expert
s_k = −∞ if expert k is unavailable   (Version A: its modality is missing)
w = softmax(s)                        routing weights
soft AES:  p(x) = Σ_k w_k p_k(x)
hard AES:  p(x) = p_{argmax_k w_k}(x)
```

- `φ(x)`: cheap meta-features (age, missingness rate, measurement density, a few severity summaries), each expert's
  probability and its entropy, the modality-availability vector (A) or a source classifier's probabilities (B).
- **Training (on the meta split only).** Soft AES: minimise the log loss of the combined prediction. Hard AES: label
  each meta sample with the expert that had the lowest per-sample log loss and train the router as a classifier.
- **Recommendation.** Soft routing as the main method (smooth, better calibrated, easier to train); hard routing as
  the interpretable variant for the specialisation analysis (§12.7).
- **Upper bound (oracle).** Pick, per sample, the expert that is right with the most confidence, using the label. That
  is not achievable, but the gap between the oracle and the best single expert tells you how much room any router
  has. If the gap is tiny, AES cannot help, and you know it before training a router.
- **Classical alternative (KNORA-E).** For a test sample, find its k nearest neighbours in the meta split and keep only
  the experts that were correct on all of them; reduce k until at least one expert remains. No training, a useful
  baseline.

### 11.11 Model soup (one architecture only)

Fine-tune `n` copies from one shared pretrained checkpoint. **Uniform soup:** `θ = (1/n) Σ θ_i`. **Greedy soup:**
sort ingredients by meta-split score, add each one to the average only if the meta score improves. It works only when
all copies stay in one low-loss basin, which requires the shared starting point (Wortsman et al. 2022; see
`dataset_and_models.md` §3.10).

### 11.12 Stitching layers and compatibility (CKA)

When block A (from expert 1) feeds block B (from expert 2), B expects inputs that look like expert 2's own
activations at that point. A stitching layer `W` maps one to the other. Because all experts see the **same stays and
hours**, you can collect paired activations on a training batch: `Z_A` (output of A in expert 1, `n × d_A`) and `Z_B`
(input of B in expert 2, `n × d_B`). Initialise with ridge regression

```
W* = (Z_Aᵀ Z_A + λ I)⁻¹ Z_Aᵀ Z_B
```

then fine-tune. To score which cut points are compatible before training, use **linear CKA** on centred activations
`X, Y`: `CKA(X, Y) = ‖Yᵀ X‖_F² / (‖Xᵀ X‖_F ‖Yᵀ Y‖_F)`, between 0 and 1. High CKA between A's output and B's expected
input means the stitch is easy (ReStNet uses CKA this way). This gives you a measurable answer to "can module
compatibility be predicted?": correlate CKA with the accuracy of the stitched model.

### 11.13 Regularised evolution and multi-objective selection

```
population ← P random legal architectures (e.g. P = 50), each trained (low fidelity) and scored
repeat until budget spent:
    sample S architectures from the population (e.g. S = 10)
    parent ← best of the sample
    child  ← mutate(parent)        swap a block family, toggle inherit/scratch, add/remove a slot, move fusion point
    train and score child; add it; remove the OLDEST member
```

Removing the oldest instead of the worst (Real et al. 2019) stops a lucky early candidate from staying forever, which
matters when scores are noisy. For two objectives (AUROC up, parameters or latency down), rank by Pareto front
(non-dominated sorting) and break ties by crowding distance, as in NSGA-II (Deb et al. 2002).

### 11.14 Zero-cost proxies

Scores computed on one mini-batch at initialisation, without training:
- **NASWOT** (Mellor et al. 2021): record which ReLUs fire for each input in the batch as a binary code `c_i`; build
  `K[i, j] = N_units − Hamming(c_i, c_j)`; score `log|det K|`. Networks that separate inputs well at initialisation
  tend to train well.
- **synflow, grad_norm** (Abdelfattah et al. 2021): saliency of all weights, or the norm of the gradient.
- **Caveat here.** NASWOT is defined for ReLU networks; attention, SiLU (Mamba) and GRU gates do not fit it cleanly,
  and inherited weights are not "at initialisation". Treat proxies as a filter you validate (Kendall τ against trained
  scores, §12.6), not as truth.

### 11.15 Surrogate: GNN over the typed architecture graph

Each architecture is a graph whose nodes are blocks (type = family, plus features such as width and inherit/scratch)
and whose edges are typed by tensor format. A relational GCN (Schlichtkrull et al. 2018) has separate weights per edge
type:

```
h_v^{(l+1)} = ReLU( W_0 h_v^{(l)} + Σ_r Σ_{u ∈ N_r(v)} (1 / |N_r(v)|) W_r h_u^{(l)} )
```

After 2–3 layers, pool the nodes into one vector and predict validation AUROC with a small MLP. Train it with a
**pairwise ranking loss** (is A better than B?), because the search only needs the order, not the exact value. This is
where "heterogeneous graph" appears legitimately in your method: a GNN over a graph with several node and edge types.

---

## 12. Statistics used, and why

### 12.1 Performance metrics

| Metric | Definition | Why it is used | Caveat |
|---|---|---|---|
| **AUROC** | Probability that a random positive gets a higher score than a random negative (= Mann–Whitney U / (n₊ n₋)) | Threshold-free, comparable across cohorts, the number every ICU benchmark reports | Insensitive to prevalence, so it can hide poor performance on the rare class |
| **AUPRC** (average precision) | Area under precision–recall; a random model scores the **prevalence** | Focuses on the positives (≈ 8–12% deaths) | Depends on prevalence, which differs between MIMIC and eICU. Always print prevalence next to it; compare models within one cohort, not across cohorts |
| **Brier score** | `(1/n) Σ (p_i − y_i)²` | Proper scoring rule: rewards both discrimination and calibration | Hard to read alone; report the scaled version `1 − Brier / Brier_prevalence` too |
| **ECE** | Bin predictions (10–15 equal-count bins); `Σ_b (n_b / n) |mean(y)_b − mean(p)_b|` | Simple calibration summary, reported by most ML papers | Depends on binning and is biased in small samples; pair it with calibration slope |
| **Calibration slope and intercept** | Fit `logit P(y=1) = a + b · logit(p)` on the test set | Slope < 1 means over-confident (typical after a shift); intercept ≠ 0 means risk is systematically too high or too low | Needs a few hundred events for stable estimates |
| **Gap** | internal − temporal, internal − eICU (for each metric) | The generalisation result itself | Its CI comes from bootstrapping the two test sets independently and taking the difference in each iteration |

For AKI (hourly predictions), compute metrics over all stay-hours but **resample stays, not hours**, when estimating
uncertainty, because hours of one stay are strongly correlated.

### 12.2 Uncertainty of one number

- **Hanley–McNeil standard error** of AUROC, from AUROC and the number of positives/negatives. Used for the quick
  estimate in the lock-in doc: SE ≈ 0.011 for ~450 deaths in ~6k stays, so a 95% CI of about ±2.1 AUROC points.
- **Bootstrap CI** (what to report in the thesis): resample patients with replacement 1,000–2,000 times, recompute the
  metric, take the 2.5th and 97.5th percentiles. On eICU, resample **hospitals** (cluster bootstrap) and then all their
  stays, because stays within a hospital share practices and are not independent.

### 12.3 Comparing two models on the same test set

Two models evaluated on the same stays have correlated errors, so their difference is much more precise than their
separate CIs suggest:

```
SE(Δ) = sqrt(SE₁² + SE₂² − 2 r SE₁ SE₂)
```

With SE₁ = SE₂ = 0.011 and a typical correlation r ≈ 0.85 between model scores, SE(Δ) ≈ 0.006, so a 95% CI on the
**difference** is about ±1.2 points instead of ±2.1. That is why every comparison in the thesis must be paired:
- **DeLong's test** for the difference between two correlated AUROCs (closed form, fast).
- **Paired bootstrap** for everything else (AUPRC, Brier, ECE): resample patients once per iteration, compute both
  models on the same resample, store the difference; the CI of the differences is the result.

### 12.4 Training and split randomness

- Each model is trained with ≥ 3 seeds, on 3 split seeds. Report mean ± SD.
- With 3 seeds, formal tests on the seed axis have almost no power. Use a simple, pre-stated rule: if two methods
  differ by less than the larger of their seed SDs on validation, call it a tie.
- The final test-set comparison is per split seed with the paired bootstrap, then summarised across split seeds.

### 12.5 Many comparisons

You compare ~10 methods (6 experts, ensemble, soup, AES, NAS winner) on several test sets and two tasks. Without a
correction, some "significant" differences will be chance.
- **Pre-register a small set of primary comparisons**: NAS winner vs AES, AES vs best single expert, NAS winner vs best
  single expert, on mortality24, on internal test and on the external test. Apply **Holm–Bonferroni** to that set.
  Everything else is reported as exploratory.
- Across several test domains (LODO folds, Version B), use the **Wilcoxon signed-rank** test or the **Friedman test
  with Nemenyi post-hoc** (Demšar, JMLR 2006). With only 4 databases these tests have very little power, so present
  them as descriptive ranks rather than proofs.

### 12.6 Ranking agreement (search diagnostics and the generalisation question)

**Kendall's τ** (and Spearman's ρ) between two rankings of the same candidates. Used for:
- proxy score vs trained validation AUROC (is the zero-cost proxy useful?);
- surrogate prediction vs true validation AUROC (is the GNN predictor useful?);
- **MIMIC-validation ranking vs eICU ranking of the same candidates**. This is one of the most informative numbers
  in the thesis: if τ is low, searching on MIMIC does not find architectures that are better on other hospitals,
  which directly answers "does synthesis generalise?".

### 12.7 Analysing AES and expert specialisation

- **Oracle gap:** oracle selector minus best single expert (§11.10). Headroom for any router.
- **Router choice distribution** on MIMIC test vs eICU (or with/without ECG/CXR), compared with the
  **Jensen–Shannon divergence**. A large divergence means the router is reacting to the shift.
- **Router entropy** (mean over samples): near zero means effectively hard selection; near `log K` means it is just
  averaging.
- **Choices per subgroup** (admission type, age band, missingness level) with a chi-square test of independence:
  does the router really specialise, or does it always pick the same expert?
- **Expert diversity:** pairwise disagreement rate and correlation of errors (Kuncheva & Whitaker 2003). Ensembles and
  AES only help when experts make different errors; if all errors are correlated above ~0.9, expect little gain.

### 12.8 Measuring the shift itself

- **Domain classifier AUROC**: train a classifier to tell MIMIC from eICU (or source from source) on the model inputs.
  0.5 means indistinguishable, near 1 means very different. Repeat on learned features of each expert to see which
  architectures encode the source most strongly (the "source shortcut" probe in Version B).
- **Per-feature missingness and mean differences** between databases, as a table. It explains which experts suffer.

### 12.9 Selection optimism

The best of N candidates on validation is biased upward (you picked the luckiest one). Report
`optimism = val score of the selected model − test score`, for the NAS winner and for a hand-picked expert. If the NAS
winner's optimism is much larger, part of its validation advantage was selection noise. The top-5 retrain with new
seeds (§4.4) reduces this.

### 12.10 Cost statistics

| Quantity | How to measure |
|---|---|
| GPU-hours | Wall-clock × number of GPUs, logged per job from the scheduler; summed per method (zoo, AES, soup, NAS) |
| Models trained | Count every trained candidate, including failed and low-fidelity ones |
| Parameters | `sum(p.numel())` |
| FLOPs per prediction | `fvcore` or `thop` on one sample |
| Latency | Median of 100 runs after warm-up, at batch size 1 and 256, same GPU |
| Peak memory | `torch.cuda.max_memory_allocated()` |
| Energy (optional) | CodeCarbon |

Headline cost figure: **Δ metric per extra GPU-hour** of NAS relative to AES, and a Pareto plot of AUROC vs GPU-hours
for all methods. That plot is the visual answer to your thesis question.

### 12.11 Sample-size sanity check before trusting a multimodal result (Version A)

If only a fraction `q` of stays has an ECG or CXR, the "has modality" subgroup in a 6k-stay test set has `6000·q`
stays and roughly `450·q` deaths. With q = 0.3 that is ~135 deaths and an AUROC CI of roughly ±4 points by
Hanley–McNeil. A multimodal gain smaller than that cannot be shown in the subgroup alone. Compute this as soon as the
coverage audit is done.

---

## 13. Decision log: what was chosen, why, and what would change it

| # | Decision | Chosen | Alternatives considered | Why | Revisit if |
|---|---|---|---|---|---|
| 1 | MIMIC-IV version | v2.2 | v3.1 | ricu/YAIB only read v2.2; benchmark numbers refer to it; keeps COVID years out of training | ricu adds v3.x support |
| 2 | Unit of a sample | First ICU stay per patient | All stays, stay-level split | YAIB splits by stay, so one patient could be in train and test; first stay removes that leak and correlated samples | You need more data than first stays give |
| 3 | Primary task | ICU mortality at 24 h | In-hospital mortality, sepsis, ARF | Most stable label across databases; comparable to YAIB; sepsis/ARF labels break across hospitals | TA requires another task |
| 4 | Secondary task | AKI (KDIGO, hourly) | LoS, sepsis | Different structure (hourly, causal) tests blocks differently; label built from labs present in all DBs | Time is short: drop it first |
| 5 | Features | YAIB's 52 | METRE's 86 + interventions, Burger's 141 | Present in all four databases with existing code; makes external tests possible | You want treatments as an extra "modality" |
| 6 | Partitions | train/val/meta/test + temporal | train/val/test | Routers, stackers, soups and calibration fitted on the experts' own validation set are over-optimistic; a separate meta split fixes that | — |
| 7 | Temporal holdout | 2017–2019 | Random split over all years | A second shift for free; the only external-style test for Version A's multimodal model | Too few stays remain for training |
| 8 | Normalisation | Train-only (pooled train in B) | Per-database normalisation | Per-database statistics need test-domain data, which is adaptation, not external validation | You decide to study adaptation |
| 9 | Missingness masks | Included, with an on/off ablation | Values only | Masks help in-domain but may encode charting habits; the ablation turns that risk into a result | — |
| 10 | Expert pool | LightGBM, GRU-D, TCN, Transformer, Mamba, Raindrop | xLSTM, SeFT, mTAND, MOMENT | Each adds a different inductive bias (§11); all fit the same input; all are public | Raindrop porting fails: drop it |
| 11 | Block interface | One token bus `[B, L, d]` + mask/time/type, adapters per family | Require same-family blocks; search raw layers | Lets any family connect to any other with a small, explicit adapter; makes adapters a measurable cost | — |
| 12 | Where modalities fuse | Only at attention, Mamba or fusion blocks | Anywhere | Convolutions and RNNs assume equally spaced steps; inserting an ECG token would corrupt them (§3.4) | — |
| 13 | Architecture representation | Typed graph (node = block family, edge = tensor format) | Flat list of genes | Validity checks for free; enables the GNN surrogate; one picture for experts and candidates | — |
| 14 | Search method | Regularised evolution, low fidelity, top-5 retrain | DARTS/one-shot, RL, Bayesian optimisation | Candidates train in minutes on a cluster; heterogeneous inherited blocks do not fit one supernet; evolution is robust to noisy scores | Candidates become expensive (Version A with fine-tuned encoders) |
| 15 | Baseline for search | Random search, same budget | None | NAS often barely beats random search (Li & Talwalkar 2019); without it no claim is defensible | — |
| 16 | Search prior | Uniform vs AES-guided, compared | Uniform only | Tests the core link between course project and thesis: does specialisation predict useful components? | — |
| 17 | Weight source | `inherit` vs `scratch` per block | Always scratch | Directly tests whether components are reusable | Inherited blocks never stitch: report and simplify |
| 18 | Selection metric (B) | Mean val AUROC over sources − λ·SD (λ = 0.5 default, ablate 0 and 1) | Pooled val AUROC only | Rewards architectures that work on every source, not just the largest | λ ablation shows no effect: use pooled AUROC |
| 19 | Encoders (A) | Non-MIMIC pretrained, frozen, precomputed; plus one fine-tuned CNN per modality | RAD-DINO, ECG-FM, CheXzero | MIMIC-pretrained encoders saw the test patients; frozen embeddings make each NAS candidate cheap | Supervisor accepts disclosed contamination |
| 20 | CXR content | Images only | Images + reports | Reports are signed after the image and can state the outcome; leakage | Report sign times are verifiable |
| 21 | ECG timing | ECGs before hour 24 minus a margin | Exact hour-24 cut-off | ECG clocks were not synchronised with MIMIC-IV | Sensitivity analysis shows no effect |
| 22 | AES (A) | Soft router, masked by modality availability, fitted on meta | Static selection, unmasked router | Experts for a missing modality cannot be used; masking makes that explicit | — |
| 23 | AES (B) | Soft router with source-classifier features | Pure architecture routing | Gives the router a reason to work: routing to the most similar hospital's expert | Oracle gap is tiny |
| 24 | Statics (B) | 5-unit bins in every source | Exact values | HiRID only has binned values; exact values elsewhere would let models identify the source | — |
| 25 | Which version | A for ~3 months, B for ~1 month | Both | A suits cross-family block connection; B suits generalisation and is less engineering. Doing both fully does not fit either timeline | Coverage audit shows A is unviable: B |
| 26 | Statistics | Paired bootstrap / DeLong, Holm on pre-registered comparisons | Unpaired CIs | Unpaired CIs of ±2 points hide real differences of ~1 point; uncorrected multiple tests produce false wins (§12.3, §12.5) | — |

---

## References

Checked this session (page or abstract read):
- MIMIC-IV-ECG v1.0: https://physionet.org/content/mimic-iv-ecg/1.0/
- Symile-MIMIC: https://www.physionet.org/content/symile-mimic/ ; Saporta et al., "Contrasting with Symile", NeurIPS 2024, https://arxiv.org/abs/2411.01053
- RAD-DINO model card: https://huggingface.co/microsoft/rad-dino ; Pérez-García et al., https://arxiv.org/abs/2401.10815
- ECG-FM: McKeen et al., https://arxiv.org/abs/2408.05178 ; https://pmc.ncbi.nlm.nih.gov/articles/PMC12530324 ; https://github.com/bowang-lab/ECG-FM
- FuseMoE: Han et al., NeurIPS 2024, https://arxiv.org/abs/2402.03226
- MedFuse: Hayat, Geras, Shamout, MLHC 2022, https://arxiv.org/abs/2207.07027
- AutoFM: https://arxiv.org/abs/2401.11252
- DeRy: Yang et al., NeurIPS 2022, https://arxiv.org/abs/2210.17409 ; https://github.com/adamdad/dery
- STAR: Thomas et al. 2024, https://arxiv.org/abs/2411.17800
- ReStNet: Wang et al. 2025, https://arxiv.org/abs/2506.09066
- Mai et al., "Revisiting Model Stitching In the Foundation Model Era", 2026, https://arxiv.org/abs/2603.12433
- Traft, "Bridging Large Gaps in Neural Network Representations with Model Stitching", UniReps 2026, https://proceedings.mlr.press/v322/traft26a.html
- Rheude, Eils, Wild, "Fusion or Confusion?", https://arxiv.org/abs/2512.22991

Found in search, not read beyond the title/listing:
- Bansal, Nakkiran, Barak, "Revisiting Model Stitching", NeurIPS 2021, https://arxiv.org/abs/2106.07682
- Pan et al., "Stitchable Neural Networks", CVPR 2023, https://arxiv.org/abs/2302.06586
- Poli et al., "Mechanistic Design and Scaling of Hybrid Architectures", ICML 2024, https://arxiv.org/abs/2403.17844
- Knyazev et al., GHN-2, NeurIPS 2021, https://arxiv.org/abs/2110.13100
- Ning et al., GATES, ECCV 2020, https://arxiv.org/abs/2004.01899
- Liu et al., UniNet, ECCV 2022, https://arxiv.org/abs/2207.05420
- Li et al., BossNAS, ICCV 2021, https://arxiv.org/abs/2103.12424
- Yin et al., BM-NAS, AAAI 2022, https://arxiv.org/abs/2104.09379
- Abdelfattah et al., Zero-Cost Proxies, ICLR 2021, https://arxiv.org/abs/2101.08134
- Li et al., GMoE, ICLR 2023, https://arxiv.org/abs/2206.04046
- Yun et al., Flex-MoE, NeurIPS 2024, https://arxiv.org/abs/2410.08245
- Yao et al., DrFuse, AAAI 2024, https://arxiv.org/abs/2403.06197
- Kim et al., HetMed, AAAI 2023, https://arxiv.org/abs/2211.15158
- Wu et al., MUSE, ICLR 2024, https://proceedings.iclr.cc/paper_files/paper/2024/file/f49d76cf84df83a611883c621c96d2d9-Paper-Conference.pdf
- Soenksen et al., HAIM, npj Digit. Med. 2022, https://arxiv.org/abs/2202.12998
- Li et al., ECGFounder, NEJM AI, https://arxiv.org/abs/2410.04133

From memory (verify before citing): Real et al. 2019 (regularised evolution); Li & Talwalkar 2019 (random search for
NAS); Liu et al. 2019 (DARTS); Mellor et al. 2021 (NASWOT); Pérez-Rúa et al. 2019 (MFAS); Dudziak et al. 2020
(BRP-NAS); Yan et al. 2020 (arch2vec); Schlichtkrull et al. 2018 (R-GCN); Csiszárik et al. 2021 (stitching
similarity); Cruz et al. 2015/2018 (META-DES, dynamic selection review); Ko et al. 2008 (KNORA); Zhou et al. 2021
(DAEL); Bai et al. 2021 (NAS-OoD); Strodthoff et al. 2021 (PTB-XL benchmark); DiffMG (Ding et al., KDD 2021);
Gulrajani & Lopez-Paz 2021; added for §11–§12: Ke et al. 2017 (LightGBM), Che et al. 2018 (GRU-D), Bai et al.
2018 (TCN), Vaswani et al. 2017 (Transformer), Gu & Dao 2023 (Mamba), Huang et al. 2017 (DenseNet), Dosovitskiy et
al. 2021 (ViT), Oquab et al. 2024 (DINOv2), Radford et al. 2021 (CLIP), Baevski et al. 2020 (wav2vec 2.0), Deb et al.
2002 (NSGA-II), Kornblith et al. 2019 (CKA), Hanley & McNeil 1982, DeLong et al. 1988, Holm 1979, Demšar 2006
(JMLR, comparing classifiers over datasets), Kuncheva & Whitaker 2003 (diversity measures). Earlier write-ups cover YAIB, ricu, Burger et al. 2024, Shazeer et al. 2017, Jamba,
BoTNet and Wortsman et al. 2022.
