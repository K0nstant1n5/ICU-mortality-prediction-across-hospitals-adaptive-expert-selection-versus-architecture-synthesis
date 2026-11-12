# MIMIC-IV → eICU: what 2023+ papers do, and what it costs vs. staying on MIMIC-IV

Written 2026-10-01. Builds on `research/dataset_and_models.md` (does not repeat it).
Numbers were read from the papers' web versions through an automated reader; the ones marked **(verify)** come
from a summary and must be checked against the PDF table before they go into the thesis.
Pages I could not read (rate limit / captcha): the full text of Allyn et al. 2026 (Critical Care, only the abstract was read),
the PMC/PubMed pages for Critical Care Medicine 2024, and arXiv 2609.30781 (missingness-aware conformal prediction).

---

## 1. The short answer

1. **Everyone harmonises the same way, at the feature level, before training.** Map each database's raw item IDs
   to shared clinical concepts (`ricu`, METRE, BlendedICU/OMOP, or a hand-picked APACHE-style list), convert units,
   clip outliers with one clinical rule set, resample to an hourly grid, impute with one rule (forward-fill +
   training-set mean), normalise with **MIMIC-only** statistics, and apply identical cohort/label definitions.
   The model never sees anything database-specific. Nobody does "model-level" adaptation as the main fix.
2. **The price of crossing is consistent: mortality loses roughly 4–6 AUROC points and about a third of its AUPRC.**
   YAIB GRU: 87.6 on MIMIC-IV → 81.7 on eICU; AUPRC 42.8 → 28.0. METRE LSTM: 0.869 → 0.824. A model trained *on*
   eICU gets 86.0 there, so most of the gap is shift, not eICU being "harder".
3. **The drop depends on the task far more than on the architecture.** Mortality transfers reasonably; respiratory
   failure, shock, AKI and especially sepsis transfer badly because their **labels** are built from treatments and
   documentation that differ between hospitals (ventilation charting, cultures, urine output).
4. **The features that help most inside MIMIC are often the ones that hurt most on eICU**: measurement counts,
   missingness patterns, variability. They encode MIMIC's charting habits, not physiology (Yamamoto et al. 2026).
   This matters directly for your expert pool (GRU-D, missingness indicators, LightGBM count features).
5. **Recent work moves from "MIMIC → eICU" to "many databases → held-out database".** Pooling sources beats
   any single source for generalisation (YAIB, Burger et al. 2024, Allyn et al. 2026). That is a stronger
   setup than one-to-one transfer, and worth considering for the thesis (§5).

---

## 2. Papers (2023 onward)

### 2.1 Benchmarks and harmonisation pipelines (the ones to build on)

| Paper | Harmonisation | MIMIC-IV in-domain | MIMIC-IV → eICU | Notes |
|---|---|---|---|---|
| **YAIB**, van de Water et al., ICLR 2024 ([arXiv 2306.05109](https://arxiv.org/abs/2306.05109)) | `ricu` concepts; 4 static + 48 dynamic features; hourly grid; forward-fill + train-mean imputation; z-scoring; adults, ≥6h stay, ≥4 measurements, no >12h gaps | GRU mortality 87.6 AUROC / 42.8 AUPRC | **81.7 AUROC / 28.0 AUPRC** (eICU in-domain: 86.0) | Pooling 3 databases "usually as good as the best single-dataset model". Fine-tuning beats scratch when target data is small (shown eICU → HiRID). AUPRC drops more than AUROC. MIMIC-IV and eICU transfer to each other better than other pairs |
| **METRE**, Liao & Voldman, J. Biomed. Inform. 2023 ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1532046423000771), [code](https://github.com/weiliao97/METRE)) | Hand-built concept mapping for exactly MIMIC-IV + eICU: 86 labs/vitals, 16 interventions, 35 static; one outlier rule set; hourly bins; binary missingness indicators; drops stays >90% missing | Mortality LSTM 0.869, CNN 0.876 | Mortality LSTM **0.824**, CNN 0.822; Shock-4h RF 0.880 → 0.782 **(verify)** | Only pipeline built specifically for this pair. Richer feature set than YAIB (interventions). Respiratory failure transferred very unevenly because ventilation is documented differently |
| **Burger et al.**, "Towards Foundation Models for Critical Care Time Series", NeurIPS 2024 workshop ([arXiv 2411.16346](https://arxiv.org/abs/2411.16346)) | Extended `ricu`: 141 concepts incl. 55 treatment variables; 9 databases | LightGBM ~95 AUROC on decompensation/circulatory/kidney | Single-source transfer loses a few points; multi-centre leave-one-out recovers most of it **(verify per-task table)** | LightGBM beat GRU/Transformer/Mamba/xLSTM by 1–2 points in-domain. Transfer is better within a continent (US↔US) than across |
| **BlendedICU**, Oliver et al., J. Biomed. Inform. 2023 ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S153204642300223X), [code](https://github.com/USM-CHU-FGuyon/BlendedICU)) | OMOP-standardised merge of MIMIC-IV, eICU, HiRID, AmsterdamUMCdb | n/a (dataset paper) | n/a | Alternative to `ricu` if you prefer Python/OMOP. Used by Allyn et al. 2026 |

### 2.2 Transfer / generalisation studies

| Paper | What they did | Key result |
|---|---|---|
| **Allyn et al.**, Critical Care 2026 ([Springer](https://link.springer.com/article/10.1186/s13054-026-06034-5)) | Temporal Pointwise Convolution (TPC) model, ~20k patients each from 4 international databases (BlendedICU); single-source vs pooling vs transfer learning; mortality + remaining LoS | Internal validation overestimates external performance; **pooling gave the best generalisation and also the best specialisation**. Only the abstract was read; numbers not extracted |
| **Meegahapola et al.**, npj Digital Medicine 2026 ([Nature](https://www.nature.com/articles/s41746-026-02364-4)) | Sepsis prediction, MIMIC-IV / eICU / HiRID, YAIB features; CNN, InceptionTime, LSTM; compares retraining, fine-tuning, MMD/CORAL domain adaptation, fusion training | Direct MIMIC → eICU deployment degrades substantially. Best strategy depends on target data size: retrain (≤8k stays), MMD/CORAL (8–32k), fusion (≥32k). **Fine-tuning consistently underperformed** |
| **Yamamoto et al.**, medRxiv 2026 ([medRxiv](https://www.medrxiv.org/content/10.64898/2026.04.05.26350209v1.full)) | Sepsis mortality, MIMIC-IV (30k) → eICU (31k), APACHE-III-aligned variables; adds measurement counts and variability features | LR latest values: 0.819 → 0.772. + counts: 0.834 → 0.752. + variability + counts: 0.844 → **0.719**. Better in-domain features ⇒ worse external. XGBoost less affected. Calibration was the most sensitive signal of shift |
| **Londschien, Burger, Rätsch, Bühlmann**, 2025 ([arXiv 2507.21783](https://arxiv.org/abs/2507.21783)) | Anchor regression / anchor boosting (invariance-regularised GBDT) across 9 databases; leave-one-database-out | Gains are small for "core" databases (~1% MSE) and larger for truly different ones. External data is "worth" ~1,500–15,000 local patients; above 10–50k local patients it barely helps |
| **ICareFM**, Burger et al., medRxiv 2025 ([medRxiv](https://www.medrxiv.org/content/10.1101/2025.07.25.25331635v1)) | Self-supervised time-to-event foundation model on 11 databases (650k stays) | Doubling local data ≈ 4× more external data in zero-shot. **Contains eICU in pretraining: never use as an expert for an eICU test** |
| **PULSE-ICU**, 2025 ([arXiv 2511.22199](https://arxiv.org/abs/2511.22199)) | Longformer pretrained on MIMIC-IV, YAIB 52-variable schema for eICU | ~0.65 AUROC zero-shot averaged over external datasets; **0.864 on eICU only after fine-tuning on 100% of eICU** (see correction in §6) |
| **Typical clinical papers**, e.g. [arXiv 2502.17978](https://arxiv.org/abs/2502.17978) (2025, sepsis-AKI mortality, XGBoost) | Same inclusion criteria in both DBs, 24 first-day tabular features, mean/KNN imputation, drop >50% missing | 0.878 internal → **0.720** on eICU. Dozens of such papers exist; they show the drop but rarely explain or fix it. Good as motivation, weak as method references |
| **Jagd, DeVries, Winther**, NeurIPS 2025 TS4H ([arXiv 2509.19885](https://arxiv.org/abs/2509.19885)) | Bi-Axial Transformer pretrained on pooled EHR, transferred to mortality | Helps mainly when target set <5,000 stays. Abstract only |

### 2.3 Not MIMIC→eICU but useful context
- **LLM-text serialisation** (ICLR 2026, [arXiv 2603.23987](https://arxiv.org/abs/2603.23987)): turns ICU series into text summaries, embeds them; smaller relative drop on new hospitals (tested on HiRID/PPICU, not eICU). Shows "representation choice affects transfer", which is your thesis's angle.
- **Implicit bias in measurement frequencies** (BMC Med. Inform. Decis. Mak. 2025, [Springer](https://link.springer.com/article/10.1186/s12911-025-03058-9)): measurement frequency and missingness vary systematically by site and patient group.

---

## 3. How the data is "edited" to work on both, step by step

This is the common pipeline across YAIB, METRE, Burger, PULSE-ICU, Meegahapola. Every step is a place where MIMIC and
eICU differ.

| Step | What is done | MIMIC/eICU difference it removes | Pitfall |
|---|---|---|---|
| 1. Concept mapping | Raw `itemid`s (MIMIC) and free-text labels (eICU `lab`, `nursecharting`, `vitalperiodic`) → shared concept ("heart rate", "creatinine") | Different schemas and naming | eICU labels are strings, sometimes hospital-specific; `ricu` / METRE already solved this, do not redo it |
| 2. Feature intersection | Keep only features present in **both** (YAIB: 48 dynamic + 4 static) | MIMIC has many more items | You throw away MIMIC's richest signals; the MIMIC-only baseline should use the **same** 52 features or the comparison is unfair (§4) |
| 3. Units and plausibility | Convert units; clip/remove physiologically impossible values with one rule set | Unit conventions, entry errors | Must be identical across DBs |
| 4. Time alignment | Time zero = ICU admission; resample to hourly bins (aggregate within bin) | MIMIC charts ~hourly; eICU `vitalperiodic` is 5-min medians from bedside monitors, many labs sparser | Hourly binning hides the frequency difference only partly; counts/masks still reveal it |
| 5. Cohort + labels | Same inclusion (adult, min stay, min measurements) and same label code (mortality, KDIGO AKI, Sepsis-3) | Different documentation of death, dialysis, cultures, antibiotics | Mortality is the most stable label; Sepsis-3 is the least (eICU microbiology sparse) |
| 6. Imputation | Forward-fill, then **MIMIC-train** mean for never-observed; optional missingness mask | — | Fitting imputation or scaling on eICU is leakage from the test domain |
| 7. Normalisation | z-score with **MIMIC-train** statistics | Different value distributions | Same as above |
| 8. (Optional) Domain adaptation | MMD/CORAL, DANN, anchor regression, fine-tuning | Residual shift | Needs eICU data (labelled or not). Any use of eICU beyond final testing changes your claim from "external validation" to "adaptation" |

---

## 4. MIMIC-IV only vs MIMIC-IV → eICU: what changes

| Aspect | Strictly MIMIC-IV | MIMIC-IV → eICU |
|---|---|---|
| Features | Can use everything (hundreds of items, notes, CXR) | Limited to the intersection (~52–140 concepts), structured only |
| Typical mortality AUROC | ~0.87–0.88 (YAIB GRU/LightGBM) | ~0.82 (YAIB, METRE) |
| AUPRC | ~0.43 | ~0.28 (prevalence differs too, so compare AUPRC with care) |
| Calibration | Good | Usually broken; recalibration needs eICU data |
| What it measures | Fit to one hospital's patients and charting | Robustness to population, practice and documentation shift |
| Model ranking | Complex models and engineered "process" features look best | Simpler / physiology-only features lose less; LightGBM remains a strong baseline |
| Effort | Single extraction | Shared pipeline (free via YAIB/`ricu`), stricter leakage discipline |
| Claim you can make | "Works at BIDMC" | "Generalises to 200 other US hospitals" |

Two consequences for the thesis:
- **Report both, always as a pair** (in-domain and external, same 52 features), and report the **gap**, not only the
  external number. The gap is the generalisation result.
- **A MIMIC-only result using all features is a different experiment.** If you want it, call it "upper bound with
  full MIMIC features" and keep it separate; do not compare it directly with the harmonised external number.

---

## 5. Things you did not ask but probably need

1. **Your thesis question has a ready-made hypothesis.** Yamamoto et al. show that in-domain gains from
   observation-process features turn into external losses. Your experts differ exactly on this axis (GRU-D and
   missingness masks exploit it; a values-only Transformer does not). Prediction: **AES on MIMIC validation will
   favour the expert that exploits MIMIC charting habits, and that expert will transfer worst.** Testing that is a
   clean, novel result. Run every expert with and without missingness masks/counts as an ablation.
2. **The router itself is a shift risk.** Per-sample expert selection learned on MIMIC may route on charting
   artefacts. Log the router's choices on eICU and compare their distribution to MIMIC; if they diverge strongly,
   that is a finding, not a bug.
3. **Pick "external validation" or "adaptation" once and keep it.** Strict MIMIC → eICU (no eICU used before
   testing) is the cleaner claim and fits your setup. If you add adaptation, Meegahapola et al. say fine-tuning is
   the weakest choice; MMD/CORAL or retraining are better, depending on eICU size.
4. **Consider multi-source leave-one-out as an extension.** The field has moved there (YAIB pooled, Burger,
   Londschien, Allyn). E.g. train on MIMIC-IV + HiRID + AUMCdb, test on eICU. It gives AES more heterogeneous
   training "domains" to specialise on, which is closer to why expert selection should help at all. Same `ricu` code.
5. **Use eICU's hospital IDs.** Per-hospital AUROC spread and leave-region-out are cheap and rarely reported.
   They tell you whether the drop is uniform or concentrated in small community hospitals.
6. **Report calibration (ECE/Brier) and AUPRC**, not only AUROC. Yamamoto et al. found calibration the most
   sensitive shift indicator; YAIB shows AUPRC drops roughly 3× more than AUROC.
7. **Stick to mortality as primary.** AKI is a reasonable secondary; avoid sepsis and respiratory failure as headline
   tasks (label definitions break across databases, see METRE ARF and Meegahapola).
8. **METRE is an alternative to YAIB** if you want interventions (vasopressors, ventilation) as features, which
   adds a "treatment" modality to strengthen the "multimodal" argument. YAIB's 52 features have no treatments;
   Burger's extended `ricu` concepts do.
9. **Be wary of the many clinical "MIMIC-IV development, eICU validation" papers** (XGBoost on first-day tabular
   features, often small disease subgroups). They are useful for showing the drop is ubiquitous, but they use
   different features and cohorts, so their numbers are not comparable to yours.

---

## 6. Correction to the earlier write-up

`dataset_and_models.md` (§1.2 and §3.4) said PULSE-ICU reached 0.864 AUROC on eICU "as an external test after
MIMIC-IV-only pretraining". The paper reports 0.864 **after fine-tuning on the full eICU training set**; zero-shot
transfer averaged around 0.65 AUROC across external datasets (exact eICU zero-shot value is only in a figure).
It is therefore not a zero-shot external baseline. The YAIB GRU (81.7) is the right zero-shot reference.

---

## Sources

- YAIB: https://arxiv.org/abs/2306.05109 ; https://openreview.net/forum?id=ox2ATRM90I
- METRE: https://www.sciencedirect.com/science/article/pii/S1532046423000771 ; https://github.com/weiliao97/METRE
- Burger et al. 2024: https://arxiv.org/abs/2411.16346
- BlendedICU: https://www.sciencedirect.com/science/article/pii/S153204642300223X ; https://github.com/USM-CHU-FGuyon/BlendedICU
- Allyn et al. 2026: https://link.springer.com/article/10.1186/s13054-026-06034-5
- Meegahapola et al. 2026: https://www.nature.com/articles/s41746-026-02364-4
- Yamamoto et al. 2026: https://www.medrxiv.org/content/10.64898/2026.04.05.26350209v1.full
- Londschien et al. 2025: https://arxiv.org/abs/2507.21783
- ICareFM 2025: https://www.medrxiv.org/content/10.1101/2025.07.25.25331635v1
- PULSE-ICU 2025: https://arxiv.org/abs/2511.22199
- Sepsis-AKI XGBoost example: https://arxiv.org/abs/2502.17978
- Jagd et al. 2025: https://arxiv.org/abs/2509.19885
- LLM text serialisation (ICLR 2026): https://arxiv.org/abs/2603.23987
- Measurement-frequency bias: https://link.springer.com/article/10.1186/s12911-025-03058-9
- Not read (blocked): Critical Care Medicine 2024 "The Impact of Multi-Institution Datasets on the Generalizability of ML Prediction Models in the ICU" (https://pubmed.ncbi.nlm.nih.gov/38958568/); arXiv 2609.30781
