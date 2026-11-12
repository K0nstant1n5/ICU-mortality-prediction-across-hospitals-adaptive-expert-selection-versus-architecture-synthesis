# MIMIC-IV train/validation/test best practices and the locked-in dataset spec

Written 2026-10-01. Builds on `dataset_and_models.md` (dataset choice, expert pool) and
`mimic_to_eicu_literature.md` (harmonisation, transfer results). Neither is repeated here.

What was checked directly, and how:
- **ricu** source (github.com/eth-mds/ricu, v0.6.3, last commit Sep 2025): its `miiv` source points at
  `physionet.org/files/mimiciv/2.2`. eICU points at v2.0.
- **YAIB-cohorts** R scripts (`base_cohort.R`, `mortality.R`, `aki.R`): cohort filters and label definitions below are
  read from the code.
- **YAIB** (`icu_benchmarks/data/split_process_data.py`, `cross_validation.py`): split logic.
- **PhysioNet MIMIC-IV v3.1 page**: years, anchor year groups, counts.
- Could not read: the YAIB paper's full text (arXiv HTML rate-limited). Numbers from it below are marked **(verify)**.

---

## 0. The short answer

1. **Use MIMIC-IV v2.2, not v3.1.** `ricu` (and so YAIB) is built for v2.2 (2008–2019, approx. 73k ICU stays).
   v3.x adds 2020–2022 (COVID era, approx. 94k stays) and fixed lab item IDs, but ricu does not support it.
   v2.2 matches every benchmark number you will cite and keeps COVID out of training.
2. **Split by patient, not by stay. YAIB does not do this for you.** YAIB's base cohort keeps *every* ICU stay of a
   patient, and its splitter groups by `stay_id`. A patient with two stays can be in train and test at once. Keep
   only each patient's **first ICU stay** (simplest, and standard in clinical papers).
3. **You need four MIMIC partitions, not three.** AES, stacking, greedy soups and calibration are all "models fitted
   on model outputs". If they are fitted on the expert's own validation set, you get optimistic routers. So:
   **train / val / meta / test**, plus a **temporal holdout** (2017–2019) and **eICU** as untouched test sets.
4. **Your test set is smaller than it looks.** With ~450 deaths in a ~6k-stay test set, a single AUROC has a 95% CI of
   about ±2 points. Differences between experts of 1 point are not visible without paired tests and repeated seeds.
   This matters most for the architecture search, which will pick a winner by noise if you let it.
5. **Correct one label detail in our earlier notes:** YAIB's mortality task is **ICU mortality**, not in-hospital
   mortality (concept `death_icu`). Keep YAIB's definition; it is what the 87.6 / 81.7 AUROC numbers refer to.

---

## 1. Version and cohort

| Decision | Choice | Why |
|---|---|---|
| MIMIC-IV version | **v2.2** (physionet.org/content/mimiciv/2.2/, still downloadable) | Only version `ricu` 0.6.x supports; same as YAIB; 2008–2019 so no COVID-era shift inside training |
| eICU version | **eICU-CRD v2.0** | What `ricu` supports |
| Extraction | `ricu` 0.6.x + YAIB-cohorts R scripts, pinned via their `renv.lock`; record the commit hashes in the thesis | Reproducibility; cohort numbers change between versions |
| Base cohort (YAIB, from code) | Exclude: invalid LoS, ICU stay < 6h, < 4 measurements, any gap > 12h without measurements, age < 18 | Identical rules in MIMIC-IV and eICU |
| **Added by us** | **First ICU stay per patient** (`subject_id` in MIMIC-IV, `uniquepid` in eICU) | Removes patient overlap across splits and correlated samples. Report how many stays this drops |
| Features | YAIB's 4 static (age, sex, height, weight) + 48 hourly dynamic features | Same in both databases |
| Never add as inputs | `los_icu`, `los_hosp` (present in YAIB's *base* cohort statics, for description only), ICD/DRG codes, discharge location, `hospital_expire_flag`, `dod`, discharge notes, eICU APACHE predictions | All are known only at or after discharge |
| v3.1 2020–2022 | Not in the core spec. Optional later as a "COVID-era" temporal test, only if you port the ricu config yourself | Not needed for the thesis question; risky engineering |

Rough expectation (not checked against data): v2.2 has approx. 73k ICU stays from approx. 50k patients, so first-stay-only
removes roughly a quarter to a third of stays before task filters. Replace this with your `attrition.csv` counts.

## 2. Tasks (as defined in YAIB-cohorts code)

| Task | Label | Prediction | Task-specific exclusions | Unit of evaluation |
|---|---|---|---|---|
| **Primary: mortality24** | Death in the ICU (`death_icu`) | Once, at hour 24, from hours 0–24 | ICU LoS < 30h; death within 30h (so there is a 6h gap between prediction time and any counted death) | One prediction per stay |
| **Secondary: AKI** | KDIGO AKI onset (ricu `aki` concept, creatinine + urine output) | Hourly, up to day 7; positive in a window around onset, roughly 6h ahead (check `outcome_window(6, 6)` in `R/aki.R` before writing it up) | AKI onset < 6h after admission; baseline creatinine > 4 mg/dL; in eICU, hospitals with zero AKI prevalence | Many predictions per stay. Report per-hour and per-stay metrics; CIs must resample stays, not hours |

AKI-specific caution: creatinine and urine output are both inputs and the label source. That is fine for early warning
only because the label is in the future; never shorten the horizon to zero, and don't evaluate hours after onset.

---

## 3. The split, locked

All splits are made **once**, at patient level, saved as a file (`subject_id → split`), and used by every model,
router, soup and NAS candidate. One seed generates it; record the seed.

```
MIMIC-IV v2.2 cohort (first ICU stay per patient, task filters applied)
│
├── Temporal holdout  ── est. stay year 2017–2019 ── final evaluation only
│
└── Development       ── est. stay year 2008–2016, stratified by label
    ├── train  60%   fit experts, NAS candidates
    ├── val    15%   early stopping, hyperparameters, NAS candidate ranking
    ├── meta   10%   AES router, stacking meta-learner, greedy-soup selection, ensemble weights, calibration
    └── test   15%   internal test, final evaluation only

eICU-CRD v2.0 cohort (first unit stay per uniquepid)  ── external test, final evaluation only
MIMIC-III ── separate source for the ablation only; never mixed into any of the above
```

**Estimating the stay year.** MIMIC-IV only gives each patient an `anchor_year_group` (e.g. "2017 - 2019") for their
shifted `anchor_year`. Estimate a stay's real year as `group start + 1 + (shifted year of intime − anchor_year)`. This is
accurate to about ±1.5 years, so call it an approximate temporal split in the thesis. The YAIB output only has `stay_id`;
join `subject_id`, `intime`, `anchor_year`, `anchor_year_group` from `icu/icustays` and `hosp/patients` yourself.

**Why these fractions.** Train needs to be large for the deep experts. Val is reused hundreds of times by the search,
so it should be as large as you can afford. Meta only fits low-capacity models (a router, a few weights, one
temperature), so 10% is enough. Test at 15% gives roughly ±2 AUROC points per model (see §5); that is acceptable only
because comparisons are paired.

**Repeated splits.** Build the split file for **3 seeds**. Run cheap things (all experts, ensembles, AES, soups) on all 3
and report mean ± SD. Run the expensive architecture search on seed 0 only, then retrain its final architecture on
seeds 1–2. Say this explicitly.

**Reproduction first.** Before switching to this protocol, reproduce YAIB's LightGBM/GRU mortality numbers with
YAIB's own protocol (all stays, stay-level 5 outer × 5 inner stratified folds, i.e. 64/16/20 per fold, from the code).
That checks your pipeline. Then expect your numbers under the first-stay, temporal protocol to differ slightly, and
report both.

---

## 4. Leakage rules

| Rule | What it prevents |
|---|---|
| Fit scaler, imputation means, missingness thresholds and any feature selection on **train only**; apply the frozen transform to val, meta, test, temporal and eICU | Test-domain leakage. YAIB fits its preprocessing on the training fold; keep that when you change splits |
| Train the AES router and stacker on **meta-split predictions** of experts that never saw meta | Routers trained on in-sample or early-stopping-set predictions learn which expert overfit, not which expert is right |
| Greedy soup selection and ensemble weights on **meta** | Same reason |
| Calibration (temperature/Platt) on **meta**, then report ECE on test, temporal and eICU **without** recalibrating on them | Keeps "external validation" honest. Recalibrating on eICU is adaptation; label it as such if you add it |
| No resampling, class balancing or filtering of val, meta or test | Changes prevalence and AUPRC |
| Pre-register the list of compared methods and metrics before the first test run, and log every evaluation on test/temporal/eICU | Stops quiet selection on test. If you look and change something, say so |
| No model trained or pretrained on eICU (ICareFM, multi-centre foundation models) | Contaminates the external test |
| Lab timestamps: ricu uses `charttime` (MIMIC) and `labresultoffset` (eICU), i.e. draw time, not result time | A lab drawn at 23:30 and resulted at 01:00 is visible at hour 24. Small and symmetric across databases; name it as a limitation rather than fix it |
| If you later add treatment features (Burger/METRE concepts), only use infusion/procedure records starting before the prediction time | Treatment end times and discharge-time charting leak the outcome |

---

## 5. Model selection when you train hundreds of models

This is the part the architecture-synthesis stage gets wrong most easily.

- **Precision of one number.** Mortality at ~8% prevalence on a 15% test split of ~40k stays is ~450 deaths in ~6k stays.
  Hanley–McNeil gives AUROC SE ≈ 0.011, so the 95% CI is ±2.1 points. Doubling the test set only shrinks it to ±1.5.
  AUPRC is noisier.
- **Compare models pairwise on the same stays**: paired bootstrap (1,000 resamples of patients; resample hospitals for
  eICU) or DeLong for AUROC. Report the difference with its CI, not two separate CIs.
- **Search on val, never on test.** With hundreds of candidates, the best val score is biased upward. Re-train the top 5
  candidates with 3 new training seeds, choose on mean val score, and only then evaluate the winner once on test.
  Report the winner's val-to-test drop as "selection optimism"; compare it with the same drop for a hand-picked expert.
- **Training-seed noise.** Train each expert with ≥ 3 seeds. If two architectures differ by less than their seed SD on
  val, treat them as tied.
- **Same budget for every method.** Count models trained and GPU hours per method (expert zoo, AES, soup, ensemble, NAS).
  That is the cost axis of the thesis and needs to be logged from the first run.

---

## 6. Reporting checklist

- Attrition table (CONSORT style) for MIMIC-IV and eICU: each filter and how many stays it removed.
- Cohort table: n, prevalence, age, sex, LoS, missingness per feature, per split (train / val / meta / test / temporal / eICU).
- For every method: AUROC, AUPRC (with prevalence), Brier, ECE, on internal test, temporal and eICU, and the gaps
  (internal − temporal, internal − eICU).
- eICU per-hospital AUROC spread (hospitals with ≥ ~10 deaths only) and per-region results.
- AES router choice distribution on MIMIC test vs eICU (see literature note §5.2).

---

## 7. Locked-in dataset spec (for Kiril to confirm)

| Item | Locked value |
|---|---|
| Training source | MIMIC-IV **v2.2**, ICU module + hosp patients table |
| External test | eICU-CRD **v2.0** |
| Ablation source | MIMIC-III v1.4 (separate runs only; never pooled with MIMIC-IV, never tested on it) |
| Pipeline | `ricu` 0.6.x + YAIB-cohorts + YAIB preprocessing, versions pinned |
| Cohort | YAIB base filters + task filters + **first ICU stay per patient** |
| Features | 4 static + 48 hourly dynamic (YAIB), MIMIC-train-only normalisation and imputation |
| Primary task | mortality24: ICU mortality, predicted at hour 24, stays ≥ 30h |
| Secondary task | AKI (KDIGO), hourly, up to 7 days |
| MIMIC partitions | Development 2008–2016 split 60/15/10/15 (train/val/meta/test), label-stratified, patient-level; temporal holdout 2017–2019 (approx. years) |
| Repeats | 3 split seeds; ≥ 3 training seeds per model; NAS on split seed 0 |
| Metrics | AUROC, AUPRC, Brier, ECE; paired patient-level bootstrap CIs (hospital-level on eICU) |
| Claim | External validation only: no eICU data used before final testing |
| Extra external tests (once access arrives) | HiRID v1.1.1 and AUMCdb v1.0.2 as additional test-only databases (§8, option B) |
| Planned extension | Pooled MIMIC-IV + HiRID + AUMCdb → eICU, with leave-one-database-out (§8, option C), started after the single-source zoo and AES work |

**Four things to confirm or push back on:**
1. ICU mortality (YAIB) rather than in-hospital mortality as the primary label. Recommended: ICU mortality, for comparability.
2. Spending ~20–25% of MIMIC on the temporal holdout. Recommended: keep it, because it is a second shift for almost no
   effort. The alternative is a random split over all years and only eICU as the shift.
3. The TA question from `dataset_and_models.md` §0 (is structured multi-source EHR "multimodal" enough) is still open
   and still decides whether this spec stands.
4. Multi-source pooling as an extension (option C below) rather than the core setup. Recommended: extension, but
   request HiRID and AUMCdb access now because AUMCdb needs a separate application.

---

## 8. Multi-source setups: MIMIC-IV + HiRID + AUMCdb (→ eICU)

Added after Kiril's follow-up. Checked: HiRID's PhysioNet page, the AmsterdamUMCdb GitHub README, and ricu's
data-source config (which versions it reads).

### 8.1 The extra databases

| | HiRID | AmsterdamUMCdb (AUMCdb) |
|---|---|---|
| Hospital | Bern University Hospital, Switzerland, one 60-bed ICU | Amsterdam UMC, Netherlands |
| Years | 2008 – June 2016 | 2003 – 2016 |
| Size | approx. 34k ICU admissions | 23,106 admissions, 20,109 patients |
| Version ricu reads | **v1.1.1** | **v1.0.2 (legacy format)**. The current release is v1.5.0 (OMOP, May 2024), which ricu does not read |
| Access | PhysioNet credentialed + DUA + a per-project research request | **Separate application** via Amsterdam Medical Data Science, not PhysioNet |
| Patient linkage | **None.** Each admission is treated as a unique patient; readmissions cannot be identified | Yes, `patientid` links admissions |
| Static features | Age, height, weight **binned in 5-unit bins** (age capped at 90) | Age, weight, height given as groups (from memory of the legacy schema; verify) |
| Resolution | 2-minute bedside data | Minute-level vitals |
| Mortality | Discharge status alive / dead / unknown | Death table; ricu's YAIB extension derives ICU death from discharge destination |

### 8.2 Three ways to use them

| Option | Train | Test | Extra cost | What it buys |
|---|---|---|---|---|
| **A. Core (locked above)** | MIMIC-IV | eICU (+ MIMIC temporal) | none | Clean, matches YAIB baselines |
| **B. More external tests** | MIMIC-IV | eICU, HiRID, AUMCdb | Two more access requests; extraction with the same ricu code | Non-US shifts (US → Switzerland, Netherlands) with no change to training |
| **C. Pooled training** | MIMIC-IV + HiRID + AUMCdb | eICU; plus leave-one-database-out over all four | B's cost + pooled preprocessing, per-source splits, source-shortcut checks | The setup the field is moving to; gives AES real domain heterogeneity |

### 8.3 Evidence for and against pooling, for this thesis

For:
- YAIB: a model pooled over databases is "usually as good as the best single-dataset model". Allyn et al. 2026: pooling
  gave the best generalisation. Burger et al. 2024: multi-centre leave-one-out recovers most of the single-source
  transfer drop **(verify)**.
- **It gives AES a reason to work.** With one source, experts differ only by architecture. With three sources you can also
  train **source experts** (same architecture, one per database) and ask whether a per-sample router picks the
  "most similar hospital". This is a much sharper AES question than routing between architectures alone.

Against:
- **Little expected gain for eICU specifically.** Londschien et al. 2025: external data is worth about 1.5–15k local
  patients, and once you have 10–50k local patients it barely helps. MIMIC-IV already gives approx. 40k development
  stays. Burger et al.: transfer works better within a continent, and HiRID and AUMCdb are European while eICU is US.
  My inference: expect at most a point or two of eICU AUROC from pooling, possibly zero.
- **Source shortcuts.** Binned age in HiRID versus exact age in MIMIC, different sampling density, different
  prevalence: a model can recognise the source and learn source-specific base rates. Mitigation: bin age, height and
  weight identically in every source (5-unit bins), resample everything to the same hourly grid, and report a
  "which-source" probe classifier on the learned features.
- **HiRID has no patient linkage**, so its patient-level split cannot be guaranteed. State it as a limitation;
  readmissions are a minority, so the effect should be small.
- **More label quirks.** ICU death is derived differently in each database (discharge status, discharge destination).
  Check prevalence per source in the cohort table before trusting pooled results.
- **Model selection gets subtle.** In domain generalisation, how you validate decides the result (Gulrajani &
  Lopez-Paz, ICLR 2021). Fix it in advance: select on the pooled validation sets of the training sources, and use
  leave-one-training-source-out only as a cross-check. Never select on eICU.

### 8.4 Rules if you run option C

- Split each source separately with the §3 fractions (train/val/meta/test, patient-level where possible). Only
  MIMIC-IV has a temporal holdout; keep MIMIC-IV training at 2008–2016 in pooled runs too.
- Normalise and impute with statistics from the **pooled training splits** only. Per-source normalisation would need
  eICU statistics at test time, which is adaptation.
- Default batch sampling: proportional to source size; ablate source-balanced sampling.
- Same feature set everywhere (YAIB's 52). A feature missing entirely from one source stays all-missing there; list
  these per source.
- Leave-one-database-out over all four (train on three, test on the fourth) is cheap once everything is extracted, and
  directly comparable to Burger et al.
- MIMIC-III is still never pooled with MIMIC-IV.

### 8.5 Recommendation

Keep **A** as the locked core. **Apply for HiRID and AUMCdb access now**, since AUMCdb is a separate application with
its own lead time. Run **B** as soon as access arrives (no training change). Run **C** as the main extension once the
single-source expert zoo and AES are working, framed around source experts versus architecture experts. If the TA
wants AES to carry the thesis, C is the strongest reason for promoting it to core; decide that with them.

## Sources

- ricu data sources config: https://github.com/eth-mds/ricu (`inst/extdata/config/data-sources.json`, NEWS.md)
- YAIB-cohorts: https://github.com/rvandewater/YAIB-cohorts (`R/base_cohort.R`, `R/mortality.R`, `R/aki.R`)
- YAIB: https://github.com/rvandewater/YAIB (`icu_benchmarks/data/split_process_data.py`); paper arXiv 2306.05109 (full text not read this time)
- MIMIC-IV v3.1 and v2.2: https://physionet.org/content/mimiciv/3.1/ , https://physionet.org/content/mimiciv/2.2/
- Hanley & McNeil, Radiology 1982 (AUROC standard error); DeLong et al., Biometrics 1988
- HiRID v1.1.1: https://physionet.org/content/hirid/1.1.1/
- AmsterdamUMCdb: https://github.com/AmsterdamUMC/AmsterdamUMCdb ; access via https://amsterdammedicaldatascience.nl/
- Gulrajani & Lopez-Paz, "In Search of Lost Domain Generalization", ICLR 2021
- Londschien et al. 2025 (arXiv 2507.21783), Burger et al. 2024 (arXiv 2411.16346), Allyn et al. 2026: see mimic_to_eicu_literature.md
