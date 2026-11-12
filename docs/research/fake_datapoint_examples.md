# What one data point looks like in each database (fabricated examples)

> **Every value, ID and timestamp in this file is made up.** None of it comes from a real record, and nothing was
> derived from MIMIC, eICU, HiRID or AUMCdb data (PhysioNet / AmsterdamUMCdb data use agreements). Fake IDs start with
> `9` or `FAKE` so they can never be confused with real ones. Column names and table layouts are the real schemas.
> Item/variable IDs marked *(verify)* are from memory; check them against `d_items`, the HiRID variable reference,
> or ricu's `inst/extdata/config/concept-dict.json` before writing them into the thesis.

## 0. The short version

A "data point" means something different at each stage:

| Stage | What one data point is | Shape |
|---|---|---|
| Raw database | One measurement row (a heart rate, a lab value, a urine bag reading) in a long event table, plus one row per stay in a stays/admissions table | Millions of long-format rows, different per DB |
| After ricu/YAIB harmonisation | One ICU stay = 4 static values + an hourly grid of 48 dynamic features + label(s) | Same columns, units and time base in all 5 DBs |
| Model input (mortality24) | One stay = one tensor of 24 hours × features, one binary label | e.g. `x_dyn: [24, 96]`, `x_static: [4]`, `y: 0/1` |
| Model input (AKI) | One stay **hour** = the history up to that hour, one binary label per hour | `x_dyn: [T, 96]`, `y: [T]`, T ≤ 168 |

The thesis models only ever see the last two stages. The raw stage matters for understanding why harmonisation
is hard (units, item IDs, time bases, sampling rates and death definitions all differ).

---

## 1. Raw shape in each database

All five examples below describe the **same imaginary patient**: a 67-year-old man, admitted to the ICU, heart rate
around 100, creatinine rising, who survives. That makes the differences between the databases easy to see.

### 1.1 MIMIC-IV v2.2 (Beth Israel Deaconess, Boston, 2008–2019)

Long format, absolute (date-shifted) timestamps, `itemid` codes looked up in `icu.d_items` / `hosp.d_labitems`.

`hosp.patients`
| subject_id | gender | anchor_age | anchor_year | anchor_year_group | dod |
|---|---|---|---|---|---|
| 90000001 | M | 67 | 2163 | 2011 - 2013 | *(null)* |

`icu.icustays`
| subject_id | hadm_id | stay_id | first_careunit | last_careunit | intime | outtime | los |
|---|---|---|---|---|---|---|---|
| 90000001 | 92000001 | 93000001 | Medical Intensive Care Unit (MICU) | Medical Intensive Care Unit (MICU) | 2163-04-11 08:42:00 | 2163-04-14 13:10:00 | 3.19 |

`icu.chartevents` (vitals, also weight/height)
| subject_id | hadm_id | stay_id | caregiver_id | charttime | storetime | itemid | value | valuenum | valueuom | warning |
|---|---|---|---|---|---|---|---|---|---|---|
| 90000001 | 92000001 | 93000001 | 99001 | 2163-04-11 09:00:00 | 2163-04-11 09:14:00 | 220045 | 104 | 104 | bpm | 0 |
| 90000001 | 92000001 | 93000001 | 99001 | 2163-04-11 09:00:00 | 2163-04-11 09:14:00 | 220179 | 118 | 118 | mmHg | 0 |
| 90000001 | 92000001 | 93000001 | 99001 | 2163-04-11 09:00:00 | 2163-04-11 09:14:00 | 220277 | 95 | 95 | % | 0 |
| 90000001 | 92000001 | 93000001 | 99001 | 2163-04-11 10:00:00 | 2163-04-11 10:05:00 | 223761 | 99.1 | 99.1 | °F | 0 |
| 90000001 | 92000001 | 93000001 | 99002 | 2163-04-11 08:55:00 | 2163-04-11 09:02:00 | 226512 | 84 | 84 | kg | 0 |

(220045 = heart rate, 220179 = non-invasive systolic BP, 220277 = SpO2, 223761 = temperature in °F, 226512 = admission
weight.) Note temperature in Fahrenheit: ricu converts it.

`hosp.labevents` (no `stay_id`; ricu attaches labs to the stay by time window)
| labevent_id | subject_id | hadm_id | specimen_id | itemid | charttime | storetime | value | valuenum | valueuom | ref_range_lower | ref_range_upper | flag | priority |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 94000001 | 90000001 | 92000001 | 95000001 | 50912 | 2163-04-11 07:30:00 | 2163-04-11 08:20:00 | 1.1 | 1.1 | mg/dL | 0.5 | 1.2 | | STAT |
| 94000002 | 90000001 | 92000001 | 95000002 | 50912 | 2163-04-12 04:10:00 | 2163-04-12 05:02:00 | 1.9 | 1.9 | mg/dL | 0.5 | 1.2 | abnormal | ROUTINE |
| 94000003 | 90000001 | 92000001 | 95000003 | 50813 | 2163-04-11 07:30:00 | 2163-04-11 07:41:00 | 2.4 | 2.4 | mmol/L | 0.5 | 2.0 | abnormal | STAT |

(50912 = creatinine, 50813 = lactate.) The 07:30 draw is *before* ICU admission at 08:42; ricu keeps it at a
negative hour, and YAIB drops hours before 0.

`icu.outputevents` (urine)
| subject_id | hadm_id | stay_id | caregiver_id | charttime | storetime | itemid | value | valueuom |
|---|---|---|---|---|---|---|---|---|
| 90000001 | 92000001 | 93000001 | 99001 | 2163-04-11 10:00:00 | 2163-04-11 10:03:00 | 226559 | 45 | ml |

(226559 = Foley catheter output.)

### 1.2 MIMIC-III v1.4 (same hospital, 2001–2012)

Same idea as MIMIC-IV, but upper-case columns, a `ROW_ID` on every table, two charting systems, and age stored
indirectly through a shifted date of birth.

`PATIENTS`
| ROW_ID | SUBJECT_ID | GENDER | DOB | DOD | DOD_HOSP | DOD_SSN | EXPIRE_FLAG |
|---|---|---|---|---|---|---|---|
| 9101 | 90000101 | M | 2081-06-02 00:00:00 | | | | 0 |

`ICUSTAYS`
| ROW_ID | SUBJECT_ID | HADM_ID | ICUSTAY_ID | DBSOURCE | FIRST_CAREUNIT | LAST_CAREUNIT | FIRST_WARDID | LAST_WARDID | INTIME | OUTTIME | LOS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 9201 | 90000101 | 92000101 | 93000101 | carevue | MICU | MICU | 52 | 52 | 2148-09-03 14:20:00 | 2148-09-06 10:05:00 | 2.82 |

Age = INTIME − DOB = 67. (Patients over 89 get a DOB about 300 years earlier; ricu caps them.)

`CHARTEVENTS`
| ROW_ID | SUBJECT_ID | HADM_ID | ICUSTAY_ID | ITEMID | CHARTTIME | STORETIME | CGID | VALUE | VALUENUM | VALUEUOM | WARNING | ERROR | RESULTSTATUS | STOPPED |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 9301 | 90000101 | 92000101 | 93000101 | 211 | 2148-09-03 15:00:00 | 2148-09-03 15:22:00 | 99101 | 104 | 104 | BPM | | | | NotStopd |
| 9302 | 90000101 | 92000101 | 93000101 | 646 | 2148-09-03 15:00:00 | 2148-09-03 15:22:00 | 99101 | 95 | 95 | % | | | | NotStopd |

The key MIMIC-III quirk: this stay is `carevue`, so heart rate is `ITEMID 211`. A `metavision` stay (2008 onward)
uses `220045`, the same ID as MIMIC-IV. ricu maps both to `hr`. That CareVue → MetaVision switch is the
"documentation shift" noted in dataset_and_models.md.

`LABEVENTS`
| ROW_ID | SUBJECT_ID | HADM_ID | ITEMID | CHARTTIME | VALUE | VALUENUM | VALUEUOM | FLAG |
|---|---|---|---|---|---|---|---|---|
| 9401 | 90000101 | 92000101 | 50912 | 2148-09-04 05:00:00 | 1.9 | 1.9 | mg/dL | abnormal |

### 1.3 eICU-CRD v2.0 (208 US hospitals, 2014–2015)

No absolute times at all: everything is **minutes since unit admission** (`...offset`). Vitals are stored *wide*
(one column per vital, a row every 5 minutes), and labs are identified by *name strings*, not numeric IDs.

`patient` (selected columns of 29)
| patientunitstayid | patienthealthsystemstayid | uniquepid | gender | age | hospitalid | unittype | admissionheight | admissionweight | hospitaladmitoffset | unitdischargeoffset | unitdischargestatus | hospitaldischargestatus | hospitaldischargeyear |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 9300001 | 9200001 | FAKE-001-0001 | Male | 67 | 999 | Med-Surg ICU | 178 | 84 | -312 | 4588 | Alive | Alive | 2015 |

Ages over 89 are the **string** `"> 89"`, so `age` is a text column. `hospitaladmitoffset = -312` means the hospital
admission was 312 minutes before ICU admission.

`vitalperiodic` (wide, every 5 minutes; selected columns)
| vitalperiodicid | patientunitstayid | observationoffset | temperature | sao2 | heartrate | respiration | systemicsystolic | systemicdiastolic | systemicmean |
|---|---|---|---|---|---|---|---|---|---|
| 9500001 | 9300001 | 5 | | 95 | 104 | 22 | | | |
| 9500002 | 9300001 | 10 | | 96 | 102 | 21 | | | |
| 9500003 | 9300001 | 15 | 37.3 | 95 | 101 | 22 | | | |

Invasive BP columns are empty when there is no arterial line; non-invasive BP lives in `vitalaperiodic`.

`lab`
| labid | patientunitstayid | labresultoffset | labtypeid | labname | labresult | labresulttext | labmeasurenamesystem | labresultrevisedoffset |
|---|---|---|---|---|---|---|---|---|
| 9600001 | 9300001 | -72 | 1 | creatinine | 1.1 | 1.1 | mg/dL | -20 |
| 9600002 | 9300001 | 1168 | 1 | creatinine | 1.9 | 1.9 | mg/dL | 1215 |
| 9600003 | 9300001 | -72 | 1 | lactate | 2.4 | 2.4 | mmol/L | -60 |

`intakeoutput` (urine is found by text in `cellpath` / `celllabel`)
| intakeoutputid | patientunitstayid | intakeoutputoffset | intaketotal | outputtotal | nettotal | cellpath | celllabel | cellvaluenumeric |
|---|---|---|---|---|---|---|---|---|
| 9700001 | 9300001 | 78 | 0 | 45 | -45 | flowsheet\|Flowsheet Cell Labels\|I&O\|Output (ml)\|Urine | Urine | 45 |

### 1.4 HiRID v1.1.1 (Bern University Hospital, Switzerland, 2008–2016)

One huge long table split into Parquet/CSV parts, very high resolution (vitals every **2 minutes**), numeric
`variableid`s, and only binned demographics.

`general_table`
| patientid | admissiontime | sex | age | discharge_status |
|---|---|---|---|---|
| 90001 | 2140-05-17 08:40:00 | M | 65 | alive |

`age` is the lower edge of a 5-year bin (65 = 65–69), so our 67-year-old becomes 65. There is no admission
weight here; weight and height come from the observation table.

`observation_tables` (raw stage)
| datetime | entertime | patientid | status | stringvalue | type | value | variableid |
|---|---|---|---|---|---|---|---|
| 2140-05-17 08:42:00 | 2140-05-17 08:42:00 | 90001 | 8 | | | 104 | 200 |
| 2140-05-17 08:44:00 | 2140-05-17 08:44:00 | 90001 | 8 | | | 103 | 200 |
| 2140-05-17 08:42:00 | 2140-05-17 08:42:00 | 90001 | 8 | | | 95 | 4000 |
| 2140-05-17 09:00:00 | 2140-05-17 09:00:00 | 90001 | 8 | | | 45 | 10020000 |
| 2140-05-18 04:10:00 | 2140-05-18 05:00:00 | 90001 | 8 | | | 168 | 20000600 |

(200 = heart rate, 4000 = SpO2, 10020000 = hourly urine, 20000600 = creatinine *(verify all IDs)*.) Creatinine is in
**µmol/L** (168 µmol/L ≈ 1.9 mg/dL); ricu converts. Values arrive every 2 minutes, so one ICU hour has about 30
heart-rate rows that ricu aggregates into one.

### 1.5 AmsterdamUMCdb v1.0.2 (Amsterdam UMC, Netherlands, 2003–2016)

Times are **milliseconds since the patient's first admission**, demographics are all groups, and item labels are
in Dutch.

`admissions` (selected columns)
| patientid | admissionid | admissioncount | location | urgency | origin | admittedat | admissionyeargroup | dischargedat | lengthofstay | destination | gender | agegroup | dateofdeath | weightgroup | heightgroup | specialty |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 90001 | 90011 | 1 | IC | 1 | Verpleegafdeling zelfde ziekenhuis | 0 | 2010-2016 | 266400000 | 74 | Verpleegafdeling zelfde ziekenhuis | Man | 60-69 | | 80-89 | 170-179 | Interne Geneeskunde |

`agegroup = 60-69`, `weightgroup = 80-89`: the model sees bin midpoints (or bin edges) here, not 67 and 84.
`dischargedat = 266400000` ms = 74 h. ICU death is derived from `destination` / `dateofdeath`.

`numericitems` (selected columns)
| admissionid | itemid | item | value | unitid | unit | measuredat | registeredat | islabresult | fluidout |
|---|---|---|---|---|---|---|---|---|---|
| 90011 | 6640 | Hartfrequentie | 104 | 15 | /min | 1080000 | 1140000 | 0 | |
| 90011 | 6709 | Saturatie (Monitor) | 95 | 34 | % | 1080000 | 1140000 | 0 | |
| 90011 | 8794 | UrineCAD | 45 | 18 | ml | 4680000 | 4800000 | 0 | 45 |
| 90011 | 6836 | Kreatinine (bloed) | 168 | 41 | µmol/l | 70200000 | 73800000 | 1 | |

(6640 = heart rate, 6709 = SpO2, 8794 = urine via catheter, 6836 = creatinine *(verify IDs and unitids)*.)
`measuredat = 1080000` ms = 18 minutes after admission. Creatinine again in µmol/L.

### 1.6 What differs, side by side

| | MIMIC-IV | MIMIC-III | eICU | HiRID | AUMCdb |
|---|---|---|---|---|---|
| Stay key | `stay_id` | `ICUSTAY_ID` | `patientunitstayid` | `patientid` | `admissionid` |
| Time base | shifted datetimes | shifted datetimes | minutes from unit admit | shifted datetimes | ms from first admit |
| Vital layout | long, `itemid` | long, two ID systems | wide, 5-min rows | long, 2-min rows | long, Dutch labels |
| Lab ID | `itemid` (int) | `ITEMID` (int) | `labname` (string) | `variableid` (int) | `itemid` (int) |
| Creatinine unit | mg/dL | mg/dL | mg/dL | µmol/L | µmol/L |
| Age | exact (capped 91) | via DOB (>89 shifted) | exact or `"> 89"` | 5-year bin | 10-year group |
| Weight/height | charted | charted | admission fields | charted | groups |
| ICU death from | death time within stay | death time within stay | `unitdischargestatus` | `discharge_status` | `destination` / `dateofdeath` |

---

## 2. After harmonisation (ricu → YAIB cohort output)

ricu maps every source to the same **concepts** with the same units, attaches each value to the stay, and
aggregates to an hourly grid (`time` = hours since ICU admission). YAIB-cohorts then writes three tables per
task. From here on, all five databases look identical; only the values and the missingness pattern differ.

### 2.1 `sta` (static, one row per stay)

| stay_id | age | sex | height | weight |
|---|---|---|---|---|
| 93000001 | 67 | Male | 178 | 84 |

(In HiRID the same patient would be `age 65`; in AUMCdb something like `age 65, weight 85, height 175` from bin
midpoints. This is the "binning" shift the lock-in note says to equalise by binning everywhere.)

### 2.2 `dyn` (dynamic, one row per stay-hour, 48 feature columns)

The 48 YAIB features (ricu concept names): `alb, alp, alt, ast, be, bicar, bili, bili_dir, bnd, bun, ca, cai, ck,
ckmb, cl, crea, crp, dbp, fgn, fio2, glu, hgb, hr, inr_pt, k, lact, lymph, map, mch, mchc, mcv, methb, mg, na, neut,
o2sat, pco2, ph, phos, plt, po2, ptt, resp, sbp, temp, tnt, urine, wbc` *(check against the YAIB feature list in your
pinned version)*. Showing 10 of them; the other 38 are mostly empty for this stay, as labs are sparse.

| stay_id | time | hr | sbp | dbp | map | resp | o2sat | temp | crea | lact | urine | … 38 more |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 93000001 | 0 | 104 | 118 | 64 | 82 | 22 | 95 | | | 2.4 | | … |
| 93000001 | 1 | 101 | 121 | 66 | 84 | 21 | 96 | 37.3 | | | 45 | … |
| 93000001 | 2 | 99 | 115 | 61 | 79 | 20 | 96 | | | | 40 | … |
| 93000001 | 3 | 98 | | | | 20 | 97 | | | | 35 | … |
| … | … | … | … | … | … | … | … | … | … | … | … | … |
| 93000001 | 19 | 108 | 109 | 58 | 75 | 24 | 94 | 37.9 | 1.9 | | 20 | … |
| … | … | … | … | … | … | … | … | … | … | … | … | … |
| 93000001 | 23 | 106 | 112 | 60 | 77 | 23 | 95 | | | | 25 | … |

Units are now the same everywhere (°C, mg/dL, mmHg, mL/h). Several raw values per hour are aggregated (median for
vitals; sum for urine). Empty cells are real missingness: about 90%+ of lab cells are empty in a typical stay.

### 2.3 `outc` (labels)

**mortality24**: one row per stay.

| stay_id | label |
|---|---|
| 93000001 | 0 |

(Stays with ICU LoS < 30 h or death before 30 h are already excluded at this point.)

**AKI**: one row per stay-hour, so the label is a sequence.

| stay_id | time | label |
|---|---|---|
| 93000001 | 0 | 0 |
| … | … | 0 |
| 93000001 | 13 | 1 |
| … | … | 1 |
| 93000001 | 25 | 0 |

Here creatinine went from 1.1 to 1.9 mg/dL (×1.7 from baseline; KDIGO stage 1 needs ≥ 1.5× or +0.3 mg/dL in 48 h),
so ricu would place AKI onset around hour 19. The label is 1 in a window before onset (about 6 h ahead; verify
`outcome_window(6, 6)` in `R/aki.R`), so the model learns to flag hours *before* creatinine is drawn.

---

## 3. Model-ready sample (after YAIB preprocessing, fitted on MIMIC-IV train only)

Preprocessing, all fitted on the MIMIC-IV training split and frozen for everything else (lock-in note §4):
1. Scale each feature with the train mean/std (`hr` mean 85, std 17 → 104 becomes +1.12).
2. Add a 0/1 missingness indicator per dynamic feature (1 = actually measured this hour).
3. Forward-fill within the stay, then fill what is still missing with the train mean (0 after scaling).
4. Sex → 0/1, static features scaled the same way.

### 3.1 One mortality24 sample (deep models: GRU-D, TCN, Transformer, Mamba)

```python
# FABRICATED example, values invented for illustration
sample = {
    "stay_id": 93000001,
    "source": "miiv",                 # miiv | eicu | mimic | hirid | aumc
    "x_static": [-0.31, 1.0, 0.62, 0.18],   # age, sex(male=1), height, weight; shape [4]
    "x_dyn":   ...,                   # shape [24, 96] = 24 hours × (48 values + 48 masks)
    "y": 0,                           # ICU death (after hour 30), 0 = survived
}

# first 3 hours, 6 of the 48 value columns and their masks:
#        hr     sbp    o2sat  temp   crea   lact | m_hr m_sbp m_o2 m_temp m_crea m_lact
# h0  [ 1.12,  -0.05, -0.21,  0.00,  0.00,  0.74 |  1    1     1    0      0      1 ]
# h1  [ 0.94,   0.09,  0.12, -0.12,  0.00,  0.74 |  1    1     1    1      0      0 ]
# h2  [ 0.82,  -0.19,  0.12, -0.12,  0.00,  0.74 |  1    1     1    0      0      0 ]
```

`crea` stays 0.00 (the train mean) until the first draw at hour 19, and its mask stays 0. `lact` is forward-filled
from hour 0 with mask 0 after that. GRU-D uses the masks (and time since last measurement) directly; for the other
deep models the masks are just extra input channels.

### 3.2 The same stay for LightGBM

Tree models get one flat row per prediction. YAIB builds it from the hourly grid with historical aggregates
(min, max, mean, count of each feature up to the prediction hour) *(check which aggregates your YAIB version adds)*:

| stay_id | age | sex | height | weight | hr_last | hr_min | hr_max | hr_mean | hr_count | crea_last | crea_count | … | y |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 93000001 | 67 | 1 | 178 | 84 | 106 | 92 | 112 | 102.4 | 24 | 1.9 | 1 | … (~250 columns) | 0 |

### 3.3 One AKI sample

Same `x_dyn` layout, but `T` is the full stay up to 168 h and `y` has one entry per hour:

```python
sample_aki = {
    "stay_id": 93000001,
    "x_static": [-0.31, 1.0, 0.62, 0.18],
    "x_dyn": ...,            # shape [T, 96], T = 74 here (stay length in hours)
    "y": [0]*13 + [1]*13 + [0]*48,   # per-hour labels, padded/masked to 168 in a batch
}
```

Hours after onset are dropped from evaluation, and confidence intervals resample stays, not hours.

### 3.4 What an eICU test sample looks like

Exactly the same structure: `source = "eicu"`, `x_dyn [24, 96]`, `y` from `unitdischargestatus`, scaled with the
**MIMIC-IV train** mean/std (not eICU's own). Typical differences the model will meet: some features never measured
in some hospitals (whole mask column 0), and different measurement frequency, e.g. hourly charted vitals vs.
5-minute monitor data aggregated to hours. That is the distribution shift the thesis measures.
