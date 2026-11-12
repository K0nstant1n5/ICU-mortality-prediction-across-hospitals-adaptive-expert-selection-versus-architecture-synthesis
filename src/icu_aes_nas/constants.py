"""Shared constants. These define the harmonised YAIB feature space used by every pipeline."""

# 48 dynamic ricu concepts used by YAIB-cohorts (R/mortality.R, R/aki.R).
DYNAMIC_VARS: tuple[str, ...] = (
    "alb", "alp", "alt", "ast", "be", "bicar", "bili", "bili_dir",
    "bnd", "bun", "ca", "cai", "ck", "ckmb", "cl", "crea", "crp",
    "dbp", "fgn", "fio2", "glu", "hgb", "hr", "inr_pt", "k", "lact",
    "lymph", "map", "mch", "mchc", "mcv", "methb", "mg", "na", "neut",
    "o2sat", "pco2", "ph", "phos", "plt", "po2", "ptt", "resp", "sbp",
    "temp", "tnt", "urine", "wbc",
)

# YAIB task statics. `sex` is categorical, the rest numeric.
STATIC_VARS: tuple[str, ...] = ("age", "sex", "height", "weight")
NUMERIC_STATICS: tuple[str, ...] = ("age", "height", "weight")

# Split names. `temporal` exists only for MIMIC-IV; `external` is every stay of a test-role source.
SPLITS: tuple[str, ...] = ("train", "val", "meta", "test", "temporal", "external")

# Token types on the bus. Pipeline A adds ECG = 2 and CXR = 3.
TOKEN_STATIC = 0
TOKEN_EHR = 1
TOKEN_ECG = 2
TOKEN_CXR = 3
