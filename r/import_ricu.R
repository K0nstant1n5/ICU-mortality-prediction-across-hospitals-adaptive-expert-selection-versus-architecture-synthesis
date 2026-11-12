# One-time conversion of raw PhysioNet CSVs into ricu's fst format.
# Usage: Rscript r/import_ricu.R <raw_dir> <src> [<src> ...]     e.g. Rscript r/import_ricu.R data/raw miiv eicu
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: import_ricu.R <raw_dir> <src> [<src> ...]")
raw <- normalizePath(args[[1]])
Sys.setenv(RICU_DATA_PATH = raw)
library(ricu)
for (src in args[-1]) {
  dir <- file.path(raw, src)
  if (!dir.exists(dir)) stop("missing ", dir)
  message("Importing ", src, " from ", dir)
  import_src(src, data_dir = dir)
}
