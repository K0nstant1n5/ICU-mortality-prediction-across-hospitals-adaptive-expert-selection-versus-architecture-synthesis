# Installs the R packages YAIB-cohorts needs that are not taken from conda-forge.
# Versions follow YAIB-cohorts' renv.lock (R 4.2.2, ricu 0.6.0).
options(repos = c(CRAN = "https://cloud.r-project.org"), Ncpus = 4)
if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")
pin <- function(pkg, version) {
  if (!requireNamespace(pkg, quietly = TRUE) || as.character(packageVersion(pkg)) != version) {
    remotes::install_version(pkg, version = version, upgrade = "never")
  }
}
pin("argparser", "0.7.1")
pin("assertthat", "0.2.1")
RICU_SHA <- "0969a52747c71ae9f47051d640caa5df2d5169a0"
installed_sha <- suppressWarnings(tryCatch(packageDescription("ricu")$RemoteSha, error = function(e) NULL))
if (!identical(installed_sha, RICU_SHA)) {
  remotes::install_github(paste0("prockenschaub/ricu-package@", RICU_SHA), upgrade = "never", build_vignettes = FALSE)
}
# Demo data packages, useful for a first dry run before the full databases are imported.
install.packages(c("mimic.demo", "eicu.demo"), repos = "https://eth-mds.github.io/physionet-demo")
cat("R packages ready. ricu version:", as.character(packageVersion("ricu")), "\n")
