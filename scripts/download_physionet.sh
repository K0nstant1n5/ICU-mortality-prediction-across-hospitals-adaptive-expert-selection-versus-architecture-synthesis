#!/usr/bin/env bash
# Downloads MIMIC-IV v2.2, eICU-CRD v2.0 and/or HiRID v1.1.1 from PhysioNet into data/raw/ in the layout ricu expects
# (README §4), then checks every file against PhysioNet's SHA256SUMS.txt.
#
# Usage:
#   bash scripts/download_physionet.sh                 # both (miiv eicu)
#   bash scripts/download_physionet.sh miiv            # only MIMIC-IV
#   bash scripts/download_physionet.sh eicu            # only eICU
#   bash scripts/download_physionet.sh hirid           # only HiRID (3 archives ricu needs, then unpacked)
#
# Login (your PhysioNet account must be credentialed and have signed both DUAs):
#   * Interactive: the script asks for your PhysioNet username and password. Nothing is saved.
#   * Unattended (e.g. sbatch): put your login in ~/.netrc (NOT in this repo), then run as normal:
#         machine physionet.org login <your_username> password <your_password>
#     and protect it:  chmod 600 ~/.netrc
#
# Safe to re-run: wget -N -c resumes partial files and skips files that are already complete.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="${IAN_DATA:-$ROOT/data}/raw"
SOURCES="${*:-miiv eicu}"

declare -A URL=( [miiv]="https://physionet.org/files/mimiciv/2.2/"
                 [eicu]="https://physionet.org/files/eicu-crd/2.0/"
                 [hirid]="https://physionet.org/files/hirid/1.1.1/" )
# HiRID: only the archives ricu's hirid config uses (zip_file in ricu's data-sources.json). The merged/imputed
# stages and parquet copies are not needed.
HIRID_FILES=(reference_data.tar.gz raw_stage/observation_tables_csv.tar.gz raw_stage/pharma_records_csv.tar.gz)

# ---- credentials -----------------------------------------------------------------------------------
AUTH=()
if [[ -f "$HOME/.netrc" ]] && grep -q "physionet.org" "$HOME/.netrc"; then
  echo ">> Using PhysioNet login from ~/.netrc"
  [[ "$(stat -c %a "$HOME/.netrc")" == "600" ]] || echo "WARNING: run 'chmod 600 ~/.netrc' so others can't read your password" >&2
elif [[ -t 0 ]]; then
  read -r -p "PhysioNet username: " PN_USER
  AUTH=(--user "$PN_USER" --ask-password)
else
  echo "No terminal and no physionet.org entry in ~/.netrc. See the header of this script." >&2
  exit 1
fi

# ---- download + verify -----------------------------------------------------------------------------
for src in $SOURCES; do
  [[ -n "${URL[$src]:-}" ]] || { echo "Unknown source '$src' (use miiv, eicu and/or hirid)" >&2; exit 1; }
  dest="$RAW/$src"
  mkdir -p "$dest"
  if [[ "$src" == hirid ]]; then
    arch="$dest/archives"; mkdir -p "$arch"
    echo ">> Downloading hirid archives into $arch"
    # one wget call (one password prompt); -x -nH --cut-dirs=3 keeps e.g. raw_stage/ under $arch
    urls=(); for f in SHA256SUMS.txt "${HIRID_FILES[@]}"; do urls+=("${URL[$src]}$f"); done
    wget -N -c -x -nH --cut-dirs=3 --no-verbose --show-progress --auth-no-challenge "${AUTH[@]}" \
         -P "$arch" "${urls[@]}"
    echo ">> Verifying hirid checksums"
    ( cd "$arch" && for f in "${HIRID_FILES[@]}"; do grep -E "[[:space:]]\*?$f\$" SHA256SUMS.txt; done \
        | sha256sum --quiet -c - ) \
      || { echo "Checksum FAILED for hirid: delete the bad archive in $arch and re-run" >&2; exit 1; }
    for f in "${HIRID_FILES[@]}"; do
      mark="$arch/.extracted_$(basename "$f")"
      [[ -f "$mark" ]] && { echo ">> already unpacked: $f"; continue; }
      echo ">> Unpacking $f into $dest"
      tar -xzf "$arch/$f" -C "$dest" && touch "$mark"
    done
    echo ">> hirid OK (archives kept in $arch; delete them to save space once ricu has imported hirid)"
    continue
  fi
  echo ">> Downloading $src from ${URL[$src]} into $dest"
  # -nH --cut-dirs=3 strips "physionet.org/files/<project>/<version>/" so files land directly in $dest
  # (miiv -> $dest/hosp/*.csv.gz and $dest/icu/*.csv.gz, eicu -> $dest/*.csv.gz)
  # --auth-no-challenge: PhysioNet answers 403 (not 401) without login, so send credentials up front (HTTPS)
  wget -r -N -c -np -nH --cut-dirs=3 --reject "index.html*" --no-verbose --show-progress --auth-no-challenge \
       "${AUTH[@]}" -P "$dest" "${URL[$src]}"

  echo ">> Verifying $src checksums"
  ( cd "$dest" && sha256sum --quiet -c SHA256SUMS.txt ) \
    && echo ">> $src OK" \
    || { echo "Checksum FAILED for $src: re-run this script to resume/repair" >&2; exit 1; }
done

chmod -R go-rwx "$RAW"   # credentialed data: only your user may read it
echo ">> Done. Raw data in $RAW. Next: bash scripts/extract_cohorts.sh \"$SOURCES\" \"mortality24 aki\""
