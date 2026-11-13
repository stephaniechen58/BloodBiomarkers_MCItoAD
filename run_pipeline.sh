#!/usr/bin/env bash
# Run the full pipeline, steps 01-04 in order. Paths are set with environment
# variables (see config.py and README.md):
#
#   ADNI_RAW_DIR              raw ADNI download (read only)
#   MCI_AD_OUTPUT_DIR         where this run writes everything
#   MCI_AD_REUSE_MODELS_FROM  optional: reuse fitted models from an earlier run (read only)
set -euo pipefail
: "${MCI_AD_OUTPUT_DIR:?Set MCI_AD_OUTPUT_DIR to the folder where results should be written}"

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "== 01_build_cohorts.py";       python3 01_build_cohorts.py
echo "== 02_survival_models.py";     python3 02_survival_models.py
echo "== 03_blood_biomarkers.py";    python3 03_blood_biomarkers.py
echo "== 04_timeROC_comparisons.R";  Rscript 04_timeROC_comparisons.R
echo "== done. Figures are in ${MCI_AD_OUTPUT_DIR}/figures/ADNI_PENN/"
