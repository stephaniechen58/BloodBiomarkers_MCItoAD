"""
Where the pipeline reads raw data from and writes its outputs to.

Every pipeline script imports its paths from here, so nothing depends on the folder you
launch it from. Paths are set with environment variables:

    ADNI_RAW_DIR              the raw ADNI download (read only)
    MCI_AD_OUTPUT_DIR         where every output of this run is written
    MCI_AD_REUSE_MODELS_FROM  optional: an earlier output folder whose fitted models and
                              predictions should be reused instead of refitting

Full run (fits every model -- takes hours):

    export ADNI_RAW_DIR=/path/to/raw_data/ADNI
    export MCI_AD_OUTPUT_DIR=/path/to/mci_ad

Test run (reuses stored models; the stored folder is only read, never written):

    export MCI_AD_REUSE_MODELS_FROM=/path/to/mci_ad
    export MCI_AD_OUTPUT_DIR=/path/to/mci_ad/test_run

MCI_AD_OUTPUT_DIR must be set. ADNI_RAW_DIR is needed only to build the cohorts
(01_build_cohorts.py). Output layout, under MCI_AD_OUTPUT_DIR:

    cleaned_data/ADNI/        derived cohorts (parquet)
    models/ADNI_PENN/         fitted forests (.joblib)
    predictions/ADNI_PENN/    per-fold prediction bundles (.npz)
    metrics/ADNI_PENN/        per-fold metrics, bootstrap AUCs (.npz), timeROC CSVs
    figures/ADNI_PENN/        every figure (.pdf, .png)
"""

import os
from pathlib import Path

RAW_DIR = Path(os.environ.get("ADNI_RAW_DIR", "/path/to/ADNI"))

_output = os.environ.get("MCI_AD_OUTPUT_DIR")
if not _output:
    raise SystemExit(
        "MCI_AD_OUTPUT_DIR is not set.\n"
        "Set it to the folder where results should be written, e.g.\n"
        "    export MCI_AD_OUTPUT_DIR=/path/to/output"
    )
OUTPUT_DIR = Path(_output)

# Written by this run
CLEANED_DIR = OUTPUT_DIR / "cleaned_data" / "ADNI"
MODELS_DIR = OUTPUT_DIR / "models" / "ADNI_PENN"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions" / "ADNI_PENN"
METRICS_DIR = OUTPUT_DIR / "metrics" / "ADNI_PENN"
FIGURES_DIR = OUTPUT_DIR / "figures" / "ADNI_PENN"
TIMEROC_INPUT_CSV = METRICS_DIR / "timeROC_input.csv"

# Fitted models and their predictions: either fitted by this run, or reused from an
# earlier run (read only).
_reuse = os.environ.get("MCI_AD_REUSE_MODELS_FROM")
REUSE_DIR = Path(_reuse) if _reuse else None
REFIT = REUSE_DIR is None
MODELS_IN = MODELS_DIR if REFIT else REUSE_DIR / "models" / "ADNI_PENN"
PREDICTIONS_IN = PREDICTIONS_DIR if REFIT else REUSE_DIR / "predictions" / "ADNI_PENN"

if not REFIT and REUSE_DIR.resolve() == OUTPUT_DIR.resolve():
    raise SystemExit(
        "MCI_AD_OUTPUT_DIR is the same folder as MCI_AD_REUSE_MODELS_FROM, so this run "
        "would overwrite the stored results.\nSet MCI_AD_OUTPUT_DIR to a new folder."
    )


def make_output_dirs():
    """Create the output folders if they don't exist yet."""
    for d in (CLEANED_DIR, MODELS_DIR, PREDICTIONS_DIR, METRICS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def check_raw_dir():
    """Fail early, with a clear message, if the raw ADNI folder can't be found."""
    if not RAW_DIR.is_dir():
        raise SystemExit(
            f"Raw ADNI folder not found: {RAW_DIR}\n"
            "Set ADNI_RAW_DIR to the folder that holds the ADNI download."
        )


def describe():
    """Print where this run reads from and writes to."""
    print(f"raw data   : {RAW_DIR}")
    print(f"outputs    : {OUTPUT_DIR}")
    if REFIT:
        print("models     : fitted by this run")
    else:
        print(f"models     : reused (read only) from {REUSE_DIR}")
