# Comparing plasma p-tau217, amyloid-beta42/40, glial fibrillary acidic protein, neurofilament light chain and amyloid imaging for predicting Alzheimer’s disease conversion in mild cognitive impairment 

## 1. Code accompanying the manuscript

Chen S*, Ellis RJ*, Patel CJ. Comparing plasma p-tau217, amyloid-beta42/40, glial fibrillary acidic protein, neurofilament light chain and amyloid imaging for predicting Alzheimer’s disease conversion in mild cognitive impairment.


\*Authors contributed equally to this work.

## 2. Pipeline
NOTE: Data are not included in this repository because of data use agreement and privacy restrictions. The data used in this study were obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI) database. Access can be requested at https://adni.loni.usc.edu.

Run the four scripts in order:

- `01_build_cohorts.py`: builds the baseline cohorts from the raw ADNI files. Covers diagnosis history (stable vs. progressive MCI), amyloid PET Centiloids, *APOE4*, censoring dates and plasma biomarkers, with biomarker-timing windows applied. Keeps one row per patient, at the earliest measurement.
- `02_survival_models.py`: the main analysis. Applies the final exclusions (N = 201; 157 stable, 44 progressive MCI). Fits random survival forests with 5-fold nested cross-validation for eight covariate sets: demographics + *APOE4* as the reference, then adding amyloid PET, p-tau217, PET + p-tau217, the blood panel (p-tau217, GFAP, NfL, Aβ42/40), the p-tau217/Aβ42 ratio, PET + the ratio, and all markers. Compares models on time-dependent AUC, IPCW C-index and Brier score with bootstrap CIs, and produces the manuscript comparison figures.
- `03_blood_biomarkers.py`: interprets the blood-panel model (partial dependence) and derives the two-cut-off plasma p-tau217 triage rule. The rule-out threshold is set at 90% sensitivity, the rule-in threshold at 90% specificity, and patients in between fall in an intermediate zone.
- `04_timeROC_comparisons.R`: time-dependent AUCs with inverse-probability-of-censoring weights (`timeROC`), 95% CIs, and pairwise model comparisons at 2, 3, 4 and 5 years.
- `05_supplemental.py` (optional, run after `02`): fits a Cox proportional-hazards model on the PET model's covariates (age, sex, education, *APOE4*, amyloid PET) using the same patients, folds and time grid as the RSF. Compares the two on C-index, time-dependent AUC and Brier score, with paired bootstrap CIs. Also tests the proportional-hazards assumption (Grambsch–Therneau test, equivalent to R's `cox.zph`) and plots the scaled Schoenfeld residuals for amyloid PET.

Each script writes files that the next one reads, so run them in order and don't skip steps. `run_pipeline.sh` runs the four main scripts; `05_supplemental.py` is not included and must be run separately. To run all four:

```bash
bash run_pipeline.sh
```

or one at a time:

```bash
python3 01_build_cohorts.py
python3 02_survival_models.py
python3 03_blood_biomarkers.py
Rscript 04_timeROC_comparisons.R
python3 05_supplemental.py   # optional: Cox vs. RSF comparison
```

`config.py` holds the input and output paths shared by the Python scripts (see *Input data*).

### Requirements

Python 3.13. The package versions used are pinned in `environment.yaml`:

```bash
conda env create -f environment.yaml
conda activate mci_ad
```

Main packages: `scikit-survival` 0.25, `scikit-learn` 1.7, `pandas` 2.3, `statsmodels` 0.14, `matplotlib`, `seaborn`.

R (developed with 4.4). Packages:

```r
install.packages(c("timeROC", "survival", "dplyr"))
```

### Input data

Set two environment variables before running:

```bash
export ADNI_RAW_DIR=/path/to/ADNI          # raw ADNI download (read only)
export MCI_AD_OUTPUT_DIR=/path/to/output   # where all results are written
```

`ADNI_RAW_DIR` should contain these ADNI files (download versions as used in the study):

- `Diagnosis/DXSUM_06Mar2025.csv`: diagnosis history
- `PET/ADSP_PHC_PET_Amyloid_Simple_07Jul2025.csv`: harmonised amyloid PET Centiloids
- `Data___Database/ADNIMERGE_07Jul2025.csv`: *APOE4* genotype
- `REGISTRY_22Aug2025.csv`: visit registry, used for censoring dates
- `Demographics/PTDEMOG_06Mar2025.csv`: date of birth
- `Biospecimen_Results/UPENN_PLASMA_FUJIREBIO_QUANTERIX_05Mar2025.csv`: plasma p-tau217, Aβ42/40, GFAP and NfL

Data are not included in this repository because of data use agreement and privacy restrictions. The data used in this study were obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI) database. Access can be requested at https://adni.loni.usc.edu.

## 3. Outputs

Results are written to `$MCI_AD_OUTPUT_DIR`:

```
cleaned_data/ADNI/       analysis cohorts (.parquet)
models/ADNI_PENN/        fitted random survival forests and supplementary Cox models (.joblib)
predictions/ADNI_PENN/   out-of-fold predictions (.npz)
metrics/ADNI_PENN/       per-fold metrics and bootstrap AUCs (.npz);
                         timeROC_input.csv, timeROC_results_sorted.csv
figures/ADNI_PENN/       all figures (.pdf, .png)
```

Figures include the model comparisons (`pet_bloodbios_{auc,cindex,bs}_comparison.pdf`, `bloodbios_demo_auc_comparison.pdf`, `demo_bloodbios_{cindex,bs}_comparison.pdf`), the two-cut-off p-tau217 rule (`two_cutoff_ptau217_90_90.png`), partial dependence (`partial_dependence_48mo.png`), biomarker distributions and the biomarker correlation matrix. `05_supplemental.py` adds the Cox vs. RSF comparisons (`supp_cph_rsf_{auc,cindex,bs}_comparison.pdf`) and the Schoenfeld residual plot for amyloid PET (`supp_cph_schoenfeld_centiloids.pdf`).

**Runtime.** The nested grid search in `02_survival_models.py` takes several hours per covariate set. To regenerate metrics and figures from models fitted earlier, without refitting, point `MCI_AD_REUSE_MODELS_FROM` at that earlier output folder. Its `models/` and `predictions/` are only read, never overwritten, and all other outputs are recomputed into `MCI_AD_OUTPUT_DIR`, which must be a different folder:

```bash
export MCI_AD_REUSE_MODELS_FROM=/path/to/previous_output
export MCI_AD_OUTPUT_DIR=/path/to/new_output
bash run_pipeline.sh
```

`05_supplemental.py` takes a few minutes. It always fits its Cox models, and with `MCI_AD_REUSE_MODELS_FROM` set it reads the PET RSF predictions from that folder.

**Pre-trained models.** The fitted models and out-of-fold predictions used in the manuscript (eight models × five folds) are available on figshare: https://doi.org/10.6084/m9.figshare.34941931. Unzip `mci_ad_models.zip` and point `MCI_AD_REUSE_MODELS_FROM` at the unzipped folder. Its `models/` and `predictions/` folders already have the layout the pipeline expects. Run `01_build_cohorts.py` first, because the cohort files come from the ADNI data and are not included.

## 4. Analysis settings

| setting | value |
|---|---|
| time origin / event | first MCI diagnosis / AD-dementia diagnosis; stable patients censored at their last visit |
| outer CV | `StratifiedKFold(n_splits=5, shuffle=True, random_state=20)`, stratified on the event |
| inner CV | `KFold(n_splits=5, shuffle=True, random_state=20)`, tuned on IPCW C-index |
| RSF grid | `n_estimators` ∈ {200, 300, 500}, `min_samples_split` ∈ {2, 5}, `min_samples_leaf` ∈ {10, 15}, `max_depth` ∈ {3, 5}, `max_features` ∈ {0.5, 0.7, 0.9, sqrt} |
| Cox grid (supplemental) | covariates standardized; ridge penalty `alpha` ∈ {0, 0.2, 0.4, 0.6, 0.8} (0 = unpenalized), `ties` ∈ {Breslow, Efron}; same inner CV and tuning metric as the RSF |
| evaluation window | 555–2154 days (≈1.5–5.9 years) after first MCI diagnosis |
| bootstrap | 1000 resamples of each test fold, percentile CIs |


## 5. License

This code is released under the MIT License; see `LICENSE`.
