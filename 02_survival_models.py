"""
Primary survival models: predicting progression from MCI to Alzheimer's dementia.

Step 2 of the pipeline and the main analysis of the study (see README.md). It takes the
baseline cohorts built by 01_build_cohorts.py, finishes cleaning them, and fits a random
survival forest for each of seven nested covariate sets, so that the contribution of
each biomarker over routinely available clinical information can be read off directly.

    time origin   first MCI diagnosis
    event         AD-dementia diagnosis; stable patients censored at their last visit
    N = 201       157 stable, 44 progressive

Model families (all include AGE_MCI, Sex, Education; prefix = the name used in filenames)

    demo          + APOE4                                   the reference model
    pet           + APOE4, PET Centiloids
    ptau          + APOE4, plasma p-tau217
    pet_ptau      + APOE4, Centiloids, p-tau217
    bio           + APOE4, p-tau217, GFAP, NfL, AB42/40      the blood panel
    ptau_amyloid  + APOE4, p-tau217/AB42 ratio
    ksink         every marker plus Centiloids               "kitchen sink"
    pet_ptau_amyloid  + APOE4, Centiloids, p-tau217/AB42 ratio   (Section 15)

Inputs
    $MCI_AD_OUTPUT_DIR/cleaned_data/ADNI/earliest_pet_df.parquet    from 01_build_cohorts.py
    $MCI_AD_OUTPUT_DIR/cleaned_data/ADNI/earliest_ptau_df.parquet
    the raw plasma CSV, for the p-tau217/AB42 ratio

Outputs
    $MCI_AD_OUTPUT_DIR/cleaned_data/ADNI/earliest_{pet,ptau}_df_rev2.parquet   read by 03_blood_biomarkers.py
    $MCI_AD_OUTPUT_DIR/models/ADNI_PENN/earliest_<prefix>_rsf_fold{0..4}.joblib
    $MCI_AD_OUTPUT_DIR/predictions/ADNI_PENN/earliest_<prefix>_rsf_fold{i}_predictions.npz
    $MCI_AD_OUTPUT_DIR/metrics/ADNI_PENN/<prefix>_metrics.npz
    $MCI_AD_OUTPUT_DIR/metrics/ADNI_PENN/timeROC_input.csv          read by 04_timeROC_comparisons.R
    $MCI_AD_OUTPUT_DIR/figures/ADNI_PENN/*.pdf, *.png

RUNTIME. Sections 5-11 each run a nested grid search over five outer folds, which takes
hours per covariate set. Each fit block (marked "if REFIT:") is followed by a scoring
block that reloads the predictions it wrote. To skip fitting, point
MCI_AD_REUSE_MODELS_FROM at a folder that already holds models/ and predictions/: those
are read (never written), and everything else is recomputed into MCI_AD_OUTPUT_DIR.
See config.py.

Section 14 writes metrics/ADNI_PENN/timeROC_input.csv, the input to 04_timeROC_comparisons.R.

Paths come from config.py: $MCI_AD_OUTPUT_DIR below is the output folder, set with the
MCI_AD_OUTPUT_DIR environment variable (raw data: ADNI_RAW_DIR). Can be run from any
folder. See README.md.
"""

# ---- Imports -------------------------------------------------------------------------

import joblib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save figures to files; no pop-up windows
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from scipy.stats import fisher_exact, chi2_contingency, mannwhitneyu

from sklearn import set_config
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.isotonic import spearmanr   # sklearn re-exports scipy's spearmanr

from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

# functions used with GridSearchCV for hyperparameter tuning
from sksurv.metrics import as_concordance_index_ipcw_scorer

# metric functions
from sksurv.metrics import (
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    brier_score,
    integrated_brier_score,
)

import config
from config import (RAW_DIR, CLEANED_DIR, MODELS_DIR, PREDICTIONS_DIR, METRICS_DIR,
                    FIGURES_DIR, MODELS_IN, PREDICTIONS_IN, REFIT, TIMEROC_INPUT_CSV)
config.check_raw_dir()
config.make_output_dirs()
config.describe()


# =====================================================================================
# 0 · Setup
#
# Imports and the shared cohort/diagnosis helpers (the same definitions used in 01_build_cohorts.py),
# plus `load_predictions`, which reads one fold's saved `.npz` prediction bundle.
# =====================================================================================

set_config(display="text")  # text representation of estimators

def build_mci_cohorts(dx, cohort = None):
    if cohort is None:
        raise ValueError("Cohort must be specified")
    if cohort == "ADNI":
        # MCI: DXMCI==1 (ADNI1) or DIAGNOSIS==2 (other cohorts)
        mci = dx[(dx.DXMCI == 1) | (dx.DIAGNOSIS == 2)]
        # Exclude patients with DXMDUE==2 (other etiologies)
        mci = mci[(mci.DXMDUE != 2)]
        # Get unique patient IDs for MCI
        mci_patients = mci.RID.unique()
        return mci, mci_patients
    
def build_ad_cohorts(dx, cohort = None): 
    if cohort is None:
        raise ValueError("Cohort must be specified")
    if cohort == "ADNI":
        # AD: DXAD==1 (ADNI1), DXAPP in [1,2] (all cohorts), or DIAGNOSIS==3 (other cohorts)
        ad = dx[(dx.DXAD == 1) | (dx.DXAPP.isin([1,2])) | (dx.DIAGNOSIS == 3)]
        # Exclude patients with DXDDUE==2 (other etiologies)
        ad = ad[ad.DXDDUE != 2]
        # Get unique patient IDs for AD
        ad_patients = ad.RID.unique()
        return ad, ad_patients

#take entry with last PET scan for each patient in both pmci and smci cohorts
def build_cohorts(pmci_pet_dx, mci_pet_dx, cohort):
    if cohort == None:
        raise ValueError("Cohort must be specified")
    if cohort == "ADNI":
        pmci_ml4h = pmci_pet_dx.sort_values(by=['RID', 'PETDATE'])
        pmci_ml4h = pmci_ml4h.groupby('RID').last().reset_index()
        mci_ml4h = mci_pet_dx.sort_values(by=['RID', 'PETDATE'])
        mci_ml4h = mci_ml4h.groupby('RID').last().reset_index()
        #print shape of both dataframes
        print(pmci_ml4h.shape, mci_ml4h.shape)
        return pmci_ml4h, mci_ml4h
    if cohort == "HABS":
        pmci_ml4h = pmci_pet_dx.sort_values(by=['Med_ID', '01_AB_FBB_Date'])
        pmci_ml4h = pmci_ml4h.groupby('Med_ID').last().reset_index()
        mci_ml4h = mci_pet_dx.sort_values(by=['Med_ID', '01_AB_FBB_Date'])
        mci_ml4h = mci_ml4h.groupby('Med_ID').last().reset_index()
        #print shape of both dataframes
        print(pmci_ml4h.shape, mci_ml4h.shape)
    return pmci_ml4h, mci_ml4h

#create_problem_list function
def create_problem_list(pmci, cohort=None):  
    problem_list = []
    if cohort is None:
        raise ValueError("Cohort must be specified")
    if cohort == "ADNI":
        for pt in pmci.RID.unique():
            pt_df = pmci[pmci.RID == pt]
            # Find first MCI diagnosis date
            mci_mask = ((pt_df.DXMCI == 1) | (pt_df.DIAGNOSIS == 2)) & (~pt_df.EXAMDATE.isna())
            mci_dates = pt_df[mci_mask].EXAMDATE.values
            if len(mci_dates) == 0:
                print("Warning: MCI date is NaN for patient " + str(pt))
                problem_list.append(pt)
                continue
            first_mci = mci_dates[0]
            # Find first AD diagnosis date
            ad_mask = ((pt_df.DXAD == 1) | (pt_df.DXAPP.isin([1,2])) | (pt_df.DIAGNOSIS == 3)) & (~pt_df.EXAMDATE.isna())
            ad_dates_df = pt_df[ad_mask]
            if ad_dates_df.shape[0] == 0:
                print("Warning: AD date is NaN for patient " + str(pt))
                problem_list.append(pt)
                continue
            else:
                first_ad = ad_dates_df.EXAMDATE.values[0]
            # If MCI diagnosis is after AD diagnosis, flag as problem
            if first_mci > first_ad:
                print("Error: MCI date is after AD date for patient " + str(pt))
                problem_list.append(pt)
        return problem_list

#get first MCI diagnosis date
def get_first_mci_date(mci_dx_df, cohort=None):
    if cohort is None:
        raise ValueError("Cohort must be specified")
    if cohort == "ADNI":
        # Get the first MCI diagnosis date for each patient
        mci_dates = mci_dx_df[(mci_dx_df.DXMCI == 1) | 
            (mci_dx_df.DIAGNOSIS == 2) & 
            (~mci_dx_df.DXDATE.isna())].groupby('RID').first().reset_index()
        return mci_dates

#get first AD diagnosis date
def get_first_ad_date(pmci_dx_df):
    first_ad_dx  = pmci_dx_df[
        ((pmci_dx_df.DXAD == 1) |  
         (pmci_dx_df.DXAPP.isin([1,2])) | 
         (pmci_dx_df.DIAGNOSIS == 3)) & 
        (~pmci_dx_df.DXDATE.isna())
    ].groupby('RID').first().reset_index()
    return first_ad_dx

def load_predictions(i, path_template):
    """
    Load saved predictions/labels for fold i.
    
    Parameters
    ----------
    i : int
        Fold index.
    path_template : str
        Template for file path. Must contain `{fold}` as placeholder.
        Example: f"{PREDICTIONS_IN}/earliest_bio_rsf_fold{fold}_predictions.npz"

    """
    filepath = path_template.format(fold=i)
    data = np.load(filepath, allow_pickle=True)

    # keep original variable names
    train_predictions      = data["train_predictions"]
    test_predictions       = data["test_predictions"]
    train_chf_predictions  = data["train_chf_predictions"]
    test_chf_predictions   = data["test_chf_predictions"]
    train_surv_predictions = data["train_surv_predictions"]
    test_surv_predictions  = data["test_surv_predictions"]
    y_train                = data["train_labels"]
    y_test                 = data["test_labels"]

    return (
        train_predictions,
        test_predictions,
        train_chf_predictions,
        test_chf_predictions,
        train_surv_predictions,
        test_surv_predictions,
        y_train,
        y_test,
    )  


# =====================================================================================
# 1 · Final cleaning of the cohorts from 01_build_cohorts.py
#
# > **Run this section before 03_blood_biomarkers.py.** It creates `AGE_MCI` and
# > `mci_event_daysdiff` — the survival time used by every model — and exports the
# > `*_rev2.parquet` files that `03_blood_biomarkers.py` loads directly.
#
# Exclusions applied here, on top of 01_build_cohorts.py:
#
# - drop patients with a missing Centiloid value;
# - restrict both tables to the **intersection** of PET and plasma patients, so every
#   model is fit on the same people and the comparisons are like-for-like;
# - drop converters whose PET scan was taken more than 3 years after their MCI diagnosis;
# - drop patients whose plasma draw was more than 6 years after their MCI diagnosis.
# =====================================================================================

base_path = str(RAW_DIR) + "/"
#load in parquet files
ptau_df = pd.read_parquet(f"{CLEANED_DIR}/earliest_ptau_df.parquet")
pet_df = pd.read_parquet(f"{CLEANED_DIR}/earliest_pet_df.parquet")

pet_df = pet_df.dropna(subset=['CENTILOIDS'])

print(ptau_df.LABEL.value_counts())
print(pet_df.LABEL.value_counts())

overlap_rids = set(ptau_df.RID).intersection(set(pet_df.RID))
print("Number of overlapping RIDs: ", len(overlap_rids))

ptau_df = ptau_df[ptau_df.RID.isin(overlap_rids)]
pet_df = pet_df[pet_df.RID.isin(overlap_rids)]

#check breakdown of LABEL in ptau_df
print(ptau_df.LABEL.value_counts())
print(pet_df.LABEL.value_counts())

print(ptau_df.head())

ptau_df['AGE_MCI'] = (ptau_df['MCI_DATE'] - ptau_df['DOB']).dt.days / 365.25
#create days diff from MCI to AD
ptau_df['mci_event_daysdiff'] = (ptau_df['EVENTDATE'] - ptau_df['MCI_DATE']).dt.days
print(ptau_df.head())

pet_df = pet_df.merge(ptau_df[['RID', 'AGE_MCI', 'DOB', 'mci_event_daysdiff']], on='RID', how='left')
print(pet_df.head())

pet_df['pet_days_diff'] = (pet_df['EVENTDATE'] - pet_df['PETDATE']).dt.days

#get days from ptau collection to AD diagnosis/censored for each patient in ptau_df
ptau_df['ptau_days_diff'] = (ptau_df['EVENTDATE'] - ptau_df['PTAUDATE']).dt.days

pet_df['pet_mci_days_diff'] = (pet_df['PETDATE'] - pet_df['MCI_DATE']).dt.days
plt.hist(pet_df['pet_mci_days_diff'], bins=20)
plt.xlabel('Days from MCI diagnosis to PET scan')
plt.ylabel('Number of patients')
plt.title('Distribution of days from MCI diagnosis to PET scan')

num_patients_long_diff = (pet_df['pet_mci_days_diff'] > 365*3).sum()
print("Number of patients with PET scan more than 3 years after MCI diagnosis: ", num_patients_long_diff)

print(pet_df[pet_df['pet_mci_days_diff'] > 365*3].LABEL.value_counts())

pet_df = pet_df[~((pet_df['pet_mci_days_diff'] > 365*3) & (pet_df['LABEL'] == 1))]
print(pet_df.LABEL.value_counts())

ptau_df['ptau_mci_days_diff'] = (ptau_df['PTAUDATE'] - ptau_df['MCI_DATE']).dt.days
plt.hist(ptau_df['ptau_mci_days_diff'], bins=20)
plt.xlabel('Days from MCI diagnosis to PTAU scan')
plt.ylabel('Number of patients')
plt.title('Distribution of days from MCI diagnosis to PTAU scan')

num_ptau_long_diff = (ptau_df['ptau_mci_days_diff'] > 365*6).sum()
print("Number of patients with ptau more than 3 years after MCI diagnosis: ", num_ptau_long_diff)

ptau_df = ptau_df[ptau_df['ptau_mci_days_diff'] <= 365*6]
print(ptau_df.LABEL.value_counts())

overlap_rids = set(ptau_df.RID).intersection(set(pet_df.RID))
print("Number of overlapping RIDs after dropping patients with long ptau/pet measurement times: ", len(overlap_rids))

ptau_df = ptau_df[ptau_df.RID.isin(overlap_rids)]
pet_df = pet_df[pet_df.RID.isin(overlap_rids)]

#check breakdown of LABEL in ptau_df
print(ptau_df.LABEL.value_counts())
print(pet_df.LABEL.value_counts())

ptau_df.to_parquet(f"{CLEANED_DIR}/earliest_ptau_df_rev2.parquet")
pet_df.to_parquet(f"{CLEANED_DIR}/earliest_pet_df_rev2.parquet")


# =====================================================================================
# 2 · Load the cleaned cohorts
#
# Re-entry point: if `*_rev2.parquet` already exists you can start here and skip
# Section 1. The plasma table is then joined to the raw biospecimen file to pick up
# `pT217_AB42_F`, the FDA-cleared p-tau217/Aβ42 ratio.
# =====================================================================================

ptau_df = pd.read_parquet(f"{CLEANED_DIR}/earliest_ptau_df_rev2.parquet")
pet_df = pd.read_parquet(f"{CLEANED_DIR}/earliest_pet_df_rev2.parquet")

main_path = str(RAW_DIR) + "/"
ptau_path = (
    main_path + "Biospecimen_Results/UPENN_PLASMA_FUJIREBIO_QUANTERIX_05Mar2025.csv"
)
adni_ptau = pd.read_csv(ptau_path, low_memory=False)

adni_ptau = adni_ptau[adni_ptau.RID.isin(ptau_df.RID)]
#get the first measurement date for each patient in adni_ptau
print(adni_ptau.shape)
adni_ptau = adni_ptau.sort_values(by=['RID', 'EXAMDATE']).groupby('RID').first().reset_index()
print(adni_ptau.shape)

print(ptau_df.shape)
ptau_df = ptau_df.merge(adni_ptau[['RID', 'pT217_AB42_F']], on='RID', how='left')
print(ptau_df.shape)
print(ptau_df.head())


# -------------------------------------------------------------------------------------
# 2.1 Cohort description (Table 1)
#
# Sex and *APOE4* compared with Fisher's exact / chi-squared; age and education with
# Mann–Whitney U; plus mean follow-up and the number at risk at each evaluation time.
# -------------------------------------------------------------------------------------

print("PTAU DF:")
print(ptau_df.groupby('LABEL')['APOE4'].value_counts())
print(ptau_df.groupby('LABEL')['Sex'].value_counts())

#Use Fisher's exact test (due to small sample size) to compare proportion of females in each group in ptau_df
# Create contingency table for Sex
contingency_sex = pd.crosstab(ptau_df['LABEL'], ptau_df['Sex'])
# Perform Fisher's exact test
oddsratio, p_value_sex = fisher_exact(contingency_sex)
print("Fisher's exact test for Sex:")
print("Odds Ratio:", oddsratio)
print("P-value:", p_value_sex)  

ptau_df['APOE4_binary'] = ptau_df['APOE4'].apply(lambda x: 1 if x > 0 else 0)
# Create contingency table for APOE4_binary
contingency_apoe = pd.crosstab(ptau_df['LABEL'], ptau_df['APOE4_binary'])
# Perform Fisher's exact test
oddsratio, p_value_apoe = fisher_exact(contingency_apoe)
print("Fisher's exact test for APOE4_binary:")
print("Odds Ratio:", oddsratio)
print("P-value:", p_value_apoe) 

contingency_apoe_full = pd.crosstab(ptau_df['LABEL'], ptau_df['APOE4'])
# Perform chi-squared test
chi2, p_value_apoe_full, dof, expected = chi2_contingency(contingency_apoe_full)
print("Chi-squared test for APOE4:")
print("Chi-squared statistic:", chi2)
print("P-value:", p_value_apoe_full)

print(ptau_df.groupby('LABEL')['APOE4'].value_counts())

print(ptau_df.groupby('LABEL')['AGE_MCI'].agg(['mean', 'std']))

#get mean and SD of education level for each group in ptau_df 
print(ptau_df.groupby('LABEL')['Education'].agg(['mean', 'std']))

print("Overall cohort:")
print("Age at MCI diagnosis: mean =", ptau_df['AGE_MCI'].mean(), ", std =", ptau_df['AGE_MCI'].std())
print("Education level: mean =", ptau_df['Education'].mean(), ", std =", ptau_df['Education'].std())

group0_age = ptau_df[ptau_df['LABEL'] == 0]['AGE_MCI']
group1_age = ptau_df[ptau_df['LABEL'] == 1]['AGE_MCI']
u_statistic_age, p_value_age = mannwhitneyu(group0_age, group1_age, alternative='two-sided')
print("Mann-Whitney U test for AGE_MCI:")
print("U statistic:", u_statistic_age)
print("P-value:", p_value_age)

# Compare Education between groups
group0_educ = ptau_df[ptau_df['LABEL'] == 0]['Education']
group1_educ = ptau_df[ptau_df['LABEL'] == 1]['Education']

#one-sided test to see if group 0 has higher education than group 1
u_statistic_educ, p_value_educ = mannwhitneyu(group0_educ, group1_educ, alternative='two-sided')
print("Mann-Whitney U test for Education:")
print("U statistic:", u_statistic_educ)
print("P-value:", p_value_educ)

print("Average follow-up time (days) for PTAU cohort: ", ptau_df['mci_event_daysdiff'].mean()/365.25, ptau_df['mci_event_daysdiff'].std()/365.25)
print("Average follow-up time (days) for PET cohort: ", pet_df['mci_event_daysdiff'].mean()/365.25, pet_df['mci_event_daysdiff'].std()/365.25)

print("Number of patients at each timepoint in ptau_df:")
time_bins = [0, 365.25*1, 365.25*2, 365.25*3, 365.25*4, 365.25*5, 365.25*6]
ptau_df['time_bin'] = pd.cut(ptau_df['ptau_days_diff'], bins=time_bins, labels=['0-1 year', '1-2 years', '2-3 years', '3-4 years', '4-5 years', '5-6 years'])
print(ptau_df.groupby('time_bin')['LABEL'].value_counts())  


# =====================================================================================
# 3 · Covariate sets
#
# One design matrix per model family, all sharing the same outcome
# `Surv(event = LABEL, time = mci_event_daysdiff)`. `merged_df` joins the plasma markers
# onto the PET table so that PET-plus-plasma models are fit on patients with both.
# See the model list in the docstring at the top of this file.
# =====================================================================================

random_state = 20

selected_covariates = ["AGE_MCI", "Sex", 'Education',"APOE4", 'CENTILOIDS']
Xt = pet_df[selected_covariates]
y_pet = Surv.from_arrays(event=pet_df.LABEL,
    time=pet_df.mci_event_daysdiff)

random_state = 20

selected_cov_ptau = ["AGE_MCI", "Sex", "Education","APOE4", "pT217_F"]
X_ptau = ptau_df[selected_cov_ptau]
y_ptau = Surv.from_arrays(event=ptau_df.LABEL,
    time=ptau_df.mci_event_daysdiff)  

random_state = 20

selected_cov_demo = ["AGE_MCI", "Sex", "Education","APOE4"]
X_demo = ptau_df[selected_cov_demo]
y_demo = Surv.from_arrays(event=ptau_df.LABEL,
    time=ptau_df.mci_event_daysdiff)  

merged_df = pet_df.merge(ptau_df[['RID','pT217_F','GFAP_Q','NfL_Q', 'AB42_AB40_F', 'AB42_F', 'AB40_F', 'pT217_AB42_F']], on='RID', how='left')
print(merged_df.shape)

random_state = 20

selected_cov_merged = ["AGE_MCI", "Sex", "Education","APOE4", "CENTILOIDS", "pT217_F"]
X_pet_ptau = merged_df[selected_cov_merged]
y_pet_ptau = Surv.from_arrays(event=merged_df.LABEL,
    time=merged_df.mci_event_daysdiff)  

random_state = 20

selected_cov_bio = ["AGE_MCI", "Sex", "Education","APOE4", "pT217_F", "GFAP_Q", "NfL_Q", "AB42_AB40_F"]
X_bio = ptau_df[selected_cov_bio]
y_bio = Surv.from_arrays(event=ptau_df.LABEL,
    time=ptau_df.mci_event_daysdiff)  

random_state = 20
selected_cov_ptau_amyloid = ["AGE_MCI", "Sex", "Education","APOE4", "pT217_AB42_F"]
X_ptau_amyloid = ptau_df[selected_cov_ptau_amyloid]
y_ptau_amyloid = Surv.from_arrays(event=ptau_df.LABEL,
    time=ptau_df.mci_event_daysdiff)

random_state = 20
selected_cov_ksink = ["AGE_MCI", "Sex", "Education","APOE4", "pT217_F","pT217_AB42_F", 
                             "GFAP_Q", "NfL_Q", "AB42_AB40_F", "CENTILOIDS"]
X_ksink = merged_df[selected_cov_ksink]
y_ksink = Surv.from_arrays(event=merged_df.LABEL,
    time=merged_df.mci_event_daysdiff)


# =====================================================================================
# 4 · Evaluation time grid
#
# Because AUC and Brier score are only estimable inside the range of observed event times,
# the grid is bounded by the 5th and 81st percentile of test-set event times, taken over
# all folds and intersected across the PET and plasma cohorts.
# =====================================================================================

X = Xt
y = y_pet['event']
skf = StratifiedKFold(n_splits=5, random_state=random_state, shuffle=True)
fifth_percentile_time = []
eightyfirst_percentile_time = []
for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(f"Fold {i}:")
    #print(f"  Train: index={train_index}")
    #print(f"  Test:  index={test_index}")
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y_pet[train_index], y_pet[test_index]
    #calculate 5th and 81st percentile of y_test times for all 5-folds
    fifth_percentile_time.append(np.percentile(y_test["time"], 5))
    eightyfirst_percentile_time.append(np.percentile(y_test["time"], 81))
    
#get max of value in fifth_percentile_time list
max_fifth_percentile_time = max(fifth_percentile_time)
#get min of value in eightyfirst_percentile_time list
min_eightyfirst_percentile_time = min(eightyfirst_percentile_time)
print(max_fifth_percentile_time, min_eightyfirst_percentile_time)
time = np.linspace(max_fifth_percentile_time, min_eightyfirst_percentile_time, 25)

print(time)

X = X_bio
y = y_bio['event']
skf = StratifiedKFold(n_splits=5, random_state=random_state, shuffle=True)
fifth_percentile_time = []
eightyfirst_percentile_time = []
for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(f"Fold {i}:")
    #print(f"  Train: index={train_index}")
    #print(f"  Test:  index={test_index}")
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y_bio[train_index], y_bio[test_index]
    #calculate 5th and 81st percentile of y_test times for all 5-folds
    fifth_percentile_time.append(np.percentile(y_test["time"], 5))
    eightyfirst_percentile_time.append(np.percentile(y_test["time"], 81))
    
#get max of value in fifth_percentile_time list
max_fifth_percentile_time = max(fifth_percentile_time)
#get min of value in eightyfirst_percentile_time list
min_eightyfirst_percentile_time = min(eightyfirst_percentile_time)
print(max_fifth_percentile_time, min_eightyfirst_percentile_time)
time_ptau = np.linspace(max_fifth_percentile_time, min_eightyfirst_percentile_time, 25)


# **Chosen grid: 555 – 2154 days (~1.5-5.9 years) after the first MCI diagnosis.**
# All models are evaluated on this common grid so their AUC and Brier curves are directly
# comparable.

# =====================================================================================
# 5 · Model — demographics only (reference)
#
# Pattern: fit → score → save metrics → reload metrics → summarise.
# =====================================================================================

X = X_demo
y = y_demo['event']

# Define internal CV for hyperparameter tuning
cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

#Best params (fold): {'estimator__max_depth': 3, 'estimator__max_features': 0.7, 'estimator__min_samples_leaf': 10, 'estimator__min_samples_split': 2, 'estimator__n_estimators': 300}
param_grid = {
    "estimator__n_estimators": [200, 300, 500],            # number of trees
    "estimator__min_samples_split": [2, 5],            # minimum samples to split a node
    "estimator__min_samples_leaf": [10, 15],             # minimum samples in a leaf node
    "estimator__max_depth": [3, 5],
    "estimator__max_features": [0.5, 0.7, 0.9, "sqrt"],    # features considered at each split
}

if REFIT:  # skipped when reusing stored models (MCI_AD_REUSE_MODELS_FROM)
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(f"Fold {i}:")

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y_ptau[train_index], y_ptau[test_index]

        # Define a new RSF model to tune
        rsf_model = RandomSurvivalForest(n_estimators=1000, random_state=random_state, n_jobs=-1)

        # Define the scorer
        #time[-1] is the most recent time point
        scorer = as_concordance_index_ipcw_scorer(rsf_model, tau=time[-1])

        # Perform grid search to find the best hyperparameters
        gcv_cindex = GridSearchCV(
            scorer,
            param_grid=param_grid,
            cv=cv,
            n_jobs=-1
        )
    
        gcv_cindex.fit(X_train, y_train)
        best_rsf = gcv_cindex.best_estimator_
    
        #save best_rsf model for each fold
        joblib.dump(best_rsf, f"{MODELS_DIR}/earliest_demo_rsf_fold{i}.joblib")
    
        #save predictions and labels
        train_predictions = best_rsf.predict(X_train)
        test_predictions = best_rsf.predict(X_test)
        train_chf_predictions = best_rsf.predict_cumulative_hazard_function(X_train)
        test_chf_predictions = best_rsf.predict_cumulative_hazard_function(X_test)
        train_surv_predictions = best_rsf.predict_survival_function(X_train)
        test_surv_predictions = best_rsf.predict_survival_function(X_test)


        np.savez_compressed(f"{PREDICTIONS_DIR}/earliest_demo_rsf_fold{i}_predictions.npz", 
                            train_predictions=train_predictions, 
                            test_predictions=test_predictions,
                            train_chf_predictions=train_chf_predictions,
                            test_chf_predictions=test_chf_predictions,
                            train_surv_predictions=train_surv_predictions,
                            test_surv_predictions=test_surv_predictions,
                            train_labels = y_train,
                            test_labels = y_test
                            )

X = X_demo
y = y_demo['event']

#Test Result Metrics
demo_rsf_c_index_list = []
demo_rsf_auc_list = []
demo_rsf_mean_auc_list = []
demo_rsf_chf_scores_list = []
demo_rsf_surv_scores_list = []
demo_rsf_bs_list = []
demo_rsf_ibs_list = []

#Train Result Metrics
demo_rsf_train_c_index_list = []
demo_rsf_train_auc_list = []
demo_rsf_train_mean_auc_list = []
demo_rsf_train_chf_scores_list = []
demo_rsf_train_surv_scores_list = []
demo_rsf_train_bs_list = []
demo_rsf_train_ibs_list = []

for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(f"Fold {i}:")
    #load rsf predictions for each fold
    loaded_data = np.load(f"{PREDICTIONS_IN}/earliest_demo_rsf_fold{i}_predictions.npz", allow_pickle=True)
    train_predictions = loaded_data['train_predictions']
    test_predictions = loaded_data['test_predictions']
    train_chf_predictions = loaded_data['train_chf_predictions']
    test_chf_predictions = loaded_data['test_chf_predictions']
    train_surv_predictions = loaded_data['train_surv_predictions']
    test_surv_predictions = loaded_data['test_surv_predictions']
    y_train = loaded_data['train_labels']
    y_test = loaded_data['test_labels']
    
#TEST METRICS
    #calculate concordance index on TEST data
    demo_rsf_c_index = concordance_index_ipcw(y_train, y_test, test_predictions)
    demo_rsf_c_index_list.append(demo_rsf_c_index)

    #calculate time-dependent risk scores on TEST data
    demo_rsf_chf_scores = np.vstack([chf(time) for chf in test_chf_predictions])
    demo_rsf_chf_scores_list.append(demo_rsf_chf_scores)

    #calculate time-dependent AUC and mean AUC on TEST data
    demo_rsf_auc, demo_rsf_mean_auc = cumulative_dynamic_auc(y_train, y_test, demo_rsf_chf_scores, time)
    demo_rsf_auc_list.append(demo_rsf_auc)
    demo_rsf_mean_auc_list.append(demo_rsf_mean_auc)
    
    #calculate survival probabilities on TEST data
    demo_rsf_surv_scores = np.vstack([fn(time) for fn in test_surv_predictions])
    demo_rsf_surv_scores_list.append(demo_rsf_surv_scores)
    
    #calculate Brier score on TEST data
    demo_rsf_bs = brier_score(y_train, y_test, demo_rsf_surv_scores, time)
    demo_rsf_bs_list.append(demo_rsf_bs)
    
    demo_rsf_ibs = integrated_brier_score(y_train, y_test, demo_rsf_surv_scores, time)
    demo_rsf_ibs_list.append(demo_rsf_ibs)
    
#TRAINING METRICS
    #calculate concordance index on TRAIN data
    demo_rsf_train_c_index = concordance_index_ipcw(y_train, y_train, train_predictions)
    demo_rsf_train_c_index_list.append(demo_rsf_train_c_index)
    
    #calculate time-dependent risk scores on TRAIN data
    demo_rsf_train_chf_scores = np.vstack([chf(time) for chf in train_chf_predictions])
    demo_rsf_train_chf_scores_list.append(demo_rsf_train_chf_scores)
    
    #calculate time-dependent AUC and mean AUC on TRAIN data
    demo_rsf_train_auc, demo_rsf_train_mean_auc = cumulative_dynamic_auc(y_train, y_train, demo_rsf_train_chf_scores, time)
    demo_rsf_train_auc_list.append(demo_rsf_train_auc)
    demo_rsf_train_mean_auc_list.append(demo_rsf_train_mean_auc)
    
    #calculate survival probabilities on TRAIN data
    demo_rsf_train_surv_scores = np.vstack([fn(time) for fn in train_surv_predictions])
    demo_rsf_train_surv_scores_list.append(demo_rsf_train_surv_scores)
    
    #calculate Brier score on TRAIN data
    demo_rsf_train_bs = brier_score(y_train, y_train, demo_rsf_train_surv_scores, time)
    demo_rsf_train_bs_list.append(demo_rsf_train_bs)

    demo_rsf_train_ibs = integrated_brier_score(y_train, y_train, demo_rsf_train_surv_scores, time)
    demo_rsf_train_ibs_list.append(demo_rsf_train_ibs)

np.savez_compressed(
    f"{METRICS_DIR}/demo_metrics.npz",
    rsf_c_index_list=demo_rsf_c_index_list,
    rsf_auc_list=demo_rsf_auc_list,
    rsf_mean_auc_list=demo_rsf_mean_auc_list,
    rsf_chf_scores_list=np.array(demo_rsf_chf_scores_list, dtype=object),
    rsf_surv_scores_list=np.array(demo_rsf_surv_scores_list, dtype=object),
    rsf_bs_list=demo_rsf_bs_list,
    rsf_ibs_list=demo_rsf_ibs_list,
    rsf_train_c_index_list=demo_rsf_train_c_index_list,
    rsf_train_auc_list=demo_rsf_train_auc_list,
    rsf_train_mean_auc_list=demo_rsf_train_mean_auc_list,
    rsf_train_chf_scores_list=np.array(demo_rsf_train_chf_scores_list, dtype=object),
    rsf_train_surv_scores_list=np.array(demo_rsf_train_surv_scores_list,dtype =object),
    rsf_train_bs_list=demo_rsf_train_bs_list,
    rsf_train_ibs_list=demo_rsf_train_ibs_list
)

demo_metrics = np.load(f"{METRICS_DIR}/demo_metrics.npz", allow_pickle=True)

demo_metrics_rsf_mean_auc_list = demo_metrics['rsf_mean_auc_list']
demo_rsf_c_index_list = demo_metrics['rsf_c_index_list']
demo_rsf_auc_list = demo_metrics['rsf_auc_list']
demo_rsf_chf_scores_list = demo_metrics['rsf_chf_scores_list']
demo_rsf_surv_scores_list = demo_metrics['rsf_surv_scores_list']
demo_rsf_bs_list=demo_metrics['rsf_bs_list']
demo_rsf_ibs_list=demo_metrics['rsf_ibs_list']
demo_rsf_train_c_index_list=demo_metrics['rsf_train_c_index_list']
demo_rsf_train_auc_list=demo_metrics['rsf_train_auc_list']
demo_rsf_train_mean_auc_list=demo_metrics['rsf_train_mean_auc_list']
demo_rsf_train_chf_scores_list=demo_metrics['rsf_train_chf_scores_list']
demo_rsf_train_surv_scores_list=demo_metrics['rsf_train_surv_scores_list']
demo_rsf_train_bs_list=demo_metrics['rsf_train_bs_list']
demo_rsf_train_ibs_list=demo_metrics['rsf_train_ibs_list']

demo_c_list =[]
for i in range(len(demo_rsf_c_index_list)):
    demo_rsf_c_index = demo_rsf_c_index_list[i][0]
    demo_c_list.append(demo_rsf_c_index)
    
#get mean aucs over the 25 timepoints per fold, there are no NaN values
demo_mean_auc_list = []
for i in range(len(demo_rsf_auc_list)):
    demo_mean_auc = np.nanmean(demo_rsf_auc_list[i])
    demo_mean_auc_list.append(demo_mean_auc)

demo_metrics_df = pd.DataFrame({
    'Fold': [0, 1, 2, 3, 4],
    'Test C-index': demo_c_list,
    'Train C-index': [x[0] for x in demo_rsf_train_c_index_list],
    'Test Mean AUC': [x for x in demo_mean_auc_list],
    'Train Mean AUC': [x for x in demo_rsf_train_mean_auc_list],
    'Test IBS': [x for x in demo_rsf_ibs_list],
    'Train IBS': [x for x in demo_rsf_train_ibs_list]
})
print(demo_metrics_df)

for t in time:
    total_cases = 0
    total_censored = 0
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        loaded_data = np.load(f"{PREDICTIONS_IN}/earliest_demo_rsf_fold{i}_predictions.npz", allow_pickle=True)
        y_test = loaded_data['test_labels']
        cases_at_t = np.sum((y_test['time'] <= t) & (y_test['event'] == 1))
        censored_at_t = np.sum((y_test['time'] <= t) & (y_test['event'] == 0))
        total_cases += cases_at_t
        total_censored += censored_at_t
    print(f"Time: {t}, Total Cases: {total_cases}, Total Censored: {total_censored}")

# (The per-timepoint AUC printout with bootstrap CIs is in Section 13, where the CIs are
# computed.)


# =====================================================================================
# 6 · Model — amyloid PET
#
# Pattern: fit → score → save metrics → reload metrics → summarise.
# =====================================================================================

X = Xt
y = y_pet['event']

# Define internal CV for hyperparameter tuning
cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

#Best params (fold): {'estimator__max_depth': 3, 'estimator__max_features': 0.7, 'estimator__min_samples_leaf': 10, 'estimator__min_samples_split': 2, 'estimator__n_estimators': 300}
param_grid = {
    "estimator__n_estimators": [200, 300, 500],            # number of trees
    "estimator__min_samples_split": [2, 5],            # minimum samples to split a node
    "estimator__min_samples_leaf": [10, 15],             # minimum samples in a leaf node
    "estimator__max_depth": [3, 5],
    "estimator__max_features": [0.5, 0.7, 0.9, "sqrt"],    # features considered at each split
}

if REFIT:  # skipped when reusing stored models (MCI_AD_REUSE_MODELS_FROM)
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(f"Fold {i}:")

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y_pet[train_index], y_pet[test_index]

        # Define a new RSF model to tune
        rsf_model = RandomSurvivalForest(n_estimators=1000, random_state=random_state, n_jobs=-1)

        # Define the scorer
        #time[-1] is the most recent time point
        scorer = as_concordance_index_ipcw_scorer(rsf_model, tau=time[-1])

        # Perform grid search to find the best hyperparameters
        gcv_cindex = GridSearchCV(
            scorer,
            param_grid=param_grid,
            cv=cv,
            n_jobs=-1
        )
    
        gcv_cindex.fit(X_train, y_train)
        best_rsf = gcv_cindex.best_estimator_
    
        #save best_rsf model for each fold
        joblib.dump(best_rsf, f"{MODELS_DIR}/earliest_pet_rsf_fold{i}.joblib")
    
        #save predictions and labels
        train_predictions = best_rsf.predict(X_train)
        test_predictions = best_rsf.predict(X_test)
        train_chf_predictions = best_rsf.predict_cumulative_hazard_function(X_train)
        test_chf_predictions = best_rsf.predict_cumulative_hazard_function(X_test)
        train_surv_predictions = best_rsf.predict_survival_function(X_train)
        test_surv_predictions = best_rsf.predict_survival_function(X_test)


        np.savez_compressed(f"{PREDICTIONS_DIR}/earliest_pet_rsf_fold{i}_predictions.npz", 
                            train_predictions=train_predictions, 
                            test_predictions=test_predictions,
                            train_chf_predictions=train_chf_predictions,
                            test_chf_predictions=test_chf_predictions,
                            train_surv_predictions=train_surv_predictions,
                            test_surv_predictions=test_surv_predictions,
                            train_labels = y_train,
                            test_labels = y_test
                            )

X = Xt
y = y_pet['event']

#Test Result Metrics
rsf_c_index_list = []
rsf_auc_list = []
rsf_mean_auc_list = []
rsf_chf_scores_list = []
rsf_surv_scores_list = []
rsf_bs_list = []
rsf_ibs_list = []

#Train Result Metrics
rsf_train_c_index_list = []
rsf_train_auc_list = []
rsf_train_mean_auc_list = []
rsf_train_chf_scores_list = []
rsf_train_surv_scores_list = []
rsf_train_bs_list = []
rsf_train_ibs_list = []

for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(f"Fold {i}:")
    #load rsf predictions for each fold
    loaded_data = np.load(f"{PREDICTIONS_IN}/earliest_pet_rsf_fold{i}_predictions.npz", allow_pickle=True)
    train_predictions = loaded_data['train_predictions']
    test_predictions = loaded_data['test_predictions']
    train_chf_predictions = loaded_data['train_chf_predictions']
    test_chf_predictions = loaded_data['test_chf_predictions']
    train_surv_predictions = loaded_data['train_surv_predictions']
    test_surv_predictions = loaded_data['test_surv_predictions']
    y_train = loaded_data['train_labels']
    y_test = loaded_data['test_labels']
    
#TEST METRICS
    #calculate concordance index on TEST data
    rsf_c_index = concordance_index_ipcw(y_train, y_test, test_predictions)
    rsf_c_index_list.append(rsf_c_index)

    #calculate time-dependent risk scores on TEST data
    rsf_chf_scores = np.vstack([chf(time) for chf in test_chf_predictions])
    rsf_chf_scores_list.append(rsf_chf_scores)

    #calculate time-dependent AUC and mean AUC on TEST data
    rsf_auc, rsf_mean_auc = cumulative_dynamic_auc(y_train, y_test, rsf_chf_scores, time)
    rsf_auc_list.append(rsf_auc)
    rsf_mean_auc_list.append(rsf_mean_auc)
    
    #calculate survival probabilities on TEST data
    rsf_surv_scores = np.vstack([fn(time) for fn in test_surv_predictions])
    rsf_surv_scores_list.append(rsf_surv_scores)
    
    #calculate Brier score on TEST data
    rsf_bs = brier_score(y_train, y_test, rsf_surv_scores, time)
    rsf_bs_list.append(rsf_bs)
    
    rsf_ibs = integrated_brier_score(y_train, y_test, rsf_surv_scores, time)
    rsf_ibs_list.append(rsf_ibs)
    
#TRAINING METRICS
    #calculate concordance index on TRAIN data
    rsf_train_c_index = concordance_index_ipcw(y_train, y_train, train_predictions)
    rsf_train_c_index_list.append(rsf_train_c_index)
    
    #calculate time-dependent risk scores on TRAIN data
    rsf_train_chf_scores = np.vstack([chf(time) for chf in train_chf_predictions])
    rsf_train_chf_scores_list.append(rsf_train_chf_scores)
    
    #calculate time-dependent AUC and mean AUC on TRAIN data
    rsf_train_auc, rsf_train_mean_auc = cumulative_dynamic_auc(y_train, y_train, rsf_train_chf_scores, time)
    rsf_train_auc_list.append(rsf_train_auc)
    rsf_train_mean_auc_list.append(rsf_train_mean_auc)
    
    #calculate survival probabilities on TRAIN data
    rsf_train_surv_scores = np.vstack([fn(time) for fn in train_surv_predictions])
    rsf_train_surv_scores_list.append(rsf_train_surv_scores)
    
    #calculate Brier score on TRAIN data
    rsf_train_bs = brier_score(y_train, y_train, rsf_train_surv_scores, time)
    rsf_train_bs_list.append(rsf_train_bs)

    rsf_train_ibs = integrated_brier_score(y_train, y_train, rsf_train_surv_scores, time)
    rsf_train_ibs_list.append(rsf_train_ibs)


np.savez_compressed(
    f"{METRICS_DIR}/pet_metrics.npz",
    rsf_c_index_list=rsf_c_index_list,
    rsf_auc_list=rsf_auc_list,
    rsf_mean_auc_list=rsf_mean_auc_list,
    rsf_chf_scores_list=np.array(rsf_chf_scores_list, dtype=object),
    rsf_surv_scores_list=np.array(rsf_surv_scores_list, dtype=object),
    rsf_bs_list=rsf_bs_list,
    rsf_ibs_list=rsf_ibs_list,
    rsf_train_c_index_list=rsf_train_c_index_list,
    rsf_train_auc_list=rsf_train_auc_list,
    rsf_train_mean_auc_list=rsf_train_mean_auc_list,
    rsf_train_chf_scores_list=np.array(rsf_train_chf_scores_list, dtype=object),
    rsf_train_surv_scores_list=np.array(rsf_train_surv_scores_list,dtype =object),
    rsf_train_bs_list=rsf_train_bs_list,
    rsf_train_ibs_list=rsf_train_ibs_list
)

rsf_metrics = np.load(f"{METRICS_DIR}/pet_metrics.npz", allow_pickle=True)

rsf_mean_auc_list = rsf_metrics['rsf_mean_auc_list']
rsf_c_index_list=rsf_metrics['rsf_c_index_list']
rsf_auc_list=rsf_metrics['rsf_auc_list']
rsf_mean_auc_list=rsf_metrics['rsf_mean_auc_list']
rsf_chf_scores_list=rsf_metrics['rsf_chf_scores_list']
rsf_surv_scores_list=rsf_metrics['rsf_surv_scores_list']
rsf_bs_list=rsf_metrics['rsf_bs_list']
rsf_ibs_list=rsf_metrics['rsf_ibs_list']
rsf_train_c_index_list=rsf_metrics['rsf_train_c_index_list']
rsf_train_auc_list=rsf_metrics['rsf_train_auc_list']
rsf_train_mean_auc_list=rsf_metrics['rsf_train_mean_auc_list']
rsf_train_chf_scores_list=rsf_metrics['rsf_train_chf_scores_list']
rsf_train_surv_scores_list=rsf_metrics['rsf_train_surv_scores_list']
rsf_train_bs_list=rsf_metrics['rsf_train_bs_list']
rsf_train_ibs_list=rsf_metrics['rsf_train_ibs_list']

rsf_c_list =[]
for i in range(len(rsf_c_index_list)):
    rsf_c_index = rsf_c_index_list[i][0]
    rsf_c_list.append(rsf_c_index)
    
#replace nan value in rsf_auc_list with np.nanmean(rsf_auc_list)
for i in range(len(rsf_auc_list)):
    if np.isnan(rsf_auc_list[i]).any():
        rsf_mean_auc_list[i] = np.nanmean(rsf_auc_list)

metrics_df = pd.DataFrame({
    'Fold': [0, 1, 2, 3, 4],
    'Test C-index': rsf_c_list,
    'Train C-index': [x[0] for x in rsf_train_c_index_list],
    'Test Mean AUC': [x for x in rsf_mean_auc_list],
    'Train Mean AUC': [x for x in rsf_train_mean_auc_list],
    'Test IBS': [x for x in rsf_ibs_list],
    'Train IBS': [x for x in rsf_train_ibs_list]
})
print(metrics_df)

print(rsf_train_mean_auc_list)


# =====================================================================================
# 7 · Model — plasma p-tau217
# =====================================================================================

X = X_ptau
y = y_ptau['event']

skf = StratifiedKFold(n_splits=5, random_state=random_state, shuffle=True)
fifth_percentile_time = []
eightyfirst_percentile_time = []
for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(f"Fold {i}:")
    #print(f"  Train: index={train_index}")
    #print(f"  Test:  index={test_index}")
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y_ptau[train_index], y_ptau[test_index]
    #calculate 5th and 81st percentile of y_test times for all 5-folds
    fifth_percentile_time.append(np.percentile(y_test["time"], 5))
    eightyfirst_percentile_time.append(np.percentile(y_test["time"], 81))
    
#get max of value in fifth_percentile_time list
max_fifth_percentile_time = max(fifth_percentile_time)
#get min of value in eightyfirst_percentile_time list
min_eightyfirst_percentile_time = min(eightyfirst_percentile_time)
print(max_fifth_percentile_time, min_eightyfirst_percentile_time)
time_ptau = np.linspace(max_fifth_percentile_time, min_eightyfirst_percentile_time, 25)


# Pattern: fit → score → save metrics → reload metrics → summarise.

X = X_ptau
y = y_ptau['event']

# Define internal CV for hyperparameter tuning
cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

#Best params (fold): {'estimator__max_depth': 3, 'estimator__max_features': 0.7, 'estimator__min_samples_leaf': 10, 'estimator__min_samples_split': 2, 'estimator__n_estimators': 300}
param_grid = {
    "estimator__n_estimators": [200, 300, 500],            # number of trees
    "estimator__min_samples_split": [2, 5],            # minimum samples to split a node
    "estimator__min_samples_leaf": [10, 15],             # minimum samples in a leaf node
    "estimator__max_depth": [3, 5],
    "estimator__max_features": [0.5, 0.7, 0.9, "sqrt"],    # features considered at each split
}

if REFIT:  # skipped when reusing stored models (MCI_AD_REUSE_MODELS_FROM)
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(f"Fold {i}:")

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y_ptau[train_index], y_ptau[test_index]

        # Define a new RSF model to tune
        rsf_model = RandomSurvivalForest(n_estimators=1000, random_state=random_state, n_jobs=-1)

        # Define the scorer
        #time[-1] is the most recent time point
        scorer = as_concordance_index_ipcw_scorer(rsf_model, tau=time[-1])

        # Perform grid search to find the best hyperparameters
        gcv_cindex = GridSearchCV(
            scorer,
            param_grid=param_grid,
            cv=cv,
            n_jobs=-1
        )
    
        gcv_cindex.fit(X_train, y_train)
        best_rsf = gcv_cindex.best_estimator_
    
        #save best_rsf model for each fold
        joblib.dump(best_rsf, f"{MODELS_DIR}/earliest_ptau_rsf_fold{i}.joblib")
    
        #save predictions and labels
        train_predictions = best_rsf.predict(X_train)
        test_predictions = best_rsf.predict(X_test)
        train_chf_predictions = best_rsf.predict_cumulative_hazard_function(X_train)
        test_chf_predictions = best_rsf.predict_cumulative_hazard_function(X_test)
        train_surv_predictions = best_rsf.predict_survival_function(X_train)
        test_surv_predictions = best_rsf.predict_survival_function(X_test)


        np.savez_compressed(f"{PREDICTIONS_DIR}/earliest_ptau_rsf_fold{i}_predictions.npz", 
                            train_predictions=train_predictions, 
                            test_predictions=test_predictions,
                            train_chf_predictions=train_chf_predictions,
                            test_chf_predictions=test_chf_predictions,
                            train_surv_predictions=train_surv_predictions,
                            test_surv_predictions=test_surv_predictions,
                            train_labels = y_train,
                            test_labels = y_test
                            )

X = X_ptau
y = y_ptau['event']

#Test Result Metrics
ptau_rsf_c_index_list = []
ptau_rsf_auc_list = []
ptau_rsf_mean_auc_list = []
ptau_rsf_chf_scores_list = []
ptau_rsf_surv_scores_list = []
ptau_rsf_bs_list = []
ptau_rsf_ibs_list = []

#Train Result Metrics
ptau_rsf_train_c_index_list = []
ptau_rsf_train_auc_list = []
ptau_rsf_train_mean_auc_list = []
ptau_rsf_train_chf_scores_list = []
ptau_rsf_train_surv_scores_list = []
ptau_rsf_train_bs_list = []
ptau_rsf_train_ibs_list = []

for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(f"Fold {i}:")
    #load rsf predictions for each fold
    loaded_data = np.load(f"{PREDICTIONS_IN}/earliest_ptau_rsf_fold{i}_predictions.npz", allow_pickle=True)
    train_predictions = loaded_data['train_predictions']
    test_predictions = loaded_data['test_predictions']
    train_chf_predictions = loaded_data['train_chf_predictions']
    test_chf_predictions = loaded_data['test_chf_predictions']
    train_surv_predictions = loaded_data['train_surv_predictions']
    test_surv_predictions = loaded_data['test_surv_predictions']
    y_train = loaded_data['train_labels']
    y_test = loaded_data['test_labels']
    
#TEST METRICS
    #calculate concordance index on TEST data
    ptau_rsf_c_index = concordance_index_ipcw(y_train, y_test, test_predictions)
    ptau_rsf_c_index_list.append(ptau_rsf_c_index)

    #calculate time-dependent risk scores on TEST data
    ptau_rsf_chf_scores = np.vstack([chf(time) for chf in test_chf_predictions])
    ptau_rsf_chf_scores_list.append(ptau_rsf_chf_scores)

    #calculate time-dependent AUC and mean AUC on TEST data
    ptau_rsf_auc, ptau_rsf_mean_auc = cumulative_dynamic_auc(y_train, y_test, ptau_rsf_chf_scores, time)
    ptau_rsf_auc_list.append(ptau_rsf_auc)
    ptau_rsf_mean_auc_list.append(ptau_rsf_mean_auc)
    
    #calculate survival probabilities on TEST data
    ptau_rsf_surv_scores = np.vstack([fn(time) for fn in test_surv_predictions])
    ptau_rsf_surv_scores_list.append(ptau_rsf_surv_scores)
    
    #calculate Brier score on TEST data
    ptau_rsf_bs = brier_score(y_train, y_test, ptau_rsf_surv_scores, time)
    ptau_rsf_bs_list.append(ptau_rsf_bs)
    
    ptau_rsf_ibs = integrated_brier_score(y_train, y_test, ptau_rsf_surv_scores, time)
    ptau_rsf_ibs_list.append(ptau_rsf_ibs)
    
#TRAINING METRICS
    #calculate concordance index on TRAIN data
    ptau_rsf_train_c_index = concordance_index_ipcw(y_train, y_train, train_predictions)
    ptau_rsf_train_c_index_list.append(ptau_rsf_train_c_index)
    
    #calculate time-dependent risk scores on TRAIN data
    ptau_rsf_train_chf_scores = np.vstack([chf(time) for chf in train_chf_predictions])
    ptau_rsf_train_chf_scores_list.append(ptau_rsf_train_chf_scores)
    
    #calculate time-dependent AUC and mean AUC on TRAIN data
    ptau_rsf_train_auc, ptau_rsf_train_mean_auc = cumulative_dynamic_auc(y_train, y_train, ptau_rsf_train_chf_scores, time)
    ptau_rsf_train_auc_list.append(ptau_rsf_train_auc)
    ptau_rsf_train_mean_auc_list.append(ptau_rsf_train_mean_auc)
    
    #calculate survival probabilities on TRAIN data
    ptau_rsf_train_surv_scores = np.vstack([fn(time) for fn in train_surv_predictions])
    ptau_rsf_train_surv_scores_list.append(ptau_rsf_train_surv_scores)
    
    #calculate Brier score on TRAIN data
    ptau_rsf_train_bs = brier_score(y_train, y_train, ptau_rsf_train_surv_scores, time)
    ptau_rsf_train_bs_list.append(ptau_rsf_train_bs)

    ptau_rsf_train_ibs = integrated_brier_score(y_train, y_train, ptau_rsf_train_surv_scores, time)
    ptau_rsf_train_ibs_list.append(ptau_rsf_train_ibs)

np.savez_compressed(
    f"{METRICS_DIR}/ptau_metrics.npz",
    rsf_c_index_list=ptau_rsf_c_index_list,
    rsf_auc_list=ptau_rsf_auc_list,
    rsf_mean_auc_list=ptau_rsf_mean_auc_list,
    rsf_chf_scores_list=np.array(ptau_rsf_chf_scores_list, dtype=object),
    rsf_surv_scores_list=np.array(ptau_rsf_surv_scores_list, dtype=object),
    rsf_bs_list=ptau_rsf_bs_list,
    rsf_ibs_list=ptau_rsf_ibs_list,
    rsf_train_c_index_list=ptau_rsf_train_c_index_list,
    rsf_train_auc_list=ptau_rsf_train_auc_list,
    rsf_train_mean_auc_list=ptau_rsf_train_mean_auc_list,
    rsf_train_chf_scores_list=np.array(ptau_rsf_train_chf_scores_list, dtype=object),
    rsf_train_surv_scores_list=np.array(ptau_rsf_train_surv_scores_list,dtype =object),
    rsf_train_bs_list=ptau_rsf_train_bs_list,
    rsf_train_ibs_list=ptau_rsf_train_ibs_list
)

ptau_metrics = np.load(f"{METRICS_DIR}/ptau_metrics.npz", allow_pickle=True)

ptau_rsf_mean_auc_list = ptau_metrics['rsf_mean_auc_list']
ptau_rsf_c_index_list = ptau_metrics['rsf_c_index_list']
ptau_rsf_auc_list = ptau_metrics['rsf_auc_list']
ptau_rsf_chf_scores_list = ptau_metrics['rsf_chf_scores_list']
ptau_rsf_surv_scores_list = ptau_metrics['rsf_surv_scores_list']
ptau_rsf_bs_list=ptau_metrics['rsf_bs_list']
ptau_rsf_ibs_list=ptau_metrics['rsf_ibs_list']
ptau_rsf_train_c_index_list=ptau_metrics['rsf_train_c_index_list']
ptau_rsf_train_auc_list=ptau_metrics['rsf_train_auc_list']
ptau_rsf_train_mean_auc_list=ptau_metrics['rsf_train_mean_auc_list']
ptau_rsf_train_chf_scores_list=ptau_metrics['rsf_train_chf_scores_list']
ptau_rsf_train_surv_scores_list=ptau_metrics['rsf_train_surv_scores_list']
ptau_rsf_train_bs_list=ptau_metrics['rsf_train_bs_list']
ptau_rsf_train_ibs_list=ptau_metrics['rsf_train_ibs_list']

ptau_c_list =[]
for i in range(len(ptau_rsf_c_index_list)):
    ptau_rsf_c_index = ptau_rsf_c_index_list[i][0]
    ptau_c_list.append(ptau_rsf_c_index)
    
#get mean aucs over the 25 timepoints per fold, there are no NaN values
ptau_mean_auc_list = []
for i in range(len(ptau_rsf_auc_list)):
    ptau_mean_auc = np.nanmean(ptau_rsf_auc_list[i])
    ptau_mean_auc_list.append(ptau_mean_auc)

print(ptau_rsf_mean_auc_list)

ptau_metrics_df = pd.DataFrame({
    'Fold': [0, 1, 2, 3, 4],
    'Test C-index': ptau_c_list,
    'Train C-index': [x[0] for x in ptau_rsf_train_c_index_list],
    'Test Mean AUC': [x for x in ptau_mean_auc_list],
    'Train Mean AUC': [x for x in ptau_rsf_train_mean_auc_list],
    'Test IBS': [x for x in ptau_rsf_ibs_list],
    'Train IBS': [x for x in ptau_rsf_train_ibs_list]
})
print(ptau_metrics_df)


# =====================================================================================
# 8 · Model — amyloid PET + p-tau217
#
# Pattern: fit → score → save metrics → reload metrics → summarise.
# =====================================================================================

X = X_pet_ptau
y = y_pet_ptau['event']

# Define internal CV for hyperparameter tuning
cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

#Best params (fold): {'estimator__max_depth': 3, 'estimator__max_features': 0.7, 'estimator__min_samples_leaf': 10, 'estimator__min_samples_split': 2, 'estimator__n_estimators': 300}
param_grid = {
    "estimator__n_estimators": [200, 300, 500],            # number of trees
    "estimator__min_samples_split": [2, 5],            # minimum samples to split a node
    "estimator__min_samples_leaf": [10, 15],             # minimum samples in a leaf node
    "estimator__max_depth": [3, 5],
    "estimator__max_features": [0.5, 0.7, 0.9, "sqrt"],    # features considered at each split
}

if REFIT:  # skipped when reusing stored models (MCI_AD_REUSE_MODELS_FROM)
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(f"Fold {i}:")

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y_pet[train_index], y_pet[test_index]

        # Define a new RSF model to tune
        rsf_model = RandomSurvivalForest(n_estimators=1000, random_state=random_state, n_jobs=-1)

        # Define the scorer
        #time[-1] is the most recent time point
        scorer = as_concordance_index_ipcw_scorer(rsf_model, tau=time[-1])

        # Perform grid search to find the best hyperparameters
        gcv_cindex = GridSearchCV(
            scorer,
            param_grid=param_grid,
            cv=cv,
            n_jobs=-1
        )
    
        gcv_cindex.fit(X_train, y_train)
        best_rsf = gcv_cindex.best_estimator_
    
        #save best_rsf model for each fold
        joblib.dump(best_rsf, f"{MODELS_DIR}/earliest_pet_ptau_rsf_fold{i}.joblib")
    
        #save predictions and labels
        train_predictions = best_rsf.predict(X_train)
        test_predictions = best_rsf.predict(X_test)
        train_chf_predictions = best_rsf.predict_cumulative_hazard_function(X_train)
        test_chf_predictions = best_rsf.predict_cumulative_hazard_function(X_test)
        train_surv_predictions = best_rsf.predict_survival_function(X_train)
        test_surv_predictions = best_rsf.predict_survival_function(X_test)


        np.savez_compressed(f"{PREDICTIONS_DIR}/earliest_pet_ptau_rsf_fold{i}_predictions.npz", 
                            train_predictions=train_predictions, 
                            test_predictions=test_predictions,
                            train_chf_predictions=train_chf_predictions,
                            test_chf_predictions=test_chf_predictions,
                            train_surv_predictions=train_surv_predictions,
                            test_surv_predictions=test_surv_predictions,
                            train_labels = y_train,
                            test_labels = y_test
                            )

merged_rsf_c_index_list = []
merged_rsf_auc_list = []
merged_rsf_mean_auc_list = []
merged_rsf_chf_scores_list = []
merged_rsf_surv_scores_list = []
merged_rsf_bs_list = []
merged_rsf_ibs_list = []

#Train Result Metrics
merged_rsf_train_c_index_list = []
merged_rsf_train_auc_list = []
merged_rsf_train_mean_auc_list = []
merged_rsf_train_chf_scores_list = []
merged_rsf_train_surv_scores_list = []
merged_rsf_train_bs_list = []
merged_rsf_train_ibs_list = []

for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(f"Fold {i}:")
    #load rsf predictions for each fold
    loaded_data = np.load(f"{PREDICTIONS_IN}/earliest_pet_ptau_rsf_fold{i}_predictions.npz", allow_pickle=True)
    train_predictions = loaded_data['train_predictions']
    test_predictions = loaded_data['test_predictions']
    train_chf_predictions = loaded_data['train_chf_predictions']
    test_chf_predictions = loaded_data['test_chf_predictions']
    train_surv_predictions = loaded_data['train_surv_predictions']
    test_surv_predictions = loaded_data['test_surv_predictions']
    y_train = loaded_data['train_labels']
    y_test = loaded_data['test_labels']
    
#TEST METRICS
    #calculate concordance index on TEST data
    merged_rsf_c_index = concordance_index_ipcw(y_train, y_test, test_predictions)
    merged_rsf_c_index_list.append(merged_rsf_c_index)

    #calculate time-dependent risk scores on TEST data
    merged_rsf_chf_scores = np.vstack([chf(time) for chf in test_chf_predictions])
    merged_rsf_chf_scores_list.append(merged_rsf_chf_scores)

    #calculate time-dependent AUC and mean AUC on TEST data
    merged_rsf_auc, merged_rsf_mean_auc = cumulative_dynamic_auc(y_train, y_test, merged_rsf_chf_scores, time)
    merged_rsf_auc_list.append(merged_rsf_auc)
    merged_rsf_mean_auc_list.append(merged_rsf_mean_auc)
    
    #calculate survival probabilities on TEST data
    merged_rsf_surv_scores = np.vstack([fn(time) for fn in test_surv_predictions])
    merged_rsf_surv_scores_list.append(merged_rsf_surv_scores)
    
    #calculate Brier score on TEST data
    merged_rsf_bs = brier_score(y_train, y_test, merged_rsf_surv_scores, time)
    merged_rsf_bs_list.append(merged_rsf_bs)
    
    merged_rsf_ibs = integrated_brier_score(y_train, y_test, merged_rsf_surv_scores, time)
    merged_rsf_ibs_list.append(merged_rsf_ibs)
    
#TRAINING METRICS
    #calculate concordance index on TRAIN data
    merged_rsf_train_c_index = concordance_index_ipcw(y_train, y_train, train_predictions)
    merged_rsf_train_c_index_list.append(merged_rsf_train_c_index)
    
    #calculate time-dependent risk scores on TRAIN data
    merged_rsf_train_chf_scores = np.vstack([chf(time) for chf in train_chf_predictions])
    merged_rsf_train_chf_scores_list.append(merged_rsf_train_chf_scores)
    
    #calculate time-dependent AUC and mean AUC on TRAIN data
    merged_rsf_train_auc, merged_rsf_train_mean_auc = cumulative_dynamic_auc(y_train, y_train, merged_rsf_train_chf_scores, time)
    merged_rsf_train_auc_list.append(merged_rsf_train_auc)
    merged_rsf_train_mean_auc_list.append(merged_rsf_train_mean_auc)
    
    #calculate survival probabilities on TRAIN data
    merged_rsf_train_surv_scores = np.vstack([fn(time) for fn in train_surv_predictions])
    merged_rsf_train_surv_scores_list.append(merged_rsf_train_surv_scores)
    
    #calculate Brier score on TRAIN data
    merged_rsf_train_bs = brier_score(y_train, y_train, merged_rsf_train_surv_scores, time)
    merged_rsf_train_bs_list.append(merged_rsf_train_bs)

    merged_rsf_train_ibs = integrated_brier_score(y_train, y_train, merged_rsf_train_surv_scores, time)
    merged_rsf_train_ibs_list.append(merged_rsf_train_ibs)

np.savez_compressed(
    f"{METRICS_DIR}/pet_ptau_metrics.npz",
    rsf_c_index_list=merged_rsf_c_index_list,
    rsf_auc_list=merged_rsf_auc_list,
    rsf_mean_auc_list=merged_rsf_mean_auc_list,
    rsf_chf_scores_list=np.array(merged_rsf_chf_scores_list, dtype=object),
    rsf_surv_scores_list=np.array(merged_rsf_surv_scores_list, dtype=object),
    rsf_bs_list=merged_rsf_bs_list,
    rsf_ibs_list=merged_rsf_ibs_list,
    rsf_train_c_index_list=merged_rsf_train_c_index_list,
    rsf_train_auc_list=merged_rsf_train_auc_list,
    rsf_train_mean_auc_list=merged_rsf_train_mean_auc_list,
    rsf_train_chf_scores_list=np.array(merged_rsf_train_chf_scores_list, dtype=object),
    rsf_train_surv_scores_list=np.array(merged_rsf_train_surv_scores_list,dtype =object),
    rsf_train_bs_list=merged_rsf_train_bs_list,
    rsf_train_ibs_list=merged_rsf_train_ibs_list
)

merged_metrics = np.load(f"{METRICS_DIR}/pet_ptau_metrics.npz", allow_pickle=True)

merged_rsf_mean_auc_list = merged_metrics['rsf_mean_auc_list']
merged_rsf_c_index_list = merged_metrics['rsf_c_index_list']
merged_rsf_auc_list = merged_metrics['rsf_auc_list']
merged_rsf_chf_scores_list = merged_metrics['rsf_chf_scores_list']
merged_rsf_surv_scores_list = merged_metrics['rsf_surv_scores_list']
merged_rsf_bs_list=merged_metrics['rsf_bs_list']
merged_rsf_ibs_list=merged_metrics['rsf_ibs_list']
merged_rsf_train_c_index_list=merged_metrics['rsf_train_c_index_list']
merged_rsf_train_auc_list=merged_metrics['rsf_train_auc_list']
merged_rsf_train_mean_auc_list=merged_metrics['rsf_train_mean_auc_list']
merged_rsf_train_chf_scores_list=merged_metrics['rsf_train_chf_scores_list']
merged_rsf_train_surv_scores_list=merged_metrics['rsf_train_surv_scores_list']
merged_rsf_train_bs_list=merged_metrics['rsf_train_bs_list']
merged_rsf_train_ibs_list=merged_metrics['rsf_train_ibs_list']

merged_c_list =[]
for i in range(len(merged_rsf_c_index_list)):
    merged_rsf_c_index = merged_rsf_c_index_list[i][0]
    merged_c_list.append(merged_rsf_c_index)
    
#get mean aucs over the 25 timepoints per fold, there are no NaN values
merged_mean_auc_list = []
for i in range(len(merged_rsf_auc_list)):
    merged_mean_auc = np.nanmean(merged_rsf_auc_list[i])
    merged_mean_auc_list.append(merged_mean_auc)

merged_metrics_df = pd.DataFrame({
    'Fold': [0, 1, 2, 3, 4],
    'Test C-index': merged_c_list,
    'Train C-index': [x[0] for x in merged_rsf_train_c_index_list],
    'Test Mean AUC': [x for x in merged_mean_auc_list],
    'Train Mean AUC': [x for x in merged_rsf_train_mean_auc_list],
    'Test IBS': [x for x in merged_rsf_ibs_list],
    'Train IBS': [x for x in merged_rsf_train_ibs_list]
})
print(merged_metrics_df)


# =====================================================================================
# 9 · Model — blood biomarker panel (p-tau217, GFAP, NfL, Aβ42/40)
# =====================================================================================


# -------------------------------------------------------------------------------------
# 9.1 Biomarker exploratory analysis
#
# Before fitting: Spearman correlations among the plasma markers and age (permutation
# p-values), baseline distributions by sMCI/pMCI with Mann–Whitney tests, and
# biomarker-versus-age scatter plots by outcome group.
# -------------------------------------------------------------------------------------

ptau_df.isna().sum()

# The strongest correlations were observed between NfL and GFAP levels (Spearman's
# rho = 0.63), p-tau217 and GFAP levels (Spearman's rho = 0.61), and p-tau217 and NfL
# levels (Spearman's rho = 0.49).

def permutation_test(x, y, num_permutations=1000):
    observed_corr, _ = spearmanr(x, y)
    count = 0
    for _ in range(num_permutations):
        permuted_y = np.random.permutation(y)
        permuted_corr, _ = spearmanr(x, permuted_y)
        if abs(permuted_corr) >= abs(observed_corr):
            count += 1
    p_value = count / num_permutations
    return observed_corr, p_value
# NfL and GFAP
nfl_gfap_corr_perm, nfl_gfap_p_perm = permutation_test(ptau_df['NfL_Q'], ptau_df['GFAP_Q'])
print(f"Permutation test - Spearman correlation between NfL and GFAP: {nfl_gfap_corr_perm:.3f}, p-value: {nfl_gfap_p_perm:.3f}")

# p-Tau217 and GFAP
ptau_gfap_corr_perm, ptau_gfap_p_perm = permutation_test(ptau_df['pT217_F'], ptau_df['GFAP_Q'])
print(f"Permutation test - Spearman correlation between p-Tau217 and GFAP: {ptau_gfap_corr_perm:.3f}, p-value: {ptau_gfap_p_perm:.3f}")

# p-Tau217 and NfL
ptau_nfl_corr_perm, ptau_nfl_p_perm = permutation_test(ptau_df['pT217_F'], ptau_df['NfL_Q'])
print(f"Permutation test - Spearman correlation between p-Tau217 and NfL: {ptau_nfl_corr_perm:.3f}, p-value: {ptau_nfl_p_perm:.3f}")        

#Age and NfL
age_nfl_corr_perm, age_nfl_p_perm = permutation_test(ptau_df['AGE_MCI'], ptau_df['NfL_Q'])
print(f"Permutation test - Spearman correlation between Age and NfL: {age_nfl_corr_perm:.3f}, p-value: {age_nfl_p_perm:.3f}")

#Age and ptau217
age_ptau217_corr_perm, age_ptau217_p_perm = permutation_test(ptau_df['AGE_MCI'], ptau_df['pT217_F'])
print(f"Permutation test -  Spearman correlation between Age and p-Tau217: {age_ptau217_corr_perm:.3f}, p-value: {age_ptau217_p_perm:.3f}") 

#Age and GFAP   
age_gfap_corr_perm, age_gfap_p_perm = permutation_test(ptau_df['AGE_MCI'], ptau_df['GFAP_Q'])
print(f"Permutation test - Spearman correlation between Age and GFAP: {age_gfap_corr_perm:.3f}, p-value: {age_gfap_p_perm:.3f}")

biomarker_cols = ["AB42_AB40_F", "pT217_F", "NfL_Q", "GFAP_Q", "pT217_AB42_F"]
p_values = {}

for col in biomarker_cols:
    group0 = ptau_df[ptau_df['LABEL'] == 0][col]
    group1 = ptau_df[ptau_df['LABEL'] == 1][col]
    stat, p_val = stats.mannwhitneyu(group0, group1, alternative='two-sided')
    p_values[col] = p_val

print("P-values for each biomarker:")
for col, p_val in p_values.items():
    print(f"{col}: {p_val:.6f}")
    
#Make mini boxplots all in one figure that show the median (center line) and interquartile range (bounds of box) as well as the 2.5th and 97.5th percentiles (whiskers)
plt.figure(figsize=(12, 8))
for i, col in enumerate(biomarker_cols):
    plt.subplot(2, 3, i+1)
    sns.boxplot(x='LABEL', y=col, data=ptau_df, palette=['lightblue', 'darkblue'], hue='LABEL')
    #create an overall title for the whole figure and no title for subplots
    plt.suptitle("Distribution of Biomarkers by Stable vs. Progressive MCI", y=1.02)
    plt.xlabel("MCI Type (0 = Stable, 1 = Progressive)")
    plt.ylabel(col)
    if col == 'AB42_AB40_F':
        plt.ylim(0, 0.2)
    if col == "pT217_AB42_F":
        plt.ylim(0, 0.175)
plt.tight_layout()

biomarker_cols = ["AB42_AB40_F", "pT217_F", "NfL_Q", "GFAP_Q", "pT217_AB42_F"]

# Clean display names
rename_map = {
    "AB42_AB40_F": "Aβ42/40",
    "pT217_F": "p-Tau217",
    "NfL_Q": "NfL",
    "GFAP_Q": "GFAP",
    "pT217_AB42_F": "p-Tau217/Aβ42"
}

p_values = {}

for col in biomarker_cols:
    group0 = ptau_df[ptau_df['LABEL'] == 0][col]
    group1 = ptau_df[ptau_df['LABEL'] == 1][col]
    stat, p_val = stats.mannwhitneyu(group0, group1, alternative='two-sided')
    p_values[col] = p_val

print("P-values for each biomarker:")
for col, p_val in p_values.items():
    print(f"{rename_map[col]}: {p_val:.6f}")
    
biomarker_cols = ["AB42_AB40_F", "pT217_F", "NfL_Q", "GFAP_Q", "pT217_AB42_F"]

rename_map = {
    "AB42_AB40_F": "Aβ42/40",
    "pT217_F": "p-Tau217",
    "NfL_Q": "NfL",
    "GFAP_Q": "GFAP",
    "pT217_AB42_F": "p-Tau217/Aβ42"
}

# Adjusted comparison: linear regression controlling for age, sex, APOE4
p_values = {}
coefficients = {}

for col in biomarker_cols:
    df_reg = ptau_df[[col, 'LABEL', 'AGE_MCI', 'Sex', 'APOE4']].dropna()
    X = df_reg[['LABEL', 'AGE_MCI', 'Sex', 'APOE4']]
    X = sm.add_constant(X)
    y = df_reg[col]
    
    model = sm.OLS(y, X).fit()
    p_values[col] = model.pvalues['LABEL']
    coefficients[col] = model.params['LABEL']

print("Adjusted p-values (controlling for age, sex, APOE4):")
for col in biomarker_cols:
    print(f"{rename_map[col]}: coef = {coefficients[col]:.4f}, p = {p_values[col]:.6f}")

# Boxplots
fig = plt.figure(figsize=(14, 9))

for i, col in enumerate(biomarker_cols):
    ax = plt.subplot(2, 3, i+1)
    sns.boxplot(x='LABEL', y=col, data=ptau_df, palette=['#A8C8E8', '#2B5C8A'], hue='LABEL', legend=False)
    
    ax.set_xticklabels(['sMCI', 'pMCI'], fontsize=11)
    ax.set_xlabel('')
    ax.set_ylabel(rename_map[col], fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if col == 'AB42_AB40_F':
        ax.set_ylim(0, 0.2)
    if col == "pT217_AB42_F":
        ax.set_ylim(0, 0.175)
    
    p_val = p_values[col]
    if p_val < 0.001:
        sig = '***'
    elif p_val < 0.01:
        sig = '**'
    elif p_val < 0.05:
        sig = '*'
    else:
        sig = None
    
    y_low, y_high = ax.get_ylim()
    y_range = y_high - y_low
    bracket_y = y_high - y_range * 0.08
    bar_height = y_range * 0.02
    
    ax.plot([0, 0, 1, 1],
            [bracket_y, bracket_y + bar_height, bracket_y + bar_height, bracket_y],
            color='black', linewidth=1.2)
    ax.text(0.5, bracket_y + bar_height + y_range * 0.01, sig,
            ha='center', va='bottom', fontsize=13, fontweight='bold')

#plt.suptitle("Distribution of Biomarkers by Stable vs. Progressive MCI",
            #fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/biomarker_boxplots.png', dpi=300, bbox_inches='tight')

print("\n* p < 0.05, ** p < 0.01, *** p < 0.001")
print("(OLS regression adjusted for age at MCI, sex, and APOE ε4)")

#same scatterplot in black and white for print: stable MCI as open circles with a
#dashed trendline, progressive MCI as filled circles with a solid trendline
#note: scatter_kws needs 'linewidths' (plural) -- seaborn already passes 'linewidth'
#to ax.scatter, and matplotlib rejects both aliases at once
plt.figure(figsize=(8, 6))

sns.regplot(x='AGE_MCI', y='pT217_F', data=ptau_df[ptau_df['LABEL'] == 0],
            scatter=True, color='black', label='Stable MCI',
            line_kws={'linestyle': '--'},
            scatter_kws={'facecolor': 'white', 'edgecolor': 'black', 's': 40, 'linewidths': 1})

sns.regplot(x='AGE_MCI', y='pT217_F', data=ptau_df[ptau_df['LABEL'] == 1],
            scatter=True, color='black', label='Progressive MCI',
            scatter_kws={'facecolor': 'black', 'edgecolor': 'black', 's': 40})

plt.ylabel("p-tau217 Levels (pg/mL)")
plt.xlabel("Age at Baseline")
plt.legend(title="MCI Type")
plt.savefig(f'{FIGURES_DIR}/ptau217_age_scatter_bw.pdf', bbox_inches='tight')

ratio_plot_df = ptau_df[ptau_df["pT217_AB42_F"]> 0]
plt.figure(figsize=(8, 6))
plt.ylim(0, 0.175)
#sns.scatterplot(x='AGE_MCI', y='pT217_F', hue='LABEL', data=ptau_df, palette=['lightblue', 'darkblue'])
#add in trendlines for each group
sns.regplot(x='AGE_MCI', y='pT217_AB42_F', data=ratio_plot_df[ratio_plot_df['LABEL'] == 0], scatter=True, color='lightblue', label='Stable MCI Trendline')
sns.regplot(x='AGE_MCI', y='pT217_AB42_F', data=ratio_plot_df[ratio_plot_df['LABEL'] == 1], scatter=True, color='darkblue', label='Progressive MCI Trendline')  
plt.title("Correlation between p-tau217/AB42 levels and AGE_MCI for Stable vs. Progressive MCI")
plt.ylabel("p-tau217/AB42")
plt.xlabel("Age at Baseline")
#plt.legend(title="MCI Type", labels=["Progressive MCI", "Stable MCI"])

plt.figure(figsize=(8, 6))
#sns.scatterplot(x='AGE_MCI', y='pT217_F', hue='LABEL', data=ptau_df, palette=['lightblue', 'darkblue'])
#add in trendlines for each group
sns.regplot(x='AGE_MCI', y='NfL_Q', data=ptau_df[ptau_df['LABEL'] == 0], scatter=True, color='lightblue', label='Stable MCI Trendline')
sns.regplot(x='AGE_MCI', y='NfL_Q', data=ptau_df[ptau_df['LABEL'] == 1], scatter=True, color='darkblue', label='Progressive MCI Trendline')  
plt.title("Correlation between p-tau217 levels and AGE_MCI for Stable vs. Progressive MCI")
plt.ylabel("NfL_Q Levels (pg/mL)")
plt.xlabel("Age at Baseline")
#plt.legend(title="MCI Type", labels=["Progressive MCI", "Stable MCI"])

plt.figure(figsize=(8, 6))
#sns.scatterplot(x='AGE_MCI', y='pT217_F', hue='LABEL', data=ptau_df, palette=['lightblue', 'darkblue'])
#add in trendlines for each group
sns.regplot(x='AGE_MCI', y='GFAP_Q', data=ptau_df[ptau_df['LABEL'] == 0], scatter=True, color='lightblue', label='Stable MCI Trendline')
sns.regplot(x='AGE_MCI', y='GFAP_Q', data=ptau_df[ptau_df['LABEL'] == 1], scatter=True, color='darkblue', label='Progressive MCI Trendline')  
plt.title("Correlation between GFAP_Q levels and AGE_MCI for Stable vs. Progressive MCI")
plt.ylabel("GFAP_Q Levels (pg/mL)")
plt.xlabel("Age at Baseline")
#plt.legend(title="MCI Type", labels=["Progressive MCI", "Stable MCI"])



# **Reading the scatter plots.**
#
# - p-Tau217 levels are higher at baseline in pMCI patients.
# - The stable and progressive distributions overlap — these people may progress after the
#   observed follow-up window, and it may also reflect people who carry amyloid pathology
#   but never develop AD dementia.

print(ptau_df.LABEL.value_counts())

plt.figure(figsize=(8, 6))
sns.scatterplot(x='pT217_F', y='AGE_MCI', hue='LABEL', data=ptau_df, palette=['lightblue', 'darkblue'])
#add in trendlines for each group
sns.regplot(x='pT217_F', y='AGE_MCI', data=ptau_df[ptau_df['LABEL'] == 0], scatter=False, color='lightblue', label='Stable MCI Trendline')
sns.regplot(x='pT217_F', y='AGE_MCI', data=ptau_df[ptau_df['LABEL'] == 1], scatter=False, color='darkblue', label='Progressive MCI Trendline')  
plt.title("Correlation between p-tau217 levels and AGE_MCI for Stable vs. Progressive MCI")
plt.xlabel("p-tau217 Levels (pg/mL)")
plt.ylabel("Age at Baseline")
plt.legend(title="MCI Type", labels=["Stable MCI", "Progressive MCI"])


# **Summary.** Individuals who progress from MCI to Alzheimer's dementia have higher
# baseline values of p-tau217, NfL and GFAP than those who do not (p < 0.05). Baseline
# amyloid-β 42/40 was slightly higher in stable MCI individuals.


# -------------------------------------------------------------------------------------
# 9.2 Fit the panel model
#
# Pattern: fit → score → save metrics → reload metrics → summarise.
# -------------------------------------------------------------------------------------

X = X_bio
y = y_bio['event']

# Define internal CV for hyperparameter tuning
cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

#Best params (fold): {'estimator__max_depth': 3, 'estimator__max_features': 0.7, 'estimator__min_samples_leaf': 10, 'estimator__min_samples_split': 2, 'estimator__n_estimators': 300}
param_grid = {
    "estimator__n_estimators": [200, 300, 500],            # number of trees
    "estimator__min_samples_split": [2, 5],            # minimum samples to split a node
    "estimator__min_samples_leaf": [10, 15],             # minimum samples in a leaf node
    "estimator__max_depth": [3, 5],
    "estimator__max_features": [0.5, 0.7, 0.9, "sqrt"],    # features considered at each split
}

if REFIT:  # skipped when reusing stored models (MCI_AD_REUSE_MODELS_FROM)
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(f"Fold {i}:")

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y_bio[train_index], y_bio[test_index]

        # Define a new RSF model to tune
        rsf_model = RandomSurvivalForest(n_estimators=1000, random_state=random_state, n_jobs=-1)

        # Define the scorer
        #time[-1] is the most recent time point
        scorer = as_concordance_index_ipcw_scorer(rsf_model, tau=time[-1])

        # Perform grid search to find the best hyperparameters
        gcv_cindex = GridSearchCV(
            scorer,
            param_grid=param_grid,
            cv=cv,
            n_jobs=-1
        )
    
        gcv_cindex.fit(X_train, y_train)
        best_rsf = gcv_cindex.best_estimator_
    
        #save best_rsf model for each fold
        joblib.dump(best_rsf, f"{MODELS_DIR}/earliest_bio_rsf_fold{i}.joblib")
    
        #save predictions and labels
        train_predictions = best_rsf.predict(X_train)
        test_predictions = best_rsf.predict(X_test)
        train_chf_predictions = best_rsf.predict_cumulative_hazard_function(X_train)
        test_chf_predictions = best_rsf.predict_cumulative_hazard_function(X_test)
        train_surv_predictions = best_rsf.predict_survival_function(X_train)
        test_surv_predictions = best_rsf.predict_survival_function(X_test)


        np.savez_compressed(f"{PREDICTIONS_DIR}/earliest_bio_rsf_fold{i}_predictions.npz", 
                            train_predictions=train_predictions, 
                            test_predictions=test_predictions,
                            train_chf_predictions=train_chf_predictions,
                            test_chf_predictions=test_chf_predictions,
                            train_surv_predictions=train_surv_predictions,
                            test_surv_predictions=test_surv_predictions,
                            train_labels = y_train,
                            test_labels = y_test
                            )

X = X_bio
y = y_bio['event']

#Test Result Metrics
bio_rsf_c_index_list = []
bio_rsf_auc_list = []
bio_rsf_mean_auc_list = []
bio_rsf_chf_scores_list = []
bio_rsf_surv_scores_list = []
bio_rsf_bs_list = []
bio_rsf_ibs_list = []

#Train Result Metrics
bio_rsf_train_c_index_list = []
bio_rsf_train_auc_list = []
bio_rsf_train_mean_auc_list = []
bio_rsf_train_chf_scores_list = []
bio_rsf_train_surv_scores_list = []
bio_rsf_train_bs_list = []
bio_rsf_train_ibs_list = []

for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(f"Fold {i}:")
    #load rsf predictions for each fold
    loaded_data = np.load(f"{PREDICTIONS_IN}/earliest_bio_rsf_fold{i}_predictions.npz", allow_pickle=True)
    train_predictions = loaded_data['train_predictions']
    test_predictions = loaded_data['test_predictions']
    train_chf_predictions = loaded_data['train_chf_predictions']
    test_chf_predictions = loaded_data['test_chf_predictions']
    train_surv_predictions = loaded_data['train_surv_predictions']
    test_surv_predictions = loaded_data['test_surv_predictions']
    y_train = loaded_data['train_labels']
    y_test = loaded_data['test_labels']
    
#TEST METRICS
    #calculate concordance index on TEST data
    bio_rsf_c_index = concordance_index_ipcw(y_train, y_test, test_predictions)
    bio_rsf_c_index_list.append(bio_rsf_c_index)

    #calculate time-dependent risk scores on TEST data
    bio_rsf_chf_scores = np.vstack([chf(time) for chf in test_chf_predictions])
    bio_rsf_chf_scores_list.append(bio_rsf_chf_scores)

    #calculate time-dependent AUC and mean AUC on TEST data
    bio_rsf_auc, bio_rsf_mean_auc = cumulative_dynamic_auc(y_train, y_test, bio_rsf_chf_scores, time)
    bio_rsf_auc_list.append(bio_rsf_auc)
    bio_rsf_mean_auc_list.append(bio_rsf_mean_auc)
    
    #calculate survival probabilities on TEST data
    bio_rsf_surv_scores = np.vstack([fn(time) for fn in test_surv_predictions])
    bio_rsf_surv_scores_list.append(bio_rsf_surv_scores)
    
    #calculate Brier score on TEST data
    bio_rsf_bs = brier_score(y_train, y_test, bio_rsf_surv_scores, time)
    bio_rsf_bs_list.append(bio_rsf_bs)
    
    bio_rsf_ibs = integrated_brier_score(y_train, y_test, bio_rsf_surv_scores, time)
    bio_rsf_ibs_list.append(bio_rsf_ibs)
    
#TRAINING METRICS
    #calculate concordance index on TRAIN data
    bio_rsf_train_c_index = concordance_index_ipcw(y_train, y_train, train_predictions)
    bio_rsf_train_c_index_list.append(bio_rsf_train_c_index)
    
    #calculate time-dependent risk scores on TRAIN data
    bio_rsf_train_chf_scores = np.vstack([chf(time) for chf in train_chf_predictions])
    bio_rsf_train_chf_scores_list.append(bio_rsf_train_chf_scores)
    
    #calculate time-dependent AUC and mean AUC on TRAIN data
    bio_rsf_train_auc, bio_rsf_train_mean_auc = cumulative_dynamic_auc(y_train, y_train, bio_rsf_train_chf_scores, time)
    bio_rsf_train_auc_list.append(bio_rsf_train_auc)
    bio_rsf_train_mean_auc_list.append(bio_rsf_train_mean_auc)
    
    #calculate survival probabilities on TRAIN data
    bio_rsf_train_surv_scores = np.vstack([fn(time) for fn in train_surv_predictions])
    bio_rsf_train_surv_scores_list.append(bio_rsf_train_surv_scores)
    
    #calculate Brier score on TRAIN data
    bio_rsf_train_bs = brier_score(y_train, y_train, bio_rsf_train_surv_scores, time)
    bio_rsf_train_bs_list.append(bio_rsf_train_bs)

    bio_rsf_train_ibs = integrated_brier_score(y_train, y_train, bio_rsf_train_surv_scores, time)
    bio_rsf_train_ibs_list.append(bio_rsf_train_ibs)

np.savez_compressed(
    f"{METRICS_DIR}/bio_metrics.npz",
    rsf_c_index_list=bio_rsf_c_index_list,
    rsf_auc_list=bio_rsf_auc_list,
    rsf_mean_auc_list=bio_rsf_mean_auc_list,
    rsf_chf_scores_list=np.array(bio_rsf_chf_scores_list, dtype=object),
    rsf_surv_scores_list=np.array(bio_rsf_surv_scores_list, dtype=object),
    rsf_bs_list=bio_rsf_bs_list,
    rsf_ibs_list=bio_rsf_ibs_list,
    rsf_train_c_index_list=bio_rsf_train_c_index_list,
    rsf_train_auc_list=bio_rsf_train_auc_list,
    rsf_train_mean_auc_list=bio_rsf_train_mean_auc_list,
    rsf_train_chf_scores_list=np.array(bio_rsf_train_chf_scores_list, dtype=object),
    rsf_train_surv_scores_list=np.array(bio_rsf_train_surv_scores_list,dtype =object),
    rsf_train_bs_list=bio_rsf_train_bs_list,
    rsf_train_ibs_list=bio_rsf_train_ibs_list
)

bio_metrics = np.load(f"{METRICS_DIR}/bio_metrics.npz", allow_pickle=True)

bio_rsf_mean_auc_list = bio_metrics['rsf_mean_auc_list']
bio_rsf_c_index_list = bio_metrics['rsf_c_index_list']
bio_rsf_auc_list = bio_metrics['rsf_auc_list']
bio_rsf_chf_scores_list = bio_metrics['rsf_chf_scores_list']
bio_rsf_surv_scores_list = bio_metrics['rsf_surv_scores_list']
bio_rsf_bs_list=bio_metrics['rsf_bs_list']
bio_rsf_ibs_list=bio_metrics['rsf_ibs_list']
bio_rsf_train_c_index_list=bio_metrics['rsf_train_c_index_list']
bio_rsf_train_auc_list=bio_metrics['rsf_train_auc_list']
bio_rsf_train_mean_auc_list=bio_metrics['rsf_train_mean_auc_list']
bio_rsf_train_chf_scores_list=bio_metrics['rsf_train_chf_scores_list']
bio_rsf_train_surv_scores_list=bio_metrics['rsf_train_surv_scores_list']
bio_rsf_train_bs_list=bio_metrics['rsf_train_bs_list']
bio_rsf_train_ibs_list=bio_metrics['rsf_train_ibs_list']

bio_c_list =[]
for i in range(len(bio_rsf_c_index_list)):
    bio_rsf_c_index = bio_rsf_c_index_list[i][0]
    bio_c_list.append(bio_rsf_c_index)
    
#get mean aucs over the 25 timepoints per fold, there are no NaN values
bio_mean_auc_list = []
for i in range(len(bio_rsf_auc_list)):
    bio_mean_auc = np.nanmean(bio_rsf_auc_list[i])
    bio_mean_auc_list.append(bio_mean_auc)

bio_metrics_df = pd.DataFrame({
    'Fold': [0, 1, 2, 3, 4],
    'Test C-index': bio_c_list,
    'Train C-index': [x[0] for x in bio_rsf_train_c_index_list],
    'Test Mean AUC': [x for x in bio_mean_auc_list],
    'Train Mean AUC': [x for x in bio_rsf_train_mean_auc_list],
    'Test IBS': [x for x in bio_rsf_ibs_list],
    'Train IBS': [x for x in bio_rsf_train_ibs_list]
})
print(bio_metrics_df)


# =====================================================================================
# 10 · Model — p-tau217/Aβ42 ratio
#
# Pattern: fit → score → save metrics → reload metrics → summarise.
# =====================================================================================

X = X_ptau_amyloid
y = y_ptau_amyloid['event']

# Define internal CV for hyperparameter tuning
cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

#Best params (fold): {'estimator__max_depth': 3, 'estimator__max_features': 0.7, 'estimator__min_samples_leaf': 10, 'estimator__min_samples_split': 2, 'estimator__n_estimators': 300}
param_grid = {
    "estimator__n_estimators": [200, 300, 500],            # number of trees
    "estimator__min_samples_split": [2, 5],            # minimum samples to split a node
    "estimator__min_samples_leaf": [10, 15],             # minimum samples in a leaf node
    "estimator__max_depth": [3, 5],
    "estimator__max_features": [0.5, 0.7, 0.9, "sqrt"],    # features considered at each split
}

if REFIT:  # skipped when reusing stored models (MCI_AD_REUSE_MODELS_FROM)
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(f"Fold {i}:")

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y_ptau_amyloid[train_index], y_ptau_amyloid[test_index]

        # Define a new RSF model to tune
        rsf_model = RandomSurvivalForest(n_estimators=1000, random_state=random_state, n_jobs=-1)

        # Define the scorer
        #time[-1] is the most recent time point
        scorer = as_concordance_index_ipcw_scorer(rsf_model, tau=time[-1])

        # Perform grid search to find the best hyperparameters
        gcv_cindex = GridSearchCV(
            scorer,
            param_grid=param_grid,
            cv=cv,
            n_jobs=-1
        )
    
        gcv_cindex.fit(X_train, y_train)
        best_rsf = gcv_cindex.best_estimator_
    
        #save best_rsf model for each fold
        joblib.dump(best_rsf, f"{MODELS_DIR}/earliest_ptau_amyloid_rsf_fold{i}.joblib")
    
        #save predictions and labels
        train_predictions = best_rsf.predict(X_train)
        test_predictions = best_rsf.predict(X_test)
        train_chf_predictions = best_rsf.predict_cumulative_hazard_function(X_train)
        test_chf_predictions = best_rsf.predict_cumulative_hazard_function(X_test)
        train_surv_predictions = best_rsf.predict_survival_function(X_train)
        test_surv_predictions = best_rsf.predict_survival_function(X_test)


        np.savez_compressed(f"{PREDICTIONS_DIR}/earliest_ptau_amyloid_rsf_fold{i}_predictions.npz", 
                            train_predictions=train_predictions, 
                            test_predictions=test_predictions,
                            train_chf_predictions=train_chf_predictions,
                            test_chf_predictions=test_chf_predictions,
                            train_surv_predictions=train_surv_predictions,
                            test_surv_predictions=test_surv_predictions,
                            train_labels = y_train,
                            test_labels = y_test
                            )

X = X_ptau_amyloid
y = y_ptau_amyloid['event']

#Test Result Metrics
ptau_amyloid_rsf_c_index_list = []
ptau_amyloid_rsf_auc_list = []
ptau_amyloid_rsf_mean_auc_list = []
ptau_amyloid_rsf_chf_scores_list = []
ptau_amyloid_rsf_surv_scores_list = []
ptau_amyloid_rsf_bs_list = []
ptau_amyloid_rsf_ibs_list = []

#Train Result Metrics
ptau_amyloid_rsf_train_c_index_list = []
ptau_amyloid_rsf_train_auc_list = []
ptau_amyloid_rsf_train_mean_auc_list = []
ptau_amyloid_rsf_train_chf_scores_list = []
ptau_amyloid_rsf_train_surv_scores_list = []
ptau_amyloid_rsf_train_bs_list = []
ptau_amyloid_rsf_train_ibs_list = []

for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(f"Fold {i}:")
    #load rsf predictions for each fold
    loaded_data = np.load(f"{PREDICTIONS_IN}/earliest_ptau_amyloid_rsf_fold{i}_predictions.npz", allow_pickle=True)
    train_predictions = loaded_data['train_predictions']
    test_predictions = loaded_data['test_predictions']
    train_chf_predictions = loaded_data['train_chf_predictions']
    test_chf_predictions = loaded_data['test_chf_predictions']
    train_surv_predictions = loaded_data['train_surv_predictions']
    test_surv_predictions = loaded_data['test_surv_predictions']
    y_train = loaded_data['train_labels']
    y_test = loaded_data['test_labels']
    
#TEST METRICS
    #calculate concordance index on TEST data
    ptau_amyloid_rsf_c_index = concordance_index_ipcw(y_train, y_test, test_predictions)
    ptau_amyloid_rsf_c_index_list.append(ptau_amyloid_rsf_c_index)

    #calculate time-dependent risk scores on TEST data
    ptau_amyloid_rsf_chf_scores = np.vstack([chf(time) for chf in test_chf_predictions])
    ptau_amyloid_rsf_chf_scores_list.append(ptau_amyloid_rsf_chf_scores)

    #calculate time-dependent AUC and mean AUC on TEST data
    ptau_amyloid_rsf_auc, ptau_amyloid_rsf_mean_auc = cumulative_dynamic_auc(y_train, y_test, ptau_amyloid_rsf_chf_scores, time)
    ptau_amyloid_rsf_auc_list.append(ptau_amyloid_rsf_auc)
    ptau_amyloid_rsf_mean_auc_list.append(ptau_amyloid_rsf_mean_auc)
    
    #calculate survival probabilities on TEST data
    ptau_amyloid_rsf_surv_scores = np.vstack([fn(time) for fn in test_surv_predictions])
    ptau_amyloid_rsf_surv_scores_list.append(ptau_amyloid_rsf_surv_scores)
    
    #calculate Brier score on TEST data
    ptau_amyloid_rsf_bs = brier_score(y_train, y_test, ptau_amyloid_rsf_surv_scores, time)
    ptau_amyloid_rsf_bs_list.append(ptau_amyloid_rsf_bs)
    
    ptau_amyloid_rsf_ibs = integrated_brier_score(y_train, y_test, ptau_amyloid_rsf_surv_scores, time)
    ptau_amyloid_rsf_ibs_list.append(ptau_amyloid_rsf_ibs)
    
#TRAINING METRICS
    #calculate concordance index on TRAIN data
    ptau_amyloid_rsf_train_c_index = concordance_index_ipcw(y_train, y_train, train_predictions)
    ptau_amyloid_rsf_train_c_index_list.append(ptau_amyloid_rsf_train_c_index)
    
    #calculate time-dependent risk scores on TRAIN data
    ptau_amyloid_rsf_train_chf_scores = np.vstack([chf(time) for chf in train_chf_predictions])
    ptau_amyloid_rsf_train_chf_scores_list.append(ptau_amyloid_rsf_train_chf_scores)
    
    #calculate time-dependent AUC and mean AUC on TRAIN data
    ptau_amyloid_rsf_train_auc, ptau_amyloid_rsf_train_mean_auc = cumulative_dynamic_auc(y_train, y_train, ptau_amyloid_rsf_train_chf_scores, time)
    ptau_amyloid_rsf_train_auc_list.append(ptau_amyloid_rsf_train_auc)
    ptau_amyloid_rsf_train_mean_auc_list.append(ptau_amyloid_rsf_train_mean_auc)
    
    #calculate survival probabilities on TRAIN data
    ptau_amyloid_rsf_train_surv_scores = np.vstack([fn(time) for fn in train_surv_predictions])
    ptau_amyloid_rsf_train_surv_scores_list.append(ptau_amyloid_rsf_train_surv_scores)
    
    #calculate Brier score on TRAIN data
    ptau_amyloid_rsf_train_bs = brier_score(y_train, y_train, ptau_amyloid_rsf_train_surv_scores, time)
    ptau_amyloid_rsf_train_bs_list.append(ptau_amyloid_rsf_train_bs)

    ptau_amyloid_rsf_train_ibs = integrated_brier_score(y_train, y_train, ptau_amyloid_rsf_train_surv_scores, time)
    ptau_amyloid_rsf_train_ibs_list.append(ptau_amyloid_rsf_train_ibs)

np.savez_compressed(
    f"{METRICS_DIR}/ptau_amyloid_metrics.npz",
    rsf_c_index_list=ptau_amyloid_rsf_c_index_list,
    rsf_auc_list=ptau_amyloid_rsf_auc_list,
    rsf_mean_auc_list=ptau_amyloid_rsf_mean_auc_list,
    rsf_chf_scores_list=np.array(ptau_amyloid_rsf_chf_scores_list, dtype=object),
    rsf_surv_scores_list=np.array(ptau_amyloid_rsf_surv_scores_list, dtype=object),
    rsf_bs_list=ptau_amyloid_rsf_bs_list,
    rsf_ibs_list=ptau_amyloid_rsf_ibs_list,
    rsf_train_c_index_list=ptau_amyloid_rsf_train_c_index_list,
    rsf_train_auc_list=ptau_amyloid_rsf_train_auc_list,
    rsf_train_mean_auc_list=ptau_amyloid_rsf_train_mean_auc_list,
    rsf_train_chf_scores_list=np.array(ptau_amyloid_rsf_train_chf_scores_list, dtype=object),
    rsf_train_surv_scores_list=np.array(ptau_amyloid_rsf_train_surv_scores_list,dtype =object),
    rsf_train_bs_list=ptau_amyloid_rsf_train_bs_list,
    rsf_train_ibs_list=ptau_amyloid_rsf_train_ibs_list
)

ptau_amyloid_metrics = np.load(f"{METRICS_DIR}/ptau_amyloid_metrics.npz", allow_pickle=True)

ptau_amyloid_rsf_mean_auc_list = ptau_amyloid_metrics['rsf_mean_auc_list']
ptau_amyloid_rsf_c_index_list = ptau_amyloid_metrics['rsf_c_index_list']
ptau_amyloid_rsf_auc_list = ptau_amyloid_metrics['rsf_auc_list']
ptau_amyloid_rsf_chf_scores_list = ptau_amyloid_metrics['rsf_chf_scores_list']
ptau_amyloid_rsf_surv_scores_list = ptau_amyloid_metrics['rsf_surv_scores_list']
ptau_amyloid_rsf_bs_list=ptau_amyloid_metrics['rsf_bs_list']
ptau_amyloid_rsf_ibs_list=ptau_amyloid_metrics['rsf_ibs_list']
ptau_amyloid_rsf_train_c_index_list=ptau_amyloid_metrics['rsf_train_c_index_list']
ptau_amyloid_rsf_train_auc_list=ptau_amyloid_metrics['rsf_train_auc_list']
ptau_amyloid_rsf_train_mean_auc_list=ptau_amyloid_metrics['rsf_train_mean_auc_list']
ptau_amyloid_rsf_train_chf_scores_list=ptau_amyloid_metrics['rsf_train_chf_scores_list']
ptau_amyloid_rsf_train_surv_scores_list=ptau_amyloid_metrics['rsf_train_surv_scores_list']
ptau_amyloid_rsf_train_bs_list=ptau_amyloid_metrics['rsf_train_bs_list']
ptau_amyloid_rsf_train_ibs_list=ptau_amyloid_metrics['rsf_train_ibs_list']

ptau_amyloid_c_list =[]
for i in range(len(ptau_amyloid_rsf_c_index_list)):
    ptau_amyloid_rsf_c_index = ptau_amyloid_rsf_c_index_list[i][0]
    ptau_amyloid_c_list.append(ptau_amyloid_rsf_c_index)
    
#get mean aucs over th 25 timepoints per fold, there are no NaN values
ptau_amyloid_mean_auc_list = []
for i in range(len(ptau_amyloid_rsf_auc_list)):
    ptau_amyloid_mean_auc = np.nanmean(ptau_amyloid_rsf_auc_list[i])
    ptau_amyloid_mean_auc_list.append(ptau_amyloid_mean_auc)

ptau_amyloid_metrics_df = pd.DataFrame({
    'Fold': [0, 1, 2, 3, 4],
    'Test C-index': ptau_amyloid_c_list,
    'Train C-index': [x[0] for x in ptau_amyloid_rsf_train_c_index_list],
    'Test Mean AUC': [x for x in ptau_amyloid_mean_auc_list],
    'Train Mean AUC': [x for x in ptau_amyloid_rsf_train_mean_auc_list],
    'Test IBS': [x for x in ptau_amyloid_rsf_ibs_list],
    'Train IBS': [x for x in ptau_amyloid_rsf_train_ibs_list]
})
print(ptau_amyloid_metrics_df)


# =====================================================================================
# 11 · Model — kitchen sink (plasma panel + ratio + PET)
# =====================================================================================

X = X_ksink
y = y_ksink['event']

# Define internal CV for hyperparameter tuning
cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

#Best params (fold): {'estimator__max_depth': 3, 'estimator__max_features': 0.7, 'estimator__min_samples_leaf': 10, 'estimator__min_samples_split': 2, 'estimator__n_estimators': 300}
param_grid = {
    "estimator__n_estimators": [200, 300, 500],            # number of trees
    "estimator__min_samples_split": [2, 5],            # minimum samples to split a node
    "estimator__min_samples_leaf": [10, 15],             # minimum samples in a leaf node
    "estimator__max_depth": [3, 5],
    "estimator__max_features": [0.5, 0.7, 0.9, "sqrt"],    # features considered at each split
}

if REFIT:  # skipped when reusing stored models (MCI_AD_REUSE_MODELS_FROM)
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(f"Fold {i}:")

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y_ksink[train_index], y_ksink[test_index]

        # Define a new RSF model to tune
        rsf_model = RandomSurvivalForest(n_estimators=1000, random_state=random_state, n_jobs=-1)

        # Define the scorer
        #time[-1] is the most recent time point
        scorer = as_concordance_index_ipcw_scorer(rsf_model, tau=time[-1])

        # Perform grid search to find the best hyperparameters
        gcv_cindex = GridSearchCV(
            scorer,
            param_grid=param_grid,
            cv=cv,
            n_jobs=-1
        )
    
        gcv_cindex.fit(X_train, y_train)
        best_rsf = gcv_cindex.best_estimator_
    
        #save best_rsf model for each fold
        joblib.dump(best_rsf, f"{MODELS_DIR}/earliest_ksink_rsf_fold{i}.joblib")
    
        #save predictions and labels
        train_predictions = best_rsf.predict(X_train)
        test_predictions = best_rsf.predict(X_test)
        train_chf_predictions = best_rsf.predict_cumulative_hazard_function(X_train)
        test_chf_predictions = best_rsf.predict_cumulative_hazard_function(X_test)
        train_surv_predictions = best_rsf.predict_survival_function(X_train)
        test_surv_predictions = best_rsf.predict_survival_function(X_test)


        np.savez_compressed(f"{PREDICTIONS_DIR}/earliest_ksink_rsf_fold{i}_predictions.npz", 
                            train_predictions=train_predictions, 
                            test_predictions=test_predictions,
                            train_chf_predictions=train_chf_predictions,
                            test_chf_predictions=test_chf_predictions,
                            train_surv_predictions=train_surv_predictions,
                            test_surv_predictions=test_surv_predictions,
                            train_labels = y_train,
                            test_labels = y_test
                            )


# Pattern: fit → score → save metrics → reload metrics → summarise.

X = X_ksink
y = y_ksink['event']

#Test Result Metrics
ksink_rsf_c_index_list = []
ksink_rsf_auc_list = []
ksink_rsf_mean_auc_list = []
ksink_rsf_chf_scores_list = []
ksink_rsf_surv_scores_list = []
ksink_rsf_bs_list = []
ksink_rsf_ibs_list = []

#Train Result Metrics
ksink_rsf_train_c_index_list = []
ksink_rsf_train_auc_list = []
ksink_rsf_train_mean_auc_list = []
ksink_rsf_train_chf_scores_list = []
ksink_rsf_train_surv_scores_list = []
ksink_rsf_train_bs_list = []
ksink_rsf_train_ibs_list = []

for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(f"Fold {i}:")
    #load rsf predictions for each fold
    loaded_data = np.load(f"{PREDICTIONS_IN}/earliest_ksink_rsf_fold{i}_predictions.npz", allow_pickle=True)
    train_predictions = loaded_data['train_predictions']
    test_predictions = loaded_data['test_predictions']
    train_chf_predictions = loaded_data['train_chf_predictions']
    test_chf_predictions = loaded_data['test_chf_predictions']
    train_surv_predictions = loaded_data['train_surv_predictions']
    test_surv_predictions = loaded_data['test_surv_predictions']
    y_train = loaded_data['train_labels']
    y_test = loaded_data['test_labels']
    
#TEST METRICS
    #calculate concordance index on TEST data
    ksink_rsf_c_index = concordance_index_ipcw(y_train, y_test, test_predictions)
    ksink_rsf_c_index_list.append(ksink_rsf_c_index)

    #calculate time-dependent risk scores on TEST data
    ksink_rsf_chf_scores = np.vstack([chf(time) for chf in test_chf_predictions])
    ksink_rsf_chf_scores_list.append(ksink_rsf_chf_scores)

    #calculate time-dependent AUC and mean AUC on TEST data
    ksink_rsf_auc, ksink_rsf_mean_auc = cumulative_dynamic_auc(y_train, y_test, ksink_rsf_chf_scores, time)
    ksink_rsf_auc_list.append(ksink_rsf_auc)
    ksink_rsf_mean_auc_list.append(ksink_rsf_mean_auc)

    #calculate survival probabilities on TEST data
    ksink_rsf_surv_scores = np.vstack([fn(time) for fn in test_surv_predictions])
    ksink_rsf_surv_scores_list.append(ksink_rsf_surv_scores)
    
    #calculate Brier score on TEST data
    ksink_rsf_bs = brier_score(y_train, y_test, ksink_rsf_surv_scores, time)
    ksink_rsf_bs_list.append(ksink_rsf_bs)
    
    ksink_rsf_ibs = integrated_brier_score(y_train, y_test, ksink_rsf_surv_scores, time)
    ksink_rsf_ibs_list.append(ksink_rsf_ibs)
    
#TRAINING METRICS
    #calculate concordance index on TRAIN data
    ksink_rsf_train_c_index = concordance_index_ipcw(y_train, y_train, train_predictions)
    ksink_rsf_train_c_index_list.append(ksink_rsf_train_c_index)
    
    #calculate time-dependent risk scores on TRAIN data
    ksink_rsf_train_chf_scores = np.vstack([chf(time) for chf in train_chf_predictions])
    ksink_rsf_train_chf_scores_list.append(ksink_rsf_train_chf_scores)
    
    #calculate time-dependent AUC and mean AUC on TRAIN data
    ksink_rsf_train_auc, ksink_rsf_train_mean_auc = cumulative_dynamic_auc(y_train, y_train, ksink_rsf_train_chf_scores, time)
    ksink_rsf_train_auc_list.append(ksink_rsf_train_auc)
    ksink_rsf_train_mean_auc_list.append(ksink_rsf_train_mean_auc)
    
    #calculate survival probabilities on TRAIN data
    ksink_rsf_train_surv_scores = np.vstack([fn(time) for fn in train_surv_predictions])
    ksink_rsf_train_surv_scores_list.append(ksink_rsf_train_surv_scores)
    
    #calculate Brier score on TRAIN data
    ksink_rsf_train_bs = brier_score(y_train, y_train, ksink_rsf_train_surv_scores, time)
    ksink_rsf_train_bs_list.append(ksink_rsf_train_bs)

    ksink_rsf_train_ibs = integrated_brier_score(y_train, y_train, ksink_rsf_train_surv_scores, time)
    ksink_rsf_train_ibs_list.append(ksink_rsf_train_ibs)

np.savez_compressed(
    f"{METRICS_DIR}/ksink_metrics.npz",
    rsf_c_index_list=ksink_rsf_c_index_list,
    rsf_auc_list=ksink_rsf_auc_list,
    rsf_mean_auc_list=ksink_rsf_mean_auc_list,
    rsf_chf_scores_list=np.array(ksink_rsf_chf_scores_list, dtype=object),
    rsf_surv_scores_list=np.array(ksink_rsf_surv_scores_list, dtype=object),
    rsf_bs_list=ksink_rsf_bs_list,
    rsf_ibs_list=ksink_rsf_ibs_list,
    rsf_train_c_index_list=ksink_rsf_train_c_index_list,
    rsf_train_auc_list=ksink_rsf_train_auc_list,
    rsf_train_mean_auc_list=ksink_rsf_train_mean_auc_list,
    rsf_train_chf_scores_list=np.array(ksink_rsf_train_chf_scores_list, dtype=object),
    rsf_train_surv_scores_list=np.array(ksink_rsf_train_surv_scores_list,dtype =object),
    rsf_train_bs_list=ksink_rsf_train_bs_list,
    rsf_train_ibs_list=ksink_rsf_train_ibs_list
)

ksink_metrics = np.load(f"{METRICS_DIR}/ksink_metrics.npz", allow_pickle=True)

ksink_rsf_mean_auc_list = ksink_metrics['rsf_mean_auc_list']
ksink_rsf_c_index_list = ksink_metrics['rsf_c_index_list']
ksink_rsf_auc_list = ksink_metrics['rsf_auc_list']
ksink_rsf_chf_scores_list = ksink_metrics['rsf_chf_scores_list']
ksink_rsf_surv_scores_list = ksink_metrics['rsf_surv_scores_list']
ksink_rsf_bs_list=ksink_metrics['rsf_bs_list']
ksink_rsf_ibs_list=ksink_metrics['rsf_ibs_list']
ksink_rsf_train_c_index_list=ksink_metrics['rsf_train_c_index_list']
ksink_rsf_train_auc_list=ksink_metrics['rsf_train_auc_list']
ksink_rsf_train_mean_auc_list=ksink_metrics['rsf_train_mean_auc_list']
ksink_rsf_train_chf_scores_list=ksink_metrics['rsf_train_chf_scores_list']
ksink_rsf_train_surv_scores_list=ksink_metrics['rsf_train_surv_scores_list']
ksink_rsf_train_bs_list=ksink_metrics['rsf_train_bs_list']
ksink_rsf_train_ibs_list=ksink_metrics['rsf_train_ibs_list']

ksink_c_list =[]
for i in range(len(ksink_rsf_c_index_list)):
    ksink_rsf_c_index = ksink_rsf_c_index_list[i][0]
    ksink_c_list.append(ksink_rsf_c_index)

#get mean aucs over the 25 timepoints per fold, there are no NaN values
ksink_mean_auc_list = []
for i in range(len(ksink_rsf_auc_list)):
    ksink_mean_auc = np.nanmean(ksink_rsf_auc_list[i])
    ksink_mean_auc_list.append(ksink_mean_auc)

ksink_rsf_overall_mean_cindex = np.mean(ksink_c_list)
ksink_rsf_overall_std_cindex = np.std(ksink_c_list, ddof=1)
print(f"All Data (Kitchen Sink) Overall Mean C-index: {ksink_rsf_overall_mean_cindex:.3f} ± {ksink_rsf_overall_std_cindex:.3f}")

loaded = np.load(
    f"{PREDICTIONS_IN}/earliest_bio_rsf_fold0_predictions.npz",
    allow_pickle=True
)
y_test_fold0 = loaded['test_labels']
total_pmci = np.sum(y_test_fold0['event'] == True)
total_smci = np.sum(y_test_fold0['event'] == False)

print(f"Total: {len(y_test_fold0)} patients (pMCI={total_pmci}, sMCI={total_smci})")
print(f"\n{'Years':<8} {'pMCI':<8} {'sMCI':<8} {'At Risk':<10}")
print("-" * 34)

for t_days in time:
    # pMCI who converted by this timepoint
    pmci = np.sum((y_test_fold0['event'] == True) & (y_test_fold0['time'] <= t_days))
    
    # sMCI censored by this timepoint
    smci_censored = np.sum((y_test_fold0['event'] == False) & (y_test_fold0['time'] <= t_days))
    
    # Still being followed
    at_risk = len(y_test_fold0) - pmci - smci_censored
    
    print(f"  {t_days/365.25:<8.2f} {pmci:<8} {smci_censored:<8} {at_risk:<10}")


# =====================================================================================
# 12 · Performance summary
#
# Mean IPCW C-index (± SD across the five folds) and mean time-dependent Brier score for
# each model. The PET + p-tau217/AB42 model is fitted later, so its numbers are printed at
# the end of Section 15. The comparison figures are drawn in Section 16.
# =====================================================================================

rsf_overall_mean_cindex = np.mean(rsf_c_list)
rsf_overall_std_cindex = np.std(rsf_c_list, ddof=1)
print(f"PET Overall Mean C-index: {rsf_overall_mean_cindex:.3f} ± {rsf_overall_std_cindex:.3f}")

ptau_rsf_overall_mean_cindex = np.mean(ptau_c_list)
ptau_rsf_overall_std_cindex = np.std(ptau_c_list, ddof=1)
print(f"p-Tau217 Overall Mean C-index: {ptau_rsf_overall_mean_cindex:.3f} ± {ptau_rsf_overall_std_cindex:.3f}")

demo_rsf_overall_mean_cindex = np.mean(demo_c_list)
demo_rsf_overall_std_cindex = np.std(demo_c_list, ddof=1)
print(f"Demographics Overall Mean C-index: {demo_rsf_overall_mean_cindex:.3f} ± {demo_rsf_overall_std_cindex:.3f}")

merged_rsf_overall_mean_cindex = np.mean(merged_c_list)
merged_rsf_overall_std_cindex = np.std(merged_c_list, ddof=1)
print(f"Merged Overall Mean C-index: {merged_rsf_overall_mean_cindex:.3f} ± {merged_rsf_overall_std_cindex:.3f}")

bio_rsf_overall_mean_cindex = np.mean(bio_c_list)
bio_rsf_overall_std_cindex = np.std(bio_c_list, ddof=1)
print(f"All Biomarkers Overall Mean C-index: {bio_rsf_overall_mean_cindex:.3f} ± {bio_rsf_overall_std_cindex:.3f}")

ptau_amyloid_rsf_overall_mean_cindex = np.mean(ptau_amyloid_c_list)
ptau_amyloid_rsf_overall_std_cindex = np.std(ptau_amyloid_c_list, ddof=1)
print(f"p-Tau217/Abeta42 Overall Mean C-index: {ptau_amyloid_rsf_overall_mean_cindex:.3f} ± {ptau_amyloid_rsf_overall_std_cindex:.3f}")

ksink_rsf_overall_mean_cindex = np.mean(ksink_c_list)
ksink_rsf_overall_std_cindex = np.std(ksink_c_list, ddof=1)
print(f"All Data (Kitchen Sink) Overall Mean C-index: {ksink_rsf_overall_mean_cindex:.3f} ± {ksink_rsf_overall_std_cindex:.3f}")

rsf_bs_arr = np.array(rsf_bs_list)
rsf_mean_bs = np.nanmean(rsf_bs_arr[:,1,:], axis=0)
rsf_std_bs = np.nanstd(rsf_bs_arr[:,1,:], axis=0)

ptau_rsf_bs_arr = np.array(ptau_rsf_bs_list)
ptau_rsf_mean_bs = np.nanmean(ptau_rsf_bs_arr[:,1,:], axis=0)
ptau_rsf_std_bs = np.nanstd(ptau_rsf_bs_arr[:,1,:], axis=0)

demo_rsf_bs_arr = np.array(demo_rsf_bs_list)
demo_rsf_mean_bs = np.nanmean(demo_rsf_bs_arr[:,1,:], axis=0)
demo_rsf_std_bs = np.nanstd(demo_rsf_bs_arr[:,1,:], axis=0)

merged_rsf_bs_arr = np.array(merged_rsf_bs_list)
merged_rsf_mean_bs = np.nanmean(merged_rsf_bs_arr[:,1,:], axis=0)
merged_rsf_std_bs = np.nanstd(merged_rsf_bs_arr[:,1,:], axis=0)

bio_rsf_bs_arr = np.array(bio_rsf_bs_list)
bio_rsf_mean_bs = np.nanmean(bio_rsf_bs_arr[:,1,:], axis=0)
bio_rsf_std_bs = np.nanstd(bio_rsf_bs_arr[:,1,:], axis=0)

ptau_amyloid_rsf_bs_arr = np.array(ptau_amyloid_rsf_bs_list)
ptau_amyloid_rsf_mean_bs = np.nanmean(ptau_amyloid_rsf_bs_arr[:,1,:], axis=0)
ptau_amyloid_rsf_std_bs = np.nanstd(ptau_amyloid_rsf_bs_arr[:,1,:], axis=0)

ksink_rsf_bs_arr = np.array(ksink_rsf_bs_list)
ksink_rsf_mean_bs = np.nanmean(ksink_rsf_bs_arr[:,1,:], axis=0)
ksink_rsf_std_bs = np.nanstd(ksink_rsf_bs_arr[:,1,:], axis=0)

print(f"PET Mean Brier Score: {np.mean(rsf_mean_bs):.3f} ± {np.mean(rsf_std_bs):.3f}")
print(f"p-Tau217 Mean Brier Score: {np.mean(ptau_rsf_mean_bs):.3f} ± {np.mean(ptau_rsf_std_bs):.3f}")
print(f"Demographics Mean Brier Score: {np.mean(demo_rsf_mean_bs):.3f} ± {np.mean(demo_rsf_std_bs):.3f}")
print(f"PET+p-Tau217 Mean Brier Score: {np.mean(merged_rsf_mean_bs):.3f} ± {np.mean(merged_rsf_std_bs):.3f}")
print(f"All Biomarkers Mean Brier Score: {np.mean(bio_rsf_mean_bs):.3f} ± {np.mean(bio_rsf_std_bs):.3f}")
print(f"p-Tau217/Abeta42 Mean Brier Score: {np.mean(ptau_amyloid_rsf_mean_bs):.3f} ± {np.mean(ptau_amyloid_rsf_std_bs):.3f}")
print(f"All Data (Kitchen Sink) Mean Brier Score: {np.mean(ksink_rsf_mean_bs):.3f} ± {np.mean(ksink_rsf_std_bs):.3f}")



# =====================================================================================
# 13 · Bootstrap confidence intervals and model comparison
#
# `bootstrap_dynamic_auc` resamples the test set 1000× per fold to obtain percentile
# confidence intervals for the time-dependent AUC. Results are cached to
# `$MCI_AD_OUTPUT_DIR/metrics/ADNI_PENN/earliest_<model>_rsf_auc_bootstrap.npz` and reused by 03_blood_biomarkers.py.
# =====================================================================================

def bootstrap_dynamic_auc(
    y_full,
    y_test,
    risk_matrix,
    time_grid,
    n_boot=1000,
    random_state=42
):
    rng = np.random.default_rng(random_state)
    n_samples = len(y_test)

    boot_aucs = []
    boot_mean_aucs = []

    for _ in range(n_boot):

        idx = rng.choice(n_samples, size=n_samples, replace=True)

        y_boot = y_test[idx]
        scores_boot = risk_matrix[idx]

        # Skip degenerate samples (all events or all censored)
        if len(np.unique(y_boot["event"])) < 2:
            continue

        try:
            auc_boot, mean_auc_boot = cumulative_dynamic_auc(
                y_full,
                y_boot,
                scores_boot,
                time_grid
            )

            # Skip if mean AUC is NaN
            if np.isnan(mean_auc_boot):
                continue

            boot_aucs.append(auc_boot)
            boot_mean_aucs.append(mean_auc_boot)

        except:
            continue

    boot_aucs = np.array(boot_aucs)
    boot_mean_aucs = np.array(boot_mean_aucs)

    if len(boot_mean_aucs) == 0:
        raise ValueError("All bootstrap samples invalid — check time grid or data.")

    auc_mean = np.nanmean(boot_aucs, axis=0)
    auc_lower = np.nanpercentile(boot_aucs, 2.5, axis=0)
    auc_upper = np.nanpercentile(boot_aucs, 97.5, axis=0)

    mean_auc_ci = np.nanpercentile(boot_mean_aucs, [2.5, 97.5])

    print("Valid bootstrap samples:", len(boot_mean_aucs))

    return auc_mean, auc_lower, auc_upper, mean_auc_ci

pet_y_test = []
pet_risk_scores = []
y_train_full = y_pet

#Step One: load fold predictions
for i in range(5):
    loaded_data = np.load(
        f"{PREDICTIONS_IN}/earliest_pet_rsf_fold{i}_predictions.npz",
        allow_pickle=True
    )

    test_chf_predictions = loaded_data['test_chf_predictions']
    y_test = loaded_data['test_labels']

    # convert CHF functions to matrix
    risk_matrix = np.vstack([chf(time) for chf in test_chf_predictions])

    pet_y_test.append(y_test)
    pet_risk_scores.append(risk_matrix)

pet_y_test = np.concatenate(pet_y_test)
pet_risk_scores = np.vstack(pet_risk_scores)

#Step Two: Compute AUC
pet_auc_original, pet_mean_auc_original = cumulative_dynamic_auc(
    y_train_full,   # survival distribution for IPCW
    pet_y_test,
    pet_risk_scores,
    time
)

#Step Three: Boostrap
pet_auc_mean, pet_auc_lower, pet_auc_upper, pet_mean_auc_ci = bootstrap_dynamic_auc(
    y_train_full,
    pet_y_test,
    pet_risk_scores,
    time,
    n_boot=1000,
    random_state=42
)

print(f"Original mean AUC: {pet_mean_auc_original:.3f}")
print(f"95% CI for mean AUC: [{pet_mean_auc_ci[0]:.3f}, {pet_mean_auc_ci[1]:.3f}]")

#save results
np.savez(
    f"{METRICS_DIR}/earliest_pet_rsf_auc_bootstrap.npz",
    pet_auc_mean=pet_auc_mean,
    pet_auc_lower=pet_auc_lower,
    pet_auc_upper=pet_auc_upper,
    pet_mean_auc_ci=pet_mean_auc_ci
)

ptau_y_test = []
ptau_risk_scores = []
y_train_full = y_ptau

#Step One: load fold predictions
for i in range(5):
    loaded_data = np.load(
        f"{PREDICTIONS_IN}/earliest_ptau_rsf_fold{i}_predictions.npz",
        allow_pickle=True
    )

    test_chf_predictions = loaded_data['test_chf_predictions']
    y_test = loaded_data['test_labels']

    # convert CHF functions to matrix
    risk_matrix = np.vstack([chf(time) for chf in test_chf_predictions])

    ptau_y_test.append(y_test)
    ptau_risk_scores.append(risk_matrix)

ptau_y_test = np.concatenate(ptau_y_test)
ptau_risk_scores = np.vstack(ptau_risk_scores)

#Step Two: Compute AUC
ptau_auc_original, ptau_mean_auc_original = cumulative_dynamic_auc(
    y_train_full,   # survival distribution for IPCW
    ptau_y_test,
    ptau_risk_scores,
    time
)

#Step Three: Boostrap
ptau_auc_mean, ptau_auc_lower, ptau_auc_upper, ptau_mean_auc_ci = bootstrap_dynamic_auc(
    y_train_full,
    ptau_y_test,
    ptau_risk_scores,
    time,
    n_boot=1000,
    random_state=42
)

#save results
np.savez(
    f"{METRICS_DIR}/earliest_ptau_rsf_auc_bootstrap.npz",
    ptau_auc_mean=ptau_auc_mean,
    ptau_auc_lower=ptau_auc_lower,
    ptau_auc_upper=ptau_auc_upper,
    ptau_mean_auc_ci=ptau_mean_auc_ci
)

print(f"Original mean AUC: {ptau_mean_auc_original:.3f}")
print(f"95% CI for mean AUC: [{ptau_mean_auc_ci[0]:.3f}, {ptau_mean_auc_ci[1]:.3f}]")

demo_y_test = []
demo_risk_scores = []
y_train_full = y_demo

#Step One: load fold predictions
for i in range(5):
    loaded_data = np.load(
        f"{PREDICTIONS_IN}/earliest_demo_rsf_fold{i}_predictions.npz",
        allow_pickle=True
    )

    test_chf_predictions = loaded_data['test_chf_predictions']
    y_test = loaded_data['test_labels']

    # convert CHF functions to matrix
    risk_matrix = np.vstack([chf(time) for chf in test_chf_predictions])

    demo_y_test.append(y_test)
    demo_risk_scores.append(risk_matrix)

demo_y_test = np.concatenate(demo_y_test)
demo_risk_scores = np.vstack(demo_risk_scores)

#Step Two: Compute AUC
demo_auc_original, demo_mean_auc_original = cumulative_dynamic_auc(
    y_train_full,   # survival distribution for IPCW
    demo_y_test,
    demo_risk_scores,
    time
)

#Step Three: Boostrap
demo_auc_mean, demo_auc_lower, demo_auc_upper, demo_mean_auc_ci = bootstrap_dynamic_auc(
    y_train_full,
    demo_y_test,
    demo_risk_scores,
    time,
    n_boot=1000,
    random_state=42
)

#save results
np.savez(
    f"{METRICS_DIR}/earliest_demo_rsf_auc_bootstrap.npz",
    demo_auc_mean=demo_auc_mean,
    demo_auc_lower=demo_auc_lower,
    demo_auc_upper=demo_auc_upper,
    demo_mean_auc_ci=demo_mean_auc_ci
)

print(f"Original mean AUC: {demo_mean_auc_original:.3f}")
print(f"95% CI for mean AUC: [{demo_mean_auc_ci[0]:.3f}, {demo_mean_auc_ci[1]:.3f}]")

demo_rsf_mean_auc = np.nanmean(demo_rsf_auc_list, axis = 0)
print("Demographics AUC at each timepoint:")
for i, t in enumerate(time):
    print(f"  {t/365.25:.2f} years: AUC = {demo_rsf_mean_auc[i]:.3f} "
          f"[{demo_auc_lower[i]:.3f}, {demo_auc_upper[i]:.3f}]")

demo_mean_auc_late = np.mean(demo_rsf_mean_auc[time >= 2.07*365.25])
print(f"Mean AUC for late timepoints (2.07 years and beyond): {demo_mean_auc_late:.3f}")

pet_ptau_y_test = []
pet_ptau_risk_scores = []
y_train_full = y_pet_ptau

#Step One: load fold predictions
for i in range(5):
    loaded_data = np.load(
        f"{PREDICTIONS_IN}/earliest_pet_ptau_rsf_fold{i}_predictions.npz",
        allow_pickle=True
    )

    test_chf_predictions = loaded_data['test_chf_predictions']
    y_test = loaded_data['test_labels']

    # convert CHF functions to matrix
    risk_matrix = np.vstack([chf(time) for chf in test_chf_predictions])

    pet_ptau_y_test.append(y_test)
    pet_ptau_risk_scores.append(risk_matrix)

pet_ptau_y_test = np.concatenate(pet_ptau_y_test)
pet_ptau_risk_scores = np.vstack(pet_ptau_risk_scores)

#Step Two: Compute AUC
pet_ptau_auc_original, pet_ptau_mean_auc_original = cumulative_dynamic_auc(
    y_train_full,   # survival distribution for IPCW
    pet_ptau_y_test,
    pet_ptau_risk_scores,
    time
)

#Step Three: Boostrap
pet_ptau_auc_mean, pet_ptau_auc_lower, pet_ptau_auc_upper, pet_ptau_mean_auc_ci = bootstrap_dynamic_auc(
    y_train_full,
    pet_ptau_y_test,
    pet_ptau_risk_scores,
    time,
    n_boot=1000,
    random_state=42
)

#save results
np.savez(
    f"{METRICS_DIR}/earliest_pet_ptau_rsf_auc_bootstrap.npz",
    pet_ptau_auc_mean=pet_ptau_auc_mean,
    pet_ptau_auc_lower=pet_ptau_auc_lower,
    pet_ptau_auc_upper=pet_ptau_auc_upper,
    pet_ptau_mean_auc_ci=pet_ptau_mean_auc_ci
)

print(f"Original mean AUC: {pet_ptau_mean_auc_original:.3f}")
print(f"95% CI for mean AUC: [{pet_ptau_mean_auc_ci[0]:.3f}, {pet_ptau_mean_auc_ci[1]:.3f}]")

ptau_amyloid_y_test = []
ptau_amyloid_risk_scores = []
y_train_full = y_ptau_amyloid

#Step One: load fold predictions
for i in range(5):
    loaded_data = np.load(
        f"{PREDICTIONS_IN}/earliest_ptau_amyloid_rsf_fold{i}_predictions.npz",
        allow_pickle=True
    )

    test_chf_predictions = loaded_data['test_chf_predictions']
    y_test = loaded_data['test_labels']

    # convert CHF functions to matrix
    risk_matrix = np.vstack([chf(time) for chf in test_chf_predictions])

    ptau_amyloid_y_test.append(y_test)
    ptau_amyloid_risk_scores.append(risk_matrix)

ptau_amyloid_y_test = np.concatenate(ptau_amyloid_y_test)
ptau_amyloid_risk_scores = np.vstack(ptau_amyloid_risk_scores)

#Step Two: Compute AUC
ptau_amyloid_auc_original, ptau_amyloid_mean_auc_original = cumulative_dynamic_auc(
    y_train_full,   # survival distribution for IPCW
    ptau_amyloid_y_test,
    ptau_amyloid_risk_scores,
    time
)

#Step Three: Boostrap
ptau_amyloid_auc_mean, ptau_amyloid_auc_lower, ptau_amyloid_auc_upper, ptau_amyloid_mean_auc_ci = bootstrap_dynamic_auc(
    y_train_full,
    ptau_amyloid_y_test,
    ptau_amyloid_risk_scores,
    time,
    n_boot=1000,
    random_state=42
)

#save results
np.savez(
    f"{METRICS_DIR}/earliest_ptau_amyloid_rsf_auc_bootstrap.npz",
    ptau_amyloid_auc_mean=ptau_amyloid_auc_mean,
    ptau_amyloid_auc_lower=ptau_amyloid_auc_lower,
    ptau_amyloid_auc_upper=ptau_amyloid_auc_upper,
    ptau_amyloid_mean_auc_ci=ptau_amyloid_mean_auc_ci
)

print(f"Original mean AUC: {ptau_amyloid_mean_auc_original:.3f}")
print(f"95% CI for mean AUC: [{ptau_amyloid_mean_auc_ci[0]:.3f}, {ptau_amyloid_mean_auc_ci[1]:.3f}]")

bio_y_test = []
bio_risk_scores = []
y_train_full = y_bio

#Step One: load fold predictions
for i in range(5):
    loaded_data = np.load(
        f"{PREDICTIONS_IN}/earliest_bio_rsf_fold{i}_predictions.npz",
        allow_pickle=True
    )

    test_chf_predictions = loaded_data['test_chf_predictions']
    y_test = loaded_data['test_labels']

    # convert CHF functions to matrix
    risk_matrix = np.vstack([chf(time) for chf in test_chf_predictions])

    bio_y_test.append(y_test)
    bio_risk_scores.append(risk_matrix)

bio_y_test = np.concatenate(bio_y_test)
bio_risk_scores = np.vstack(bio_risk_scores)

#Step Two: Compute AUC
bio_auc_original, bio_mean_auc_original = cumulative_dynamic_auc(
    y_train_full,   # survival distribution for IPCW
    bio_y_test,
    bio_risk_scores,
    time
)

#Step Three: Boostrap
bio_auc_mean, bio_auc_lower, bio_auc_upper, bio_mean_auc_ci = bootstrap_dynamic_auc(
    y_train_full,
    bio_y_test,
    bio_risk_scores,
    time,
    n_boot=1000,
    random_state=42
)

#save results
np.savez(
    f"{METRICS_DIR}/earliest_bio_rsf_auc_bootstrap.npz",
    bio_auc_mean=bio_auc_mean,
    bio_auc_lower=bio_auc_lower,
    bio_auc_upper=bio_auc_upper,
    bio_mean_auc_ci=bio_mean_auc_ci
)

print(f"Original mean AUC: {bio_mean_auc_original:.3f}")
print(f"95% CI for mean AUC: [{bio_mean_auc_ci[0]:.3f}, {bio_mean_auc_ci[1]:.3f}]")

ksink_y_test = []
ksink_risk_scores = []
y_train_full = y_ksink

#Step One: load fold predictions
for i in range(5):
    loaded_data = np.load(
        f"{PREDICTIONS_IN}/earliest_ksink_rsf_fold{i}_predictions.npz",
        allow_pickle=True
    )

    test_chf_predictions = loaded_data['test_chf_predictions']
    y_test = loaded_data['test_labels']

    # convert CHF functions to matrix
    risk_matrix = np.vstack([chf(time) for chf in test_chf_predictions])

    ksink_y_test.append(y_test)
    ksink_risk_scores.append(risk_matrix)

ksink_y_test = np.concatenate(ksink_y_test)
ksink_risk_scores = np.vstack(ksink_risk_scores)

#Step Two: Compute AUC
ksink_auc_original, ksink_mean_auc_original = cumulative_dynamic_auc(
    y_train_full,   # survival distribution for IPCW
    ksink_y_test,
    ksink_risk_scores,
    time
)

#Step Three: Boostrap
ksink_auc_mean, ksink_auc_lower, ksink_auc_upper, ksink_mean_auc_ci = bootstrap_dynamic_auc(
    y_train_full,
    ksink_y_test,
    ksink_risk_scores,
    time,
    n_boot=1000,
    random_state=42
)

#save results
np.savez(
    f"{METRICS_DIR}/earliest_ksink_rsf_auc_bootstrap.npz",
    ksink_auc_mean=ksink_auc_mean,
    ksink_auc_lower=ksink_auc_lower,
    ksink_auc_upper=ksink_auc_upper,
    ksink_mean_auc_ci=ksink_mean_auc_ci
)

print(f"Original mean AUC: {ksink_mean_auc_original:.3f}")
print(f"95% CI for mean AUC: [{ksink_mean_auc_ci[0]:.3f}, {ksink_mean_auc_ci[1]:.3f}]")

pet_bootstrap = np.load(f"{METRICS_DIR}/earliest_pet_rsf_auc_bootstrap.npz")
ptau_bootstrap = np.load(f"{METRICS_DIR}/earliest_ptau_rsf_auc_bootstrap.npz")
demo_bootstrap = np.load(f"{METRICS_DIR}/earliest_demo_rsf_auc_bootstrap.npz")
pet_ptau_bootstrap = np.load(f"{METRICS_DIR}/earliest_pet_ptau_rsf_auc_bootstrap.npz")
ptau_amyloid_bootstrap = np.load(f"{METRICS_DIR}/earliest_ptau_amyloid_rsf_auc_bootstrap.npz")
bio_bootstrap = np.load(f"{METRICS_DIR}/earliest_bio_rsf_auc_bootstrap.npz")
ksink_bootstrap = np.load(f"{METRICS_DIR}/earliest_ksink_rsf_auc_bootstrap.npz")

pet_auc_lower = pet_bootstrap["pet_auc_lower"]
pet_auc_upper = pet_bootstrap["pet_auc_upper"]
ptau_auc_lower = ptau_bootstrap["ptau_auc_lower"]
ptau_auc_upper = ptau_bootstrap["ptau_auc_upper"]
demo_auc_lower = demo_bootstrap["demo_auc_lower"]
demo_auc_upper = demo_bootstrap["demo_auc_upper"]
pet_ptau_auc_lower = pet_ptau_bootstrap["pet_ptau_auc_lower"]
pet_ptau_auc_upper = pet_ptau_bootstrap["pet_ptau_auc_upper"]
ptau_amyloid_auc_lower = ptau_amyloid_bootstrap["ptau_amyloid_auc_lower"]
ptau_amyloid_auc_upper = ptau_amyloid_bootstrap["ptau_amyloid_auc_upper"]   
bio_auc_lower = bio_bootstrap["bio_auc_lower"]
bio_auc_upper = bio_bootstrap["bio_auc_upper"]
ksink_auc_lower = ksink_bootstrap["ksink_auc_lower"]
ksink_auc_upper = ksink_bootstrap["ksink_auc_upper"]


# =====================================================================================
# 14 · Export pooled test-fold risk scores for 04_timeROC_comparisons.R
#
# One row per patient: survival time, event indicator, and each model's out-of-fold
# risk score (rsf.predict, the cumulative hazard). Read by 04_timeROC_comparisons.R for the
# timeROC significance tests.
# =====================================================================================

file_prefix = {
    'Demographics': 'demo',
    'PET': 'pet',
    'p-Tau217': 'ptau',
    'PET + p-Tau217': 'pet_ptau',
    'Biomarker Panel': 'bio',
    'p-Tau217/Abeta42': 'ptau_amyloid',
    'Kitchen sink': 'ksink',
}

all_T = []
all_delta = []
all_risk = {prefix: [] for prefix in file_prefix.values()}

for i in range(5):
    # labels are the same patients for every model; take them from the panel model
    loaded = np.load(f"{PREDICTIONS_IN}/earliest_bio_rsf_fold{i}_predictions.npz", allow_pickle=True)
    y_test = loaded['test_labels']
    all_T.append(y_test['time'])
    all_delta.append(y_test['event'].astype(int))

    for name, prefix in file_prefix.items():
        loaded_model = np.load(
            f"{PREDICTIONS_IN}/earliest_{prefix}_rsf_fold{i}_predictions.npz", allow_pickle=True
        )
        all_risk[prefix].append(loaded_model['test_predictions'])

df_export = pd.DataFrame({
    'time': np.concatenate(all_T),
    'event': np.concatenate(all_delta),
})
for name, prefix in file_prefix.items():
    df_export[f'risk_{prefix}'] = np.concatenate(all_risk[prefix])

print(f"Shape: {df_export.shape}, events: {df_export['event'].sum()}")
for name, prefix in file_prefix.items():
    col = f'risk_{prefix}'
    print(f"  {name:<20}: min={df_export[col].min():.3f}, max={df_export[col].max():.3f}, "
          f"mean={df_export[col].mean():.3f}, NaN={df_export[col].isna().sum()}")

df_export.to_csv(TIMEROC_INPUT_CSV, index=False)
print(f"wrote {TIMEROC_INPUT_CSV}")


# =====================================================================================
# 15 · Model — amyloid PET + p-tau217/Aβ42 ratio (pet_ptau_amyloid)
#
# The eighth covariate set: AGE_MCI, Sex, Education, APOE4, p-tau217/Aβ42 ratio and PET
# Centiloids. Same pattern as Sections 5-11: fit -> score -> save metrics -> reload ->
# per-fold C-index, then bootstrap CIs for the AUC (as in Section 13). Plotted in Section 16.
# =====================================================================================

#PET + ptau217/AB42 ratio + demographics
random_state = 20
selected_cov_pet_ptau_amyloid = ["AGE_MCI", "Sex", "Education","APOE4", "pT217_AB42_F", "CENTILOIDS"]
X_pet_ptau_amyloid = merged_df[selected_cov_pet_ptau_amyloid]
y_pet_ptau_amyloid = Surv.from_arrays(event=merged_df.LABEL,
    time=merged_df.mci_event_daysdiff)

X = X_pet_ptau_amyloid
y = y_pet_ptau_amyloid['event']

# Define internal CV for hyperparameter tuning
cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

#Best params (fold): {'estimator__max_depth': 3, 'estimator__max_features': 0.7, 'estimator__min_samples_leaf': 10, 'estimator__min_samples_split': 2, 'estimator__n_estimators': 300}
param_grid = {
    "estimator__n_estimators": [200, 300, 500],            # number of trees
    "estimator__min_samples_split": [2, 5],            # minimum samples to split a node
    "estimator__min_samples_leaf": [10, 15],             # minimum samples in a leaf node
    "estimator__max_depth": [3, 5],
    "estimator__max_features": [0.5, 0.7, 0.9, "sqrt"],    # features considered at each split
}

if REFIT:  # skipped when reusing stored models (MCI_AD_REUSE_MODELS_FROM)
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(f"Fold {i}:")

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y_pet_ptau_amyloid[train_index], y_pet_ptau_amyloid[test_index]

        # Define a new RSF model to tune
        rsf_model = RandomSurvivalForest(n_estimators=1000, random_state=random_state, n_jobs=-1)

        # Define the scorer
        #time[-1] is the most recent time point
        scorer = as_concordance_index_ipcw_scorer(rsf_model, tau=time[-1])

        # Perform grid search to find the best hyperparameters
        gcv_cindex = GridSearchCV(
            scorer,
            param_grid=param_grid,
            cv=cv,
            n_jobs=-1
        )
    
        gcv_cindex.fit(X_train, y_train)
        best_rsf = gcv_cindex.best_estimator_
    
        #save best_rsf model for each fold
        joblib.dump(best_rsf, f"{MODELS_DIR}/earliest_pet_ptau_amyloid_rsf_fold{i}.joblib")
    
        #save predictions and labels
        train_predictions = best_rsf.predict(X_train)
        test_predictions = best_rsf.predict(X_test)
        train_chf_predictions = best_rsf.predict_cumulative_hazard_function(X_train)
        test_chf_predictions = best_rsf.predict_cumulative_hazard_function(X_test)
        train_surv_predictions = best_rsf.predict_survival_function(X_train)
        test_surv_predictions = best_rsf.predict_survival_function(X_test)



        np.savez_compressed(f"{PREDICTIONS_DIR}/earliest_pet_ptau_amyloid_rsf_fold{i}_predictions.npz", 
                            train_predictions=train_predictions, 
                            test_predictions=test_predictions,
                            train_chf_predictions=train_chf_predictions,
                            test_chf_predictions=test_chf_predictions,
                            train_surv_predictions=train_surv_predictions,
                            test_surv_predictions=test_surv_predictions,
                            train_labels = y_train,
                            test_labels = y_test
                            )

#Metrics Calculation
X = X_pet_ptau_amyloid
y = y_pet_ptau_amyloid['event']

#Test Result Metrics
pet_ptau_amyloid_rsf_c_index_list = []
pet_ptau_amyloid_rsf_auc_list = []
pet_ptau_amyloid_rsf_mean_auc_list = []
pet_ptau_amyloid_rsf_chf_scores_list = []
pet_ptau_amyloid_rsf_surv_scores_list = []
pet_ptau_amyloid_rsf_bs_list = []
pet_ptau_amyloid_rsf_ibs_list = []

#Train Result Metrics
pet_ptau_amyloid_rsf_train_c_index_list = []
pet_ptau_amyloid_rsf_train_auc_list = []
pet_ptau_amyloid_rsf_train_mean_auc_list = []
pet_ptau_amyloid_rsf_train_chf_scores_list = []
pet_ptau_amyloid_rsf_train_surv_scores_list = []
pet_ptau_amyloid_rsf_train_bs_list = []
pet_ptau_amyloid_rsf_train_ibs_list = []

for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(f"Fold {i}:")
    #load rsf predictions for each fold
    loaded_data = np.load(f"{PREDICTIONS_IN}/earliest_pet_ptau_amyloid_rsf_fold{i}_predictions.npz", allow_pickle=True)
    train_predictions = loaded_data['train_predictions']
    test_predictions = loaded_data['test_predictions']
    train_chf_predictions = loaded_data['train_chf_predictions']
    test_chf_predictions = loaded_data['test_chf_predictions']
    train_surv_predictions = loaded_data['train_surv_predictions']
    test_surv_predictions = loaded_data['test_surv_predictions']
    y_train = loaded_data['train_labels']
    y_test = loaded_data['test_labels']
    
#TEST METRICS
    #calculate concordance index on TEST data
    pet_ptau_amyloid_rsf_c_index = concordance_index_ipcw(y_train, y_test, test_predictions)
    pet_ptau_amyloid_rsf_c_index_list.append(pet_ptau_amyloid_rsf_c_index)

    #calculate time-dependent risk scores on TEST data
    pet_ptau_amyloid_rsf_chf_scores = np.vstack([chf(time) for chf in test_chf_predictions])
    pet_ptau_amyloid_rsf_chf_scores_list.append(pet_ptau_amyloid_rsf_chf_scores)

    #calculate time-dependent AUC and mean AUC on TEST data
    pet_ptau_amyloid_rsf_auc, pet_ptau_amyloid_rsf_mean_auc = cumulative_dynamic_auc(y_train, y_test, pet_ptau_amyloid_rsf_chf_scores, time)
    pet_ptau_amyloid_rsf_auc_list.append(pet_ptau_amyloid_rsf_auc)
    pet_ptau_amyloid_rsf_mean_auc_list.append(pet_ptau_amyloid_rsf_mean_auc)
    
    #calculate survival probabilities on TEST data
    pet_ptau_amyloid_rsf_surv_scores = np.vstack([fn(time) for fn in test_surv_predictions])
    pet_ptau_amyloid_rsf_surv_scores_list.append(pet_ptau_amyloid_rsf_surv_scores)
    
    #calculate Brier score on TEST data
    pet_ptau_amyloid_rsf_bs = brier_score(y_train, y_test, pet_ptau_amyloid_rsf_surv_scores, time)
    pet_ptau_amyloid_rsf_bs_list.append(pet_ptau_amyloid_rsf_bs)
    
    pet_ptau_amyloid_rsf_ibs = integrated_brier_score(y_train, y_test, pet_ptau_amyloid_rsf_surv_scores, time)
    pet_ptau_amyloid_rsf_ibs_list.append(pet_ptau_amyloid_rsf_ibs)
    
#TRAINING METRICS
    #calculate concordance index on TRAIN data
    pet_ptau_amyloid_rsf_train_c_index = concordance_index_ipcw(y_train, y_train, train_predictions)
    pet_ptau_amyloid_rsf_train_c_index_list.append(pet_ptau_amyloid_rsf_train_c_index)
    
    #calculate time-dependent risk scores on TRAIN data
    pet_ptau_amyloid_rsf_train_chf_scores = np.vstack([chf(time) for chf in train_chf_predictions])
    pet_ptau_amyloid_rsf_train_chf_scores_list.append(pet_ptau_amyloid_rsf_train_chf_scores)
    
    #calculate time-dependent AUC and mean AUC on TRAIN data
    pet_ptau_amyloid_rsf_train_auc, pet_ptau_amyloid_rsf_train_mean_auc = cumulative_dynamic_auc(y_train, y_train, pet_ptau_amyloid_rsf_train_chf_scores, time)
    pet_ptau_amyloid_rsf_train_auc_list.append(pet_ptau_amyloid_rsf_train_auc)
    pet_ptau_amyloid_rsf_train_mean_auc_list.append(pet_ptau_amyloid_rsf_train_mean_auc)
    
    #calculate survival probabilities on TRAIN data
    pet_ptau_amyloid_rsf_train_surv_scores = np.vstack([fn(time) for fn in train_surv_predictions])
    pet_ptau_amyloid_rsf_train_surv_scores_list.append(pet_ptau_amyloid_rsf_train_surv_scores)
    
    #calculate Brier score on TRAIN data
    pet_ptau_amyloid_rsf_train_bs = brier_score(y_train, y_train, pet_ptau_amyloid_rsf_train_surv_scores, time)
    pet_ptau_amyloid_rsf_train_bs_list.append(pet_ptau_amyloid_rsf_train_bs)

    pet_ptau_amyloid_rsf_train_ibs = integrated_brier_score(y_train, y_train, pet_ptau_amyloid_rsf_train_surv_scores, time)
    pet_ptau_amyloid_rsf_train_ibs_list.append(pet_ptau_amyloid_rsf_train_ibs)

np.savez_compressed(
    f"{METRICS_DIR}/pet_ptau_amyloid_metrics.npz",
    rsf_c_index_list=pet_ptau_amyloid_rsf_c_index_list,
    rsf_auc_list=pet_ptau_amyloid_rsf_auc_list,
    rsf_mean_auc_list=pet_ptau_amyloid_rsf_mean_auc_list,
    rsf_chf_scores_list=np.array(pet_ptau_amyloid_rsf_chf_scores_list, dtype=object),
    rsf_surv_scores_list=np.array(pet_ptau_amyloid_rsf_surv_scores_list, dtype=object),
    rsf_bs_list=pet_ptau_amyloid_rsf_bs_list,
    rsf_ibs_list=pet_ptau_amyloid_rsf_ibs_list,
    rsf_train_c_index_list=pet_ptau_amyloid_rsf_train_c_index_list,
    rsf_train_auc_list=pet_ptau_amyloid_rsf_train_auc_list,
    rsf_train_mean_auc_list=pet_ptau_amyloid_rsf_train_mean_auc_list,
    rsf_train_chf_scores_list=np.array(pet_ptau_amyloid_rsf_train_chf_scores_list, dtype=object),
    rsf_train_surv_scores_list=np.array(pet_ptau_amyloid_rsf_train_surv_scores_list,dtype =object),
    rsf_train_bs_list=pet_ptau_amyloid_rsf_train_bs_list,
    rsf_train_ibs_list=pet_ptau_amyloid_rsf_train_ibs_list
)

#load metrics
pet_ptau_amyloid_metrics = np.load(f"{METRICS_DIR}/pet_ptau_amyloid_metrics.npz", allow_pickle=True)

pet_ptau_amyloid_rsf_mean_auc_list = pet_ptau_amyloid_metrics['rsf_mean_auc_list']
pet_ptau_amyloid_rsf_c_index_list = pet_ptau_amyloid_metrics['rsf_c_index_list']
pet_ptau_amyloid_rsf_auc_list = pet_ptau_amyloid_metrics['rsf_auc_list']
pet_ptau_amyloid_rsf_chf_scores_list = pet_ptau_amyloid_metrics['rsf_chf_scores_list']
pet_ptau_amyloid_rsf_surv_scores_list = pet_ptau_amyloid_metrics['rsf_surv_scores_list']
pet_ptau_amyloid_rsf_bs_list=pet_ptau_amyloid_metrics['rsf_bs_list']
pet_ptau_amyloid_rsf_ibs_list=pet_ptau_amyloid_metrics['rsf_ibs_list']
pet_ptau_amyloid_rsf_train_c_index_list=pet_ptau_amyloid_metrics['rsf_train_c_index_list']
pet_ptau_amyloid_rsf_train_auc_list=pet_ptau_amyloid_metrics['rsf_train_auc_list']
pet_ptau_amyloid_rsf_train_mean_auc_list=pet_ptau_amyloid_metrics['rsf_train_mean_auc_list']
pet_ptau_amyloid_rsf_train_chf_scores_list=pet_ptau_amyloid_metrics['rsf_train_chf_scores_list']
pet_ptau_amyloid_rsf_train_surv_scores_list=pet_ptau_amyloid_metrics['rsf_train_surv_scores_list']
pet_ptau_amyloid_rsf_train_bs_list=pet_ptau_amyloid_metrics['rsf_train_bs_list']
pet_ptau_amyloid_rsf_train_ibs_list=pet_ptau_amyloid_metrics['rsf_train_ibs_list']

#C-index is one value per fold
pet_ptau_amyloid_c_list =[]
for i in range(len(pet_ptau_amyloid_rsf_c_index_list)):
    pet_ptau_amyloid_rsf_c_index = pet_ptau_amyloid_rsf_c_index_list[i][0]
    pet_ptau_amyloid_c_list.append(pet_ptau_amyloid_rsf_c_index)
    
#get mean aucs over th 25 timepoints per fold, there are no NaN values
pet_ptau_amyloid_mean_auc_list = []
for i in range(len(pet_ptau_amyloid_rsf_auc_list)):
    pet_ptau_amyloid_mean_auc = np.nanmean(pet_ptau_amyloid_rsf_auc_list[i])
    pet_ptau_amyloid_mean_auc_list.append(pet_ptau_amyloid_mean_auc)

#calculate AUCs and 95 CIs for PET+ ptau model
pet_ptau_amyloid_y_test = []
pet_ptau_amyloid_risk_scores = []
y_train_full = y_pet_ptau_amyloid

#Step One: load fold predictions
for i in range(5):
    loaded_data = np.load(
        f"{PREDICTIONS_IN}/earliest_pet_ptau_amyloid_rsf_fold{i}_predictions.npz",
        allow_pickle=True
    )

    test_chf_predictions = loaded_data['test_chf_predictions']
    y_test = loaded_data['test_labels']

    # convert CHF functions to matrix
    risk_matrix = np.vstack([chf(time) for chf in test_chf_predictions])

    pet_ptau_amyloid_y_test.append(y_test)
    pet_ptau_amyloid_risk_scores.append(risk_matrix)

pet_ptau_amyloid_y_test = np.concatenate(pet_ptau_amyloid_y_test)
pet_ptau_amyloid_risk_scores = np.vstack(pet_ptau_amyloid_risk_scores)

#Step Two: Compute AUC
pet_ptau_amyloid_auc_original, pet_ptau_amyloid_mean_auc_original = cumulative_dynamic_auc(
    y_train_full,   # survival distribution for IPCW
    pet_ptau_amyloid_y_test,
    pet_ptau_amyloid_risk_scores,
    time
)

#Step Three: Boostrap
pet_ptau_amyloid_auc_mean, pet_ptau_amyloid_auc_lower, pet_ptau_amyloid_auc_upper, pet_ptau_amyloid_mean_auc_ci = bootstrap_dynamic_auc(
    y_train_full,
    pet_ptau_amyloid_y_test,
    pet_ptau_amyloid_risk_scores,
    time,
    n_boot=1000,
    random_state=42
)

#save results
np.savez(
    f"{METRICS_DIR}/earliest_pet_ptau_amyloid_rsf_auc_bootstrap.npz",
    pet_ptau_amyloid_auc_mean=pet_ptau_amyloid_auc_mean,
    pet_ptau_amyloid_auc_lower=pet_ptau_amyloid_auc_lower,
    pet_ptau_amyloid_auc_upper=pet_ptau_amyloid_auc_upper,
    pet_ptau_amyloid_mean_auc_ci=pet_ptau_amyloid_mean_auc_ci
)

print(f"Original mean AUC: {pet_ptau_amyloid_mean_auc_original:.3f}")
print(f"95% CI for mean AUC: [{pet_ptau_amyloid_mean_auc_ci[0]:.3f}, {pet_ptau_amyloid_mean_auc_ci[1]:.3f}]")

# Performance summary for this model, in the same form as Section 12
pet_ptau_amyloid_rsf_overall_mean_cindex = np.mean(pet_ptau_amyloid_c_list)
pet_ptau_amyloid_rsf_overall_std_cindex = np.std(pet_ptau_amyloid_c_list, ddof=1)
print(f"PET + p-Tau217/Abeta42 Overall Mean C-index: {pet_ptau_amyloid_rsf_overall_mean_cindex:.3f} ± {pet_ptau_amyloid_rsf_overall_std_cindex:.3f}")

pet_ptau_amyloid_rsf_bs_arr = np.array(pet_ptau_amyloid_rsf_bs_list)
pet_ptau_amyloid_rsf_mean_bs = np.nanmean(pet_ptau_amyloid_rsf_bs_arr[:,1,:], axis=0)
pet_ptau_amyloid_rsf_std_bs = np.nanstd(pet_ptau_amyloid_rsf_bs_arr[:,1,:], axis=0)
print(f"PET + p-Tau217/Abeta42 Mean Brier Score: {np.mean(pet_ptau_amyloid_rsf_mean_bs):.3f} ± {np.mean(pet_ptau_amyloid_rsf_std_bs):.3f}")


# =====================================================================================
# 16 · Manuscript comparison figures
#
#   bloodbios_demo_auc_comparison.pdf      AUC: baseline, p-tau217, ratio, blood panel
#   pet_bloodbios_auc_comparison.pdf       AUC: PET and blood-biomarker models
#   pet_bloodbios_bs_comparison.pdf        Brier score, same set
#   pet_bloodbios_cindex_comparison.pdf    C-index, PET and blood-biomarker models
#   demo_bloodbios_cindex_comparison.pdf   C-index: baseline, p-tau217, ratio, blood panel
#   demo_bloodbios_bs_comparison.pdf       Brier score, same four
#
# auc_std and add_complete_risk_table are the two plotting helpers used here.
# =====================================================================================

def auc_std(auc_list):
    """Std of per-fold mean AUC (one AUC per fold, averaged across timepoints)."""
    fold_means = np.nanmean(auc_list, axis=1)
    return np.nanstd(fold_means)


def add_complete_risk_table(ax, time, y_test_folds, fig, step=1, fontsize=10):
    """
    y_test_folds: list of structured arrays (one per fold), each with 'event' and 'time' fields.
    Counts are summed across all folds at each time point.
    """
    time_years = time / 365.25
    display_idx = list(range(0, len(time), step))
    
    cases = np.zeros(len(time), dtype=int)
    censored = np.zeros(len(time), dtype=int)
    
    for y_test in y_test_folds:
        for i, t_days in enumerate(time):
            cases[i] += np.sum((y_test['event'] == True) & (y_test['time'] > t_days))
            censored[i] += np.sum((y_test['event'] == False) & (y_test['time'] > t_days))
    
    xlim = ax.get_xlim()
    
    y_cases = -0.18
    y_censored = -0.26
    
    case_color = '#8B0000'
    censor_color = '#2F4F7F'
    
    ax.text(-0.02, y_cases, 'Cases', transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold', color=case_color, ha='right', va='center')
    ax.text(-0.02, y_censored, 'Censored', transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold', color=censor_color, ha='right', va='center')
    
    for i in display_idx:
        x_frac = (time_years[i] - xlim[0]) / (xlim[1] - xlim[0])
        
        ax.text(x_frac, y_cases, str(cases[i]), transform=ax.transAxes,
                fontsize=fontsize, color=case_color, ha='center', va='center')
        ax.text(x_frac, y_censored, str(censored[i]), transform=ax.transAxes,
                fontsize=fontsize, color=censor_color, ha='center', va='center')

# Load all 5 folds
y_test_all_folds = []
for fold in range(5):
    loaded = np.load(
        f"{PREDICTIONS_IN}/earliest_bio_rsf_fold{fold}_predictions.npz",
        allow_pickle=True
    )
    y_test_all_folds.append(loaded['test_labels'])
    
# Consistent palette across all thesis figures
PALETTE = {
    'p-Tau217': '#2ca02c',       # green
    'Aβ42': '#1f77b4',            # blue
    'p-Tau217/Aβ42' : '#e377c2',  # pink
    'Blood biomarkers': '#9467bd', # purple 
}

# ── Your existing AUROC plot ──
fig, ax = plt.subplots(figsize=(12, 7))

# Your existing plotting code

demo_rsf_mean_auc = np.nanmean(demo_rsf_auc_list, axis=0)
ax.plot(time/365.25, demo_rsf_mean_auc, "o-", label=f"Baseline (mean AUC = {np.mean(demo_rsf_mean_auc):.3f} ± {auc_std(demo_rsf_auc_list):.3f})")
ax.fill_between(time/365.25, demo_auc_lower, demo_auc_upper, alpha=0.2)

ptau_rsf_mean_auc = np.nanmean(ptau_rsf_auc_list, axis=0)
ax.plot(time/365.25, ptau_rsf_mean_auc, "o-", color = PALETTE['p-Tau217'], label=f"p-Tau217 (mean AUC = {np.mean(ptau_rsf_mean_auc):.3f} ± {auc_std(ptau_rsf_auc_list):.3f})")
ax.fill_between(time/365.25, ptau_auc_lower, ptau_auc_upper, color = PALETTE['p-Tau217'], alpha=0.2)

ptau_amyloid_rsf_mean_auc = np.nanmean(ptau_amyloid_rsf_auc_list, axis=0)
ax.plot(time/365.25, ptau_amyloid_rsf_mean_auc, "o-", color = PALETTE['p-Tau217/Aβ42'], label=f"p-Tau217/Aβ42 (mean AUC = {np.mean(ptau_amyloid_rsf_mean_auc):.3f} ± {auc_std(ptau_amyloid_rsf_auc_list):.3f})")
ax.fill_between(time/365.25, ptau_amyloid_auc_lower, ptau_amyloid_auc_upper, color = PALETTE['p-Tau217/Aβ42'], alpha=0.2)

bio_rsf_mean_auc = np.nanmean(bio_rsf_auc_list, axis=0)
ax.plot(time/365.25, bio_rsf_mean_auc, "o-", color=PALETTE['Blood biomarkers'], label=f"Blood biomarkers (mean AUC = {np.mean(bio_rsf_mean_auc):.3f} ± {auc_std(bio_rsf_auc_list):.3f})")
ax.fill_between(time/365.25, bio_auc_lower, bio_auc_upper, color=PALETTE['Blood biomarkers'], alpha=0.2)

ax.set_ylabel("AUROC", fontsize=14, fontweight='bold')
ax.set_xlabel("Follow-up time (years)", fontsize=14, fontweight='bold')
ax.legend(loc="lower center", fontsize=9)
ax.grid(True)
ax.set_ylim(0.2, 1.0)

# ── Add risk table summed across all 5 folds ──
plt.subplots_adjust(bottom=0.30)
add_complete_risk_table(ax, time, y_test_all_folds, fig, step=1)
plt.savefig(f"{FIGURES_DIR}/bloodbios_demo_auc_comparison.pdf")

# Load all 5 folds
y_test_all_folds = []
for fold in range(5):
    loaded = np.load(
        f"{PREDICTIONS_IN}/earliest_bio_rsf_fold{fold}_predictions.npz",
        allow_pickle=True
    )
    y_test_all_folds.append(loaded['test_labels'])

# Consistent palette across all thesis figures
PALETTE = {
    'PET' : '#ff7f0e',             # orange
    'p-Tau217': '#2ca02c',       # green
    'PET + p-Tau217': '#d62728',  # red
    'p-Tau217/Aβ42': '#e377c2',  # pink
    'PET + p-Tau217/Aβ42': '#17becf', # cyan  
    'Blood biomarkers': '#9467bd', # purple 
    'PET + biomarkers': '#8c564b', # brown 
}

# ── Your existing AUROC plot ──
fig, ax = plt.subplots(figsize=(12, 7))

# Your existing plotting code

rsf_mean_auc = np.nanmean(rsf_auc_list, axis=0)
ax.plot(time/365.25, rsf_mean_auc, "o-", color = PALETTE['PET'], label=f"PET (mean AUC = {np.mean(rsf_mean_auc):.3f}± {auc_std(rsf_auc_list):.3f})")
ax.fill_between(time/365.25, pet_auc_lower,pet_auc_upper, color = PALETTE['PET'], alpha=0.2)

ptau_rsf_mean_auc = np.nanmean(ptau_rsf_auc_list, axis=0)
ax.plot(time/365.25, ptau_rsf_mean_auc, "o-", color = PALETTE['p-Tau217'], label=f"p-Tau217 (mean AUC = {np.mean(ptau_rsf_mean_auc):.3f})")
ax.fill_between(time/365.25, ptau_auc_lower, ptau_auc_upper, color = PALETTE['p-Tau217'], alpha=0.2)

merged_rsf_mean_auc = np.nanmean(merged_rsf_auc_list, axis=0)
ax.plot(time/365.25, merged_rsf_mean_auc, "o-", color = PALETTE['PET + p-Tau217'], label=f"PET + p-Tau217 (mean AUC = {np.mean(merged_rsf_mean_auc):.3f} ± {auc_std(merged_rsf_auc_list):.3f})")
ax.fill_between(time/365.25,pet_ptau_auc_lower,pet_ptau_auc_upper,color = PALETTE['PET + p-Tau217'], alpha=0.2)

ptau_amyloid_rsf_mean_auc = np.nanmean(ptau_amyloid_rsf_auc_list, axis=0)
ax.plot(time/365.25, ptau_amyloid_rsf_mean_auc, "o-", color = PALETTE['p-Tau217/Aβ42'], label=f"p-Tau217/Aβ42 (mean AUC = {np.mean(ptau_amyloid_rsf_mean_auc):.3f} ± {auc_std(ptau_amyloid_rsf_auc_list):.3f})")
ax.fill_between(time/365.25, ptau_amyloid_auc_lower, ptau_amyloid_auc_upper, color = PALETTE['p-Tau217/Aβ42'], alpha=0.2)

pet_ptau_amyloid_rsf_mean_auc = np.nanmean(pet_ptau_amyloid_rsf_auc_list, axis=0)
ax.plot(time/365.25, pet_ptau_amyloid_rsf_mean_auc, "o-", color = PALETTE['PET + p-Tau217/Aβ42'], label=f"PET + p-Tau217/Aβ42 (mean AUC = {np.mean(pet_ptau_amyloid_rsf_mean_auc):.3f} ± {auc_std(pet_ptau_amyloid_rsf_auc_list):.3f})")
ax.fill_between(time/365.25, pet_ptau_amyloid_auc_lower, pet_ptau_amyloid_auc_upper, color = PALETTE['PET + p-Tau217/Aβ42'], alpha=0.2)

bio_rsf_mean_auc = np.nanmean(bio_rsf_auc_list, axis=0)
ax.plot(time/365.25, bio_rsf_mean_auc, "o-", color = PALETTE['Blood biomarkers'], label=f"Blood biomarkers (mean AUC = {np.mean(bio_rsf_mean_auc):.3f} ± {auc_std(bio_rsf_auc_list):.3f})")
ax.fill_between(time/365.25, bio_auc_lower, bio_auc_upper, color = PALETTE['Blood biomarkers'], alpha=0.2)

ksink_rsf_mean_auc = np.nanmean(ksink_rsf_auc_list, axis=0)
ax.plot(time/365.25, ksink_rsf_mean_auc, "o-", color = PALETTE['PET + biomarkers'], label=f"PET + blood biomarkers (mean AUC = {np.mean(ksink_rsf_mean_auc):.3f} ± {auc_std(ksink_rsf_auc_list):.3f})")
ax.fill_between(time/365.25, ksink_auc_lower, ksink_auc_upper, color = PALETTE['PET + biomarkers'], alpha=0.2)

ax.set_ylabel("AUROC", fontsize=14, fontweight='bold')
ax.set_xlabel("Follow-up time (years)", fontsize=14, fontweight='bold')
ax.legend(loc="lower center", fontsize=9)
ax.grid(True)
ax.set_ylim(0.3, 1.0)

# ── Add risk table from all folds ──
plt.subplots_adjust(bottom=0.30)  # increased from 0.25
add_complete_risk_table(ax, time, y_test_all_folds, fig, step=1)
plt.savefig(f"{FIGURES_DIR}/pet_bloodbios_auc_comparison.pdf")

# Load all 5 folds
y_test_all_folds = []
for fold in range(5):
    loaded = np.load(
        f"{PREDICTIONS_IN}/earliest_bio_rsf_fold{fold}_predictions.npz",
        allow_pickle=True
    )
    y_test_all_folds.append(loaded['test_labels'])

# Consistent palette across all thesis figures
PALETTE = {
    'PET' : '#ff7f0e',             # orange
    'p-Tau217': '#2ca02c',       # green
    'PET + p-Tau217': '#d62728',  # red
    'p-Tau217/Aβ42': '#e377c2',  # pink
    'PET + p-Tau217/Aβ42': '#17becf', # cyan  
    'Blood biomarkers': '#9467bd', # purple 
    'PET + biomarkers': '#8c564b', # brown 
}

# ── Brier Score plot ──
fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)

rsf_bs_arr = np.array(rsf_bs_list)  
rsf_mean_bs = np.nanmean(rsf_bs_arr[:,1,:], axis=0)
rsf_std_bs = np.nanstd(rsf_bs_arr[:,1,:], axis=0)

ax.plot(time/365.25, rsf_mean_bs, "o-", color = PALETTE['PET'], label=f"PET (mean IBS = {np.mean(rsf_ibs_list):.3f})")
ax.fill_between(time/365.25, rsf_mean_bs - rsf_std_bs, rsf_mean_bs + rsf_std_bs, color = PALETTE['PET'], alpha=0.2)

ptau_rsf_bs_arr = np.array(ptau_rsf_bs_list)  
ptau_rsf_mean_bs = np.nanmean(ptau_rsf_bs_arr[:,1,:], axis=0)
ptau_rsf_std_bs = np.nanstd(ptau_rsf_bs_arr[:,1,:], axis=0)

ax.plot(time/365.25, ptau_rsf_mean_bs, "o-", color = PALETTE['p-Tau217'], label=f"p-Tau217 (mean IBS = {np.mean(ptau_rsf_ibs_list):.3f})")
ax.fill_between(time/365.25, ptau_rsf_mean_bs - ptau_rsf_std_bs, ptau_rsf_mean_bs + ptau_rsf_std_bs, color = PALETTE['p-Tau217'], alpha=0.2)

merged_rsf_bs_arr = np.array(merged_rsf_bs_list) 
merged_rsf_mean_bs = np.nanmean(merged_rsf_bs_arr[:,1,:], axis=0)
merged_rsf_std_bs = np.nanstd(merged_rsf_bs_arr[:,1,:], axis=0)

ax.plot(time/365.25, merged_rsf_mean_bs, "o-", color = PALETTE['PET + p-Tau217'], label=f"PET+p-Tau217 (mean IBS = {np.mean(merged_rsf_ibs_list):.3f})")
ax.fill_between(time/365.25, merged_rsf_mean_bs - merged_rsf_std_bs, merged_rsf_mean_bs + merged_rsf_std_bs, color = PALETTE['PET + p-Tau217'], alpha=0.2)

ptau_amyloid_rsf_bs_arr = np.array(ptau_amyloid_rsf_bs_list)
ptau_amyloid_rsf_mean_bs = np.nanmean(ptau_amyloid_rsf_bs_arr[:,1,:], axis=0)
ptau_amyloid_rsf_std_bs = np.nanstd(ptau_amyloid_rsf_bs_arr[:,1,:], axis=0)

ax.plot(time/365.25, ptau_amyloid_rsf_mean_bs, "o-", color = PALETTE['p-Tau217/Aβ42'], label=f"p-Tau217/Abeta42 (mean IBS = {np.mean(ptau_amyloid_rsf_ibs_list):.3f})")
ax.fill_between(time/365.25, ptau_amyloid_rsf_mean_bs - ptau_amyloid_rsf_std_bs, ptau_amyloid_rsf_mean_bs + ptau_amyloid_rsf_std_bs, color = PALETTE['p-Tau217/Aβ42'], alpha=0.2)


bio_rsf_bs_arr = np.array(bio_rsf_bs_list) 
bio_rsf_mean_bs = np.nanmean(bio_rsf_bs_arr[:,1,:], axis=0)
bio_rsf_std_bs = np.nanstd(bio_rsf_bs_arr[:,1,:], axis=0)

ax.plot(time/365.25, bio_rsf_mean_bs, "o-", color = PALETTE['Blood biomarkers'], label=f"Blood biomarkers (mean IBS = {np.mean(bio_rsf_ibs_list):.3f})")
ax.fill_between(time/365.25, bio_rsf_mean_bs - bio_rsf_std_bs, bio_rsf_mean_bs + bio_rsf_std_bs, color = PALETTE['Blood biomarkers'], alpha=0.2)

ksink_rsf_bs_arr = np.array(ksink_rsf_bs_list) 
ksink_rsf_mean_bs = np.nanmean(ksink_rsf_bs_arr[:,1,:], axis=0)
ksink_rsf_std_bs = np.nanstd(ksink_rsf_bs_arr[:,1,:], axis=0)

ax.plot(time/365.25, ksink_rsf_mean_bs, "o-", color = PALETTE['PET + biomarkers'], label=f"PET + biomarkers (mean IBS = {np.mean(ksink_rsf_ibs_list):.3f})")
ax.fill_between(time/365.25, ksink_rsf_mean_bs - ksink_rsf_std_bs, ksink_rsf_mean_bs + ksink_rsf_std_bs, color = PALETTE['PET + biomarkers'], alpha=0.2)


ax.set_xlabel("Follow-up time (years)", fontsize=12, fontweight='bold')
ax.set_ylabel("Time-dependent Brier Score", fontsize=12, fontweight='bold')
ax.tick_params(axis='both', which='major', labelsize=12, width=2)
legend_properties = {'weight': 'bold'}
ax.legend(loc="lower center", prop=legend_properties)
plt.grid(True)
plt.savefig(f"{FIGURES_DIR}/pet_bloodbios_bs_comparison.pdf", bbox_inches='tight')

#C-index plot
import itertools

model_names = ['PET', 'p-Tau217', 'PET + p-Tau217', 'p-Tau217/Aβ42',
                'PET + p-Tau217/Aβ42', 'Blood biomarkers', 'PET + biomarkers']

all_c_lists = [rsf_c_list, ptau_c_list, merged_c_list, ptau_amyloid_c_list,
               pet_ptau_amyloid_c_list, bio_c_list, ksink_c_list]

# Built from actual list lengths rather than assuming 5 per model,
# in case fold counts differ across models
model_labels = list(itertools.chain.from_iterable(
    [name] * len(lst) for name, lst in zip(model_names, all_c_lists)
))
combined_c_list = list(itertools.chain.from_iterable(all_c_lists))

PALETTE = {
    'PET': '#ff7f0e',                    # orange
    'p-Tau217': '#2ca02c',                # green
    'PET + p-Tau217': '#d62728',          # red
    'p-Tau217/Aβ42': '#e377c2',           # pink
    'PET + p-Tau217/Aβ42': '#17becf',     # cyan
    'Blood biomarkers': '#9467bd',        # purple
    'PET + biomarkers': '#8c564b',        # brown
}

fig, ax = plt.subplots(figsize=(10, 6))

sns.boxplot(x=model_labels, y=combined_c_list, order=model_names,
            palette=PALETTE, showmeans=False,
            meanprops={'marker': '^', 'markerfacecolor': 'black', 'markeredgecolor': 'black', 'markersize': 8},
            ax=ax)
sns.stripplot(x=model_labels, y=combined_c_list, order=model_names,
              color='darkorange', size=5, alpha=0.7, ax=ax)

ax.set_ylabel("C-index", fontsize=14, fontweight='bold')
ax.set_xlabel("")
ax.tick_params(axis='y', labelsize=12)
ax.tick_params(axis='x', labelsize=11, width=0, rotation=20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/pet_bloodbios_cindex_comparison.pdf", bbox_inches='tight')

# Define colors to match AUROC plot
model_colors = {
    'Baseline': '#1f77b4',
    'p-Tau217': '#2ca02c',
    'p-Tau217/Aβ42' : '#e377c2',  # pink
    'Blood\nBiomarkers': '#9467bd'
}

order = ['Baseline', 'p-Tau217', 'p-Tau217/Aβ42', 'Blood\nBiomarkers']
palette = [model_colors[m] for m in order]

# Use wrapped labels for the data
model = ['Baseline']*5 + ['p-Tau217']*5 + ['p-Tau217/Aβ42']*5 + ['Blood\nBiomarkers']*5
combined_c_list = demo_c_list + ptau_c_list + ptau_amyloid_c_list + bio_c_list

plt.figure(figsize=(8, 6))

sns.boxplot(x=model, y=combined_c_list, palette=palette, showmeans=True,
            order=order)
sns.stripplot(x=model, y=combined_c_list, color='#ff7f0e', size=6, alpha=0.8,
              order=order, edgecolor='none', linewidth=0.5)

plt.ylabel("C-index", fontsize=14, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold')
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/demo_bloodbios_cindex_comparison.pdf", bbox_inches='tight')

# Consistent palette across all thesis figures
PALETTE = {
    'demo': '#1f77b4',            # blue
    'p-Tau217': '#2ca02c',       # green
    'p-Tau217/Aβ42' : '#e377c2',  # pink
    'Blood biomarkers': '#9467bd' # purple 
}

fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)

demo_rsf_bs_arr = np.array(demo_rsf_bs_list)  
demo_rsf_mean_bs = np.nanmean(demo_rsf_bs_arr[:,1,:], axis=0)
demo_rsf_std_bs = np.nanstd(demo_rsf_bs_arr[:,1,:], axis=0)

ax.plot(time/365.25, demo_rsf_mean_bs, "o-", label=f"Baseline (mean IBS = {np.mean(demo_rsf_ibs_list):.3f})", color=PALETTE['demo'])
ax.fill_between(time/365.25, demo_rsf_mean_bs - demo_rsf_std_bs, demo_rsf_mean_bs + demo_rsf_std_bs, alpha=0.2, color=PALETTE['demo'])

ptau_rsf_bs_arr = np.array(ptau_rsf_bs_list)  
ptau_rsf_mean_bs = np.nanmean(ptau_rsf_bs_arr[:,1,:], axis=0)
ptau_rsf_std_bs = np.nanstd(ptau_rsf_bs_arr[:,1,:], axis=0)

ax.plot(time/365.25, ptau_rsf_mean_bs, "o-", label=f"p-Tau217 (mean IBS = {np.mean(ptau_rsf_ibs_list):.3f})", color=PALETTE['p-Tau217'])
ax.fill_between(time/365.25, ptau_rsf_mean_bs - ptau_rsf_std_bs, ptau_rsf_mean_bs + ptau_rsf_std_bs, alpha=0.2, color=PALETTE['p-Tau217'])

ptau_amyloid_rsf_bs_arr = np.array(ptau_amyloid_rsf_bs_list)
ptau_amyloid_rsf_mean_bs = np.nanmean(ptau_amyloid_rsf_bs_arr[:,1,:], axis=0)
ptau_amyloid_rsf_std_bs = np.nanstd(ptau_amyloid_rsf_bs_arr[:,1,:], axis=0)

ax.plot(time/365.25, ptau_amyloid_rsf_mean_bs, "o-", label=f"p-Tau217/Abeta42 (mean IBS = {np.mean(ptau_amyloid_rsf_ibs_list):.3f})", color=PALETTE['p-Tau217/Aβ42'])
ax.fill_between(time/365.25, ptau_amyloid_rsf_mean_bs - ptau_amyloid_rsf_std_bs, ptau_amyloid_rsf_mean_bs + ptau_amyloid_rsf_std_bs, alpha=0.2, color=PALETTE['p-Tau217/Aβ42'])

bio_rsf_bs_arr = np.array(bio_rsf_bs_list) 
bio_rsf_mean_bs = np.nanmean(bio_rsf_bs_arr[:,1,:], axis=0)
bio_rsf_std_bs = np.nanstd(bio_rsf_bs_arr[:,1,:], axis=0)

ax.plot(time/365.25, bio_rsf_mean_bs, "o-", label=f"p-Tau217, NfL, GFAP, Abeta42/40 (mean IBS = {np.mean(bio_rsf_ibs_list):.3f})", color=PALETTE['Blood biomarkers'])
ax.fill_between(time/365.25, bio_rsf_mean_bs - bio_rsf_std_bs, bio_rsf_mean_bs + bio_rsf_std_bs, alpha=0.2, color=PALETTE['Blood biomarkers'])

ax.set_xlabel("Follow-up time (years)", fontsize=12,fontweight='bold')
ax.set_ylabel("Time-dependent Brier Score",fontsize=12,fontweight='bold')
ax.tick_params(axis='both', which='major', labelsize=12, width=2)
legend_properties = {'weight': 'bold'}
ax.legend(loc="lower center", prop=legend_properties)
#plt.title("Time-dependent Brier Score over 5 folds")
plt.grid(True)
plt.savefig(f"{FIGURES_DIR}/demo_bloodbios_bs_comparison.pdf")
