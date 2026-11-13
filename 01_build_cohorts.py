"""
Build the ADNI baseline cohorts for the MCI-to-AD progression analysis.

Step 1 of the pipeline (see README.md). It reads the raw ADNI
downloads and produces two patient-level tables -- one row per patient, holding that
patient's earliest amyloid-PET scan or earliest plasma draw near their first MCI
diagnosis, plus demographics, APOE4 genotype and the survival outcome.

    LABEL = 1  (pMCI)  patient received an AD-dementia diagnosis during follow-up
    LABEL = 0  (sMCI)  patient remained stable; censored at their last recorded visit
    MCI_DATE           date of first MCI diagnosis -- the time origin for the models
    EVENTDATE          AD diagnosis (pMCI) or last visit (sMCI)

Inputs (under base_path)
    Diagnosis/DXSUM_06Mar2025.csv                            diagnosis history
    PET/ADSP_PHC_PET_Amyloid_Simple_07Jul2025.csv            harmonised PET Centiloids
    Data___Database/ADNIMERGE_07Jul2025.csv                  APOE4 genotype
    REGISTRY_22Aug2025.csv                                   visit registry (censoring)
    Demographics/PTDEMOG_06Mar2025.csv                       date of birth
    Biospecimen_Results/UPENN_PLASMA_FUJIREBIO_QUANTERIX_05Mar2025.csv   plasma markers

Outputs (under out_path)
    earliest_pet_df.parquet     291 patients
    earliest_ptau_df.parquet    239 patients

Their intersection, after the further exclusions in Section 1 of
02_survival_models.py, is the N=201 analysis cohort (157 sMCI / 44 pMCI).

Usage
    python 01_build_cohorts.py                    # paths from config.py / environment
    python 01_build_cohorts.py <raw_dir> <out_dir>

"""

import os
import sys

import pandas as pd

from config import RAW_DIR, CLEANED_DIR

# ---- Paths --------------------------------------------------------------------------
# Set in config.py; override with the ADNI_RAW_DIR / MCI_AD_OUTPUT_DIR environment
# variables, or pass both folders on the command line.
base_path = str(RAW_DIR) + "/"
out_path = f"{CLEANED_DIR}/"

if len(sys.argv) == 3:  # optional override, e.g. for a dry run into a scratch directory
    base_path = sys.argv[1].rstrip("/") + "/"
    out_path = sys.argv[2].rstrip("/") + "/"

if not os.path.isdir(base_path):
    raise SystemExit(f"Raw ADNI folder not found: {base_path}\n"
                     "Set ADNI_RAW_DIR (or pass <raw_dir> <out_dir>).")
os.makedirs(out_path, exist_ok=True)


# =====================================================================================
# Section 1 · Cohort-definition helpers
#
# The diagnosis coding changed between ADNI phases, so cohort membership has to be
# tested against both the ADNI1 columns (DXMCI, DXAD, DXAPP) and the DIAGNOSIS column
# used from ADNI2/GO onward.
# =====================================================================================

def build_mci_cohorts(dx, cohort = None):
    """All MCI visits, excluding MCI attributed to a non-AD etiology (DXMDUE == 2)."""
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
    """All AD-dementia visits, excluding dementia of a non-AD etiology (DXDDUE == 2)."""
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


def create_problem_list(pmci, cohort=None):
    """Flag pMCI patients whose record is internally inconsistent.

    A patient is flagged when they have no dated MCI visit, no dated AD visit, or an
    AD diagnosis that precedes their MCI diagnosis. These are dropped downstream.
    """
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


def get_first_mci_date(mci_dx_df, cohort=None):
    """Earliest dated MCI visit per patient."""
    if cohort is None:
        raise ValueError("Cohort must be specified")
    if cohort == "ADNI":
        # Get the first MCI diagnosis date for each patient
        mci_dates = mci_dx_df[(mci_dx_df.DXMCI == 1) |
            (mci_dx_df.DIAGNOSIS == 2) &
            (~mci_dx_df.DXDATE.isna())].groupby('RID').first().reset_index()
        return mci_dates


def get_first_ad_date(pmci_dx_df):
    """Earliest dated AD-dementia visit per patient."""
    first_ad_dx  = pmci_dx_df[
        ((pmci_dx_df.DXAD == 1) |
         (pmci_dx_df.DXAPP.isin([1,2])) |
         (pmci_dx_df.DIAGNOSIS == 3)) &
        (~pmci_dx_df.DXDATE.isna())
    ].groupby('RID').first().reset_index()
    return first_ad_dx


# =====================================================================================
# Section 2 · Load diagnoses and split stable vs. progressive MCI
#
# A patient appearing in both the MCI and the AD set progressed (pMCI); an MCI patient
# who never appears in the AD set is stable (sMCI).
# =====================================================================================

dx = pd.read_csv(base_path + "Diagnosis/DXSUM_06Mar2025.csv")

mci, mci_patients = build_mci_cohorts(dx=dx, cohort="ADNI")
ad, ad_patients = build_ad_cohorts(dx=dx, cohort="ADNI")
print(mci.RID.nunique(), ad.RID.nunique())
#progressive MCI Patient IDs
pmci_patients = list(set(mci_patients) & set(ad_patients))
#dataframe for progressive MCI patients
pmci = dx[dx.RID.isin(pmci_patients)]
print(len(pmci_patients), pmci.RID.nunique())

# Sort by patient and exam date
pmci = pmci.sort_values(by=['RID', 'EXAMDATE'], ascending=True)

#identify patients with MCI diagnosis after AD diagnosis
problem_list = create_problem_list(pmci, cohort="ADNI")
print(len(problem_list))

#stable MCI is defined by subtraction
smci = mci[~mci.RID.isin(pmci_patients)]
print(smci.shape, smci.RID.nunique())


# =====================================================================================
# Section 3 · Attach amyloid PET (ADSP harmonised Centiloids)
#
# ADSP_PHC_PET_Amyloid_Simple carries Centiloid values harmonised across tracers and
# ADNI phases, with age at scan, sex and years of education. For pMCI the frame is
# restricted to scans taken strictly before the AD diagnosis -- a scan taken after
# conversion cannot inform a prediction made at baseline.
# =====================================================================================

pet_path = "PET/ADSP_PHC_PET_Amyloid_Simple_07Jul2025.csv"
adsp = pd.read_csv(base_path + pet_path)

#create function to get smci and pmci cohorts
def get_mci_pmci_cohorts(df, smci_patients, pmci_patients):
    df = df[~df.RID.isin(problem_list)]
    print(df.shape)
    pmci_df = df[df.RID.isin(pmci_patients)]
    smci_df = df[df.RID.isin(smci_patients)]
    smci_df = smci_df[~smci_df.RID.isin(pmci_patients)]
    #sting saying how many patients are in MCI and pMCI cohorts
    print(f"sMCI cohort has {smci_df.RID.nunique()} patients, pMCI cohort has {pmci_df.RID.nunique()} patients")
    return df, smci_df, pmci_df

adsp, adsp_mci, adsp_pmci = get_mci_pmci_cohorts(adsp, mci_patients, pmci_patients)
print(adsp_mci.RID.nunique(), adsp_pmci.RID.nunique())

pmci = pmci.rename(columns={'EXAMDATE': 'DXDATE'})
pmci = pmci.sort_values(by=['RID', 'DXDATE'], ascending=True)

first_mci_dx = get_first_mci_date(pmci, cohort="ADNI")
#rename DXDATE to MCI_DATE
first_mci_dx = first_mci_dx.rename(columns={'DXDATE': 'MCI_DATE'})

first_ad_dx = get_first_ad_date(pmci)
#rename DXDATE to AD_DATE
first_ad_dx = first_ad_dx.rename(columns={'DXDATE': 'AD_DATE'})

adsp_pmci = adsp_pmci.rename(columns={'PHC_SCANDATE': 'PETDATE'})
adsp_pmci = adsp_pmci.sort_values(by=['RID', 'PETDATE'])
print(adsp_pmci.head(15))

adsp_pmci = adsp_pmci.rename(columns={'PHC_Age_PET': 'AGE_PET',
                'PHC_Sex': 'Sex',
                'PHC_Education': 'Education',
                'PHC_Diagnosis': 'Diagnosis',
                'PHC_CENTILOIDS': 'CENTILOIDS'})
print(adsp_pmci.head())

#select for AGE_PET, Sex, Education, Diagnosis, CENTILOIDS columns
print(adsp_pmci.loc[:,['RID', 'PETDATE', 'AGE_PET', 'Sex', 'Education', 'Diagnosis', 'CENTILOIDS']])

#merge with first MCI and AD diagnosis dates
print(adsp_pmci.shape)
adsp_pmci_merged = adsp_pmci.merge(first_ad_dx.loc[:,['RID','AD_DATE']], on='RID', how='left')
print(adsp_pmci_merged.shape)
adsp_pmci_merged = adsp_pmci_merged.merge(first_mci_dx.loc[:,['RID','MCI_DATE']], on='RID', how='left')
print(adsp_pmci_merged.shape)
#filter so that everyone has a PET scan before their first AD diagnosis date
#(these are ISO date strings, so the comparison is lexicographic but still correct)
adsp_pmci_final = adsp_pmci_merged.loc[adsp_pmci_merged.PETDATE < adsp_pmci_merged.AD_DATE]
print(adsp_pmci_final.shape)

print(adsp_pmci_merged.loc[:,['RID', 'PETDATE', 'AGE_PET', 'MCI_DATE', 'AD_DATE', 'Diagnosis', 'CENTILOIDS']].head(10))


# ---- Section 3.1 · Same merge for the stable-MCI cohort -----------------------------

mci = mci.rename(columns={'EXAMDATE': 'DXDATE'})
mci = mci.sort_values(by=['RID', 'DXDATE'], ascending=True)
first_smci_dx = get_first_mci_date(mci, cohort="ADNI")
#rename DXDATE to MCI_DATE
first_smci_dx = first_smci_dx.rename(columns={'DXDATE': 'MCI_DATE'})
print(first_smci_dx.shape, first_smci_dx.RID.nunique())

#merge the columns RID and MCI_DATE from first_mci_dx to adsp_mci (pet df)
adsp_mci = adsp_mci.rename(columns={'PHC_SCANDATE': 'PETDATE'})
adsp_mci_final = adsp_mci.merge(first_smci_dx.loc[:,['RID','MCI_DATE']], on='RID', how='left')
print(adsp_mci_final.shape, adsp_mci_final.RID.nunique())

adsp_mci_final = adsp_mci_final.rename(columns={'PHC_Age_PET': 'AGE_PET',
                'PHC_Sex': 'Sex',
                'PHC_Education': 'Education',
                'PHC_Diagnosis': 'Diagnosis',
                'PHC_CENTILOIDS': 'CENTILOIDS'})


# =====================================================================================
# Section 4 · Add APOE4 genotype from ADNIMERGE
#
# APOE4 is the count of e4 alleles (0, 1 or 2). ADNIMERGE holds one row per visit, so
# it is reduced to one row per patient before merging; the merge is left-joined so the
# PET row count is unchanged.
# =====================================================================================

adnimerge = pd.read_csv(base_path + "Data___Database/ADNIMERGE_07Jul2025.csv")
adnimerge, adnimerge_mci, adnimerge_pmci = get_mci_pmci_cohorts(adnimerge, mci_patients, pmci_patients)

adnimerge_pmci = adnimerge_pmci.loc[:, ['RID', 'APOE4','EXAMDATE']]
#sort by RID and EXAMDATE
adnimerge_pmci = adnimerge_pmci.sort_values(by=['RID', 'EXAMDATE'])
adnimerge_pmci = adnimerge_pmci.drop_duplicates(subset=['RID'])
#check shape before merging
print(adnimerge_pmci.shape)

#add APOE4 status to adsp_pmci_final by merging on RID, keep shape of adsp_pmci_final the same as before merging
pmci_pet_df = adsp_pmci_final.merge(adnimerge_pmci, on='RID', how='left')
print(pmci_pet_df.shape, pmci_pet_df.RID.nunique())

#select columns RID, AGE_PET, Sex, Education, Diagnosis, PETDATE, CENTILOIDS, APOE4, AD_DATE, MCI_DATE
pmci_pet_df = pmci_pet_df.loc[:, ['RID', 'AGE_PET', 'Sex', 'Education',
    'Diagnosis', 'PETDATE', 'CENTILOIDS',
    'APOE4', 'AD_DATE', 'MCI_DATE']]

#check if there is any PETDATE after AD_DATE
print(pmci_pet_df[pmci_pet_df.PETDATE > pmci_pet_df.AD_DATE].shape)
#check if there is any MCI_DATE after AD_DATE
print(pmci_pet_df[pmci_pet_df.MCI_DATE > pmci_pet_df.AD_DATE].shape)
print(pmci_pet_df.shape, pmci_pet_df.RID.nunique())


# ---- Section 4.1 · Same merge for the stable-MCI cohort -----------------------------

adnimerge_mci = adnimerge_mci.loc[:, ['RID', 'APOE4','EXAMDATE']]
#sort by RID and EXAMDATE
adnimerge_mci = adnimerge_mci.sort_values(by=['RID', 'EXAMDATE'])
adnimerge_mci = adnimerge_mci.drop_duplicates(subset=['RID'])
#check shape before merging
print(adnimerge_mci.shape)

smci_pet_df = adsp_mci_final.merge(adnimerge_mci, on='RID', how='left')
print(smci_pet_df.shape, smci_pet_df.RID.nunique())


# =====================================================================================
# Section 5 · Censoring date for stable MCI
#
# Stable patients have no event, so they are censored at their last recorded visit in
# the registry, stored as EVENTDATE. Two exclusions follow: scans dated after the
# censoring visit, and patients whose MCI diagnosis post-dates their last visit --
# checking the clinical database showed those patients have only a single entry, so
# using MCI_DATE as the time origin would give a survival time of zero.
# =====================================================================================

registry = pd.read_csv(base_path + "/REGISTRY_22Aug2025.csv")

registry['USERDATE'] = pd.to_datetime(registry['USERDATE'], errors='coerce')

registry = registry.sort_values(by=['RID', 'USERDATE'], ascending=True)
last_visit_date = registry.groupby('RID').last().reset_index()

smci_pet_df = smci_pet_df.merge(last_visit_date.loc[:, ['RID', 'USERDATE']], on='RID', how='left')
smci_pet_df = smci_pet_df.rename(columns={'USERDATE': 'EVENTDATE'})
print(smci_pet_df.shape)

#select columns RID, AGE_PET, Sex, Education, Diagnosis, PETDATE, CENTILOIDS, APOE4, MCI_DATE, EVENTDATE
smci_pet_df = smci_pet_df.loc[:, ['RID', 'AGE_PET', 'Sex', 'Education',
    'Diagnosis', 'PETDATE', 'CENTILOIDS',
    'APOE4', 'MCI_DATE', 'EVENTDATE']]

#add a LABEL column with value 0 for all rows
smci_pet_df['LABEL'] = 0

#check if there is any PETDATE after EVENTDATE
print(smci_pet_df[smci_pet_df.PETDATE > smci_pet_df.EVENTDATE].shape)
#check if there is any MCI_DATE after EVENTDATE
print(smci_pet_df[smci_pet_df.MCI_DATE > smci_pet_df.EVENTDATE].shape)
print(smci_pet_df.shape, smci_pet_df.RID.nunique())

#drop rows where PETDATE is after EVENTDATE
smci_pet_df = smci_pet_df[smci_pet_df.PETDATE <= smci_pet_df.EVENTDATE]
print(smci_pet_df.shape, smci_pet_df.RID.nunique())
#get IDs where MCI_DATE is after EVENTDATE (computed after the scan filter)
mci_after_event_IDs = smci_pet_df[smci_pet_df.MCI_DATE > smci_pet_df.EVENTDATE].RID.unique()

#check mci_after_event_IDs in the original mci dataframe
print(mci[mci.RID.isin(mci_after_event_IDs)].loc[:, ['RID', 'DXDATE', 'DIAGNOSIS']])

#drop mci_after_event_IDs in smci_pet_df
smci_pet_df = smci_pet_df[~smci_pet_df.RID.isin(mci_after_event_IDs)]
print(smci_pet_df.shape, smci_pet_df.RID.nunique())

print(smci_pet_df.head())

#check percentage of patients that have PETDATE after MCI date
print(len(smci_pet_df[smci_pet_df.PETDATE > smci_pet_df.MCI_DATE])/smci_pet_df.shape[0])

print(pmci_pet_df.head())

#check percentage of patients that have PETDATE after MCI date
print(len(pmci_pet_df[pmci_pet_df.PETDATE > pmci_pet_df.MCI_DATE])/pmci_pet_df.shape[0])


# =====================================================================================
# Section 6 · Add plasma biomarkers
#
#   pT217_F      plasma p-tau217 (Fujirebio)
#   GFAP_Q       glial fibrillary acidic protein (Quanterix)
#   NfL_Q        neurofilament light (Quanterix)
#   AB42_AB40_F  amyloid-beta 42/40 ratio
#   AB42_F, AB40_F
#
# Both modalities are restricted to patients who have PET *and* plasma. Demographics,
# APOE4 and the outcome dates are carried over from the PET frames; date of birth comes
# from PTDEMOG so age at draw can be computed.
# =====================================================================================

main_path = base_path
ptau_path = (
    main_path + "Biospecimen_Results/UPENN_PLASMA_FUJIREBIO_QUANTERIX_05Mar2025.csv"
)
ptau_df = pd.read_csv(ptau_path, low_memory=False)

#check how many overlaping IDs there are between ptau_df and pmci_pet_df and smci_pet_df
pmci_ptau_IDs = set(ptau_df.RID.unique()) & set(pmci_pet_df.RID.unique())
smci_ptau_IDs = set(ptau_df.RID.unique()) & set(smci_pet_df.RID.unique())
print(f"Number of overlapping IDs between ptau_df and pmci_pet_df: {len(pmci_ptau_IDs)}")
print(f"Number of overlapping IDs between ptau_df and smci_pet_df: {len(smci_ptau_IDs)}")

#subset to overlapping RIDs for pMCI
pmci_pet_df = pmci_pet_df[pmci_pet_df.RID.isin(pmci_ptau_IDs)]
print(pmci_pet_df.shape, pmci_pet_df.RID.nunique())
pmci_ptau_df = ptau_df[ptau_df.RID.isin(pmci_ptau_IDs)]
print(pmci_ptau_df.shape, pmci_ptau_df.RID.nunique())

#subset to overlapping RIDs for sMCI
smci_pet_df = smci_pet_df[smci_pet_df.RID.isin(smci_ptau_IDs)]
print(smci_pet_df.shape, smci_pet_df.RID.nunique())
smci_ptau_df = ptau_df[ptau_df.RID.isin(smci_ptau_IDs)]
print(smci_ptau_df.shape, smci_ptau_df.RID.nunique())

# ---- stable MCI ---------------------------------------------------------------------
demo_df = (
    smci_pet_df
    .sort_values('MCI_DATE')   # optional
    .groupby('RID')
    .first()                   # take one row per RID
    .reset_index()
)

demo_df = demo_df[['RID', 'Sex', 'Education', 'Diagnosis', 'APOE4', 'MCI_DATE', 'EVENTDATE']]

smci_ptau_df = smci_ptau_df.loc[:, ['RID', 'EXAMDATE', 'pT217_F','GFAP_Q','NfL_Q', 'AB42_AB40_F', 'AB42_F', 'AB40_F']]

#add in all columns beside PETDATE and Centiloids to smci_ptau_df
smci_ptau_df = smci_ptau_df.merge(demo_df.loc[:, ['RID', 'Sex', 'Education', 'Diagnosis', 'APOE4', 'MCI_DATE', 'EVENTDATE']], on='RID', how='left')
#rename EXAMDATE to PTAUDATE
smci_ptau_df = smci_ptau_df.rename(columns={'EXAMDATE': 'PTAUDATE'})
print(smci_ptau_df.shape, smci_ptau_df.RID.nunique())

#add in PTDOB from PTDEMOG to smci_ptau_df
demog_df = pd.read_csv(base_path + "Demographics/PTDEMOG_06Mar2025.csv")

demog_df = (
    demog_df
    .groupby('RID')
    .first()
    .reset_index()[['RID','PTDOB']]
)

smci_ptau_df = smci_ptau_df.merge(demog_df, on='RID', how='left')

# Convert MM/YYYY -> datetime with day=15
smci_ptau_df['PTDOB'] = pd.to_datetime(
    smci_ptau_df['PTDOB'] + '/15',
        format='%m/%Y/%d',
        errors='coerce'
)

#calculate age at time of pTau measurement and add as a column to smci_ptau_df
smci_ptau_df['PTAUDATE'] = pd.to_datetime(smci_ptau_df['PTAUDATE'])
smci_ptau_df['AGE_PTAU'] = (
    (smci_ptau_df['PTAUDATE'] - smci_ptau_df['PTDOB'])
    .dt.days / 365.25
)
#add in Label column to smci_ptau_df with value 0 for all rows
smci_ptau_df['LABEL'] = 0

print(smci_ptau_df.shape, smci_ptau_df.RID.nunique())
print(smci_ptau_df.head(10))

# ---- progressive MCI ----------------------------------------------------------------
#get data from the clinical dataset for the pMCI patients to merge with the pTau dataset
pmci_demo_df = (
    pmci_pet_df
    .sort_values('AD_DATE')   # optional
    .groupby('RID')
    .first()                   # take one row per RID
    .reset_index()
)

pmci_demo_df = pmci_demo_df[['RID', 'Sex', 'Education', 'Diagnosis', 'APOE4', 'MCI_DATE', 'AD_DATE']]

pmci_ptau_df = pmci_ptau_df.loc[:, ['RID', 'EXAMDATE', 'pT217_F','GFAP_Q','NfL_Q', 'AB42_AB40_F', 'AB42_F', 'AB40_F']]

#add in all columns beside PETDATE and Centiloids to pmci_ptau_df
pmci_ptau_df = pmci_ptau_df.merge(pmci_demo_df.loc[:, ['RID', 'Sex', 'Education', 'Diagnosis', 'APOE4', 'MCI_DATE', 'AD_DATE']], on='RID', how='left')

#rename EXAMDATE to PTAUDATE
pmci_ptau_df = pmci_ptau_df.rename(columns={'EXAMDATE': 'PTAUDATE'})

#get data from the demogrpahics df with DOB
demog_df = (
    demog_df
    .groupby('RID')
    .first()
    .reset_index()[['RID','PTDOB']]
)
pmci_ptau_df = pmci_ptau_df.merge(demog_df, on='RID', how='left')

# Convert MM/YYYY -> datetime with day=15
pmci_ptau_df['PTDOB'] = pd.to_datetime(
    pmci_ptau_df['PTDOB'] + '/15',
        format='%m/%Y/%d',
        errors='coerce'
)

#the pMCI arm calls it DOB from here; the sMCI arm keeps PTDOB until Section 9
pmci_ptau_df = pmci_ptau_df.rename(columns={'PTDOB':'DOB'})

#calculate Age at time of biomarker measurement
pmci_ptau_df['PTAUDATE'] = pd.to_datetime(pmci_ptau_df['PTAUDATE'])
pmci_ptau_df['AGE_PTAU'] = (
    (pmci_ptau_df['PTAUDATE'] - pmci_ptau_df['DOB'])
    .dt.days / 365.25
)

#add in LABEL = 1
pmci_ptau_df['LABEL'] = 1

print(pmci_ptau_df.shape, pmci_ptau_df.RID.nunique())
print(pmci_ptau_df.head())


# =====================================================================================
# Section 7 · Plasma biomarker-timing filters
#
# A plasma draw only counts as baseline if it was taken close enough to the MCI
# diagnosis: within 2 years before it for pMCI, 4 years for sMCI (p-tau217 varies less
# over time in patients who do not progress). Draws dated after the event are dropped.
# These windows reproduce the published N=201 cohort; see README.md.
# =====================================================================================

#--- pMCI: keep draws from up to 2 years before the MCI date onward ---
pmci_ptau_df["PTAUDATE"] = pd.to_datetime(pmci_ptau_df["PTAUDATE"])
pmci_ptau_df["MCI_DATE"] = pd.to_datetime(pmci_ptau_df["MCI_DATE"])
pmci_ptau_df["ptau_mci_daysdiff"] = (pmci_ptau_df["PTAUDATE"] - pmci_ptau_df["MCI_DATE"]).dt.days
print(pmci_ptau_df.shape, pmci_ptau_df.RID.nunique())
pmci_ptau_df = pmci_ptau_df[pmci_ptau_df["ptau_mci_daysdiff"] >= -365*2]
print(pmci_ptau_df.shape, pmci_ptau_df.RID.nunique())

#--- sMCI: keep draws from up to 4 years before the MCI date onward ---
smci_ptau_df["PTAUDATE"] = pd.to_datetime(smci_ptau_df["PTAUDATE"])
smci_ptau_df["MCI_DATE"] = pd.to_datetime(smci_ptau_df["MCI_DATE"])
smci_ptau_df["ptau_mci_daysdiff"] = (smci_ptau_df["PTAUDATE"] - smci_ptau_df["MCI_DATE"]).dt.days
print(smci_ptau_df.shape, smci_ptau_df.RID.nunique())
smci_ptau_df = smci_ptau_df[smci_ptau_df["ptau_mci_daysdiff"] >= -365*4]
print(smci_ptau_df.shape, smci_ptau_df.RID.nunique())

#--- drop draws taken after the event ---
pmci_ptau_df["AD_DATE"] = pd.to_datetime(pmci_ptau_df["AD_DATE"])
pmci_ptau_df = pmci_ptau_df[pmci_ptau_df["PTAUDATE"] <= pmci_ptau_df["AD_DATE"]]
print(pmci_ptau_df.shape, pmci_ptau_df.RID.nunique())

smci_ptau_df["EVENTDATE"] = pd.to_datetime(smci_ptau_df["EVENTDATE"])
smci_ptau_df = smci_ptau_df[smci_ptau_df["PTAUDATE"] <= smci_ptau_df["EVENTDATE"]]
print(smci_ptau_df.shape, smci_ptau_df.RID.nunique())

#rename AD_DATE to EVENTDATE in pmci_ptau_df
pmci_ptau_df = pmci_ptau_df.rename(columns={'AD_DATE': 'EVENTDATE'})
pmci_pet_df = pmci_pet_df.rename(columns={'AD_DATE': 'EVENTDATE'})

#add LABEL column
pmci_pet_df["LABEL"] = 1

#rename PTDOB to DOB in smci_ptau_df
smci_ptau_df = smci_ptau_df.rename(columns={'PTDOB': 'DOB'})


# =====================================================================================
# Section 8 · Baseline window, earliest measurement per patient, final cohorts
#
#   1. keep PET scans within 1 year (pMCI) / 2 years (sMCI) before the MCI date;
#   2. collapse to one row per patient -- the earliest scan / draw;
#   3. concatenate pMCI and sMCI;
#   4. drop patients with no APOE4 genotype.
# =====================================================================================

#check how many PET scans are after MCI diagnosis date in both pmci and smci cohorts
print(f"Number of PET scans after MCI diagnosis date in pMCI cohort: {len(pmci_pet_df[pmci_pet_df.PETDATE > pmci_pet_df.MCI_DATE])}")
print(f"Number of PET scans after MCI diagnosis date in sMCI cohort: {len(smci_pet_df[smci_pet_df.PETDATE > smci_pet_df.MCI_DATE])}")

#create a column for PETDATE - MCI_DATE in days
pmci_pet_df["PETDATE"] = pd.to_datetime(pmci_pet_df["PETDATE"])
pmci_pet_df["MCI_DATE"] = pd.to_datetime(pmci_pet_df["MCI_DATE"])
pmci_pet_df["pet_mci_daysdiff"] = (pmci_pet_df["PETDATE"] - pmci_pet_df["MCI_DATE"]).dt.days

#subset to people with days diff > 0 or within 1 year before mci diagnosis date
# FIX: originally written as (d >= 0) | (d >= -365*1). The first test is subsumed
# by the second, so the filter is the second test alone. No change in behaviour.
print(pmci_pet_df.shape, pmci_pet_df.RID.nunique())
pmci_pet_df = pmci_pet_df[pmci_pet_df["pet_mci_daysdiff"] >= -365*1]
print(pmci_pet_df.shape, pmci_pet_df.RID.nunique())

#create a column for PETDATE - MCI_DATE in days
smci_pet_df["PETDATE"] = pd.to_datetime(smci_pet_df["PETDATE"])
smci_pet_df["MCI_DATE"] = pd.to_datetime(smci_pet_df["MCI_DATE"])
smci_pet_df["pet_mci_daysdiff"] = (smci_pet_df["PETDATE"] - smci_pet_df["MCI_DATE"]).dt.days

#subset to people with days diff > 0 or within 2 years before mci diagnosis date (since pet values for sMCI should not vary as much over time)
print(smci_pet_df.shape, smci_pet_df.RID.nunique())
smci_pet_df = smci_pet_df[smci_pet_df["pet_mci_daysdiff"] >= -365*2]
print(smci_pet_df.shape, smci_pet_df.RID.nunique())

#subset to earliest PETDATE
pmci_pet_df = pmci_pet_df.sort_values(by=['RID', 'PETDATE'])
pmci_pet_df = pmci_pet_df.groupby('RID').first().reset_index()
print(pmci_pet_df.shape, pmci_pet_df.RID.nunique())

smci_pet_df = smci_pet_df.sort_values(by=['RID', 'PETDATE'])
smci_pet_df = smci_pet_df.groupby('RID').first().reset_index()
print(smci_pet_df.shape, smci_pet_df.RID.nunique())

#subset to earliest PTAUDATE
pmci_ptau_df = pmci_ptau_df.sort_values(by=['RID', 'PTAUDATE'])
pmci_ptau_df = pmci_ptau_df.groupby('RID').first().reset_index()
print(pmci_ptau_df.shape, pmci_ptau_df.RID.nunique())

smci_ptau_df = smci_ptau_df.sort_values(by=['RID', 'PTAUDATE'])
smci_ptau_df = smci_ptau_df.groupby('RID').first().reset_index()
print(smci_ptau_df.shape, smci_ptau_df.RID.nunique())

#concat PET dataframes
all_pet_df = pd.concat([pmci_pet_df, smci_pet_df], axis=0)
print(all_pet_df.shape, all_pet_df.RID.nunique())

#concat pTau dataframes
all_ptau_df = pd.concat([pmci_ptau_df, smci_ptau_df], axis=0)
print(all_ptau_df.shape, all_ptau_df.RID.nunique())

#check for NA values in all_pet_df
print(all_pet_df.isna().sum())

#drop people with NA values for APOE4 status in all_ptau_df
all_ptau_df = all_ptau_df[~all_ptau_df.APOE4.isna()]
print(all_ptau_df.shape, all_ptau_df.RID.nunique())
print(all_ptau_df.LABEL.value_counts())

#drop people with NA values for APOE4 status in all_pet_df
all_pet_df = all_pet_df[~all_pet_df.APOE4.isna()]
print(all_pet_df.shape, all_pet_df.RID.nunique())
print(all_pet_df.LABEL.value_counts())

print(all_ptau_df.head())
print(all_pet_df.head())

#change EVENTDATE from object to datetime
all_pet_df["EVENTDATE"] = pd.to_datetime(all_pet_df["EVENTDATE"])
all_ptau_df["EVENTDATE"] = pd.to_datetime(all_ptau_df["EVENTDATE"])


# ---- Section 8.1 · Export -----------------------------------------------------------
# These two files are the input to 02_survival_models.py.

#save to parquet files
all_ptau_df.to_parquet(out_path + "earliest_ptau_df.parquet")
all_pet_df.to_parquet(out_path + "earliest_pet_df.parquet")
print("wrote " + out_path + "earliest_ptau_df.parquet")
print("wrote " + out_path + "earliest_pet_df.parquet")
