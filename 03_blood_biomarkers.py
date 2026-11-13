"""
Step 3 of the pipeline (see README.md). 

Paths come from config.py ($MCI_AD_OUTPUT_DIR = output folder; raw data: ADNI_RAW_DIR).
Models are read from MCI_AD_REUSE_MODELS_FROM when it is set.

Inputs (all written by 02_survival_models.py -- run it first)
    $MCI_AD_OUTPUT_DIR/cleaned_data/ADNI/earliest_ptau_df_rev2.parquet
    $MCI_AD_OUTPUT_DIR/models/ADNI_PENN/earliest_bio_rsf_fold{0..4}.joblib          panel model
    the raw UPENN plasma CSV, for the p-tau217/AB42 ratio

Outputs
    partial_dependence_48mo.png        Figure 4A
    two_cutoff_ptau217_90_90.png       Figure 5
    two_cutoff_ptau217.png             overview of the 95/95, 90/90, 85/85 rules
    printed tables: zone counts, PPV/NPV for the 90/90 rule

"""

# ---- Imports -------------------------------------------------------------------------

import joblib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save figures to files; no pop-up windows
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline


import config
from config import RAW_DIR, CLEANED_DIR, MODELS_IN, FIGURES_DIR
config.check_raw_dir()
config.make_output_dirs()
config.describe()


# =====================================================================================
# 1 · Load the analysis cohort
#
# earliest_ptau_df_rev2 is the plasma cohort written by Section 1 of 02_survival_models.py: one
# row per patient, at their earliest plasma draw. We re-attach pT217_AB42_F (the
# p-tau217/AB42 ratio) from the raw biospecimen file, taking each patient's first
# measurement -- identical to Section 2 of 02_survival_models.py.
# =====================================================================================

#load in cleaned dataframes
ptau_df = pd.read_parquet(f"{CLEANED_DIR}/earliest_ptau_df_rev2.parquet")

#load in the ptau dataset to get amyloid beta 42 data
main_path = str(RAW_DIR) + "/"
ptau_path = (
    main_path + "Biospecimen_Results/UPENN_PLASMA_FUJIREBIO_QUANTERIX_05Mar2025.csv"
)
adni_ptau = pd.read_csv(ptau_path, low_memory=False)

#subset adni_ptau to only include RIDs in ptau_df
adni_ptau = adni_ptau[adni_ptau.RID.isin(ptau_df.RID)]
#get the first measurement date for each patient in adni_ptau
print(adni_ptau.shape)
adni_ptau = adni_ptau.sort_values(by=['RID', 'EXAMDATE']).groupby('RID').first().reset_index()
print(adni_ptau.shape)

#add in amyloid beta 42 and pT217_AB42_F data to ptau_df
print(ptau_df.shape)
ptau_df = ptau_df.merge(adni_ptau[['RID', 'pT217_AB42_F']], on='RID', how='left')
print(ptau_df.shape)
print(ptau_df.head())


# =====================================================================================
# 2 · Covariate set for the blood panel
#
#     bio    AGE_MCI, Sex, Education, APOE4 + p-tau217, GFAP, NfL, AB42/40
#            -- the blood PANEL, the same covariates as the bio model in
#               02_survival_models.py; used by Section 3
# =====================================================================================

#all biomarkers + demographics
selected_cov_bio = ["AGE_MCI", "Sex", "Education","APOE4", "pT217_F", "GFAP_Q", "NfL_Q", "AB42_AB40_F"]
X_bio = ptau_df[selected_cov_bio]


# =====================================================================================
# 3 · Partial-dependence plots  
#
# For each panel marker: set that marker to the same value for every patient, keep
# everything else as observed, and average the model's predicted probability of
# converting by 48 months. Sweeping the value across the marker's range traces how
# predicted risk depends on that marker alone.
#
#   - Done separately with each of the five fold models. The line is their mean; the
#     shaded band is the min-max across folds (model-to-model spread, not a CI).
#   - Curves are lightly smoothed with a cubic spline and clipped to [0, 1].
#   - Grid: 1st-99th percentile of each marker; for AB42/40, the full range of
#     non-negative values (negative entries are excluded).
#   - The tick marks along the bottom ("rug") show each patient's actual value,
#     coloured by outcome: blue = stable MCI, gold = progressive MCI.
# =====================================================================================

# ── Load fold models ──
fold_models = [
    joblib.load(f"{MODELS_IN}/earliest_bio_rsf_fold{i}.joblib")
    for i in range(5)
]

t_days = 4 * 365.25  # 48 months

biomarkers = {
    'pT217_F': 'p-Tau217 (pg/mL)',
    'GFAP_Q': 'GFAP (pg/mL)',
    'NfL_Q': 'NfL (pg/mL)',
    'AB42_AB40_F': 'Aβ42/40'
}

# ── Outcome labels for rug coloring (0 = sMCI, 1 = pMCI) ──
labels = ptau_df.loc[X_bio.index, 'LABEL'].values

sMCI_color = '#4E79A7'   # blue
pMCI_color = '#DAA520'   # yellow (goldenrod — brighter yellow washes out on white)

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.flatten()

for ax, (col, label) in zip(axes, biomarkers.items()):
    # Create grid across observed range
    if col == 'AB42_AB40_F':
        positive_vals = X_bio[col][X_bio[col] >= 0]
        grid = np.linspace(positive_vals.min(), positive_vals.max(), 100)
    else:
        grid = np.linspace(X_bio[col].quantile(0.01), X_bio[col].quantile(0.99), 100)

    # For each grid value, predict risk for ALL patients
    # with that feature replaced, then average
    fold_curves = []

    for model in fold_models:
        grid_risks = []

        for val in grid:
            X_modified = X_bio.copy()
            X_modified[col] = val

            surv_fns = model.predict_survival_function(X_modified)
            mean_risk = np.mean([1 - fn(t_days) for fn in surv_fns])
            grid_risks.append(mean_risk)

        fold_curves.append(grid_risks)

    # Average across folds
    mean_curve = np.mean(fold_curves, axis=0)
    lower = np.min(fold_curves, axis=0)
    upper = np.max(fold_curves, axis=0)

    # Smooth
    spline = UnivariateSpline(grid, mean_curve, k=3, s=0.001)
    grid_smooth = np.linspace(grid.min(), grid.max(), 300)
    curve_smooth = np.clip(spline(grid_smooth), 0, 1)

    spline_lower = UnivariateSpline(grid, lower, k=3, s=0.001)
    spline_upper = UnivariateSpline(grid, upper, k=3, s=0.001)
    lower_smooth = np.clip(spline_lower(grid_smooth), 0, 1)
    upper_smooth = np.clip(spline_upper(grid_smooth), 0, 1)

    # Plot
    ax.plot(grid_smooth, curve_smooth, lw=2.5, color='#2E75B6')
    ax.fill_between(grid_smooth, lower_smooth, upper_smooth,
                     color='#2E75B6', alpha=0.15)

    # ── Rug plot, split and colored by outcome label ──
    rug_data = X_bio[col].values
    rug_labels = labels

    if col == 'AB42_AB40_F':
        valid_mask = X_bio[col].values >= 0
        rug_data = rug_data[valid_mask]
        rug_labels = rug_labels[valid_mask]

    rug_y = ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 0

    ax.plot(rug_data[rug_labels == 0],
            np.full((rug_labels == 0).sum(), rug_y),
            '|', color=sMCI_color, alpha=0.5, markersize=5, label='sMCI')

    ax.plot(rug_data[rug_labels == 1],
            np.full((rug_labels == 1).sum(), rug_y),
            '|', color=pMCI_color, alpha=0.5, markersize=5, label='pMCI')

    # Force x-axis to match grid range
    ax.set_xlim(grid.min(), grid.max())

    ax.set_xlabel(label, fontsize=12)
    ax.set_ylabel('Predicted 48-month\nconversion probability', fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ── Single shared legend for the rug colors ──
handles, legend_labels = axes[0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc='upper center', ncol=2,
           bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=11)

plt.suptitle('Partial Dependence: Predicted probability of AD conversion at 48 months',
             fontsize=14, y=1.05)
plt.tight_layout(w_pad=3)
plt.savefig(f'{FIGURES_DIR}/partial_dependence_48mo.png', dpi=300, bbox_inches='tight')


# =====================================================================================
# 4 · Two-cut-off thresholds for plasma p-tau217
#
# Instead of one cut-off, use two:
#
#     p-tau217 <= LOWER  ->  low risk      LOWER is chosen for high sensitivity:
#                                          few converters fall below it (rule-out)
#     p-tau217 >= UPPER  ->  high risk     UPPER is chosen for high specificity:
#                                          few stable patients sit above it (rule-in)
#     in between         ->  intermediate  refer for confirmatory amyloid PET
#
# "90/90" means LOWER gives 90% sensitivity and UPPER gives 90% specificity.
# Everything here uses observed outcomes, not model predictions.
# =====================================================================================

# -------------------------------------------------------------------------------------
# 4.1 · The 4-year outcome, and sensitivity/specificity at every cut-off
#
# Positive = converted to AD within 4 years. Negative = followed for more than 4 years
# without converting. Patients censored before 4 years have an unknown 4-year outcome
# and are left out of all of Section 4.
# -------------------------------------------------------------------------------------

# ── Step 1: Define binary outcome at 4 years ──
# Use actual outcomes, not model predictions
t_cutoff = 4 * 365.25

# Converter within 4 years = positive
binary_outcome = (ptau_df['LABEL'] == 1) & (ptau_df['mci_event_daysdiff'] <= t_cutoff)
# Known outcome: either converted by 4yr or followed past 4yr
known = binary_outcome | (ptau_df['mci_event_daysdiff'] > t_cutoff)

ptau_vals = ptau_df.loc[known, 'pT217_F'].values
outcomes = binary_outcome[known].values.astype(int)

print(f"Patients with known 4-year outcome: {len(outcomes)}")
print(f"Converters: {outcomes.sum()}")
print(f"Non-converters: {(outcomes == 0).sum()}")

# ── Step 2: Calculate sensitivity and specificity at every threshold ──
thresholds = np.sort(np.unique(ptau_vals))
sensitivities = []
specificities = []

for thresh in thresholds:
    predicted_pos = ptau_vals >= thresh
    tp = np.sum(predicted_pos & (outcomes == 1))
    fn = np.sum(~predicted_pos & (outcomes == 1))
    tn = np.sum(~predicted_pos & (outcomes == 0))
    fp = np.sum(predicted_pos & (outcomes == 0))

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    sensitivities.append(sens)
    specificities.append(spec)

sensitivities = np.array(sensitivities)
specificities = np.array(specificities)

# ── Step 3: Find thresholds for different sensitivity/specificity targets ──
target_configs = [
    {'name': 'Sensitivity 95% - Specificity 95%', 'sens_target': 0.95, 'spec_target': 0.95},
    {'name': 'Sensitivity 90% - Specificity 90%', 'sens_target': 0.90, 'spec_target': 0.90},
    {'name': 'Sensitivity 85% - Specificity 85%', 'sens_target': 0.85, 'spec_target': 0.85},
]

def find_threshold_for_sensitivity(target_sens, thresholds, sensitivities):
    """Lower threshold: highest threshold that achieves >= target sensitivity."""
    valid = sensitivities >= target_sens
    if not np.any(valid):
        return thresholds[0], sensitivities[0]
    idx = np.where(valid)[0][-1]  # highest threshold meeting criteria
    return thresholds[idx], sensitivities[idx]

def find_threshold_for_specificity(target_spec, thresholds, specificities):
    """Upper threshold: lowest threshold that achieves >= target specificity."""
    valid = specificities >= target_spec
    if not np.any(valid):
        return thresholds[-1], specificities[-1]
    idx = np.where(valid)[0][0]  # lowest threshold meeting criteria
    return thresholds[idx], specificities[idx]

# -------------------------------------------------------------------------------------
# 4.2 · Overview figure: the 95/95, 90/90 and 85/85 rules side by side
#
# One panel per rule: every patient's p-tau217, split by outcome, coloured by zone,
# with a zone-by-outcome count table underneath. Exploratory -- not in the
# manuscript, but saved to the figures folder. NB its colours are the reverse of Figure 5:
# here blue = high risk and red = low risk.
# -------------------------------------------------------------------------------------

# ── Step 4: Create visualization ──
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, rule in zip(axes, target_configs):
    # Find thresholds
    lower_thresh, actual_sens = find_threshold_for_sensitivity(
        rule['sens_target'], thresholds, sensitivities)
    upper_thresh, actual_spec = find_threshold_for_specificity(
        rule['spec_target'], thresholds, specificities)

    # If lower > upper, swap (shouldn't happen but safety check)
    if lower_thresh > upper_thresh:
        lower_thresh, upper_thresh = upper_thresh, lower_thresh

    print(f"\n{rule['name']}:")
    print(f"  Lower cutoff (high sensitivity): {lower_thresh:.3f}")
    print(f"  Upper cutoff (high specificity): {upper_thresh:.3f}")

    # Classify patients into three zones
    high_risk = ptau_vals >= upper_thresh       # above upper = clearly positive
    low_risk = ptau_vals <= lower_thresh        # below lower = clearly negative
    gray_zone = ~high_risk & ~low_risk          # intermediate

    # Create x-axis: converter (1) vs non-converter (0) with jitter
    jitter = np.random.RandomState(42).uniform(-0.2, 0.2, len(outcomes))
    x_pos = outcomes + jitter

    # Color by zone
    colors = np.where(high_risk, '#3366CC',           # blue = above upper cutoff
              np.where(low_risk, '#CC3333',            # red = below lower cutoff
                       '#999999'))                      # gray = intermediate

    ax.scatter(x_pos, ptau_vals, c=colors, alpha=0.6, s=30, edgecolors='none')

    # Draw threshold lines
    ax.axhline(y=upper_thresh, color='#3366CC', ls='--', lw=1.5, alpha=0.7)
    ax.axhline(y=lower_thresh, color='#CC3333', ls='--', lw=1.5, alpha=0.7)

    # Annotate thresholds
    ax.text(1.35, upper_thresh, f'upper cut-off {upper_thresh:.3f} pg/ml',
            fontsize=8, color='#3366CC', va='center')
    ax.text(1.35, lower_thresh, f'lower cut-off {lower_thresh:.3f} pg/ml',
            fontsize=8, color='#CC3333', va='center')

    # Labels
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Non-converter', 'Converter'], fontsize=10)
    ax.set_ylabel('Plasma p-Tau217', fontsize=11)
    ax.set_title(rule['name'], fontsize=11)
    ax.set_xlim(-0.5, 2.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── Confusion matrix below plot ──
    # Count patients in each zone × outcome
    n_conv = outcomes.sum()
    n_nonconv = (outcomes == 0).sum()

    high_conv = np.sum(high_risk & (outcomes == 1))
    high_nonconv = np.sum(high_risk & (outcomes == 0))
    gray_conv = np.sum(gray_zone & (outcomes == 1))
    gray_nonconv = np.sum(gray_zone & (outcomes == 0))
    low_conv = np.sum(low_risk & (outcomes == 1))
    low_nonconv = np.sum(low_risk & (outcomes == 0))

    print(f"  High risk (above upper): {high_conv + high_nonconv} "
          f"({high_conv} converters, {high_nonconv} non-converters)")
    print(f"  Gray zone: {gray_conv + gray_nonconv} "
          f"({gray_conv} converters, {gray_nonconv} non-converters)")
    print(f"  Low risk (below lower): {low_conv + low_nonconv} "
          f"({low_conv} converters, {low_nonconv} non-converters)")

    # Add table below
    table_data = [
        [high_conv, gray_conv, low_conv, n_conv],
        [high_nonconv, gray_nonconv, low_nonconv, n_nonconv],
        [high_conv + high_nonconv, gray_conv + gray_nonconv,
         low_conv + low_nonconv, len(outcomes)]
    ]

    row_labels = ['Converter', 'Non-converter', 'Total']
    col_labels = [f'p-Tau217+\n(>{upper_thresh:.2f})',
                  f'Intermediate',
                  f'p-Tau217−\n(<{lower_thresh:.2f})',
                  'Total']

    # Color cells
    cell_colors = [
        ['#CCE5FF', '#E6E6E6', '#FFCCCC', 'white'],  # converters
        ['#CCE5FF', '#E6E6E6', '#FFCCCC', 'white'],  # non-converters
        ['white', 'white', 'white', 'white']           # total
    ]

    table = ax.table(cellText=table_data,
                      rowLabels=row_labels,
                      colLabels=col_labels,
                      cellColours=cell_colors,
                      loc='bottom',
                      bbox=[0.0, -0.45, 1.0, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(8)

plt.subplots_adjust(bottom=0.25)
plt.suptitle('Two-Cutoff Approach for p-Tau217 Classification\n(4-year conversion outcome)',
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/two_cutoff_ptau217.png', dpi=300, bbox_inches='tight')

# -------------------------------------------------------------------------------------
# 4.3 · The 90/90 rule  ->  Figure 5
#
# Published cut-offs: 0.199 and 0.553 pg/mL. Red = high risk, grey = intermediate,
# blue = low risk. The PPV/NPV in the bubbles count the intermediate zone in (PPV over
# high + intermediate, NPV over low + intermediate). The zone-only PPV/NPV are in 4.4.
# -------------------------------------------------------------------------------------

# ── Find 90/90 thresholds ──
lower_thresh, _ = find_threshold_for_sensitivity(0.90, thresholds, sensitivities)
upper_thresh, _ = find_threshold_for_specificity(0.90, thresholds, specificities)

if lower_thresh > upper_thresh:
    lower_thresh, upper_thresh = upper_thresh, lower_thresh

# ── Classify patients ──
high_risk = ptau_vals >= upper_thresh
low_risk = ptau_vals <= lower_thresh
gray_zone = ~high_risk & ~low_risk

n_total = len(outcomes)
n_high = high_risk.sum()
n_gray = gray_zone.sum()
n_low = low_risk.sum()

# ── NPV and PPV including intermediate zone ──
conv_low_n = np.sum(low_risk & (outcomes == 1))
conv_gray_n = np.sum(gray_zone & (outcomes == 1))
conv_high_n = np.sum(high_risk & (outcomes == 1))

nonconv_low_n = np.sum(low_risk & (outcomes == 0))
nonconv_gray_n = np.sum(gray_zone & (outcomes == 0))
nonconv_high_n = np.sum(high_risk & (outcomes == 0))

npv_combined = (nonconv_low_n + nonconv_gray_n) / (n_low + n_gray)
ppv_combined = (conv_high_n + conv_gray_n) / (n_high + n_gray)

print(f"NPV (low + intermediate): {npv_combined:.1%}  ({nonconv_low_n + nonconv_gray_n}/{n_low + n_gray})")
print(f"PPV (high + intermediate): {ppv_combined:.1%}  ({conv_high_n + conv_gray_n}/{n_high + n_gray})")

# ── Colors ──
color_high = '#CC3333'
color_inter = '#999999'
color_low = '#3366CC'

colors = np.where(high_risk, color_high,
          np.where(low_risk, color_low, color_inter))

# ── Plot ──
fig, ax = plt.subplots(figsize=(10, 8))

np.random.seed(42)
jitter = np.random.uniform(-0.2, 0.2, len(outcomes))
x_pos = outcomes + jitter

ax.scatter(x_pos, ptau_vals, c=colors, alpha=0.6, s=35, edgecolors='none', zorder=3)

# ── Threshold lines ──
ax.axhline(y=upper_thresh, color='#CC3333', ls='--', lw=1.5, alpha=0.7, zorder=2)
ax.axhline(y=lower_thresh, color='#3366CC', ls='--', lw=1.5, alpha=0.7, zorder=2)

# ── Bubble annotations ──
def add_bubble(ax, y_pos, zone_text, pct_text, bg_color, text_color):
    """Draw a rounded label box for one zone, with a bold PPV/NPV figure beside it."""
    bbox_props = dict(boxstyle="round,pad=0.4", facecolor=bg_color,
                      edgecolor='gray', alpha=0.85, linewidth=0.8)
    ax.text(1.65, y_pos, zone_text, fontsize=9, ha='center', va='center',
            bbox=bbox_props, zorder=5)
    ax.text(2.15, y_pos, pct_text, fontsize=10, ha='left', va='center',
            fontweight='bold', color=text_color, zorder=5)

high_y = upper_thresh + (ptau_vals.max() - upper_thresh) * 0.45
add_bubble(ax, high_y,
           f'High risk:\n{n_high/n_total:.1%} patients (n={n_high})',
           f'PPV: {ppv_combined:.1%}',
           '#FFCCCC', '#CC3333')

inter_y = (upper_thresh + lower_thresh) / 2
add_bubble(ax, inter_y,
           f'Intermediate risk:\n{n_gray/n_total:.1%} patients (n={n_gray})',
           '',
           '#E6E6E6', '#666666')

low_y = lower_thresh * 0.35
add_bubble(ax, low_y,
           f'Low risk:\n{n_low/n_total:.1%} patients (n={n_low})',
           f'NPV: {npv_combined:.1%}',
           '#CCE5FF', '#3366CC')

# ── Formatting ──
ax.set_xticks([0, 1])
ax.set_xticklabels(['Stable MCI', 'Progressive MCI'], fontsize=11)
ax.set_ylabel('Plasma p-Tau217 (pg/mL)', fontsize=12, fontweight='bold')
ax.set_xlim(-0.55, 2.65)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_title('Two-Cutoff p-Tau217 Screening for MCI-to-AD Progression\n(4-year outcome)',
             fontsize=13, fontweight='bold', pad=15)

# ── Confusion matrix table ──
n_conv = outcomes.sum()
n_nonconv = (outcomes == 0).sum()

table_data = [
    [conv_high_n, conv_gray_n, conv_low_n, n_conv],
    [nonconv_high_n, nonconv_gray_n, nonconv_low_n, n_nonconv],
    [n_high, n_gray, n_low, n_total]
]

row_labels = ['Converter', 'Non-converter', 'Total']
col_labels = [f'p-Tau217+\n(>{upper_thresh:.2f})',
              'Intermediate',
              f'p-Tau217−\n(<{lower_thresh:.2f})',
              'Total']

cell_colors = [
    ['#FFCCCC', '#E6E6E6', '#CCE5FF', 'white'],
    ['#FFCCCC', '#E6E6E6', '#CCE5FF', 'white'],
    ['white', 'white', 'white', 'white']
]

table = ax.table(cellText=table_data,
                  rowLabels=row_labels,
                  colLabels=col_labels,
                  cellColours=cell_colors,
                  loc='bottom',
                  bbox=[0.1, -0.32, 0.65, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(9)

plt.subplots_adjust(bottom=0.28)
plt.savefig(f'{FIGURES_DIR}/two_cutoff_ptau217_90_90.png', dpi=300, bbox_inches='tight')

# -------------------------------------------------------------------------------------
# 4.4 · 90/90 zone summary
#
# Here PPV and NPV are zone-only: how often the confident calls are right. Expected
# output: NPV 94.1%, PPV 64.3%. "PET avoided" = share of
# patients in the low or high zone, who would not need a confirmatory scan.
# -------------------------------------------------------------------------------------

# Using the 90/90 thresholds already computed
high = ptau_vals >= upper_thresh
low = ptau_vals <= lower_thresh
gray = ~high & ~low

# ── Low risk zone ──
low_total = low.sum()
low_converters = outcomes[low].sum()        # false negatives
low_stable = (outcomes[low] == 0).sum()     # true negatives

false_neg_rate = low_converters / low_total if low_total > 0 else 0
npv = low_stable / low_total if low_total > 0 else 0

print(f"LOW RISK (p-Tau217 <= {lower_thresh:.3f})")
print(f"  Total: {low_total}")
print(f"  True negatives (sMCI): {low_stable}")
print(f"  False negatives (pMCI missed): {low_converters}")
print(f"  False negative rate: {false_neg_rate:.1%}")
print(f"  NPV: {npv:.1%}")

# ── High risk zone ──
high_total = high.sum()
high_converters = outcomes[high].sum()          # true positives
high_stable = (outcomes[high] == 0).sum()       # false positives

false_pos_rate = high_stable / high_total if high_total > 0 else 0
ppv = high_converters / high_total if high_total > 0 else 0

print(f"\nHIGH RISK (p-Tau217 >= {upper_thresh:.3f})")
print(f"  Total: {high_total}")
print(f"  True positives (pMCI caught): {high_converters}")
print(f"  False positives (sMCI flagged): {high_stable}")
print(f"  False positive rate: {false_pos_rate:.1%}")
print(f"  PPV: {ppv:.1%}")

# ── Intermediate zone ──
gray_total = gray.sum()
gray_converters = outcomes[gray].sum()
gray_stable = (outcomes[gray] == 0).sum()

print(f"\nINTERMEDIATE (p-Tau217 {lower_thresh:.3f} - {upper_thresh:.3f})")
print(f"  Total: {gray_total}")
print(f"  Converters: {gray_converters}")
print(f"  Stable: {gray_stable}")
print(f"  Conversion rate: {gray_converters/gray_total:.1%}" if gray_total > 0 else "")

# ── Overall summary ──
print(f"\n{'='*40}")
print(f"SUMMARY")
print(f"{'='*40}")
print(f"  NPV (low risk):  {npv:.1%} — {npv:.1%} of low-risk patients truly stable")
print(f"  PPV (high risk):  {ppv:.1%} — {ppv:.1%} of high-risk patients truly convert")
print(f"  PET avoided:      {(low_total + high_total)}/{len(outcomes)} ({(low_total + high_total)/len(outcomes):.1%})")
