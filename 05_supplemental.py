"""
Supplemental analysis: Cox proportional-hazards model vs. random survival forest.

Step 5 of the pipeline (optional, run after 02_survival_models.py). Fits a Cox model on
the same covariates as the PET random survival forest -- age at MCI diagnosis, sex,
education, APOE4 and amyloid PET Centiloids -- and scores it on exactly the same
patients, folds and time grid, so the two models can be compared head to head.

    cohort        the N = 201 analysis cohort from 02_survival_models.py (157 sMCI, 44 pMCI)
    folds         StratifiedKFold(5, shuffle=True, random_state=20) on the event indicator,
                  the same split the PET RSF used
    time grid     the 25-point grid from 02 (5th-81st percentile of test-fold times,
                  ~555-2154 days)
    Cox model     standardise -> CoxPHSurvivalAnalysis, with the ridge penalty (alpha)
                  and tie handling tuned by inner 5-fold CV on the IPCW C-index, the same
                  nested scheme as the RSF

Metrics are summarised the same way as for the RSF models in 02_survival_models.py:

    mean AUC      fold-averaged AUC curve, averaged over the 25 time points;
                  ± SD of the five per-fold mean AUCs (auc_std in 02, Section 16)
    mean C-index  mean of the five per-fold IPCW C-indexes; ± SD across folds (ddof=1)
    mean Brier    fold-averaged Brier curve, averaged over time; ± the across-fold SD at
                  each time point, averaged over time (02, Section 12)
    IBS           mean of the five per-fold integrated Brier scores

Inputs
    $MCI_AD_OUTPUT_DIR/cleaned_data/ADNI/earliest_pet_df_rev2.parquet       from 02_survival_models.py
    <predictions>/earliest_pet_rsf_fold{0..4}_predictions.npz              from 02_survival_models.py
        (<predictions> is PREDICTIONS_IN in config.py, so MCI_AD_REUSE_MODELS_FROM works)

Outputs
    $MCI_AD_OUTPUT_DIR/models/ADNI_PENN/earliest_pet_cph_fold{i}.joblib
    $MCI_AD_OUTPUT_DIR/predictions/ADNI_PENN/earliest_pet_cph_fold{i}_predictions.npz
    $MCI_AD_OUTPUT_DIR/figures/ADNI_PENN/supp_cph_rsf_auc_comparison.pdf
    $MCI_AD_OUTPUT_DIR/figures/ADNI_PENN/supp_cph_rsf_cindex_comparison.pdf
    $MCI_AD_OUTPUT_DIR/figures/ADNI_PENN/supp_cph_rsf_bs_comparison.pdf
    $MCI_AD_OUTPUT_DIR/figures/ADNI_PENN/supp_cph_schoenfeld_centiloids.pdf
"""

# ---- Imports -------------------------------------------------------------------------

import joblib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save figures to files; no pop-up windows
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import chi2, rankdata

from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import (
    as_concordance_index_ipcw_scorer,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    brier_score,
    integrated_brier_score,
)

import config
from config import (CLEANED_DIR, MODELS_DIR, PREDICTIONS_DIR, FIGURES_DIR,
                    PREDICTIONS_IN)
config.make_output_dirs()
config.describe()

RANDOM_STATE = 20          # same seed as 02_survival_models.py
N_BOOT = 1000
COVARIATES = ["AGE_MCI", "Sex", "Education", "APOE4", "CENTILOIDS"]
MODELS = ["RSF", "CoxPH"]
# RSF keeps the PET orange used in the thesis figures; Cox gets a neutral gray.
PALETTE = {"RSF": "#ff7f0e", "CoxPH": "#7f7f7f"}


# =====================================================================================
# 1 · Cohort, folds and time grid -- rebuilt exactly as in 02_survival_models.py
# =====================================================================================

pet_df = pd.read_parquet(f"{CLEANED_DIR}/earliest_pet_df_rev2.parquet")

X = pet_df[COVARIATES].astype(float)
y = Surv.from_arrays(event=pet_df.LABEL, time=pet_df.mci_event_daysdiff)

n, n_events = len(y), int(y["event"].sum())
print(f"Cohort: N = {n}, progressors = {n_events}, stable = {n - n_events}")
if (n, n_events) != (201, 44):
    print("WARNING: this is not the 201 / 44 cohort reported in the manuscript.")

skf = StratifiedKFold(n_splits=5, random_state=RANDOM_STATE, shuffle=True)
folds = list(skf.split(X, y["event"]))

# Evaluation grid: 5th-81st percentile of test-fold times, as in Section 4 of 02.
p05 = max(np.percentile(y[test]["time"], 5) for _, test in folds)
p81 = min(np.percentile(y[test]["time"], 81) for _, test in folds)
time = np.linspace(p05, p81, 25)
print(f"Time grid: {time[0]:.0f} - {time[-1]:.0f} days")


# =====================================================================================
# 2 · Fit the Cox model (nested CV, same scheme as the RSF)
#
# Covariates are standardised inside the pipeline so the ridge penalty treats them
# equally (Centiloids span ~200 units, sex spans 1). alpha = 0 is the ordinary,
# unpenalised Cox model. Tuning uses the IPCW C-index, as for the RSF.
# =====================================================================================

inner_cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
param_grid = {
    "estimator__coxphsurvivalanalysis__alpha": [0, 0.2, 0.4, 0.6, 0.8],
    "estimator__coxphsurvivalanalysis__ties": ["breslow", "efron"],
}

for i, (train_idx, test_idx) in enumerate(folds):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    cox = make_pipeline(StandardScaler(), CoxPHSurvivalAnalysis())
    scorer = as_concordance_index_ipcw_scorer(cox, tau=time[-1])
    gcv = GridSearchCV(scorer, param_grid=param_grid, cv=inner_cv, n_jobs=-1)
    gcv.fit(X_train, y_train)
    best = gcv.best_estimator_
    print(f"Fold {i}: best params {gcv.best_params_}")

    joblib.dump(best, f"{MODELS_DIR}/earliest_pet_cph_fold{i}.joblib")
    np.savez_compressed(
        f"{PREDICTIONS_DIR}/earliest_pet_cph_fold{i}_predictions.npz",
        train_predictions=best.predict(X_train),
        test_predictions=best.predict(X_test),
        train_chf_predictions=best.predict_cumulative_hazard_function(X_train),
        test_chf_predictions=best.predict_cumulative_hazard_function(X_test),
        train_surv_predictions=best.predict_survival_function(X_train),
        test_surv_predictions=best.predict_survival_function(X_test),
        train_labels=y_train,
        test_labels=y_test,
    )


# =====================================================================================
# 3 · Score Cox and the PET RSF the same way (test folds), as in 02
#
# IPCW C-index on the risk score, time-dependent AUC on the cumulative hazard at each
# grid time, Brier score / IBS on the survival function.
# =====================================================================================

def score_model(path_template, model_name):
    """Per-fold test metrics plus the pooled out-of-fold predictions."""
    res = {"c": [], "auc": [], "bs": [], "ibs": [], "y_test": [],
           "chf": [], "risk": []}
    for i, (_, test_idx) in enumerate(folds):
        d = np.load(path_template.format(fold=i), allow_pickle=True)
        y_train, y_test = d["train_labels"], d["test_labels"]
        # Guard against folds that don't line up with the RSF's.
        if not np.array_equal(y_test, y[test_idx]):
            raise SystemExit(f"{model_name} fold {i}: test patients differ from this "
                             "script's folds -- predictions are not comparable.")

        chf = np.vstack([fn(time) for fn in d["test_chf_predictions"]])
        surv = np.vstack([fn(time) for fn in d["test_surv_predictions"]])

        res["c"].append(concordance_index_ipcw(y_train, y_test, d["test_predictions"])[0])
        res["auc"].append(cumulative_dynamic_auc(y_train, y_test, chf, time)[0])
        res["bs"].append(brier_score(y_train, y_test, surv, time)[1])
        res["ibs"].append(integrated_brier_score(y_train, y_test, surv, time))
        res["y_test"].append(y_test)
        res["chf"].append(chf)
        res["risk"].append(d["test_predictions"])

    res["auc"], res["bs"] = np.array(res["auc"]), np.array(res["bs"])
    res["pooled_y"] = np.concatenate(res["y_test"])
    res["chf"] = np.vstack(res["chf"])
    res["risk"] = np.concatenate(res["risk"])
    return res


scores = {
    "RSF": score_model(f"{PREDICTIONS_IN}/earliest_pet_rsf_fold{{fold}}_predictions.npz", "RSF"),
    "CoxPH": score_model(f"{PREDICTIONS_DIR}/earliest_pet_cph_fold{{fold}}_predictions.npz", "CoxPH"),
}
pooled_y = scores["RSF"]["pooled_y"]
assert np.array_equal(pooled_y, scores["CoxPH"]["pooled_y"])

# 02-style summaries
for name, s in scores.items():
    s["auc_curve"] = np.nanmean(s["auc"], axis=0)        # fold-averaged AUC per time
    s["fold_auc"] = np.nanmean(s["auc"], axis=1)         # per-fold mean AUC
    s["bs_curve"] = np.nanmean(s["bs"], axis=0)
    s["bs_sd"] = np.nanstd(s["bs"], axis=0)
    s["mean_auc"] = np.mean(s["auc_curve"])
    s["mean_auc_sd"] = np.nanstd(s["fold_auc"])          # auc_std() in 02
    s["mean_c"] = np.mean(s["c"])
    s["mean_c_sd"] = np.std(s["c"], ddof=1)
    s["mean_bs"] = np.mean(s["bs_curve"])
    s["mean_bs_sd"] = np.mean(s["bs_sd"])
    s["mean_ibs"] = np.mean(s["ibs"])

per_fold = pd.DataFrame([
    {"model": name, "fold": i, "c_index": s["c"][i], "mean_auc": s["fold_auc"][i],
     "mean_brier": np.mean(s["bs"][i]), "ibs": s["ibs"][i]}
    for name, s in scores.items() for i in range(len(folds))
])
print("\nPer-fold test metrics")
print(per_fold.pivot(index="fold", columns="model").round(3).to_string())


# =====================================================================================
# 4 · Pooled out-of-fold bootstrap (as in Section 13 of 02), paired across models
#
# Test-fold predictions are pooled across the five folds and the 201 patients are
# resampled 1000x. The same resample is used for both models, which gives a CI for the
# difference (Cox - RSF). The per-time AUC percentiles are the CI bands in the AUC plot.
# =====================================================================================

rng = np.random.default_rng(42)
for s in scores.values():
    s["pooled_auc"] = cumulative_dynamic_auc(y, pooled_y, s["chf"], time)[1]
    s["pooled_c"] = concordance_index_ipcw(y, pooled_y, s["risk"], tau=time[-1])[0]
    s["boot_curves"], s["boot_auc"], s["boot_c"] = [], [], []

for _ in range(N_BOOT):
    idx = rng.choice(n, size=n, replace=True)
    yb = pooled_y[idx]
    if yb["event"].sum() == 0 or yb["event"].all():
        continue
    try:
        draws = {}
        for name, s in scores.items():
            curve, mean_auc = cumulative_dynamic_auc(y, yb, s["chf"][idx], time)
            c = concordance_index_ipcw(y, yb, s["risk"][idx], tau=time[-1])[0]
            draws[name] = (curve, mean_auc, c)
    except ValueError:
        continue
    if any(np.isnan([v[1], v[2]]).any() for v in draws.values()):
        continue
    for name, (curve, mean_auc, c) in draws.items():
        scores[name]["boot_curves"].append(curve)
        scores[name]["boot_auc"].append(mean_auc)
        scores[name]["boot_c"].append(c)

for s in scores.values():
    for k in ("boot_curves", "boot_auc", "boot_c"):
        s[k] = np.array(s[k])
    s["auc_lower"] = np.nanpercentile(s["boot_curves"], 2.5, axis=0)
    s["auc_upper"] = np.nanpercentile(s["boot_curves"], 97.5, axis=0)
print(f"Valid bootstrap samples: {len(scores['RSF']['boot_auc'])}")


def ci(a):
    lo, hi = np.percentile(a, [2.5, 97.5])
    return f"[{lo:.3f}, {hi:.3f}]"


def two_sided_p(diff):
    """Bootstrap p-value for H0: difference = 0."""
    return min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))


cph, rsf = scores["CoxPH"], scores["RSF"]
d_auc = cph["boot_auc"] - rsf["boot_auc"]
d_c = cph["boot_c"] - rsf["boot_c"]


def pm(s, key):
    return f"{s[key]:.3f} ± {s[key + '_sd']:.3f}"


summary = pd.DataFrame([
    {"metric": "Mean AUC (± SD of per-fold mean AUC)",
     "CoxPH": pm(cph, "mean_auc"), "RSF": pm(rsf, "mean_auc"), "Cox - RSF [95% CI], p": ""},
    {"metric": "Mean C-index, IPCW (± SD across folds)",
     "CoxPH": pm(cph, "mean_c"), "RSF": pm(rsf, "mean_c"), "Cox - RSF [95% CI], p": ""},
    {"metric": "Mean Brier score (± mean across-fold SD)",
     "CoxPH": pm(cph, "mean_bs"), "RSF": pm(rsf, "mean_bs"), "Cox - RSF [95% CI], p": ""},
    {"metric": "Integrated Brier score (mean over folds)",
     "CoxPH": f"{cph['mean_ibs']:.3f}", "RSF": f"{rsf['mean_ibs']:.3f}",
     "Cox - RSF [95% CI], p": ""},
    {"metric": "Pooled mean AUC [95% bootstrap CI]",
     "CoxPH": f"{cph['pooled_auc']:.3f} {ci(cph['boot_auc'])}",
     "RSF": f"{rsf['pooled_auc']:.3f} {ci(rsf['boot_auc'])}",
     "Cox - RSF [95% CI], p":
         f"{cph['pooled_auc'] - rsf['pooled_auc']:+.3f} {ci(d_auc)}, p = {two_sided_p(d_auc):.3f}"},
    {"metric": "Pooled C-index [95% bootstrap CI]",
     "CoxPH": f"{cph['pooled_c']:.3f} {ci(cph['boot_c'])}",
     "RSF": f"{rsf['pooled_c']:.3f} {ci(rsf['boot_c'])}",
     "Cox - RSF [95% CI], p":
         f"{cph['pooled_c'] - rsf['pooled_c']:+.3f} {ci(d_c)}, p = {two_sided_p(d_c):.3f}"},
])
print("\nCoxPH vs RSF -- age, sex, education, APOE4 + amyloid PET")
print(summary.to_string(index=False))


# =====================================================================================
# 5 · Proportional-hazards check (Grambsch-Therneau test, as R's cox.zph)
#
# Cox assumes each covariate's hazard ratio is constant over follow-up. The test:
#   1. fit one ordinary (unpenalised, Efron ties) Cox model on all 201 patients;
#   2. at each progression, take every covariate's Schoenfeld residual: the patient
#      who progressed minus the risk-weighted average of everyone still at risk then.
#      If PH holds these residuals show no trend over time;
#   3. score-test whether adding a "covariate x g(time)" term would improve the fit,
#      with g(time) = rank of follow-up time. That is a test of whether the residuals
#      trend with time, per covariate (1 df) and for all covariates jointly (GLOBAL).
# A small p-value means that covariate's effect changes over follow-up. "direction"
# says which way the log hazard ratio drifts: "strengthens" = it increases later in
# follow-up, "weakens" = it decreases (for a risk factor, its effect fades).
# Matches survival::cox.zph(coxph(..., ties = "efron"), transform = "rank") in R
# (checked against survival 3.8.3: identical p-values to 3 decimals).
# =====================================================================================

def cox_zph_rank(X, time_, event, beta):
    """Grambsch-Therneau score test of proportional hazards, rank time transform."""
    X, time_, event = np.asarray(X, float), np.asarray(time_, float), np.asarray(event, bool)
    risk = np.exp(X @ beta)
    ev = np.where(event)[0]
    g = rankdata(time_)[ev]                 # rank among ALL follow-up times, as in R
    g = g - g.mean()
    resid, info = [], []
    for i in ev:
        at_risk = time_ >= time_[i]
        w = risk[at_risk] / risk[at_risk].sum()
        xbar = w @ X[at_risk]
        resid.append(X[i] - xbar)                         # Schoenfeld residual
        xc = X[at_risk] - xbar
        info.append((xc * w[:, None]).T @ xc)             # risk-set covariance
    resid, info = np.array(resid), np.array(info)
    U = resid.T @ g                                       # score for covariate x g(t)
    I11 = info.sum(0)
    I12 = np.einsum("k,kij->ij", g, info)
    I22 = np.einsum("k,kij->ij", g ** 2, info)
    var_U = I22 - I12 @ np.linalg.solve(I11, I12)
    stat = U ** 2 / np.diag(var_U)
    global_stat = U @ np.linalg.solve(var_U, U)
    return U, stat, global_stat, resid, ev


ph_fit = sm.PHReg(y["time"], X, status=y["event"].astype(int), ties="efron").fit()
U, zph_stat, zph_global, zph_resid, zph_ev = cox_zph_rank(
    X, y["time"], y["event"], np.asarray(ph_fit.params))

ph_check = pd.DataFrame({
    "HR (per unit)": np.exp(np.asarray(ph_fit.params)),
    "chisq": zph_stat,
    "p": chi2.sf(zph_stat, 1),
    "direction": np.where(U > 0, "strengthens", "weakens"),
}, index=COVARIATES)
ph_check.loc["GLOBAL"] = [np.nan, zph_global, chi2.sf(zph_global, len(COVARIATES)), ""]
ph_check["PH holds (p >= 0.05)"] = np.where(ph_check["p"] >= 0.05, "yes", "NO")
print("\nProportional-hazards check (Grambsch-Therneau, rank of time; = R cox.zph)")
print(ph_check.round(3).to_string())


# =====================================================================================
# 6 · Comparison figures, in the style of Section 16 of 02
#
#   supp_cph_rsf_auc_comparison.pdf      AUROC over time, bootstrap 95% CI bands,
#                                        cases / censored still at risk underneath
#   supp_cph_rsf_cindex_comparison.pdf   per-fold IPCW C-index, box + points
#   supp_cph_rsf_bs_comparison.pdf       Brier score over time, ± SD across folds
# =====================================================================================

def add_complete_risk_table(ax, time, y_test_folds, fig, step=1, fontsize=10):
    """Same helper as in 02: counts summed across all folds at each time point."""
    time_years = time / 365.25
    display_idx = list(range(0, len(time), step))

    cases = np.zeros(len(time), dtype=int)
    censored = np.zeros(len(time), dtype=int)
    for y_test in y_test_folds:
        for i, t_days in enumerate(time):
            cases[i] += np.sum((y_test['event'] == True) & (y_test['time'] > t_days))
            censored[i] += np.sum((y_test['event'] == False) & (y_test['time'] > t_days))

    xlim = ax.get_xlim()
    y_cases, y_censored = -0.18, -0.26
    case_color, censor_color = '#8B0000', '#2F4F7F'

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


years = time / 365.25

# ── AUROC ──
fig, ax = plt.subplots(figsize=(12, 7))
for name in MODELS:
    s = scores[name]
    ax.plot(years, s["auc_curve"], "o-", color=PALETTE[name],
            label=f"{name} (mean AUC = {s['mean_auc']:.3f} ± {s['mean_auc_sd']:.3f})")
    ax.fill_between(years, s["auc_lower"], s["auc_upper"], color=PALETTE[name], alpha=0.2)
ax.set_ylabel("AUROC", fontsize=14, fontweight='bold')
ax.set_xlabel("Follow-up time (years)", fontsize=14, fontweight='bold')
ax.legend(loc="lower center", fontsize=11)
ax.grid(True)
ax.set_ylim(0.3, 1.0)
plt.subplots_adjust(bottom=0.30)
add_complete_risk_table(ax, time, scores["RSF"]["y_test"], fig, step=1)
plt.savefig(f"{FIGURES_DIR}/supp_cph_rsf_auc_comparison.pdf")
plt.close(fig)

# ── C-index ──
model_labels = [name for name in MODELS for _ in scores[name]["c"]]
combined_c_list = [c for name in MODELS for c in scores[name]["c"]]

fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x=model_labels, y=combined_c_list, order=MODELS, hue=model_labels,
            palette=PALETTE, legend=False, showmeans=True,
            meanprops={'marker': '^', 'markerfacecolor': 'black',
                       'markeredgecolor': 'black', 'markersize': 8},
            ax=ax)
sns.stripplot(x=model_labels, y=combined_c_list, order=MODELS,
              color='black', size=6, alpha=0.7, ax=ax)
for j, name in enumerate(MODELS):
    s = scores[name]
    ax.text(j, 1.02, f"{s['mean_c']:.3f} ± {s['mean_c_sd']:.3f}",
            transform=ax.get_xaxis_transform(), ha='center', fontsize=12)
ax.set_ylabel("C-index", fontsize=14, fontweight='bold')
ax.set_xlabel("")
ax.tick_params(axis='y', labelsize=12)
ax.tick_params(axis='x', labelsize=12, width=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/supp_cph_rsf_cindex_comparison.pdf", bbox_inches='tight')
plt.close(fig)

# ── Brier score ──
fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
for name in MODELS:
    s = scores[name]
    ax.plot(years, s["bs_curve"], "o-", color=PALETTE[name],
            label=f"{name} (mean IBS = {s['mean_ibs']:.3f})")
    ax.fill_between(years, s["bs_curve"] - s["bs_sd"], s["bs_curve"] + s["bs_sd"],
                    color=PALETTE[name], alpha=0.2)
ax.set_xlabel("Follow-up time (years)", fontsize=12, fontweight='bold')
ax.set_ylabel("Time-dependent Brier Score", fontsize=12, fontweight='bold')
ax.tick_params(axis='both', which='major', labelsize=12, width=2)
ax.legend(loc="lower center", prop={'weight': 'bold'})
ax.grid(True)
plt.savefig(f"{FIGURES_DIR}/supp_cph_rsf_bs_comparison.pdf", bbox_inches='tight')
plt.close(fig)

# ── Scaled Schoenfeld residuals for amyloid PET (as R's plot(cox.zph(fit))) ──
# Each point is one progression. Scaled residuals estimate the log hazard ratio at that
# time, beta(t) (Grambsch & Therneau 1994): scaled = beta + n_events * residual @ Var(beta).
# Under proportional hazards the smooth is flat at the Cox estimate; a downward slope
# means the Centiloid effect weakens over follow-up.
j = COVARIATES.index("CENTILOIDS")
beta = np.asarray(ph_fit.params)
scaled = beta + len(zph_ev) * zph_resid @ np.asarray(ph_fit.cov_params())
t_ev = np.asarray(y["time"])[zph_ev] / 365.25
smooth = sm.nonparametric.lowess(scaled[:, j], t_ev, frac=2 / 3)

fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
ax.axhline(0, color="black", lw=0.8)
ax.scatter(t_ev, scaled[:, j], color=PALETTE["CoxPH"], alpha=0.6, s=25,
           label="Scaled Schoenfeld residual (one per progression)")
ax.plot(smooth[:, 0], smooth[:, 1], color="black", lw=2,
        label="LOWESS smooth: estimated β(t)")
ax.axhline(beta[j], color=PALETTE["RSF"], ls="--", lw=2,
           label=f"Cox estimate, constant β = {beta[j]:.3f}")
ax.set_xlabel("Follow-up time (years)", fontsize=12, fontweight='bold')
ax.set_ylabel("β(t), amyloid PET\n(log HR per Centiloid)", fontsize=12, fontweight='bold')
ax.set_title(f"Proportional-hazards check, amyloid PET "
             f"(Grambsch–Therneau p = {ph_check.loc['CENTILOIDS', 'p']:.3f})", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.savefig(f"{FIGURES_DIR}/supp_cph_schoenfeld_centiloids.pdf", bbox_inches='tight')
plt.close(fig)

print(f"\nWrote figures to {FIGURES_DIR}")
