library(timeROC)
library(survival)
library(dplyr)

# Usage:  Rscript 04_timeROC_comparisons.R [input_csv] [output_csv]
# input_csv defaults to $MCI_AD_OUTPUT_DIR/metrics/ADNI_PENN/timeROC_input.csv, the file
# written by Section 14 of 02_survival_models.py (same environment variable as the
# Python scripts; see config.py). output_csv defaults to timeROC_results_sorted.csv in the
# same folder as the input.
args <- commandArgs(trailingOnly = TRUE)
output_dir <- Sys.getenv("MCI_AD_OUTPUT_DIR", unset = "/path/to/output")
input_csv  <- if (length(args) >= 1) args[1] else
  file.path(output_dir, "metrics", "ADNI_PENN", "timeROC_input.csv")
output_csv <- if (length(args) >= 2) args[2] else
  file.path(dirname(input_csv), "timeROC_results_sorted.csv")
if (!file.exists(input_csv)) {
  stop("timeROC input not found: ", input_csv,
       "\nRun 02_survival_models.py first, set MCI_AD_OUTPUT_DIR, or pass the CSV path.")
}
cat("reading", input_csv, "\n")

df <- read.csv(input_csv)
cat("Loaded", nrow(df), "patients,", sum(df$event), "events\n")

# Timepoints in days
times <- c(2*365.25, 3*365.25, 4*365.25, 5*365.25)

# ── Compute timeROC for each model ──
# marker = cumulative hazard from RSF (higher = higher risk)
#Model with Age, Sex, APOE4, and Education (baseline)
ROC_demo <- timeROC(T=df$time, delta=df$event, marker=df$risk_demo,
                    cause=1, weighting="marginal", times=times, iid=TRUE)
#Baseline + PET Model
ROC_pet <- timeROC(T=df$time, delta=df$event, marker=df$risk_pet,
                   cause=1, weighting="marginal", times=times, iid=TRUE)
#Baseline + p-Tau217 Model
ROC_ptau <- timeROC(T=df$time, delta=df$event, marker=df$risk_ptau,
                    cause=1, weighting="marginal", times=times, iid=TRUE)
#Baseline + PET + p-Tau217 Model
ROC_pet_ptau <- timeROC(T=df$time, delta=df$event, marker=df$risk_pet_ptau,
                        cause=1, weighting="marginal", times=times, iid=TRUE)
#Baseline + p-Tau217, amyloid-beta 42/40, NfL, GFAP Model
ROC_bio <- timeROC(T=df$time, delta=df$event, marker=df$risk_bio,
                   cause=1, weighting="marginal", times=times, iid=TRUE)
#Baseline + p-Tau217/abeta42 Model
ROC_ptau_amyloid <- timeROC(T=df$time, delta=df$event, marker=df$risk_ptau_amyloid,
                            cause=1, weighting="marginal", times=times, iid=TRUE)
#Baseline + all blood biomarkers + PET
ROC_ksink <- timeROC(T=df$time, delta=df$event, marker=df$risk_ksink,
                     cause=1, weighting="marginal", times=times, iid=TRUE)

# ── Print AUCs ──
cat("\n========== AUCs ==========\n")
cat("Demographics:\n"); print(ROC_demo)
cat("PET:\n"); print(ROC_pet)
cat("p-Tau217:\n"); print(ROC_ptau)
cat("PET + p-Tau217:\n"); print(ROC_pet_ptau)
cat("Biomarker Panel:\n"); print(ROC_bio)
cat("p-Tau217/Abeta42:\n"); print(ROC_ptau_amyloid)
cat("All blood biomarkers + PET:\n"); print(ROC_ksink)

# ── Confidence intervals ──
cat("\n========== Confidence Intervals ==========\n")
cat("Baseline:\n"); confint(ROC_demo)
cat("PET:\n"); confint(ROC_pet)
cat("p-Tau217:\n"); confint(ROC_ptau)
cat("PET + p-Tau217:\n"); confint(ROC_pet_ptau)
cat("Biomarker Panel:\n"); confint(ROC_bio)
cat("p-Tau217/Abeta42:\n"); confint(ROC_ptau_amyloid)
cat("All blood biomarkers + PET:\n"); confint(ROC_ksink)

# ── Pairwise comparisons ──
cat("\n========== Pairwise Comparisons ==========\n")

cat("\n--- PET vs Demographics ---\n")
compare(ROC_pet, ROC_demo, adjusted = TRUE)

cat("\n--- p-Tau217 vs Demographics ---\n")
compare(ROC_ptau, ROC_demo, adjusted = TRUE)

cat("\n--- PET + p-Tau217 vs Demographics ---\n")
compare(ROC_pet_ptau, ROC_demo, adjusted = TRUE)

cat("\n--- Biomarker Panel vs Demographics ---\n")
compare(ROC_bio, ROC_demo, adjusted = TRUE)

cat("\n--- p-Tau217 vs PET ---\n")
compare(ROC_ptau, ROC_pet, adjusted = TRUE)

cat("\n--- PET + p-Tau217 vs p-Tau217 ---\n")
compare(ROC_pet_ptau, ROC_ptau, adjusted = TRUE)

cat("\n--- p-Tau217 vs p-Tau217/Abeta42 ---\n")
compare(ROC_ptau, ROC_ptau_amyloid, adjusted = TRUE)

cat("\n--- p-Tau217 vs Biomarker Panel ---\n")
compare(ROC_ptau, ROC_bio, adjusted = TRUE)

cat("\n--- p-Tau217 vs all Blood Biomarkers and PET ---\n")
compare(ROC_ptau, ROC_ksink, adjusted = TRUE)

cat("\n--- p-Tau217/Abeta42 vs PET ---\n")
compare(ROC_ptau_amyloid, ROC_pet, adjusted = TRUE)

cat("\n--- Biomarker Panel vs PET ---\n")
compare(ROC_bio, ROC_pet, adjusted = TRUE)


# ── Save results to CSV for Python ──
comparisons <- list(
  list("PET", "Demographics", ROC_pet, ROC_demo),
  list("p-Tau217", "Demographics", ROC_ptau, ROC_demo),
  list("Biomarker Panel", "Demographics", ROC_bio, ROC_demo),
  list("PET + p-Tau217", "Demographics", ROC_pet_ptau, ROC_demo),
  list("p-Tau217", "PET", ROC_ptau, ROC_pet),
  list("PET + p-Tau217", "PET", ROC_pet_ptau, ROC_pet),
  list("PET + p-Tau217", "p-Tau217", ROC_pet_ptau, ROC_ptau),
  list("p-Tau217/Abeta42", "p-Tau217", ROC_ptau_amyloid, ROC_ptau),
  list("p-Tau217/Abeta42", "PET", ROC_ptau_amyloid, ROC_pet),
  list("PET + p-Tau217", "Biomarker Panel", ROC_pet_ptau, ROC_bio),
  list("Biomarker Panel", "PET", ROC_bio, ROC_pet),
  list("Biomarker Panel", "p-Tau217", ROC_bio, ROC_ptau)
)

results_df <- data.frame()
for (comp in comparisons) {
  res <- compare(comp[[3]], comp[[4]], adjusted=TRUE)
  
  for (i in seq_along(times)) {
    results_df <- rbind(results_df, data.frame(
      model_a = comp[[2]],
      model_b = comp[[1]],
      time_days = times[i],
      time_years = times[i] / 365.25,
      AUC_a = comp[[4]]$AUC[i],
      AUC_b = comp[[3]]$AUC[i],
      delta_AUC = comp[[3]]$AUC[i] - comp[[4]]$AUC[i],
      p_unadjusted = res$p_values_AUC[1, i],  # row 1 = Non-adjusted
      p_adjusted = res$p_values_AUC[2, i]      # row 2 = Adjusted
    ))
  }
}

results_df <- results_df %>%
  mutate(comparison = paste(model_a, model_b, sep = " vs ")) %>%
  arrange(p_adjusted)

write.csv(results_df, output_csv, row.names=FALSE)
cat("wrote", output_csv, "\n")
print(results_df)
