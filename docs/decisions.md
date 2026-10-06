# Decisions

**D1 Data: a 5% Bernoulli sample (seed 42) of the full v2.1 file, staged from Jitesh's Mac**
Criteo, HuggingFace and the sklift S3 mirror are all blocked from the cloud workspace and from the Mac sandbox. Jitesh downloaded the official file, and the sample was taken with DuckDB on his machine (`scripts/make_sample.py`). It has 699,009 rows. Full-file totals are kept in `reports/full_dataset_stats.json`. The data is not committed (non-commercial research license).

**D2 SRM p = 0.036 is not an alarm**
The full file is at 0.85000 treated. The sample's 0.8509 is Bernoulli sampling noise, and SRM alarms conventionally use p < 0.001.

**D3 The balance check failed, so adjusted estimates are the headline**
Out-of-fold AUC predicting assignment from features is 0.508, and standardized mean differences reach 0.046 (about 13 SE). On a permuted A/A, adjustment moves estimates by under 1 SE; on the real assignment it moves visit by 4.4 SE. The unadjusted lift is reported but not used for the decision. Three adjusted estimators with different assumptions (Lin, CUPAC, AIPW) are shown side by side.

**D4 Variance reduction reported on the real experiment, not the A/A**
The permuted A/A shows a larger variance reduction (33.5% visit, 9.1% conversion), because the treatment effect itself adds variance the features can't explain. The real-experiment numbers (25.0% and 2.0%) are what this test actually gained, and the power table uses those.

**D5 mSPRT mixing scale fixed before seeing real results**
On simulated data, tau at a 1% lift made mSPRT almost never reject. Tau was set to a 5% relative lift before the real-data run, so it was not tuned on the outcome.

**D6 Peeking evaluated by permutation A/A on the real outcomes**
Keeping the real outcome distribution, including the rare conversions, is a better test than Gaussian simulation. Group-sequential boundaries are calibrated by Brownian-motion simulation for K = 20 equally spaced looks.

**D7 IV headline uses the covariate-adjusted 2SLS**
Wald assumes clean randomization, which the balance check rejects. 2SLS with the 12 features as controls is the more defensible number, and the naive overstatement is reported as a range across both (1.4 to 2.0x).

**D8 statsmodels as the reference implementation**
The estimators are hand-written with numpy so they scale to millions of rows. statsmodels' OLS HC2 is used in a test to verify them to 1e-10.
