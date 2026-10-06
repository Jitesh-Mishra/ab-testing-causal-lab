# Probe prep (answers come only from what this run did)

Read `notebooks/readout.ipynb` end to end and run `pytest -q` before an interview.

## How the code works, in 5 lines
1. `estimators.py` has the plain-numpy estimators: SRM chi-square, difference in means with Welch SE, Lin regression adjustment with HC2 SEs, AIPW, MDE and sample-size formulas, Wald IV and 2SLS.
2. `sequential.py` computes running z-statistics at 20 looks using cumulative sums. It calibrates Pocock and O'Brien-Fleming boundaries by simulating Brownian motion, and implements mSPRT.
3. `readout.py` loads the sample, runs the health checks (SRM, balance AUC, mean differences), and computes ITT under five estimators. CUPAC and AIPW nuisances are cross-fitted LightGBM models that never see assignment, except the propensity model.
4. The same file runs 2,000 permutation A/A replays and bootstrap replays for peeking, then IV vs the naive contrasts, and writes `reports/`.
5. `build_notebook.py` turns `reports/` into the executed readout notebook. The tests check each estimator on simulations where the truth is known.

## Probes

**Q: Walk me through your readout.**
- 15s: I checked the split first, and it failed balance. Features predict assignment, and adjustment moved the visit lift from +25% to about +17 to 19%. On the adjusted numbers, ads lift visits and conversions significantly. I'd ship conditional on cost, which isn't in the data.

**Q: The SRM p-value is 0.036. Isn't that a failed SRM?**
- No. The full 14M file is at exactly 85.00%, and mine is a 5% random sample, so 85.09% is sampling noise. SRM alarms use p < 0.001, because with millions of users and many tests you'd otherwise chase noise constantly. The real problem was elsewhere: covariate balance.

**Q: How do you know the imbalance matters?**
- I ran the same adjustment on a permuted assignment, where randomization is clean by construction. The estimate moved under 1 SE. On the real assignment it moved 4.4 SE for visits. That difference is the imbalance. Criteo's erratum says features are advertiser-dependent and incrementality varied by advertiser, which fits pooled tests with different setups.

**Q: What is CUPAC and why did it help visits but not conversions?**
- CUPED adjusts with a pre-period metric. With no pre-period here, CUPAC uses a cross-fitted ML prediction of the outcome from pre-treatment features as the covariate. Variance falls by about the R² of that prediction: 25% for visits, but only 2% for a 0.2% conversion rate the features can't predict.

**Q: Why are naive exposed vs unexposed comparisons wrong?**
- Exposure is chosen by the auction, which targets high-intent users who would visit anyway. Assignment is random and only affects outcomes through exposure (control users are withheld), so it's a valid instrument. The effect on the exposed is ITT / exposure rate. The naive comparison was 1.4 to 2.0x too big.

**Q: Assumptions behind the IV?**
- Relevance (first stage 3.6 pts, F about 22,000), exclusion (withholding is the only difference), as-good-as-random assignment (violated mildly, hence 2SLS with features), and one-sided non-compliance (0 control exposures in 2.1M), which makes the LATE the effect on the treated who were exposed.

**Q: What does peeking do and how do you fix it?**
- Checking 20 times and stopping at the first p < 0.05 gave 28.5% false positives instead of 5%. Group-sequential boundaries spend alpha across looks: Pocock uses a flat threshold of about 2.5 here, and O'Brien-Fleming is strict early and lenient late. mSPRT gives always-valid p-values for any stopping rule but is conservative with few looks. I'd pick O'Brien-Fleming for a planned weekly review, and mSPRT for a dashboard people refresh whenever they like.

**Q: How do you know your estimators are right?**
- Simulations with known truth: CI coverage, unbiasedness, IV recovery, AIPW under feature-dependent assignment, peeking calibration, MDE giving 80% power. Plus statsmodels HC2 agreement to 1e-10. That's also how I'd validate AI-generated statistical code: test it against a case where you know the answer.

**Q: What would you do with more access?**
- Get advertiser or test id and stratify rather than model the imbalance. Get spend for cost per incremental conversion. Run the conversion readout on all 14M rows (MDE falls from 21% to 4.7%). Build an uplift model for who to target.
