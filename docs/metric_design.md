# Metric design for the ad-incrementality experiment

> This is a documented design written for this public dataset, not live product work. The Criteo data has four outcome fields, so the metric set is constrained by what exists. Metrics a real team would add are marked *not in data*.

## Decision the experiment informs
Should the advertiser keep paying for retargeting ads for this audience? That is, do ads cause extra visits and conversions, and are they worth what they cost?

## Unit and assignment
- **Unit:** user. **Assignment:** treatment (eligible to be shown ads) vs control (withheld), designed at 85/15.
- **Exposure is not assigned.** Only 3.6% of treated users actually see an ad, because the auction decides. Every primary metric is therefore **intent-to-treat (ITT)** over all assigned users. The effect on users who saw an ad is a secondary, IV-based estimate.

## Metrics

| Role | Metric | Definition | Why |
|---|---|---|---|
| **Primary** | Conversion rate (ITT) | conversions / assigned users | Closest to the advertiser's value; the ship decision rests on it |
| Secondary | Visit rate (ITT) | visits / assigned users | 16x more frequent than conversion, so it has far more power; an early read and a mechanism check |
| Secondary | Effect on the exposed (LATE) | ITT / exposure rate, via IV | What "an impression is worth"; needed for bid pricing |
| Guardrail | Sample ratio | treated share vs 85% design | Broken assignment invalidates everything else |
| Guardrail | Covariate balance | out-of-fold AUC of predicting T from X; standardized mean differences | Checks that randomization holds in the pooled data |
| Guardrail | Control exposure | share of control users exposed | Must be 0 for the IV to identify the effect on the exposed |
| Guardrail | Conversion per visit | conversions / visits by arm | Ads that buy low-intent visits would lower it: quality of incremental traffic |
| Guardrail (*not in data*) | Cost per incremental conversion | spend / (ITT × users) | The real ship criterion; needs spend |
| Guardrail (*not in data*) | Unsubscribe / ad-hide rate | user annoyance | Long-term user cost |

## Trade-offs
- **Conversion vs visit as primary:** conversion is what pays, but at a 0.2% base rate it needs about 12M users to detect a 5% lift (`reports/power.csv`). Visit detects 5% with about 0.6M. The rule: decide on conversion, use visit to read direction early, and never ship on visit alone.
- **ITT vs effect on the exposed:** ITT is what the business gets from the policy "run ads for this audience". The per-exposure effect is larger and is the right input for bidding, but it relies on IV assumptions (below).
- **Peeking:** the readout is fixed-horizon. Anyone who wants to look early uses the group-sequential boundaries or mSPRT in `abx.sequential`, chosen before launch.

## IV assumptions (for the effect on the exposed)
1. **Assignment is as good as random:** violated mildly in the pooled data (see the balance guardrail), so the 2SLS adds features as controls.
2. **Exclusion:** assignment affects outcomes only through exposure. This is plausible, because control users are simply withheld from ads.
3. **Relevance:** first stage of 3.6 points, F of about 22,000.
4. **One-sided non-compliance:** no control user is exposed (verified: 0 of 2.1M in the full data), so the IV estimate is the effect on the treated who were exposed.
