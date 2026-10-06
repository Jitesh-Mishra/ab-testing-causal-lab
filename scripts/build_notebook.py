"""Build and execute notebooks/readout.ipynb from reports/ (run after python -m abx.readout)."""
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
    md("# Experiment readout: do retargeting ads cause visits and conversions?\n"
       "Criteo Uplift v2.1, 5% random sample (699,009 users). Everything below is read from `reports/`, produced by "
       "`python -m abx.readout`. Metric design: `docs/metric_design.md`."),
    code("import json, pandas as pd\nfrom IPython.display import Image, display\n"
         "pd.set_option('display.float_format', lambda v: f'{v:,.5f}')\n"
         "R = json.load(open('../reports/readout.json'))\nfull = json.load(open('../reports/full_dataset_stats.json'))"),
    md("## 1. Health checks: can we trust the split?"),
    code("srm = R['srm']\nprint(f\"sample: {srm['n']:,} users, treated share {srm['observed_share']:.4f} vs designed 0.85, SRM p = {srm['p_value']:.3f}\")\n"
         "print(f\"full dataset: {full['full_rows']:,} users, treated share {full['treat_share']:.4f}, control users exposed: {full['control_exposed']}\")\n"
         "print(f\"balance: out-of-fold AUC predicting T from features = {R['balance_auc_t_from_features']:.4f}; \"\n"
         "      f\"max |std mean diff| = {R['max_abs_std_mean_diff']:.3f} (one SE = {R['smd_se']:.4f})\")\n"
         "pd.DataFrame(R['rates_by_arm'])"),
    md("**Read:** the SRM p of 0.036 is a property of this random 5% sample. The full dataset is at 0.8500, and a p of 0.036 "
       "is far from the usual 0.001 alarm level, so this is **not an SRM**. Balance is the real finding: features predict "
       "assignment slightly but very significantly (AUC 0.508, mean differences up to 13 SEs). Criteo's erratum says feature "
       "distributions are advertiser-dependent and incrementality varied by advertiser. The likely cause is pooling of "
       "tests with different setups (a hypothesis; advertiser id is not in the data). **Consequence: the "
       "unadjusted difference in means is not trustworthy on its own. Adjust for features before reading the lift.**"),
    md("## 2. Effect of assignment (ITT), with and without adjustment"),
    code("itt = pd.read_csv('../reports/itt.csv')\nitt['shift_vs_unadj_in_se'] = itt.groupby('outcome', group_keys=False).apply(\n"
         "    lambda g: (g.estimate - g.estimate.iloc[0]) / g.se.iloc[0])\nitt"),
    code("display(Image('../reports/figures/effects-light.png'))\n"
         "pd.DataFrame({y: R['itt'][y]['permuted_check'] for y in ['visit', 'conversion']})"),
    md("**Read:** on a permuted A/A (clean randomization by construction), adjustment moves the estimate by under 1 SE, as it "
       "should. On the real assignment it moves the visit effect by about 4.4 SEs, which is imbalance correction, not noise. The "
       "adjusted estimators (Lin, CUPAC, AIPW) agree with each other, and adjustment cuts the variance of the visit estimate by "
       "25% (2% for the rare conversion outcome)."),
    md("## 3. Power"),
    code("pd.read_csv('../reports/power.csv')"),
    md("## 4. Peeking"),
    code("display(Image('../reports/figures/peeking-light.png'))\npd.read_csv('../reports/peeking.csv')"),
    md("**Read:** checking a fixed-horizon test 20 times and stopping at the first p < 0.05 gives false positives 5 to 6 times "
       "the nominal rate. Pocock and O'Brien-Fleming boundaries bring it back to about 5%. mSPRT is valid at any stopping time, so "
       "it is conservative with only 20 looks. On the real effect all methods reach 100% power, and the bootstrap "
       "`avg_stop_fraction` shows how early each would stop."),
    md("## 5. Effect on users who actually saw an ad (IV)"),
    code("display(Image('../reports/figures/iv-light.png'))\n"
         "pd.DataFrame({y: {'naive exposed-unexposed': R['iv'][y]['naive']['exposed_vs_unexposed_all'],\n"
         "                 'IV Wald': R['iv'][y]['wald']['late'], 'IV 2SLS + features': R['iv'][y]['tsls_with_features']['late'],\n"
         "                 'naive / Wald': R['iv'][y]['naive']['exposed_vs_unexposed_all'] / R['iv'][y]['wald']['late'],\n"
         "                 'naive / 2SLS': R['iv'][y]['naive']['exposed_vs_unexposed_all'] / R['iv'][y]['tsls_with_features']['late'],\n"
         "                 'first stage': R['iv'][y]['wald']['itt_d']} for y in ['visit', 'conversion']})"),
    md("## 6. Recommendation\n"
       "**Ship (keep running ads for this audience), conditional on cost.** After adjusting for the feature imbalance, assignment "
       "to ads raises visits by about 0.7 percentage points (about +17 to 19%) and conversions by about 0.09 points (about +42 to 49%). "
       "Every adjusted 95% CI excludes zero. For users who actually see an ad, the effect is about 20 to 27 points on visits and "
       "about 3 points on conversions. Dashboards that compare exposed with unexposed users overstate this 1.4 to 2.0x, so bids "
       "priced from them would overpay.\n\n"
       "**Before acting:** (1) get spend to compute cost per incremental conversion, the real ship criterion; "
       "(2) re-run with advertiser id or test id to stratify the imbalance away instead of modelling it; (3) treat the size of "
       "the conversion lift as uncertain. It is estimated on about 2,000 conversions in this sample."),
]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = ROOT / "notebooks" / "readout.ipynb"
NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT / "notebooks")}}).execute()
nbf.write(nb, out)
print("wrote", out)
