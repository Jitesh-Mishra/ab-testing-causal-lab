"""Reproducible 5% Bernoulli sample of Criteo Uplift v2.1 (seed 42) + full-dataset totals.

Download criteo-research-uplift-v2.1.csv.gz from https://ailab.criteo.com/criteo-uplift-prediction-dataset/
(non-commercial research use; cite Diemert et al. 2018), then:
    python scripts/make_sample.py /path/to/criteo-research-uplift-v2.1.csv.gz
The dataset itself is not redistributed in this repo.
"""
import json
import sys
from pathlib import Path

import duckdb

src = sys.argv[1]
out = Path(__file__).resolve().parents[1]
con = duckdb.connect()
con.sql(f"CREATE TABLE crit AS SELECT * FROM read_csv_auto('{src}')")
tot = con.sql("SELECT count(*), avg(treatment), avg(visit), avg(conversion), avg(exposure), "
              "sum(exposure*(1-treatment)) FROM crit").fetchone()
(out / "data").mkdir(exist_ok=True)
con.sql(f"COPY (SELECT * FROM crit USING SAMPLE 5 PERCENT (bernoulli, 42)) TO '{out / 'data' / 'criteo_sample.parquet'}' "
        "(FORMAT parquet, COMPRESSION zstd)")
json.dump(dict(zip(["full_rows", "treat_share", "visit", "conversion", "exposure", "control_exposed"], tot)),
          open(out / "reports" / "full_dataset_stats.json", "w"), indent=1)
print(tot)
