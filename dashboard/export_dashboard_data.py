"""Export the small, pre-aggregated tables behind the Tableau and Power BI dashboards.

Run from the repository root:  python dashboard/export_dashboard_data.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dashboard" / "data"
OUT.mkdir(parents=True, exist_ok=True)
def save(df, name):
    df.to_csv(OUT / name, index=False)
    print(name); print(df.to_string(index=False)); print()

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
adm = pd.read_csv("https://stats.oarc.ucla.edu/stat/data/binary.csv")
save(adm.groupby("rank")["admit"].agg(["mean", "size"]).reset_index().rename(columns={"rank": "Institution rank", "mean": "Admission rate", "size": "Applicants"})
     .assign(**{"Institution rank": lambda d: "Rank " + d["Institution rank"].astype(str), "Admission rate": lambda d: d["Admission rate"].round(4)}), "adm_by_rank.csv")
band = pd.cut(adm["gpa"], [2.2, 3.0, 3.3, 3.6, 3.8, 4.0], labels=["1. Up to 3.0", "2. 3.0-3.3", "3. 3.3-3.6", "4. 3.6-3.8", "5. 3.8-4.0"])
save(adm.groupby(band, observed=True)["admit"].mean().round(4).reset_index().rename(columns={"gpa": "GPA band", "admit": "Admission rate"}), "adm_by_gpa.csv")
X, y = adm.drop(columns="admit"), adm["admit"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
prep = ColumnTransformer([("num", StandardScaler(), ["gre", "gpa"]), ("rank", OneHotEncoder(drop="first", categories=[[1, 2, 3, 4]]), ["rank"])])
logit = make_pipeline(prep, LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
names = ["GRE (+1 SD)", "GPA (+1 SD)", "Rank 2 vs rank 1", "Rank 3 vs rank 1", "Rank 4 vs rank 1"]
save(pd.DataFrame({"Factor": names, "Odds ratio": np.exp(logit[-1].coef_[0]).round(2)}), "adm_odds_ratios.csv")
bal = make_pipeline(prep, LogisticRegression(max_iter=1000, class_weight="balanced"))
oof = cross_val_predict(bal, X_tr, y_tr, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
rows = []
for t in np.round(np.arange(0.30, 0.71, 0.05), 2):
    p = oof >= t
    rows += [{"Threshold": f"{t:.2f}", "Metric": "Recall", "Value": round(p[y_tr == 1].mean(), 4)},
             {"Threshold": f"{t:.2f}", "Metric": "Precision", "Value": round(y_tr[p].mean() if p.any() else 0, 4)}]
save(pd.DataFrame(rows), "adm_thresholds.csv")
auc = roc_auc_score(y_te, bal.fit(X_tr, y_tr).predict_proba(X_te)[:, 1])
save(pd.DataFrame({"metric": ["Applicants", "Admitted", "Rank 1 admit rate", "Rank 4 admit rate", "Test ROC-AUC", "Recall at 0.40"],
                   "value": [f"{len(adm)}", f"{y.mean():.1%}", f"{adm[adm['rank'] == 1].admit.mean():.1%}", f"{adm[adm['rank'] == 4].admit.mean():.1%}",
                             f"{auc:.2f}", f"{(oof >= 0.4)[y_tr == 1].mean():.0%}"]}), "adm_kpis.csv")
