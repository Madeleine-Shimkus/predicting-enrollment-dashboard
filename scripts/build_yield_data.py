"""Builds the three yield files the dashboard reads, without Colab.

This is the same pipeline as Capstone_Yield_Model.ipynb (steps 4-8), minus the plots and
the IPEDS download. It needs the IPEDS admissions counts as a CSV with columns
unitid, year, number_applied, number_admitted, number_enrolled.

    python scripts/build_yield_data.py                                   # 15 schools (default)
    python scripts/build_yield_data.py data/raw/admissions_yield.csv     # all 93, once you have it

Writes data/yield_model_data.csv, data/yield_within_effects.csv, data/yield_across_effects.csv,
the same three files the notebook saves to Google Drive.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features import CANDIDATES  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
SCORECARD_FILE = ROOT / "data" / "raw" / "scorecard_93_schools.csv"
ADMISSIONS_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "raw" / "admissions_15_schools.csv"
OUT = ROOT / "data"

SCHOOL_NAMES = {123961: "USC", 131496: "Georgetown", 139658: "Emory", 152080: "Notre Dame",
                164924: "Boston College", 165015: "Brandeis", 168148: "Tufts", 179867: "WashU",
                190415: "Cornell", 193900: "NYU", 198419: "Duke", 199847: "Wake Forest",
                217156: "Brown", 221999: "Vanderbilt", 227757: "Rice"}

TARGET = "yield"
MIN_TARGET_YEAR = 2010
N_MAX = 12
MAX_CORR = 0.8
MIN_COVERAGE = 0.8
N_TEST_YEARS = 6

# ---------------- step 4: merge + yield ----------------
scorecard_df = pd.read_csv(SCORECARD_FILE, low_memory=False)
adm_df = pd.read_csv(ADMISSIONS_FILE)
scorecard_df["fall"] = scorecard_df["year"].str[:4].astype(int)

latest = scorecard_df.sort_values("fall").groupby("UNITID").last()
traits = pd.DataFrame({
    "Religious": latest["RELAFFIL"].notna().astype(int),
    "Catholic": latest["RELAFFIL"].isin([30]).astype(int),
    "City campus": (latest["LOCALE"] // 10 == 1).astype(int),
    "Region": latest["REGION"].map({1: "New England", 2: "Mid East", 3: "Great Lakes", 4: "Plains",
                                    5: "Southeast", 6: "Southwest", 7: "Rocky Mtns", 8: "Far West"}),
    "INSTNM_latest": latest["INSTNM"],
})

merged = (scorecard_df.merge(adm_df, left_on=["UNITID", "fall"], right_on=["unitid", "year"],
                             how="left", suffixes=("", "_adm"))
          .drop(columns=["unitid", "year_adm"])
          .merge(traits, left_on="UNITID", right_index=True, how="left"))

model_df = (merged.dropna(subset=["number_admitted", "number_enrolled"])
            .drop_duplicates(["UNITID", "fall"]).sort_values(["UNITID", "fall"]).reset_index(drop=True))
model_df["yield"] = model_df["number_enrolled"] / model_df["number_admitted"] * 100
model_df = model_df[model_df["yield"].between(1, 100)]
model_df["school"] = model_df["UNITID"].map(SCHOOL_NAMES).fillna(model_df["INSTNM_latest"])

# ---------------- step 5: candidate predictors (last year's values) ----------------
df = model_df.copy()
num = lambda c: pd.to_numeric(df[c], errors="coerce") if c in df.columns else pd.Series(np.nan, index=df.index)

df["COST_FILLED"] = num("COSTT4_A").fillna(
    num("TUITIONFEE_IN") + num("ROOMBOARD_ON") + num("BOOKSUPPLY") + num("OTHEREXPENSE_ON"))
df["ENDOW_PER_STUDENT"] = num("ENDOWEND") / num("UGDS")
df["NONWHITE_SHARE"] = 1 - num("UGDS_WHITE").fillna(num("UGDS_WHITENH"))
df["EARN_FILLED"] = num("MD_EARN_WNE_P10").groupby(df["UNITID"]).ffill()
df["APPS"] = num("number_applied")

candidates = CANDIDATES

consecutive = df.groupby("UNITID")["fall"].shift(1) == df["fall"] - 1
def lag(s):
    return s.groupby(df["UNITID"]).shift(1).where(consecutive)

X = pd.DataFrame(index=df.index)
for code, (label, use_log) in candidates.items():
    s = num(code)
    if use_log:
        s = np.log(s.where(s > 0))
    X[label] = lag(s)
X["Test required"] = (num("ADMCON7") == 1).astype(float).where(num("ADMCON7").notna())

TRAITS = ["Religious", "Catholic", "City campus"]
data = pd.concat([df[["UNITID", "school", "fall", "Region"] + TRAITS],
                  df[TARGET].rename("target"), X], axis=1)
data = data[data["target"].notna() & (data["fall"] >= MIN_TARGET_YEAR)].reset_index(drop=True)
LAGGED = list(X.columns)
print(f"{len(data)} school-years, {data['UNITID'].nunique()} schools, falls {data['fall'].min()}-{data['fall'].max()}")

# ---------------- step 6: screen ----------------
def two_way_demean(s, d=data):
    s = s - s.groupby(d["UNITID"]).transform("mean")
    return s - s.groupby(d["fall"]).transform("mean")

feats = [f for f in LAGGED if data[f].notna().mean() >= MIN_COVERAGE]

# ---------------- step 7: forward selection ----------------
def fe_design(d, features, school_levels, year_levels):
    S = pd.get_dummies(pd.Categorical(d["UNITID"], categories=school_levels)).values.astype(float)
    Y = pd.get_dummies(pd.Categorical(d["fall"], categories=year_levels)).values[:, 1:].astype(float)
    return np.column_stack([S, Y, d[features].values]) if features else np.column_stack([S, Y])

def cv_rmse(d, features):
    errs = []
    for ty in sorted(d["fall"].unique())[-N_TEST_YEARS:]:
        tr = d[d["fall"] < ty]
        te = d[(d["fall"] == ty) & d["UNITID"].isin(tr["UNITID"])].copy()
        if len(tr) < 50 or te.empty:
            continue
        trX, teX = tr.copy(), te.copy()
        if features:
            med = tr[features].median()
            trX[features] = trX[features].fillna(med); teX[features] = teX[features].fillna(med)
        schools, years = sorted(tr["UNITID"].unique()), sorted(tr["fall"].unique())
        teX["fall"] = years[-1]
        beta, *_ = np.linalg.lstsq(fe_design(trX, features, schools, years), trX["target"].values, rcond=None)
        pred = fe_design(teX, features, schools, years) @ beta
        errs.append((te["target"].values - pred) ** 2)
    return np.sqrt(np.concatenate(errs).mean())

def forward_select(d, pool):
    corr = two_way_demean(d[pool].fillna(d[pool].median()), d).corr()
    selected, remaining = [], list(pool)
    history = [("Baseline (school + year only)", cv_rmse(d, []))]
    while len(selected) < N_MAX and remaining:
        scores = {f: cv_rmse(d, selected + [f]) for f in remaining
                  if not any(abs(corr.loc[f, s]) > MAX_CORR for s in selected)}
        if not scores:
            break
        best = min(scores, key=scores.get)
        selected.append(best); remaining.remove(best)
        history.append((best, scores[best]))
    return selected, history

selected, history = forward_select(data, feats)
hist = pd.DataFrame(history, columns=["variable added", "RMSE (yield pts)"])
best_k = int(hist["RMSE (yield pts)"].iloc[1:].idxmin())
print(hist.round(3).to_string())

# ---------------- step 8: final models ----------------
FINAL = selected[:max(best_k, 6)]
ACROSS = [f for f in feats if f not in ("Applications", "Undergrad enrollment")]

def standardize(d, cols):
    d = d.copy()
    d[cols] = d[cols].fillna(d[cols].median())
    d[cols] = (d[cols] - d[cols].mean()) / d[cols].std()
    return d

def q(cols):
    return " + ".join(f"Q('{c}')" for c in cols)

def clustered(formula, d):
    return smf.ols(formula, d).fit(cov_type="cluster", cov_kwds={"groups": d["UNITID"]})

d_w = standardize(data, FINAL)
within = clustered(f"target ~ {q(FINAL)} + C(UNITID) + C(fall)", d_w)

corr_a = data[ACROSS].corr().abs()
keep = []
for f in ACROSS:
    if all(corr_a.loc[f, k] <= MAX_CORR for k in keep):
        keep.append(f)
d_a = standardize(data, keep)
# with few schools a trait can be all-0 or all-1; leave those out so the fit still runs
traits_used = [t for t in TRAITS if data[t].nunique() > 1]
region_term = " + C(Region)" if data["Region"].nunique() > 1 else ""
across = clustered(f"target ~ {q(keep)}" + (f" + {q(traits_used)}" if traits_used else "")
                   + f"{region_term} + C(fall)", d_a)

def tidy(fit, cols):
    names = [f"Q('{c}')" for c in cols]
    out = pd.DataFrame({"effect (yield pts per 1 SD)": fit.params[names].values,
                        "p-value": fit.pvalues[names].values}, index=cols)
    return out.reindex(out["effect (yield pts per 1 SD)"].abs().sort_values(ascending=False).index)

res_w = tidy(within, FINAL)
res_a = tidy(across, keep + traits_used)
print(res_w.round(3)); print(res_a.round(3))

data.to_csv(OUT / "yield_model_data.csv", index=False)
res_w.to_csv(OUT / "yield_within_effects.csv")
res_a.to_csv(OUT / "yield_across_effects.csv")
print("Wrote", OUT / "yield_model_data.csv", "and the two effects files")
