"""The train/test-split yield model from Yield_TTS.ipynb, packaged for the dashboard.

Schools (not years) are split: 20% of schools, always including Boston College, are held
out, so every prediction for them comes from a model that never saw that school.
Predictors are last year's values of 8 variables plus year. Two models (Ridge regression
and gradient boosting) are averaged into a "Blend".

To update: replace data/yield_data.csv with a new yield_data.csv (the one the notebook reads).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

DATA_FILE = Path(__file__).parent / "data" / "yield_data.csv"
BC_ID = 164924
MIN_YEAR = 2010
TEST_SIZE = 0.20
HEADLINE_SEED = 42

# label -> (candidate Scorecard columns, take log?)   same as the notebook
VARIABLES = {
    "Median debt":           (["DEBT_MDN", "GRAD_DEBT_MDN"], True),
    "Net price $0-30k":      (["NPT41_PRIV"], True),
    "Federal loan share":    (["PCTFLOAN"], False),
    "Instruction $/student": (["INEXPFTE"], True),
    "Avg faculty salary":    (["AVGFACSAL"], True),
    "Net price (all)":       (["NPT4_PRIV"], True),
    "Student-faculty ratio": (["STUFACR"], False),
    "6-yr grad rate":        (["C150_4"], False),
}
# How each variable is shown to a person
FRIENDLY = {
    "Median debt": "Median student debt at graduation",
    "Net price $0-30k": "Yearly cost after aid, families earning under $30k",
    "Federal loan share": "Students taking federal loans",
    "Instruction $/student": "Instruction spending per student",
    "Avg faculty salary": "Average faculty salary (per month)",
    "Net price (all)": "Yearly cost after aid, average aided student",
    "Student-faculty ratio": "Students per faculty member",
    "6-yr grad rate": "Six-year graduation rate",
}
SHARES = {"Federal loan share", "6-yr grad rate"}
MODELS = ["Training mean", "Training year mean", "Ridge", "Boosting", "Blend"]


def version():
    return DATA_FILE.stat().st_mtime


def load():
    """Returns (data used for modeling, raw values by school-year, list of features, column used per feature)."""
    df = pd.read_csv(DATA_FILE, low_memory=False)
    df = df.sort_values(["UNITID", "fall"]).drop_duplicates(["UNITID", "fall"]).reset_index(drop=True)
    num = lambda c: pd.to_numeric(df[c], errors="coerce") if c in df.columns else pd.Series(np.nan, index=df.index)
    fall = df["fall"]
    ok = (fall - df.groupby("UNITID")["fall"].shift(1)) == 1

    def lag(s):
        return s.groupby(df["UNITID"]).shift(1).where(ok)

    F, raw, used = pd.DataFrame(index=df.index), pd.DataFrame(index=df.index), {}
    for label, (cands, use_log) in VARIABLES.items():
        col = next((c for c in cands if c in df.columns and num(c).notna().any()), None)
        if col is None:
            continue
        used[label] = col
        raw[label] = num(col)
        s = np.log(raw[label].where(raw[label] > 0)) if use_log else raw[label]
        F[label] = lag(s)
    feats = list(F.columns)
    df = pd.concat([df, F], axis=1)
    df["y"] = num("yield")
    data = df[df["y"].notna() & (df["fall"] >= MIN_YEAR)].reset_index(drop=True)
    raw = pd.concat([df[["UNITID", "school", "fall"]], raw], axis=1)
    return data, raw, feats, used


def transform(label, value):
    """A raw value (dollars, share, ratio) -> what the model uses."""
    use_log = VARIABLES[label][1]
    return float(np.log(value)) if use_log and value > 0 else (np.nan if use_log else float(value))


# ---------------------------------------------------------------- models (same settings as the notebook)
class Blend:
    def __init__(self, tr, feats):
        self.feats = [c for c in feats if tr[c].notna().any()]
        self.med = tr[self.feats].median()
        filled = tr[self.feats].fillna(self.med)
        self.mu, self.sd = filled.mean(), filled.std().replace(0, np.nan).fillna(1)
        self.years = sorted(tr["fall"].unique())
        Xtr = self._ridge_X(tr)
        folds = list(GroupKFold(n_splits=5).split(Xtr, tr["y"], groups=tr["UNITID"]))
        self.ridge = RidgeCV(alphas=np.logspace(-1, 4, 30), cv=folds).fit(Xtr, tr["y"])
        self.gbm_cols = self.feats + ["fall"]
        self.gbm = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=20,
                                                 l2_regularization=1.0, random_state=0).fit(tr[self.gbm_cols], tr["y"])
        self.train_mean = tr["y"].mean()
        self.year_mean = tr.groupby("fall")["y"].mean()

    def _ridge_X(self, d):
        Z = (d[self.feats].fillna(self.med) - self.mu) / self.sd
        D = pd.DataFrame({f"yr{y}": (d["fall"] == y).astype(float).values for y in self.years[1:]}, index=d.index)
        return pd.concat([Z, D], axis=1)

    def predict_all(self, te):
        r = np.clip(self.ridge.predict(self._ridge_X(te)), 1, 100)
        b = np.clip(self.gbm.predict(te[self.gbm_cols]), 1, 100)
        g = np.full(len(te), self.train_mean)
        yr = te["fall"].map(self.year_mean).fillna(self.train_mean).values
        return {"Training mean": g, "Training year mean": yr, "Ridge": r, "Boosting": b, "Blend": (r + b) / 2}

    def predict(self, te):
        return self.predict_all(te)["Blend"]


def headline_split(data):
    """The notebook's headline split: BC + a random 20% of schools held out (seed 42)."""
    all_schools = np.array(sorted(data["UNITID"].unique()))
    other = all_schools[all_schools != BC_ID]
    n_other = round(len(all_schools) * TEST_SIZE) - 1
    rng = np.random.default_rng(HEADLINE_SEED)
    test_ids = np.concatenate(([BC_ID], rng.choice(other, size=n_other, replace=False)))
    te = data[data["UNITID"].isin(test_ids)].copy()
    tr = data[~data["UNITID"].isin(test_ids)].copy()
    return tr, te


def evaluate(data, feats):
    tr, te = headline_split(data)
    model = Blend(tr, feats)
    preds = model.predict_all(te)
    scores = pd.DataFrame({m: {"R2": r2_score(te["y"], p), "MAE": mean_absolute_error(te["y"], p)}
                           for m, p in preds.items()}).T.reindex(MODELS)
    out = te[["UNITID", "school", "fall", "y"]].rename(columns={"y": "actual"}).copy()
    out["predicted"] = preds["Blend"]
    by_school = out.groupby(["UNITID", "school"], as_index=False)[["actual", "predicted"]].mean()
    school_r2 = r2_score(by_school["actual"], by_school["predicted"])
    pi = permutation_importance(model.gbm, te[model.gbm_cols], te["y"], n_repeats=20, random_state=0,
                                scoring="neg_mean_absolute_error")
    importance = pd.Series(pi.importances_mean, index=model.gbm_cols).drop("fall").sort_values(ascending=False)
    return dict(model=model, train=tr, test=te, scores=scores, out=out, by_school=by_school,
                school_r2=school_r2, importance=importance)
