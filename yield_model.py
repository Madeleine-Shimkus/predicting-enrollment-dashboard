"""Loads the yield model results and refits the within-school model for the dashboard.

The three files in data/ come straight from Capstone_Yield_Model.ipynb (it saves them to
Google Drive), or from scripts/build_yield_data.py. To update the dashboard, replace them.

Yield = first-year students who enrolled / students admitted, in percent.
Predictors are last year's values. Effects are yield points per 1 standard deviation.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

DATA = Path(__file__).parent / "data"
MODEL_FILE = DATA / "yield_model_data.csv"
WITHIN_FILE = DATA / "yield_within_effects.csv"
ACROSS_FILE = DATA / "yield_across_effects.csv"
FILES = [MODEL_FILE, WITHIN_FILE, ACROSS_FILE]

BC_ID = 164924
PEER_IDS = [123961, 131496, 139658, 152080, 165015, 168148, 179867,
            190415, 193900, 198419, 199847, 217156, 221999, 227757]
TRAITS = ["Religious", "Catholic", "City campus"]

# Variables the notebook logs before standardizing (dollar amounts and counts).
LOGGED = {"Applications", "Net price (all)", "Net price $0–30k", "Net price $110k+", "Sticker price",
          "Endowment per student", "Median earnings (10 yr)", "Instruction $/student", "Undergrad enrollment"}
# Variables stored as a 0-1 share.
SHARES = {"Pell share", "International share", "Non-white share", "6-yr grad rate", "Retention rate",
          "Federal loan share"}


def version():
    """Changes whenever any data file changes, so Streamlit's cache refreshes."""
    return tuple(f.stat().st_mtime for f in FILES)


def load():
    data = pd.read_csv(MODEL_FILE)
    data["group"] = np.where(data["UNITID"] == BC_ID, "Boston College",
                             np.where(data["UNITID"].isin(PEER_IDS), "Original peer", "Other selective private"))
    within = _effects(WITHIN_FILE)
    across = _effects(ACROSS_FILE)
    return data, within, across


def _effects(path):
    e = pd.read_csv(path, index_col=0)
    e.columns = ["effect", "p_value"]
    e.index.name = "variable"
    return e.reset_index()


def describe_sd(data, var):
    """One standard deviation in plain units, e.g. 'about 28%' or '6 pts'."""
    sd = data[var].std()
    if var in LOGGED:
        return f"about {np.exp(sd) - 1:.0%} higher"
    if var in SHARES:
        return f"{sd * 100:.1f} pts higher"
    if var == "Test required":
        return "test required (vs. not)"
    if var == "SAT average":
        return f"{sd:.0f} SAT points higher"
    return f"{sd:.2f} higher"


def _q(cols):
    return " + ".join(f"Q('{c}')" for c in cols)


def fit_within(data, features):
    """Same within-school model as the notebook: school + year fixed effects, standardized predictors."""
    d = data.copy()
    d[features] = d[features].fillna(d[features].median())
    d[features] = (d[features] - d[features].mean()) / d[features].std()
    fit = smf.ols(f"target ~ {_q(features)} + C(UNITID) + C(fall)", d).fit(
        cov_type="cluster", cov_kwds={"groups": d["UNITID"]})
    base = smf.ols("target ~ C(UNITID) + C(fall)", d).fit()
    d["predicted"] = fit.fittedvalues
    d["school_year_only"] = base.fittedvalues
    explained = 1 - fit.ssr / base.ssr   # share of year-to-year movement the variables explain
    return d, explained


def what_if_change(data, within, changes):
    """Change in yield (pts) from moving each variable by the amount given in plain units.

    changes: {variable: amount}. Logged variables take a % change (10 = +10%),
    shares take percentage points, SAT takes points, Test required takes -1/0/+1.
    """
    total = 0.0
    for var, amount in changes.items():
        sd = data[var].std()
        if var in LOGGED:
            dx = np.log1p(amount / 100)
        elif var in SHARES:
            dx = amount / 100
        else:
            dx = amount
        total += float(within.set_index("variable").loc[var, "effect"]) * dx / sd
    return total


PLAIN = {"Median earnings (10 yr)": "graduates' earnings", "Net price $0–30k": "net price for low-income families",
         "Net price $110k+": "net price for high-income families", "Net price (all)": "net price",
         "Instruction $/student": "instruction spending", "Endowment per student": "endowment per student",
         "6-yr grad rate": "graduation rate", "SAT average": "average SAT", "Test required": "requiring tests",
         "City campus": "a city campus", "Catholic": "being Catholic", "Religious": "a religious affiliation"}


def plain(var):
    """Plain-English name for headlines."""
    return PLAIN.get(var, var.lower())


# ---------------------------------------------------------------- forecasting
DOLLARS = {"Net price (all)", "Net price $0–30k", "Net price $110k+", "Sticker price", "Endowment per student",
           "Median earnings (10 yr)", "Instruction $/student"}


def to_plain(var, x):
    """Model scale -> what a person would type (dollars, %, counts)."""
    if var in LOGGED:
        return float(np.exp(x))
    if var in SHARES:
        return float(x) * 100
    return float(x)


def from_plain(var, v):
    if var in LOGGED:
        return float(np.log(v))
    if var in SHARES:
        return v / 100
    return float(v)


def _fe_fit(d, features, extra="C(UNITID)"):
    return smf.ols(f"target ~ {_q(features)} + {extra} + C(fall)", d).fit()


def _filled(d, features, med):
    d = d.copy()
    d[features] = d[features].fillna(med)
    return d


def forecaster(data, features):
    """Within-school model on raw (unstandardized) inputs, for predicting a new fall.

    A new fall's market-wide shift isn't known yet, so it reuses the latest year's effect,
    the same rule the notebook's cross-validation uses.
    """
    med = data[features].median()
    fit = _fe_fit(_filled(data, features, med), features)
    return fit, med


def predict(fit, med, rows, features, fall_effect_year):
    r = rows.copy()
    r[features] = r[features].fillna(med)
    r["fall"] = fall_effect_year
    return fit.predict(r)


def backtest(data, features, n_years=6):
    """Each of the last n falls predicted using only earlier falls. Returns predictions + typical miss."""
    out = []
    for ty in sorted(data["fall"].unique())[-n_years:]:
        tr = data[data["fall"] < ty]
        te = data[(data["fall"] == ty) & data["UNITID"].isin(tr["UNITID"])].copy()
        fit, med = forecaster(tr, features)
        te["forecast"] = predict(fit, med, te, features, ty - 1).values
        out.append(te[["UNITID", "school", "fall", "target", "forecast"]])
    bt = pd.concat(out, ignore_index=True)
    rmse = float(np.sqrt(((bt["target"] - bt["forecast"]) ** 2).mean()))
    return bt, rmse


def across_fit(data, numeric, traits):
    """Across-school model on raw inputs: year effects + region + traits, no school effects."""
    med = data[numeric].median()
    extra = " + ".join([f"Q('{t}')" for t in traits] + (["C(Region)"] if data["Region"].nunique() > 1 else []))
    fit = smf.ols(f"target ~ {_q(numeric)} + {extra} + C(fall)", _filled(data, numeric, med)).fit()
    return fit, med


LAG = "Last year yield"


def add_lag(data):
    """Adds last fall's yield (only when last fall exists). Used by the forecast, not the effect charts."""
    d = data.sort_values(["UNITID", "fall"]).copy()
    consecutive = d.groupby("UNITID")["fall"].shift(1) == d["fall"] - 1
    d[LAG] = d.groupby("UNITID")["target"].shift(1).where(consecutive)
    return d.sort_index()
