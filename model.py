"""Data loading and the forecasting model behind the dashboard.

The model predicts how much a school's first-year applications grow from one
fall to the next (growth in log applications), then turns that back into a
count. It is a Ridge regression trained on BC and its 14 peers.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_FILE = Path(__file__).parent / "data" / "bc_peers.csv"
BC_ID = 164924
FIRST_TEST_YEAR = 2014  # expanding-window backtest starts here

# Model inputs and the plain-English names shown in the dashboard.
FEATURES = {
    "net_price_chg": "Change in net price (after aid)",
    "low_inc_price_chg": "Change in net price for $0-30k families",
    "cost_chg": "Change in sticker price (cost of attendance)",
    "pell_chg": "Change in Pell share",
    "sat_chg": "Change in average SAT",
    "endow_chg": "Change in endowment",
    "prev_admit_rate": "Last year's admit rate",
    "prev_growth": "Last year's application growth",
    "covid": "COVID year (fall 2020)",
    "test_optional": "Test-optional surge (fall 2021)",
    "bc_essay": "BC adds essay (fall 2013)",
}


def load_data() -> pd.DataFrame:
    d = pd.read_csv(DATA_FILE).sort_values(["unitid", "fall"]).reset_index(drop=True)
    by_school = d.groupby("unitid")

    d["log_apps"] = np.log(d["number_applied"])
    d["growth"] = by_school["log_apps"].diff()
    d["prev_growth"] = d.groupby("unitid")["growth"].shift(1)

    # Year-over-year changes. Dollar amounts use log change (about % change).
    for name, col, use_log in [
        ("net_price", "npt4_priv", True),
        ("low_inc_price", "npt41_priv", True),
        ("cost", "costt4_a", True),
        ("endow", "endowend", True),
        ("pell", "pctpell", False),
        ("sat", "sat_avg", False),
    ]:
        values = np.log(d[col].where(d[col] > 0)) if use_log else d[col]
        d[f"{name}_chg"] = values.groupby(d["unitid"]).diff()

    d["prev_admit_rate"] = by_school["adm_rate"].shift(1)
    d["covid"] = (d["fall"] == 2020).astype(int)
    d["test_optional"] = (d["fall"] == 2021).astype(int)
    d["bc_essay"] = ((d["unitid"] == BC_ID) & (d["fall"] == 2013)).astype(int)
    return d


def _new_model():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30)))


def training_rows(d: pd.DataFrame) -> pd.DataFrame:
    return d.dropna(subset=list(FEATURES) + ["growth"])


def fit(d: pd.DataFrame):
    rows = training_rows(d)
    return _new_model().fit(rows[list(FEATURES)], rows["growth"])


def predict_apps(model, rows: pd.DataFrame, last_year_apps) -> np.ndarray:
    growth = model.predict(rows[list(FEATURES)])
    return np.asarray(last_year_apps) * np.exp(growth)


def backtest(d: pd.DataFrame) -> pd.DataFrame:
    """Expanding window: to predict fall Y, train only on years before Y."""
    rows = training_rows(d)
    out = []
    for year in range(FIRST_TEST_YEAR, int(rows["fall"].max()) + 1):
        train, test = rows[rows["fall"] < year], rows[rows["fall"] == year]
        if test.empty:
            continue
        m = _new_model().fit(train[list(FEATURES)], train["growth"])
        last = np.exp(test["log_apps"] - test["growth"])
        out.append(pd.DataFrame({
            "unitid": test["unitid"], "instnm": test["instnm"], "fall": year,
            "actual": test["number_applied"],
            "predicted": predict_apps(m, test, last),
            "no_change_guess": last,
        }))
    return pd.concat(out, ignore_index=True)


def factor_importance(model, d: pd.DataFrame) -> pd.DataFrame:
    """Mean |SHAP| for a linear model: |coef| x average distance from the mean.

    For a linear model this is exactly what the shap package would report,
    so we skip the extra dependency.
    """
    rows = training_rows(d)[list(FEATURES)]
    scaler, ridge = model.named_steps["standardscaler"], model.named_steps["ridgecv"]
    z = scaler.transform(rows)
    shap_values = z * ridge.coef_
    return pd.DataFrame({
        "feature": list(FEATURES),
        "label": list(FEATURES.values()),
        "importance": np.abs(shap_values).mean(axis=0),
        "direction": np.where(ridge.coef_ >= 0, "Raises applications", "Lowers applications"),
    }).sort_values("importance", ascending=False)


# ---------------------------------------------------------------------------
# Team regression: which prior-year factors predict application growth?
# OLS with school + year fixed effects, standardized predictors, standard
# errors clustered by school (t-distribution, 14 df), 95% CI.

FE_PREDICTORS = {
    "adm_rate": "Admit rate",
    "yield_rate": "Yield rate",
    "c150_4": "6-year graduation rate",
    "adm_rate_chg": "Change in admit rate",
    "momentum": "Application growth momentum (prior 2 years)",
    "pctpell": "Share of students on Pell grants",
    "test_required": "Standardized test required",
    "pctfloan": "Share of students with federal loans",
    "log_inst_spend": "Instructional spending per student, log",
    "log_net_price": "Net price, log",
    "stufacr": "Student-to-faculty ratio",
}


def fixed_effects_regression(d: pd.DataFrame) -> pd.DataFrame:
    import statsmodels.formula.api as smf

    d = d.sort_values(["unitid", "fall"]).copy()
    by_school = d.groupby("unitid")
    d["yield_rate"] = d["number_enrolled"] / d["number_admitted"]
    d["adm_rate_chg"] = by_school["adm_rate"].diff()
    d["momentum"] = d["log_apps"] - by_school["log_apps"].shift(2)
    d["test_required"] = (d["admcon7"] == 1).astype(float).where(d["admcon7"].notna())
    d["log_inst_spend"] = np.log(d["inexpfte"])
    d["log_net_price"] = np.log(d["npt4_priv"])

    lagged = []
    for col in FE_PREDICTORS:  # every predictor uses the prior year's value
        d[f"L_{col}"] = d.groupby("unitid")[col].shift(1)
        lagged.append(f"L_{col}")

    rows = d.dropna(subset=lagged + ["growth"]).copy()
    for col in lagged:
        rows[col] = (rows[col] - rows[col].mean()) / rows[col].std()

    formula = "growth ~ " + " + ".join(lagged) + " + C(unitid) + C(fall)"
    fit = smf.ols(formula, rows).fit(cov_type="cluster", cov_kwds={"groups": rows["unitid"]}, use_t=True)
    ci = fit.conf_int().loc[lagged]
    return pd.DataFrame({
        "label": [f"{name} (prior year)" if "prior 2" not in name else name for name in FE_PREDICTORS.values()],
        "coef": fit.params[lagged].values,
        "low": ci[0].values,
        "high": ci[1].values,
        "p_value": fit.pvalues[lagged].values,
        "n": int(fit.nobs),
    })
