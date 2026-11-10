"""Analyses from the team's Colab notebook, packaged for the dashboard.

Growth here is in percent (log change x 100), the same as the notebook.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BC_ID = 164924
SHORT_NAMES = {123961: "USC", 131496: "Georgetown", 139658: "Emory", 152080: "Notre Dame",
               164924: "Boston College", 165015: "Brandeis", 168148: "Tufts", 179867: "WashU",
               190415: "Cornell", 193900: "NYU", 198419: "Duke", 199847: "Wake Forest",
               217156: "Brown", 221999: "Vanderbilt", 227757: "Rice"}


def growth_table(d: pd.DataFrame) -> pd.DataFrame:
    g = d.dropna(subset=["number_applied"]).sort_values(["unitid", "fall"]).copy()
    g["school"] = g["unitid"].map(SHORT_NAMES)
    consecutive = g.groupby("unitid")["fall"].diff() == 1
    g["growth_pct"] = (np.log(g["number_applied"]).groupby(g["unitid"]).diff() * 100).where(consecutive)
    g = g.dropna(subset=["growth_pct"])
    total = g.groupby("fall")["growth_pct"].transform("sum")
    count = g.groupby("fall")["growth_pct"].transform("count")
    g["peer_avg_pct"] = (total - g["growth_pct"]) / (count - 1)  # average of the OTHER schools
    g["rel_growth_pct"] = g["growth_pct"] - g["peer_avg_pct"]
    return g


def year_share(g: pd.DataFrame, col: str) -> float:
    """Share of the variation in `col` explained by which year it was."""
    return g.groupby("fall")[col].transform("mean").var() / g[col].var()


def _fe_fit(formula, rows):
    return smf.ols(formula, rows).fit(cov_type="cluster", cov_kwds={"groups": rows["unitid"]})


PRICE_MEASURES = {
    "npt4_priv": "Net price (all families)",
    "npt41_priv": "Net price, $0-30k",
    "npt42_priv": "Net price, $30-48k",
    "npt43_priv": "Net price, $48-75k",
    "npt44_priv": "Net price, $75-110k",
    "npt45_priv": "Net price, $110k+",
    "costt4_a": "Sticker price (cost of attendance)",
}


def net_price_effects(g: pd.DataFrame) -> pd.DataFrame:
    """Effect of a 10% higher price last year on this year's growth (school + year FE)."""
    ten_pct = np.log(1.10)
    rows = []
    for col, label in PRICE_MEASURES.items():
        x = g.copy()
        x["lp_prev"] = np.log(x[col].where(x[col] > 0)).groupby(x["unitid"]).shift(1)
        x = x.dropna(subset=["growth_pct", "lp_prev"])
        fit = _fe_fit("growth_pct ~ lp_prev + C(unitid) + C(fall)", x)
        lo, hi = fit.conf_int().loc["lp_prev"] * ten_pct
        rows.append({"measure": label, "effect": fit.params["lp_prev"] * ten_pct, "low": lo, "high": hi,
                     "p_value": fit.pvalues["lp_prev"], "n": int(fit.nobs)})
    return pd.DataFrame(rows)


def net_price_binned(g: pd.DataFrame):
    """Net price vs. growth with each school's and each year's average removed."""
    x = g.copy()
    x["lp_prev"] = np.log(x["npt4_priv"]).groupby(x["unitid"]).shift(1)
    x = x.dropna(subset=["growth_pct", "lp_prev"])
    for c in ["growth_pct", "lp_prev"]:
        x[c + "_r"] = smf.ols(f"{c} ~ C(unitid) + C(fall)", x).fit().resid
    x["bin"] = pd.qcut(x["lp_prev_r"], 10, labels=False)
    bins = x.groupby("bin")[["lp_prev_r", "growth_pct_r"]].mean()
    return x, bins


# Schools that dropped their test requirement for the same cycle as BC (fall 2021).
# Georgetown never went test-optional and Wake Forest already was, so both are left out.
TEST_OPTIONAL_2021 = [165015, 217156, 190415, 227757, 152080, 179867]


def test_optional_event(g: pd.DataFrame, event_year: int = 2021) -> pd.DataFrame:
    x = g[g["fall"].between(event_year - 3, event_year + 2)].copy()
    x["years_since"] = x["fall"] - event_year
    bc = x[x["unitid"] == BC_ID].set_index("years_since")["growth_pct"]
    others = x[x["unitid"].isin(TEST_OPTIONAL_2021)].groupby("years_since")["growth_pct"].mean()
    return pd.DataFrame({"Boston College": bc, "Other test-optional schools": others})


def retention_vs_graduation(d: pd.DataFrame):
    x = d.dropna(subset=["ret_ft4", "c150_4"]).copy()
    x["Retention"] = x["ret_ft4"] * 100
    x["Graduation"] = x["c150_4"] * 100
    x["school"] = x["unitid"].map(SHORT_NAMES)
    school_avg = x.groupby("school")[["Retention", "Graduation"]].mean().reset_index()
    within = x[["Retention", "Graduation"]] - x.groupby("unitid")[["Retention", "Graduation"]].transform("mean")
    within["school"] = x["school"]
    r_between = school_avg[["Retention", "Graduation"]].corr().iloc[0, 1]
    r_within = within[["Retention", "Graduation"]].corr().iloc[0, 1]
    return x, school_avg, within, r_between, r_within


# Best-subset search results, copied from the team's notebook (slide 5).
# Peer-relative growth, % pts per 1 SD; R^2 = 0.13.
BEST_SUBSET_8 = pd.DataFrame({
    "variable": ["Admit rate", "Yield", "2-yr momentum", "Pell share", "Median debt",
                 "Admit-rate change", "Federal loan share", "Student-faculty ratio"],
    "effect": [3.5, 2.3, -1.6, 1.6, -1.5, -1.5, -1.3, -1.1],
    "stars": ["***", "***", "**", "***", "***", "**", "**", "**"],
})
