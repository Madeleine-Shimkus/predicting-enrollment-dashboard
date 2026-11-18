"""Predictor definitions shared by the dashboard and scripts/build_yield_data.py.

Matches step 5 of Capstone_Yield_Model.ipynb. Also builds next fall's inputs (this fall's
Scorecard values) so the dashboard can forecast a year the model hasn't seen.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).parent / "data" / "raw"
SCORECARD_FILE = RAW / "scorecard_93_schools.csv"

# Scorecard code -> (label used everywhere, take log?)
CANDIDATES = {
    "SAT_AVG":           ("SAT average", False),
    "APPS":              ("Applications", True),
    "NPT4_PRIV":         ("Net price (all)", True),
    "NPT41_PRIV":        ("Net price $0–30k", True),
    "NPT45_PRIV":        ("Net price $110k+", True),
    "COST_FILLED":       ("Sticker price", True),
    "ENDOW_PER_STUDENT": ("Endowment per student", True),
    "PCTPELL":           ("Pell share", False),
    "UGDS_NRA":          ("International share", False),
    "NONWHITE_SHARE":    ("Non-white share", False),
    "C150_4":            ("6-yr grad rate", False),
    "RET_FT4":           ("Retention rate", False),
    "EARN_FILLED":       ("Median earnings (10 yr)", True),
    "INEXPFTE":          ("Instruction $/student", True),
    "PCTFLOAN":          ("Federal loan share", False),
    "UGDS":              ("Undergrad enrollment", True),
}


def add_derived(df):
    """Adds the combined columns the candidates use. df has Scorecard columns + number_applied."""
    num = lambda c: pd.to_numeric(df[c], errors="coerce") if c in df.columns else pd.Series(np.nan, index=df.index)
    df = df.copy()
    df["COST_FILLED"] = num("COSTT4_A").fillna(
        num("TUITIONFEE_IN") + num("ROOMBOARD_ON") + num("BOOKSUPPLY") + num("OTHEREXPENSE_ON"))
    df["ENDOW_PER_STUDENT"] = num("ENDOWEND") / num("UGDS")
    df["NONWHITE_SHARE"] = 1 - num("UGDS_WHITE").fillna(num("UGDS_WHITENH"))
    df["EARN_FILLED"] = num("MD_EARN_WNE_P10").groupby(df["UNITID"]).ffill()
    df["APPS"] = num("number_applied")
    return df


def transformed(df):
    """Each candidate as the model sees it (logged where needed), same-year values."""
    out = pd.DataFrame(index=df.index)
    for code, (label, use_log) in CANDIDATES.items():
        s = pd.to_numeric(df[code], errors="coerce") if code in df.columns else pd.Series(np.nan, index=df.index)
        out[label] = np.log(s.where(s > 0)) if use_log else s
    return out


def admissions():
    """Every admissions file in data/raw (Colab's admissions_yield.csv and/or the 15-school one)."""
    files = sorted(RAW.glob("admissions*.csv"))
    if not files:
        return pd.DataFrame(columns=["unitid", "year", "number_applied", "number_admitted", "number_enrolled"])
    adm = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return adm.drop_duplicates(["unitid", "year"], keep="first")


def inputs_for_fall(fall):
    """Model inputs for predicting `fall`: last fall's values (test policy from this fall)."""
    sc = pd.read_csv(SCORECARD_FILE, low_memory=False)
    sc["fall"] = sc["year"].str[:4].astype(int)
    adm = admissions()
    sc = sc.merge(adm, left_on=["UNITID", "fall"], right_on=["unitid", "year"], how="left", suffixes=("", "_adm"))
    sc = add_derived(sc.sort_values(["UNITID", "fall"]))
    prev = sc[sc["fall"] == fall - 1].set_index("UNITID")
    X = transformed(prev)
    now = sc[sc["fall"] == fall].set_index("UNITID")
    X["Test required"] = (pd.to_numeric(now["ADMCON7"], errors="coerce") == 1).astype(float).reindex(X.index)
    return X
