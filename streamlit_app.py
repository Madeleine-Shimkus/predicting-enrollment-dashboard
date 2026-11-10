"""Boston College application forecast dashboard.

Run it locally with:  streamlit run streamlit_app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import model as M

st.set_page_config(page_title="BC Application Forecast", page_icon=":mortar_board:", layout="wide")

BC_COLOR, PEER_COLOR, PRED_COLOR = "#8a100b", "#9aa0a6", "#2b6cb0"


@st.cache_data
def get_data():
    d = M.load_data()
    return d, M.backtest(d)


@st.cache_resource
def get_model(d):
    return M.fit(d)


d, bt = get_data()
model = get_model(d)

bc = d[d["unitid"] == M.BC_ID]
last_year = int(bc.dropna(subset=["number_applied"])["fall"].max())  # last fall with real counts
next_year = last_year + 1
next_row = bc[bc["fall"] == next_year].copy()
last_apps = float(bc.loc[bc["fall"] == last_year, "number_applied"].iloc[0])
baseline = float(M.predict_apps(model, next_row, [last_apps])[0])

bc_bt = bt[bt["unitid"] == M.BC_ID]
bc_error = float((abs(bc_bt["actual"] - bc_bt["predicted"]) / bc_bt["actual"]).mean())
all_error = float((abs(bt["actual"] - bt["predicted"]) / bt["actual"]).mean())
naive_error = float((abs(bt["actual"] - bt["no_change_guess"]) / bt["actual"]).mean())

st.title("What drives applications to Boston College?")
st.caption(
    "BC and 14 peer schools, fall 2005-2024. Data: College Scorecard (prices, aid, SAT, endowment) "
    "and IPEDS (application counts, which run through fall 2023)."
)

tab_forecast, tab_drivers, tab_whatif, tab_peers = st.tabs(
    ["Forecast", "What matters most", "What if?", "BC vs. peers"]
)

# ---------------------------------------------------------------- Forecast
with tab_forecast:
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Predicted BC applications, fall {next_year}", f"{baseline:,.0f}",
              f"{baseline / last_apps - 1:+.1%} vs. fall {last_year}")
    c2.metric(f"Actual BC applications, fall {last_year}", f"{last_apps:,.0f}")
    c3.metric("Typical miss for BC (backtest)", f"±{bc_error:.0%}",
              help="Average percent error when each year was predicted using only earlier years.")

    fig = go.Figure()
    fig.add_scatter(x=bc["fall"], y=bc["number_applied"], name="Actual", mode="lines+markers",
                    line=dict(color=BC_COLOR, width=3))
    fig.add_scatter(x=bc_bt["fall"], y=bc_bt["predicted"], name="Model prediction",
                    mode="lines+markers", line=dict(color=PRED_COLOR, width=2, dash="dash"))
    fig.add_scatter(x=[last_year, next_year], y=[last_apps, baseline], name=f"Forecast {next_year}",
                    mode="lines+markers", line=dict(color=PRED_COLOR, width=2, dash="dot"),
                    marker=dict(size=12, symbol="star"))
    for year, label in [(2013, "Essay added"), (2020, "COVID"), (2021, "Test-optional")]:
        fig.add_vline(x=year, line=dict(color="#ccc", dash="dot"))
        fig.add_annotation(x=year, y=1.04, yref="paper", text=label, showarrow=False, font=dict(size=11))
    fig.update_layout(title="BC first-year applications: actual vs. predicted", hovermode="x unified",
                      yaxis_title="Applications", yaxis_tickformat=",", xaxis_dtick=2,
                      legend=dict(orientation="h", y=-0.15), margin=dict(t=70))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Each predicted point was made using only data from earlier years (expanding-window backtest). "
        f"Across all 15 schools the model's average miss is {all_error:.1%}, vs. {naive_error:.1%} "
        f"for just guessing \"same as last year\"."
    )

# ---------------------------------------------------------------- Drivers
with tab_drivers:
    imp = M.factor_importance(model, d)
    fig = px.bar(imp.sort_values("importance"), x="importance", y="label", color="direction",
                 orientation="h", color_discrete_map={"Raises applications": "#2f855a",
                                                      "Lowers applications": BC_COLOR})
    fig.update_layout(title="Which factors move application growth the most",
                      xaxis_title="Average impact on yearly growth (mean |SHAP|, log points)",
                      yaxis_title=None, legend_title=None, legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        "- **Model:** Ridge regression (scikit-learn) on all 15 schools, predicting each year's growth in applications.\n"
        "- **Bar length:** how much that factor typically pushes the prediction up or down (mean |SHAP|).\n"
        "- **Color:** whether more of that factor tends to raise or lower applications.\n"
        "- One-time events (COVID, test-optional, BC's essay) are included so they don't distort the other factors."
    )

# ---------------------------------------------------------------- What-if
with tab_whatif:
    st.subheader(f"What if BC changed things for fall {next_year}?")
    st.caption("Sliders start at what actually happened. Move them to see how the forecast responds.")

    def pct(feature):  # log change -> percent, for slider defaults
        return float(np.expm1(next_row[feature].iloc[0]) * 100)

    left, right = st.columns([1, 1])
    with left:
        net = st.slider("Net price after aid, % change", -15.0, 15.0, round(pct("net_price_chg"), 1), 0.5)
        low = st.slider("Net price for $0-30k families, % change", -50.0, 30.0, round(pct("low_inc_price_chg"), 1), 1.0,
                        help="Lower = more generous aid for low-income families.")
        cost = st.slider("Sticker price, % change", -10.0, 15.0, round(pct("cost_chg"), 1), 0.5)
        sat = st.slider("Average SAT, point change", -30, 30, int(next_row["sat_chg"].iloc[0]), 1)
        endow = st.slider("Endowment, % change", -20.0, 25.0, round(pct("endow_chg"), 1), 1.0)

    scenario = next_row.copy()
    scenario["net_price_chg"] = np.log1p(net / 100)
    scenario["low_inc_price_chg"] = np.log1p(low / 100)
    scenario["cost_chg"] = np.log1p(cost / 100)
    scenario["sat_chg"] = sat
    scenario["endow_chg"] = np.log1p(endow / 100)
    what_if = float(M.predict_apps(model, scenario, [last_apps])[0])
    if abs(what_if - baseline) < 25:  # slider rounding only, no real change
        what_if = baseline

    with right:
        st.metric(f"Predicted applications, fall {next_year}", f"{what_if:,.0f}",
                  f"{what_if - baseline:+,.0f} vs. baseline forecast")
        gauge = go.Figure(go.Bar(x=["Baseline", "Your scenario"], y=[baseline, what_if],
                                 marker_color=[PEER_COLOR, BC_COLOR], text=[f"{baseline:,.0f}", f"{what_if:,.0f}"],
                                 textposition="outside"))
        gauge.update_layout(yaxis_tickformat=",", yaxis_range=[0, max(baseline, what_if) * 1.2],
                            height=320, margin=dict(t=20))
        st.plotly_chart(gauge, use_container_width=True)
    st.caption("College rankings aren't in the College Scorecard data, so there is no ranking slider yet.")

# ---------------------------------------------------------------- Peers
with tab_peers:
    peers = d.dropna(subset=["number_applied"]).copy()
    peers["school"] = np.where(peers["unitid"] == M.BC_ID, "Boston College", "Peer")
    peers["name"] = peers["instnm"].str.replace("University of ", "").str.replace(" University", "")
    fig = px.scatter_geo(peers, lat="latitude", lon="longitude", size="number_applied", color="school",
                         hover_name="instnm", hover_data={"number_applied": ":,", "latitude": False,
                                                          "longitude": False, "school": False},
                         animation_frame="fall", scope="usa", size_max=40,
                         color_discrete_map={"Boston College": BC_COLOR, "Peer": PEER_COLOR})
    fig.update_layout(title="Applications by school, 2005-2023 (press play)", legend_title=None, height=520)
    st.plotly_chart(fig, use_container_width=True)

    med = peers[peers["unitid"] != M.BC_ID].groupby("fall")["number_applied"].median().reset_index()
    line = go.Figure()
    line.add_scatter(x=bc["fall"], y=bc["number_applied"], name="Boston College", line=dict(color=BC_COLOR, width=3))
    line.add_scatter(x=med["fall"], y=med["number_applied"], name="Peer median", line=dict(color=PEER_COLOR, width=3))
    line.update_layout(title="BC vs. peer median applications", yaxis_tickformat=",", hovermode="x unified")
    st.plotly_chart(line, use_container_width=True)
