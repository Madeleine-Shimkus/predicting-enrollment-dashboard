"""Boston College application forecast dashboard.

Run it locally with:  streamlit run streamlit_app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import findings as F
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

(tab_forecast, tab_growth, tab_drivers, tab_price, tab_test, tab_retention, tab_whatif,
 tab_peers) = st.tabs(["Forecast", "Growth vs. peers", "What matters most", "Net price",
                       "Test-optional", "Retention vs. grad", "What if?", "Map"])

g = F.growth_table(d)

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


# ---------------------------------------------------------------- Growth vs. peers
with tab_growth:
    st.subheader("Most of the swings hit every school at once")
    share_raw = F.year_share(g, "growth_pct")
    share_no_covid = F.year_share(g[~g["fall"].isin([2020, 2021])], "growth_pct")
    c1, c2 = st.columns(2)
    c1.metric("Growth explained just by the year", f"{share_raw:.0%}",
              help="Share of the variation in yearly growth explained by each year's all-school average.")
    c2.metric("Same, leaving out 2020-21", f"{share_no_covid:.0%}",
              help="COVID and test-optional drive most of the shared swings.")

    fig = go.Figure()
    for uid, part in g.groupby("unitid"):
        if uid == M.BC_ID:
            continue
        fig.add_scatter(x=part["fall"], y=part["growth_pct"], mode="lines", name=part["school"].iloc[0],
                        line=dict(color=PEER_COLOR, width=1), opacity=0.45, showlegend=False)
    avg = g.groupby("fall")["growth_pct"].mean()
    fig.add_scatter(x=avg.index, y=avg.values, name="All-school average", line=dict(color="black", width=2.5, dash="dash"))
    bc_g = g[g["unitid"] == M.BC_ID]
    fig.add_scatter(x=bc_g["fall"], y=bc_g["growth_pct"], name="Boston College", line=dict(color=BC_COLOR, width=3))
    fig.add_hline(y=0, line=dict(color="#888", width=1))
    fig.update_layout(title="Year-over-year application growth (gray = each peer)", yaxis_title="Growth (%)",
                      hovermode="closest", legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)

    events = {2013: "Essay added", 2020: "Early Decision + COVID", 2021: "Test-optional"}
    bars = go.Figure(go.Bar(
        x=bc_g["fall"], y=bc_g["rel_growth_pct"],
        marker_color=np.where(bc_g["rel_growth_pct"] >= 0, "#2f855a", BC_COLOR),
        hovertemplate="Fall %{x}<br>BC vs. peers: %{y:+.1f} pts<extra></extra>"))
    for year, label in events.items():
        val = float(bc_g.loc[bc_g["fall"] == year, "rel_growth_pct"].iloc[0])
        bars.add_annotation(x=year, y=val, text=label, showarrow=True, arrowhead=0, ay=-30 if val > 0 else 30)
    bars.update_layout(title="BC's growth minus the peer average (peer-relative growth)",
                       yaxis_title="Percentage points", xaxis_dtick=2)
    st.plotly_chart(bars, use_container_width=True)
    st.markdown(
        "- Big years for all schools (like the 2021 test-optional jump) show up as the dashed line.\n"
        "- Subtracting the peer average leaves what's special about BC. Its biggest gaps line up with its own policy changes.\n"
        "- Peer-relative growth is the target in our best-subset search."
    )

# ---------------------------------------------------------------- Drivers
@st.cache_data
def get_fe_results(d):
    return M.fixed_effects_regression(d)


with tab_drivers:
    fe = get_fe_results(d).iloc[::-1]  # plot top-to-bottom in table order
    fe["significant"] = np.where(fe["p_value"] < 0.05, "Significant (95% CI excludes 0)", "Not significant")
    fig = go.Figure()
    for label, color in [("Significant (95% CI excludes 0)", PRED_COLOR), ("Not significant", PEER_COLOR)]:
        part = fe[fe["significant"] == label]
        fig.add_scatter(
            x=part["coef"], y=part["label"], mode="markers", name=label,
            marker=dict(size=11, color=color),
            error_x=dict(type="data", symmetric=False, array=part["high"] - part["coef"],
                         arrayminus=part["coef"] - part["low"], color=color, thickness=2, width=6),
            customdata=np.c_[part["low"], part["high"], part["p_value"]],
            hovertemplate="%{y}<br>Effect: %{x:+.3f}<br>95% CI: %{customdata[0]:+.3f} to %{customdata[1]:+.3f}"
                          "<br>p = %{customdata[2]:.3f}<extra></extra>",
        )
    fig.add_vline(x=0, line=dict(color="#888", dash="dash"))
    fig.update_layout(
        title="Which prior-year factors predict application growth?",
        xaxis_title="Change in application growth (log points) per 1 SD increase in predictor",
        yaxis_title=None, height=520, legend=dict(orientation="h", y=-0.2), margin=dict(l=10),
        yaxis=dict(categoryorder="array", categoryarray=list(fe["label"])),
    )
    st.plotly_chart(fig, use_container_width=True)
    n_obs = int(fe["n"].iloc[0])
    st.markdown(
        f"- **Model:** OLS regression with school and year fixed effects ({n_obs} school-years, statsmodels). "
        "Predictors are standardized, so bars compare directly.\n"
        "- **Error bars:** 95% confidence intervals, clustered by school.\n"
        "- **Significant:** prior-year admit rate (+), change in admit rate (−) and 2-year momentum (−). "
        "Surges tend to cool off the next year.\n"
        "- Price, aid, debt and student-faculty ratio don't clearly move growth once school and year are accounted for."
    )


    with st.expander("Best-subset search: best combination of 8 predictors (from our notebook)"):
        bs = F.BEST_SUBSET_8.iloc[::-1]
        fig = go.Figure(go.Bar(x=bs["effect"], y=bs["variable"], orientation="h",
                               marker_color=np.where(bs["effect"] > 0, "#3E9C9C", "#E07B54"),
                               text=[f"{v:+.1f} {s}" for v, s in zip(bs["effect"], bs["stars"])],
                               textposition="outside"))
        fig.update_layout(xaxis_title="Change in peer-relative growth (% pts per 1 SD)", xaxis_range=[-5, 5],
                          height=380, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Tried every combination of up to 10 candidate variables and kept the best fit. "
                   "R² = 0.13 with 8 predictors. *** p<0.01, ** p<0.05.")

    with st.expander("Forecast model's view (Ridge regression + SHAP)"):
        imp = M.factor_importance(model, d)
        fig = px.bar(imp.sort_values("importance"), x="importance", y="label", color="direction",
                     orientation="h", color_discrete_map={"Raises applications": "#2f855a",
                                                          "Lowers applications": BC_COLOR})
        fig.update_layout(xaxis_title="Average impact on yearly growth (mean |SHAP|, log points)",
                          yaxis_title=None, legend_title=None, legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("This is the model behind the Forecast and What if? tabs. "
                   "Bar length shows how much each factor typically moves the prediction.")


# ---------------------------------------------------------------- Net price
@st.cache_data
def get_price_results(g):
    return F.net_price_effects(g), F.net_price_binned(g)


with tab_price:
    st.subheader("Does a higher net price slow application growth?")
    effects, (points, bins) = get_price_results(g)
    left, right = st.columns(2)
    with left:
        e = effects.iloc[::-1]
        colors = np.where(e["p_value"] < 0.05, BC_COLOR, PEER_COLOR)
        fig = go.Figure(go.Scatter(
            x=e["effect"], y=e["measure"], mode="markers", marker=dict(size=11, color=colors),
            error_x=dict(type="data", symmetric=False, array=e["high"] - e["effect"],
                         arrayminus=e["effect"] - e["low"], color="#999"),
            hovertemplate="%{y}<br>%{x:+.2f} pts<extra></extra>"))
        fig.add_vline(x=0, line=dict(color="#555"))
        fig.update_layout(title="Effect of a 10% higher price last year (red = significant)",
                          xaxis_title="Change in next-year growth (% pts)", height=420)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = go.Figure()
        fig.add_scatter(x=points["lp_prev_r"] * 100, y=points["growth_pct_r"], mode="markers",
                        marker=dict(color="#d0d0d0", size=6), name="School-years")
        fig.add_scatter(x=bins["lp_prev_r"] * 100, y=bins["growth_pct_r"], mode="lines+markers",
                        line=dict(color=BC_COLOR, width=2), marker=dict(size=8), name="Average of each tenth")
        fig.update_layout(title="Overall net price: no clear pattern",
                          xaxis_title="Net price last year vs. school's usual (%)",
                          yaxis_title="Growth vs. usual (% pts)", height=420, legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        "- Each dot is one regression with school and year fixed effects, clustered by school.\n"
        "- Net price for most income groups has no clear effect. The intervals all cross zero.\n"
        "- Only the $110k+ price is significant, and it's positive. Pricier schools may just be in higher demand, so this isn't causal."
    )

# ---------------------------------------------------------------- Test-optional
with tab_test:
    st.subheader("How big was the test-optional effect?")
    ev = F.test_optional_event(g)
    fig = go.Figure()
    bc_higher = ev["Boston College"] >= ev["Other test-optional schools"]
    for col, color in [("Other test-optional schools", "#4BA3A3"), ("Boston College", "#B5323C")]:
        above = bc_higher if col == "Boston College" else ~bc_higher
        fig.add_scatter(x=ev.index, y=ev[col], name=col, mode="lines+markers+text", line=dict(color=color, width=3),
                        text=[f"{v:+.0f}%" for v in ev[col]], textfont=dict(color=color),
                        textposition=["top center" if a else "bottom center" for a in above])
    fig.add_vline(x=0, line=dict(color="gray", dash="dash"))
    fig.add_hline(y=0, line=dict(color="black", width=1))
    fig.update_layout(xaxis=dict(tickvals=list(ev.index),
                                 ticktext=[f"{x:+d} yr" if x else "Went test-optional (fall 2021)" for x in ev.index]),
                      yaxis_title="Application growth (%)", legend=dict(orientation="h", y=-0.2),
                      title="Boston College vs. other schools that went test-optional for fall 2021")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        "- Comparison group: Brandeis, Brown, Cornell, Notre Dame, Rice and WashU, which dropped testing the same cycle as BC.\n"
        f"- They grew **{ev.loc[0, 'Other test-optional schools']:+.0f}%** the year they went test-optional. "
        f"BC grew **{ev.loc[0, 'Boston College']:+.0f}%**, partly bouncing back from **{ev.loc[-1, 'Boston College']:+.0f}%** "
        "the year before (Early Decision added).\n"
        "- Growth was back to normal within 1-2 years, so fall 2021 is flagged in the models."
    )

# ---------------------------------------------------------------- Retention vs. graduation
with tab_retention:
    st.subheader("Are retention and graduation rates saying the same thing?")
    rv, school_avg, within, r_between, r_within = F.retention_vs_graduation(d)
    left, right = st.columns(2)
    with left:
        fig = px.scatter(school_avg, x="Retention", y="Graduation", text="school", trendline="ols",
                         color=np.where(school_avg["school"] == "Boston College", "Boston College", "Peer"),
                         color_discrete_map={"Boston College": BC_COLOR, "Peer": "#1F4E79"})
        fig.update_traces(textposition="top center", textfont_size=10)
        fig.update_layout(title=f"Across schools: nearly the same ranking (r = {r_between:.2f})",
                          xaxis_title="Retention rate (%)", yaxis_title="6-yr graduation rate (%)",
                          showlegend=False, height=430)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.scatter(within, x="Retention", y="Graduation", hover_name="school", opacity=0.6,
                         color_discrete_sequence=["#1F4E79"])
        fig.update_layout(title=f"Within a school over time: barely related (r = {r_within:.2f})",
                          xaxis_title="Retention vs. school's usual (pts)",
                          yaxis_title="Graduation vs. school's usual (pts)", height=430)
        st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        "- Across schools the two rates rank schools the same way, so they're redundant for comparing schools.\n"
        "- Within a school they barely change year to year, and neither predicts application growth.\n"
        "- We keep graduation rate as a school-quality descriptor, not a growth predictor."
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
