"""What drives applications to Boston College? A scrolling data story.

Run it locally with:  streamlit run streamlit_app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import findings as F
import model as M

st.set_page_config(page_title="What Drives Applications to BC?", page_icon=":eagle:", layout="wide")

# ---------------------------------------------------------------- look and feel
MAROON, GOLD, INK, MUTED, TEAL = "#8A100B", "#C59B2E", "#1A1A1A", "#B5B5B5", "#3E6A8A"
POS, NEG = "#3E6A8A", "#B23A2E"
PEER_PALETTE = ["#3E6A8A", "#D08C2E", "#5B8C5A", "#7A5C99", "#8C8C8C", "#C2668A", "#2E9AA0"]

st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Libre+Franklin:wght@400;600;800&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap" rel="stylesheet">
    <style>
      html, body, .stApp { background: #FFFFFF; }
      .stMarkdown p, .stMarkdown li { font-family: 'Source Serif 4', Georgia, serif; font-size: 1.06rem; line-height: 1.55; color: #222; }
      .block-container { max-width: 1040px; padding-top: 3rem; }
      [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none; }
      .kicker { font-family: 'Libre Franklin', sans-serif; font-weight: 600; font-size: .8rem; color: #8A100B;
                text-transform: uppercase; letter-spacing: .06em; }
      .headline { font-family: 'Libre Franklin', sans-serif; font-weight: 800; font-size: 3rem; line-height: 1.05;
                  letter-spacing: -0.025em; color: #111; margin: .4rem 0 .8rem; max-width: 860px; }
      .dek { font-family: 'Source Serif 4', Georgia, serif; font-size: 1.3rem; line-height: 1.45; color: #333; max-width: 720px; }
      .byline { font-family: 'Libre Franklin', sans-serif; font-size: .85rem; color: #666; margin: 1.1rem 0 1.6rem;
                padding-bottom: 1.1rem; border-bottom: 1px solid #DDD; }
      .byline b { color: #111; font-weight: 600; }
      .section { margin-top: 3.6rem; border-top: 3px solid #111; padding-top: .7rem; }
      .section .num { font-family: 'Libre Franklin', sans-serif; font-weight: 800; color: #8A100B; font-size: .95rem; }
      .section h2 { font-family: 'Libre Franklin', sans-serif !important; font-weight: 800; font-size: 1.75rem;
                    letter-spacing: -0.015em; line-height: 1.15; color: #111; margin: .15rem 0 .5rem; padding: 0; }
      .section .lede { font-family: 'Source Serif 4', Georgia, serif; font-size: 1.1rem; line-height: 1.55; color: #333;
                       max-width: 700px; }
      .note { font-family: 'Source Serif 4', Georgia, serif; font-size: 1.06rem; line-height: 1.55; color: #222;
              max-width: 700px; margin: .3rem 0 1rem; }
      .num-big { font-family: 'Libre Franklin', sans-serif; font-weight: 800; font-size: 2.3rem; color: #111;
                 line-height: 1; letter-spacing: -0.02em; }
      .num-label { font-family: 'Libre Franklin', sans-serif; font-size: .85rem; color: #666; margin-top: .35rem;
                   max-width: 230px; }
      .stat-wrap { padding: .6rem 0 .4rem; }
      .source { font-family: 'Libre Franklin', sans-serif; font-size: .78rem; color: #888; }
    </style>
    """,
    unsafe_allow_html=True,
)


def style(fig, height=430, legend=True):
    fig.update_layout(
        template="simple_white", height=height, font=dict(family="Libre Franklin, Arial, sans-serif", size=13, color="#333"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=30, b=10),
        hoverlabel=dict(bgcolor="white", font_size=13), showlegend=legend,
        legend=dict(orientation="h", y=-0.18, x=0, title=None),
    )
    fig.update_xaxes(showgrid=False, linecolor="#999", ticks="outside", tickcolor="#999")
    fig.update_yaxes(gridcolor="#E6E6E6", griddash="dot", linecolor="rgba(0,0,0,0)", zeroline=False)
    return fig


def chapter(num, anchor, title, lede):
    st.markdown(
        f'<div class="section" id="{anchor}"><div class="num">{num}</div><h2>{title}</h2>'
        f'<div class="lede">{lede}</div></div>',
        unsafe_allow_html=True,
    )


def takeaway(text):
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def stat(col, value, label):
    col.markdown(f'<div class="stat-wrap"><div class="num-big">{value}</div><div class="num-label">{label}</div></div>',
                 unsafe_allow_html=True)


# ---------------------------------------------------------------- data
@st.cache_data
def get_data(data_version):  # data_version changes when the CSV changes, so the cache refreshes
    d = M.load_data()
    return d, M.backtest(d), F.growth_table(d)


@st.cache_resource
def get_model(data_version):
    return M.fit(M.load_data())


@st.cache_data
def get_regressions(data_version):
    d, _, g = get_data(data_version)
    return M.fixed_effects_regression(d), F.net_price_effects(g)


version = Path(M.DATA_FILE).stat().st_mtime
d, bt, g = get_data(version)
model = get_model(version)
fe, price_fx = get_regressions(version)

bc = d[d["unitid"] == M.BC_ID]
bc_g = g[g["unitid"] == M.BC_ID]
last_year = int(bc.dropna(subset=["number_applied"])["fall"].max())
next_year = last_year + 1
next_row = bc[bc["fall"] == next_year].copy()
last_apps = float(bc.loc[bc["fall"] == last_year, "number_applied"].iloc[0])
first_apps = float(bc.loc[bc["fall"] == 2005, "number_applied"].iloc[0])
baseline = float(M.predict_apps(model, next_row, [last_apps])[0])
share_all = F.year_share(g, "growth_pct")
share_no_covid = F.year_share(g[~g["fall"].isin([2020, 2021])], "growth_pct")
EVENTS = {2013: "Essay added", 2020: "Early Decision + COVID", 2021: "Test-optional"}

# ---------------------------------------------------------------- opening
st.markdown(
    f"""
    <div class="kicker">Boston College admissions, 2005-2023</div>
    <div class="headline">What moves applications to Boston College?</div>
    <div class="dek">BC went from {first_apps:,.0f} applicants to {last_apps:,.0f} in under twenty years. The climb was
    uneven, and the biggest drops lined up with decisions BC made itself. We compared BC with 14 peer schools to see
    which factors matter.</div>
    <div class="byline">By <b>Ellie El-Fishawy, Francesco Noble, Yennie Park, Madeleine Shimkus and Nolan Senecal</b>
    &nbsp;·&nbsp; Data science capstone &nbsp;·&nbsp; Data: College Scorecard, IPEDS</div>
    """,
    unsafe_allow_html=True,
)
c1, c2, c3, c4 = st.columns(4)
stat(c1, f"{last_apps:,.0f}", f"BC applications, fall {last_year}")
stat(c2, f"{last_apps / first_apps - 1:+.0%}", "Growth since 2005")
stat(c3, f"{bc_g['growth_pct'].abs().max():.0f}%", "Biggest one-year swing")
stat(c4, f"{baseline:,.0f}", f"Our forecast for fall {next_year}")

# ---------------------------------------------------------------- 1. a bumpy ride
chapter(1, "ride", "The climb was not a straight line",
        "Pick a few peers to compare. The <b>indexed</b> view sets every school to 100 in 2005, so you can see "
        "who grew fastest regardless of size.")
names = sorted(g["school"].unique().tolist())
left, right = st.columns([3, 1])
with right:
    picks = st.multiselect("Compare BC with", [n for n in names if n != "Boston College"],
                           default=["Notre Dame", "Georgetown", "Tufts"])
    indexed = st.toggle("Indexed (2005 = 100)", value=False)
    show_events = st.toggle("Show BC policy changes", value=True)
with left:
    apps = d.dropna(subset=["number_applied"]).copy()
    apps["school"] = apps["unitid"].map(F.SHORT_NAMES)
    apps = apps[apps["school"].isin(picks + ["Boston College"])]
    if indexed:
        base = apps[apps["fall"] == 2005].set_index("school")["number_applied"]
        apps["value"] = apps["number_applied"] / apps["school"].map(base) * 100
    else:
        apps["value"] = apps["number_applied"]
    fig = go.Figure()
    peer_colors = PEER_PALETTE
    for i, school in enumerate(picks):
        part = apps[apps["school"] == school]
        fig.add_scatter(x=part["fall"], y=part["value"], name=school, mode="lines",
                        line=dict(color=peer_colors[i % len(peer_colors)], width=2), opacity=0.75)
    part = apps[apps["school"] == "Boston College"]
    fig.add_scatter(x=part["fall"], y=part["value"], name="Boston College", mode="lines+markers",
                    line=dict(color=MAROON, width=4), marker=dict(size=7))
    if show_events:
        for year, label in EVENTS.items():
            y = float(part.loc[part["fall"] == year, "value"].iloc[0])
            fig.add_annotation(x=year, y=y, text=label, showarrow=True, arrowhead=0, ax=0, ay=-45,
                               font=dict(color=MAROON, size=12), bgcolor="rgba(251,248,243,.9)")
    fig.update_yaxes(title="Index (2005 = 100)" if indexed else "First-year applications",
                     tickformat=",")
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(style(fig, 460), use_container_width=True)
takeaway("BC lost more than a quarter of its applicants when it added an essay in 2013, and again dipped when it added "
         "Early Decision in 2020. Both times, it bounced back.")

with st.expander("See all 15 schools on a map (press play)"):
    mp = d.dropna(subset=["number_applied"]).copy()
    mp["school"] = mp["unitid"].map(F.SHORT_NAMES)
    mp["group"] = np.where(mp["unitid"] == M.BC_ID, "Boston College", "Peer")
    fig = px.scatter_geo(mp, lat="latitude", lon="longitude", size="number_applied", color="group",
                         hover_name="school", hover_data={"number_applied": ":,", "latitude": False,
                                                          "longitude": False, "group": False},
                         animation_frame="fall", scope="usa", size_max=42,
                         color_discrete_map={"Boston College": MAROON, "Peer": GOLD})
    fig.update_geos(bgcolor="rgba(0,0,0,0)", landcolor="#F1EADF", subunitcolor="#fff")
    st.plotly_chart(style(fig, 520), use_container_width=True)

# ---------------------------------------------------------------- 2. everyone moves together
chapter(2, "everyone", "In 2020 and 2021, every school moved together",
        "Each gray line is one school's yearly growth. Drag the slider to spotlight a year and see how the 15 "
        "schools did. In some years nearly everyone rises or falls at once.")
year = st.select_slider("Spotlight a year", options=sorted(g["fall"].unique()), value=2021)
left, right = st.columns([3, 2])
with left:
    fig = go.Figure()
    for uid, part in g.groupby("unitid"):
        if uid == M.BC_ID:
            continue
        fig.add_scatter(x=part["fall"], y=part["growth_pct"], mode="lines", name=part["school"].iloc[0],
                        line=dict(color=MUTED, width=1), opacity=0.5, showlegend=False,
                        hovertemplate="%{fullData.name}<br>%{x}: %{y:+.1f}%<extra></extra>")
    avg = g.groupby("fall")["growth_pct"].mean()
    fig.add_scatter(x=avg.index, y=avg.values, name="15-school average", line=dict(color=INK, width=2.5, dash="dash"))
    fig.add_scatter(x=bc_g["fall"], y=bc_g["growth_pct"], name="Boston College", line=dict(color=MAROON, width=3.5))
    fig.add_vrect(x0=year - 0.5, x1=year + 0.5, fillcolor="#999", opacity=0.15, line_width=0)
    fig.add_hline(y=0, line=dict(color="#BBBBBB", width=1))
    fig.update_yaxes(title="Growth vs. prior year (%)", ticksuffix="%")
    st.plotly_chart(style(fig, 440), use_container_width=True)
with right:
    yr = g[g["fall"] == year].sort_values("growth_pct")
    fig = go.Figure(go.Bar(
        x=yr["growth_pct"], y=yr["school"], orientation="h",
        marker_color=[MAROON if s == "Boston College" else (POS if v >= 0 else NEG)
                      for s, v in zip(yr["school"], yr["growth_pct"])],
        text=[f"{v:+.0f}%" for v in yr["growth_pct"]], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:+.1f}%<extra></extra>"))
    rng = max(abs(yr["growth_pct"]).max() * 1.6, 10)
    fig.update_xaxes(range=[-rng, rng], ticksuffix="%")
    fig.update_layout(title=dict(text=f"Fall {year}: {(yr['growth_pct'] > 0).sum()} of {len(yr)} schools grew",
                                 font=dict(size=15)))
    st.plotly_chart(style(fig, 440, legend=False), use_container_width=True)
c1, c2, _ = st.columns([1, 1, 1])
stat(c1, f"{share_all:.0%}", "of growth is explained just by which year it was")
stat(c2, f"{share_no_covid:.0%}", "once you leave out the 2020-21 COVID and test-optional years")
takeaway("Shared shocks like COVID and test-optional move every school at once. Outside those two years, "
         "the market mostly isn't the story, so the next question is what's unique to BC.")

# ---------------------------------------------------------------- 3. BC's own choices
chapter(3, "choices", "BC's biggest drops followed its own policy changes",
        "Here we subtract the average of the other 14 schools from BC's growth each year. What's left is "
        "what made BC different. Hover a bar for details.")
fig = go.Figure(go.Bar(
    x=bc_g["fall"], y=bc_g["rel_growth_pct"],
    marker_color=np.where(bc_g["rel_growth_pct"] >= 0, POS, NEG),
    customdata=np.c_[bc_g["growth_pct"], bc_g["peer_avg_pct"]],
    hovertemplate="Fall %{x}<br>BC: %{customdata[0]:+.1f}%<br>Peers: %{customdata[1]:+.1f}%"
                  "<br><b>Gap: %{y:+.1f} pts</b><extra></extra>"))
for yr_, label in {**EVENTS, 2015: "Rebound"}.items():
    v = float(bc_g.loc[bc_g["fall"] == yr_, "rel_growth_pct"].iloc[0])
    fig.add_annotation(x=yr_, y=v, text=f"<b>{label}</b><br>{v:+.0f} pts", showarrow=False,
                       yshift=28 if v > 0 else -28, font=dict(size=12, color=INK))
fig.add_hline(y=0, line=dict(color="#BBBBBB"))
fig.update_yaxes(title="BC growth minus peer average (pts)")
fig.update_xaxes(dtick=2)
st.plotly_chart(style(fig, 440, legend=False), use_container_width=True)
takeaway("The essay (2013) and Early Decision (2020) each cost BC 15-38 points of growth against its peers. "
         "Admissions policy moved BC's numbers more than anything in the financial data.")

# ---------------------------------------------------------------- 4. test-optional
chapter(4, "testoptional", "Going test-optional gave a one-year bump",
        "Six peers dropped their test requirement the same cycle BC did (fall 2021). Compare BC with their "
        "average, or turn on each school individually.")
ev = F.test_optional_event(g)
ev_long = F.test_optional_event(g, long=True)
show_each = st.toggle("Show each school", value=False)
fig = go.Figure()
if show_each:
    for i, (school, part) in enumerate(ev_long[ev_long["school"] != "Boston College"].groupby("school")):
        fig.add_scatter(x=part["years_since"], y=part["growth_pct"], name=school, mode="lines+markers",
                        line=dict(width=1.5, color=PEER_PALETTE[i]), opacity=0.7)
fig.add_scatter(x=ev.index, y=ev["Other test-optional schools"], name="Average of the six peers",
                mode="lines+markers", line=dict(color=TEAL, width=3.5, dash="dot" if show_each else "solid"),
                marker=dict(size=9))
fig.add_scatter(x=ev.index, y=ev["Boston College"], name="Boston College", mode="lines+markers+text",
                line=dict(color=MAROON, width=4), marker=dict(size=10),
                text=[f"{v:+.0f}%" for v in ev["Boston College"]],
                textposition=["top center" if b >= p else "bottom center"
                              for b, p in zip(ev["Boston College"], ev["Other test-optional schools"])],
                textfont=dict(color=MAROON, size=13))
fig.add_vrect(x0=-0.4, x1=0.4, fillcolor="#999", opacity=0.12, line_width=0,
              annotation_text="Went test-optional", annotation_position="bottom")
fig.add_hline(y=0, line=dict(color="#BBBBBB"))
fig.update_xaxes(tickvals=list(ev.index), ticktext=[f"{x:+d} yr" if x else "Fall 2021" for x in ev.index])
fig.update_yaxes(title="Application growth (%)", ticksuffix="%")
st.plotly_chart(style(fig, 460), use_container_width=True)
c1, c2, c3 = st.columns(3)
stat(c1, f"{ev.loc[0, 'Other test-optional schools']:+.0f}%", "Peers' growth the year they went test-optional")
stat(c2, f"{ev.loc[0, 'Boston College']:+.0f}%", f"BC's growth, after {ev.loc[-1, 'Boston College']:+.0f}% the year before")
stat(c3, "1-2 yrs", "Until growth was back to normal")
takeaway("Going test-optional brought a one-time surge, not a new growth rate. That's why fall 2021 is flagged "
         "in our models.")

# ---------------------------------------------------------------- 5. price
chapter(5, "price", "Price barely moves applications",
        "If a school gets more expensive, do fewer students apply the next year? Pick a price measure to see "
        "each school-year, with every school's usual level and every year's trend removed.")
label_to_col = {v: k for k, v in F.PRICE_MEASURES.items()}
measure = st.radio("Price measure", list(label_to_col), horizontal=True, index=0)
points, bins = F.net_price_binned(g, label_to_col[measure])
left, right = st.columns([3, 2])
with left:
    fig = go.Figure()
    fig.add_scatter(x=points["lp_prev_r"] * 100, y=points["growth_pct_r"], mode="markers",
                    name="School-years (BC in red)",
                    marker=dict(color=np.where(points["unitid"] == M.BC_ID, MAROON, "#CBCBCB"), size=8,
                                line=dict(width=0)),
                    customdata=np.c_[points["school"], points["fall"]],
                    hovertemplate="%{customdata[0]} %{customdata[1]}<br>Price vs usual: %{x:+.1f}%"
                                  "<br>Growth vs usual: %{y:+.1f} pts<extra></extra>")
    fig.add_scatter(x=bins["lp_prev_r"] * 100, y=bins["growth_pct_r"], mode="lines+markers",
                    name="Average of each tenth", line=dict(color=INK, width=3), marker=dict(size=9))
    fig.add_hline(y=0, line=dict(color="#BBBBBB")); fig.add_vline(x=0, line=dict(color="#BBBBBB"))
    fig.update_xaxes(title="Price last year vs. school's usual (%)", ticksuffix="%")
    fig.update_yaxes(title="Growth vs. usual (pts)")
    st.plotly_chart(style(fig, 430), use_container_width=True)
with right:
    e = price_fx.iloc[::-1]
    sig = e["p_value"] < 0.05
    fig = go.Figure(go.Scatter(
        x=e["effect"], y=e["measure"], mode="markers",
        marker=dict(size=[16 if m == measure else 10 for m in e["measure"]],
                    color=[TEAL if s else MUTED for s in sig], line=dict(width=0)),
        error_x=dict(type="data", symmetric=False, array=e["high"] - e["effect"],
                     arrayminus=e["effect"] - e["low"], color="#BBBBBB", thickness=2),
        hovertemplate="%{y}<br>%{x:+.2f} pts per +10% price<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=INK, width=1))
    fig.update_xaxes(title="Pts per +10% price")
    fig.update_layout(title=dict(text="Effect with 95% CI (blue = significant)", font=dict(size=14)))
    st.plotly_chart(style(fig, 430, legend=False), use_container_width=True)
takeaway("For most families' price brackets the effect is indistinguishable from zero. The only significant one, "
         "the $110k+ price, goes the <i>wrong</i> way, likely because schools in high demand can charge more.")

# ---------------------------------------------------------------- 6. what predicts growth
chapter(6, "predicts", "Last year's admit rate is the strongest signal",
        "We put 11 of last year's characteristics into one model with school and year fixed effects. "
        "Each dot shows how much growth changes when that factor is one standard deviation higher.")
only_sig = st.toggle("Only show significant factors", value=False)
f = fe.iloc[::-1].copy()
f["sig"] = f["p_value"] < 0.05
if only_sig:
    f = f[f["sig"]]
fig = go.Figure(go.Scatter(
    x=f["coef"], y=f["label"], mode="markers",
    marker=dict(size=13, color=[(POS if c > 0 else NEG) if s else MUTED for c, s in zip(f["coef"], f["sig"])]),
    error_x=dict(type="data", symmetric=False, array=f["high"] - f["coef"], arrayminus=f["coef"] - f["low"],
                 color="#BBBBBB", thickness=2.5, width=6),
    customdata=np.c_[f["low"], f["high"], f["p_value"]],
    hovertemplate="%{y}<br>Effect: %{x:+.3f}<br>95% CI %{customdata[0]:+.3f} to %{customdata[1]:+.3f}"
                  "<br>p = %{customdata[2]:.3f}<extra></extra>"))
fig.add_vline(x=0, line=dict(color=INK, width=1, dash="dot"))
fig.update_xaxes(title="Change in application growth (log points) per 1 SD")
fig.update_yaxes(categoryorder="array", categoryarray=list(f["label"]))
st.plotly_chart(style(fig, 260 if only_sig else 500, legend=False), use_container_width=True)
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**Less selective last year → more growth.** Schools with higher admit rates had room to attract "
                "more applicants.")
with c2:
    st.markdown("**Big drops in admit rate → slower growth next year.** Getting pickier can scare some applicants off.")
with c3:
    st.markdown("**Surges cool off.** Two strong years tend to be followed by a slower one.")
with st.expander("Other ways we checked: best-subset search and the forecast model"):
    l2, r2 = st.columns(2)
    with l2:
        bs = F.BEST_SUBSET_8.iloc[::-1]
        fig = go.Figure(go.Bar(x=bs["effect"], y=bs["variable"], orientation="h",
                               marker_color=np.where(bs["effect"] > 0, POS, NEG),
                               text=[f"{v:+.1f} {s}" for v, s in zip(bs["effect"], bs["stars"])],
                               textposition="outside"))
        fig.update_xaxes(range=[-5, 5], title="Peer-relative growth, pts per 1 SD")
        fig.update_layout(title=dict(text="Best 8-variable combination (R² = 0.13)", font=dict(size=14)))
        st.plotly_chart(style(fig, 360, legend=False), use_container_width=True)
    with r2:
        imp = M.factor_importance(model, d)
        fig = px.bar(imp.sort_values("importance"), x="importance", y="label", orientation="h", color="direction",
                     color_discrete_map={"Raises applications": POS, "Lowers applications": NEG})
        fig.update_xaxes(title="Mean |SHAP| (Ridge forecast model)")
        fig.update_yaxes(title=None)
        fig.update_layout(title=dict(text="Forecast model's view", font=dict(size=14)))
        st.plotly_chart(style(fig, 360), use_container_width=True)

# ---------------------------------------------------------------- 7. quality signals
chapter(7, "quality", "Retention and graduation rates overlap",
        "Retention and graduation rates both measure how well a school serves students. Toggle between comparing "
        "schools to each other and following each school over time.")
rv, school_avg, within, r_between, r_within = F.retention_vs_graduation(d)
view = st.segmented_control("View", ["Across schools", "Within a school over time"], default="Across schools")
if view == "Within a school over time":
    fig = px.scatter(within, x="Retention", y="Graduation", hover_name="school", opacity=0.7,
                     color=np.where(within["school"] == "Boston College", "Boston College", "Peer"),
                     color_discrete_map={"Boston College": MAROON, "Peer": GOLD})
    fig.update_traces(marker=dict(size=9))
    fig.update_xaxes(title="Retention vs. school's usual (pts)")
    fig.update_yaxes(title="Graduation vs. school's usual (pts)")
    note = f"r = {r_within:.2f}: year to year, the two barely move together."
else:
    fig = px.scatter(school_avg, x="Retention", y="Graduation", text="school", trendline="ols",
                     color=np.where(school_avg["school"] == "Boston College", "Boston College", "Peer"),
                     color_discrete_map={"Boston College": MAROON, "Peer": GOLD})
    fig.update_traces(marker=dict(size=14), textposition="top center", selector=dict(mode="markers+text"))
    fig.update_xaxes(title="Average retention rate (%)")
    fig.update_yaxes(title="Average 6-year graduation rate (%)")
    note = f"r = {r_between:.2f}: schools rank almost the same on both."
st.plotly_chart(style(fig, 440, legend=False), use_container_width=True)
takeaway(f"{note} The two rates are redundant for comparing schools, and neither predicts application growth, "
         "so we keep graduation rate as a descriptor only.")

# ---------------------------------------------------------------- 8. looking ahead
chapter(8, "ahead", f"A forecast for fall {next_year}",
        f"Our forecast model predicts fall {next_year} applications from price, aid, SAT and endowment changes. "
        "Move the sliders to test a scenario. They start at what actually happened.")
bc_bt = bt[bt["unitid"] == M.BC_ID]
bc_error = float((abs(bc_bt["actual"] - bc_bt["predicted"]) / bc_bt["actual"]).mean())


def pct(feature):
    return float(np.expm1(next_row[feature].iloc[0]) * 100)


left, right = st.columns([2, 3])
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
if abs(what_if - baseline) < 25:
    what_if = baseline
with right:
    fig = go.Figure()
    fig.add_scatter(x=bc["fall"], y=bc["number_applied"], name="Actual", mode="lines+markers",
                    line=dict(color=MAROON, width=3.5))
    fig.add_scatter(x=bc_bt["fall"], y=bc_bt["predicted"], name="Model (predicted each year in advance)",
                    mode="lines", line=dict(color=TEAL, width=2, dash="dash"))
    fig.add_scatter(x=[last_year, next_year], y=[last_apps, what_if], name=f"Your scenario for {next_year}",
                    mode="lines+markers", line=dict(color=GOLD, width=3, dash="dot"),
                    marker=dict(size=[0, 18], symbol="star", color=GOLD))
    lo, hi = what_if * (1 - bc_error), what_if * (1 + bc_error)
    fig.add_scatter(x=[next_year, next_year], y=[lo, hi], mode="lines", line=dict(color=GOLD, width=8),
                    opacity=0.35, name=f"Typical miss (±{bc_error:.0%})")
    fig.update_yaxes(title="Applications", tickformat=",")
    fig.update_xaxes(range=[2004.5, next_year + 0.8])
    st.plotly_chart(style(fig, 400), use_container_width=True)
c1, c2, c3 = st.columns(3)
stat(c1, f"{what_if:,.0f}", f"Predicted applications, fall {next_year}")
stat(c2, f"{what_if - baseline:+,.0f}", "vs. baseline forecast")
stat(c3, f"±{bc_error:.0%}", "Typical miss for BC in past years")
takeaway("Even big price moves shift the forecast only modestly, which matches the rest of the story: "
         "BC's admissions policies and market-wide shocks matter more than price. College rankings aren't in "
         "our data, so there's no ranking slider yet.")

st.markdown("<br><br>", unsafe_allow_html=True)
st.caption("Methods: application growth = log change in first-year applications (IPEDS). Predictors from College "
           "Scorecard, lagged one year. Regressions use school + year fixed effects with standard errors clustered "
           "by school (statsmodels). Forecast model: Ridge regression (scikit-learn) with an expanding-window backtest.")
