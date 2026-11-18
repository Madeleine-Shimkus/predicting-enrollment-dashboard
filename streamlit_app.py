"""Why do admitted students choose Boston College? A scrolling data story about yield.

Run it locally with:  streamlit run streamlit_app.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import features as F
import yield_model as Y

st.set_page_config(page_title="Who Says Yes to BC?", page_icon=":eagle:", layout="wide")

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




DARK_PEER, LIGHT = "#555555", "#D0D0D0"


def sig_word(p):
    return "clear" if p < 0.05 else ("weak" if p < 0.10 else "not significant")


# ---------------------------------------------------------------- data
@st.cache_data
def get_data(data_version):  # data_version changes when any data file changes, so the cache refreshes
    data, within, across = Y.load()
    fitted, explained = Y.fit_within(data, within["variable"].tolist())
    return data, within, across, fitted, explained


@st.cache_resource
def get_models(data_version):
    data, within, across, _, _ = get_data(data_version)
    feats = within["variable"].tolist()
    numeric = [v for v in across["variable"] if v not in Y.TRAITS]
    traits = [v for v in across["variable"] if v in Y.TRAITS]
    # the forecast also uses last year's yield, so a school's recent level carries forward
    fc_feats = feats + [Y.LAG]
    lagged = Y.add_lag(data)
    fit, med = Y.forecaster(lagged, fc_feats)
    bt, rmse = Y.backtest(lagged, fc_feats)
    afit, amed = Y.across_fit(data, numeric, traits)
    nxt = int(data["fall"].max()) + 1
    X = F.inputs_for_fall(nxt)
    # anything missing for next fall (e.g. applications outside our admissions file): carry last value forward
    last = data.sort_values("fall").groupby("UNITID").last()
    cols = [c for c in X.columns if c in last.columns]
    X = X.reindex(last.index)
    X[cols] = X[cols].fillna(last[cols])
    X[Y.LAG] = last["target"]
    return dict(fc_feats=fc_feats, feats=feats, numeric=numeric, traits=traits, fit=fit, med=med, bt=bt, rmse=rmse,
                afit=afit, amed=amed, X=X)


data, within, across, fitted, explained = get_data(Y.version())
M = get_models(Y.version())
next_year = int(data["fall"].max()) + 1


def forecast_for(uid, values):
    row = pd.DataFrame([{**values, "UNITID": uid}])
    return float(Y.predict(M["fit"], M["med"], row, M["fc_feats"], next_year - 1).iloc[0])


bc_inputs = M["X"].loc[Y.BC_ID, M["fc_feats"]].to_dict()
bc_forecast = forecast_for(Y.BC_ID, bc_inputs)

n_schools = data["UNITID"].nunique()
first_year, last_year = int(data["fall"].min()), int(data["fall"].max())
bc = data[data["UNITID"] == Y.BC_ID].sort_values("fall")
bc_last = float(bc.loc[bc["fall"] == last_year, "target"].iloc[0])
bc_first = float(bc.loc[bc["fall"] == first_year, "target"].iloc[0])
latest = data[data["fall"] == last_year].sort_values("target", ascending=False).reset_index(drop=True)
bc_rank = int(latest.index[latest["UNITID"] == Y.BC_ID][0]) + 1
peer_med = data[data["UNITID"].isin(Y.PEER_IDS)].groupby("fall")["target"].median()
all_med = data[data["UNITID"] != Y.BC_ID].groupby("fall")["target"].median()
big_sample = n_schools > 15
field = f"{n_schools - 1} other selective private colleges" if big_sample else "its 14 peer schools"

# ---------------------------------------------------------------- opening
st.markdown(
    f"""
    <div class="kicker">Boston College admissions, fall {first_year}-{last_year}</div>
    <div class="headline">Who says yes to Boston College?</div>
    <div class="dek">Getting admitted is half the story. Yield is the share of admitted students who actually
    enroll, and it decides how big each class is and how many offers BC has to make. In fall {last_year},
    {bc_last:.0f}% of the students BC admitted chose to come. We compared BC with {field} to see what
    moves that number.</div>
    <div class="byline">By <b>Ellie El-Fishawy, Francesco Noble, Yennie Park, Madeleine Shimkus and Nolan Senecal</b>
    &nbsp;·&nbsp; Data science capstone &nbsp;·&nbsp; Data: College Scorecard, IPEDS</div>
    """,
    unsafe_allow_html=True,
)
c1, c2, c3, c4 = st.columns(4)
stat(c1, f"{bc_last:.0f}%", f"BC yield, fall {last_year}")
stat(c2, f"{bc_last - bc_first:+.0f} pts", f"Change since {first_year}")
stat(c3, f"#{bc_rank} of {len(latest)}", f"BC's rank in fall {last_year}")
stat(c4, f"{bc_forecast:.0f}%", f"Our forecast for fall {next_year} (±{M['rmse']:.0f} pts)")

# ---------------------------------------------------------------- 1. over time
gap_now = bc_last - peer_med.loc[last_year]
chapter(1, "time", f"BC's yield {'rose' if bc_last > bc_first else 'slipped'} from {bc_first:.0f}% to {bc_last:.0f}%",
        "Each line is the share of admitted students who enrolled that fall. Add any school to compare.")
names = sorted(data.loc[data["UNITID"] != Y.BC_ID, "school"].unique().tolist())
left, right = st.columns([3, 1])
with right:
    default = [n for n in ["Notre Dame", "Georgetown"] if n in names]
    picks = st.multiselect("Compare BC with", names, default=default)
    show_peer = st.toggle("14-peer median", value=True)
    show_all = st.toggle("All-school median", value=big_sample)
with left:
    fig = go.Figure()
    for i, school in enumerate(picks):
        part = data[data["school"] == school]
        fig.add_scatter(x=part["fall"], y=part["target"], name=school, mode="lines",
                        line=dict(color=PEER_PALETTE[i % len(PEER_PALETTE)], width=2), opacity=0.8)
    if show_peer:
        fig.add_scatter(x=peer_med.index, y=peer_med.values, name="14-peer median",
                        line=dict(color=INK, width=2.5, dash="dash"))
    if show_all:
        fig.add_scatter(x=all_med.index, y=all_med.values, name=f"All {n_schools - 1} other schools (median)",
                        line=dict(color=DARK_PEER, width=2, dash="dot"))
    fig.add_scatter(x=bc["fall"], y=bc["target"], name="Boston College", mode="lines+markers",
                    line=dict(color=MAROON, width=4), marker=dict(size=7))
    fig.update_yaxes(title="Yield (share of admits who enroll)", ticksuffix="%")
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(style(fig, 450), width="stretch")
takeaway(f"In fall {last_year} BC's yield was {abs(gap_now):.0f} points {'above' if gap_now >= 0 else 'below'} "
         f"the typical peer. Yield mostly reflects how strongly students already prefer a school, so it moves "
         f"slowly from year to year.")

# ---------------------------------------------------------------- 2. ranking
chapter(2, "rank", f"Where BC lands among {len(latest)} schools",
        "Every school's yield for one entering class, highest to lowest. Drag the slider to change the year.")
yr = st.select_slider("Entering class (fall)", options=sorted(data["fall"].unique()), value=last_year)
one = data[data["fall"] == yr].sort_values("target", ascending=False).reset_index(drop=True)
one["rank"] = one.index + 1
colors = [MAROON if u == Y.BC_ID else (DARK_PEER if u in Y.PEER_IDS else LIGHT) for u in one["UNITID"]]
fig = go.Figure(go.Bar(x=one["rank"], y=one["target"], marker_color=colors, marker_line_width=0,
                       customdata=one[["school"]], hovertemplate="#%{x} %{customdata[0]}: %{y:.1f}%<extra></extra>"))
r = one.loc[one["UNITID"] == Y.BC_ID].iloc[0]
fig.add_annotation(x=r["rank"], y=r["target"], text=f"<b>Boston College</b><br>#{int(r['rank'])}, {r['target']:.0f}%",
                   showarrow=True, arrowhead=0, ax=0, ay=-40, font=dict(color=MAROON, size=13))
fig.update_xaxes(title="Rank", showticklabels=len(one) <= 20, dtick=1 if len(one) <= 20 else None)
fig.update_yaxes(title="Yield", ticksuffix="%")
if len(one) <= 20:
    fig.update_xaxes(tickvals=one["rank"], ticktext=one["school"], tickangle=-45, title=None)
st.plotly_chart(style(fig, 420, legend=False), width="stretch")
st.markdown('<div class="source">Maroon = Boston College · dark gray = the 14 original peers'
            + (" · light gray = other selective private colleges" if big_sample else "") + "</div>",
            unsafe_allow_html=True)

# ---------------------------------------------------------------- 3. within a school
top = within.iloc[0]
sig_w = within[within["p_value"] < 0.05]
chapter(3, "within", f"{Y.plain(top['variable']).capitalize()} {'up' if top['effect'] > 0 else 'down'}, yield {'up' if top['effect'] > 0 else 'down'} the next year"
        if top["p_value"] < 0.05 else "Year to year, no single factor moves yield much",
        "This model follows each school over time. It removes every school's usual yield and every year's "
        "market-wide shift, then asks what last year's numbers add. Each bar is the change in yield when a "
        "factor is one standard deviation higher than usual. Hover for what that means in real units.")
only_sig = st.toggle("Only show clear effects (p < 0.05)", value=False, key="w_sig")


def effect_bars(e, height):
    e = e.iloc[::-1]
    sd_text = [Y.describe_sd(data, v) if v in data.columns and v not in Y.TRAITS else "yes vs. no" for v in e["variable"]]
    fig = go.Figure(go.Bar(
        x=e["effect"], y=e["variable"], orientation="h", marker_line_width=0,
        marker_color=[(MAROON if v > 0 else POS) if p < 0.05 else LIGHT for v, p in zip(e["effect"], e["p_value"])],
        text=[f"{v:+.1f}" for v in e["effect"]], textposition="outside", cliponaxis=False,
        customdata=np.c_[sd_text, e["p_value"]],
        hovertemplate="<b>%{y}</b><br>%{x:+.2f} yield pts when last year's value is<br>%{customdata[0]}"
                      "<br>p = %{customdata[1]:.3f}<extra></extra>"))
    lim = max(e["effect"].abs().max() * 1.35, 1)
    fig.add_vline(x=0, line=dict(color=INK, width=1))
    fig.update_xaxes(range=[-lim, lim], title="Change in yield (% pts)")
    fig.update_yaxes(categoryorder="array", categoryarray=list(e["variable"]))
    return style(fig, height, legend=False)


w = sig_w if only_sig and len(sig_w) else within
st.plotly_chart(effect_bars(w, 120 + 42 * len(w)), width="stretch")
c1, c2, _ = st.columns(3)
stat(c1, f"{explained:.0%}", "of the year-to-year movement (beyond each school's usual level) these factors explain")
stat(c2, f"{len(sig_w)} of {len(within)}", "factors with a clear effect (p < 0.05)")
lines = [f"<b>{r.variable}</b> ({r.effect:+.1f} pts per SD)" for r in sig_w.itertuples()]
takeaway("Clear effects: " + ("; ".join(lines) if lines else "none at the 5% level") +
         ". Colored bars are significant; maroon raises yield, blue lowers it. Gray bars could be zero. "
         "Admit rate is left out on purpose: admitting fewer students raises yield almost automatically.")

# ---------------------------------------------------------------- 4. across schools
chapter(4, "across", "What kind of school keeps the students it admits",
        "This model compares schools with each other instead. Pick a factor to see each school's average "
        "against its average yield.")
numeric_across = [v for v in across["variable"] if v not in Y.TRAITS]
left, right = st.columns([3, 2])
with left:
    var = st.selectbox("Factor", numeric_across, index=0)
    avg = data.groupby(["UNITID", "school", "group"], as_index=False)[[var, "target"]].mean().dropna()
    x = np.exp(avg[var]) if var in Y.LOGGED else (avg[var] * 100 if var in Y.SHARES else avg[var])
    avg["x"] = x
    fig = go.Figure()
    for grp, color, size in [("Other selective private", LIGHT, 8), ("Original peer", DARK_PEER, 10),
                             ("Boston College", MAROON, 15)]:
        part = avg[avg["group"] == grp]
        if part.empty:
            continue
        fig.add_scatter(x=part["x"], y=part["target"], mode="markers" if grp != "Boston College" else "markers+text",
                        name=grp if grp != "Original peer" else "14 original peers", text=part["school"],
                        textposition="top center", textfont=dict(color=MAROON, size=13),
                        marker=dict(color=color, size=size, line=dict(width=0)),
                        hovertemplate="%{hovertext}<br>%{x:,.3~f}<br>Yield %{y:.1f}%<extra></extra>",
                        hovertext=part["school"])
    if len(avg) > 3:
        b = np.polyfit(avg["x"], avg["target"], 1)
        xs = np.linspace(avg["x"].min(), avg["x"].max(), 50)
        fig.add_scatter(x=xs, y=np.polyval(b, xs), mode="lines", name="Trend", line=dict(color=INK, width=1.5, dash="dot"))
    fig.update_xaxes(title=var + (" (%)" if var in Y.SHARES else ""), type="log" if var in Y.LOGGED else "linear")
    fig.update_yaxes(title="Average yield", ticksuffix="%")
    st.plotly_chart(style(fig, 440), width="stretch")
with right:
    a = across.head(10)
    st.plotly_chart(effect_bars(a, 120 + 38 * len(a)), width="stretch")
st.markdown('<div class="source">Right: the 10 largest effects in the across-school model (year effects and region '
            'held fixed). School traits like Catholic or city campus are yes vs. no, not per SD.</div>',
            unsafe_allow_html=True)
sig_a = across[across["p_value"] < 0.05].head(4)
takeaway("Across schools, the clearest differences are " +
         ", ".join(f"<b>{Y.plain(r.variable)}</b> ({r.effect:+.1f})" for r in sig_a.itertuples()) +
         ". These describe what high-yield schools look like, not what would change BC's yield on its own.")

# ---------------------------------------------------------------- 5. forecast
def slider_for(var, value, key):
    """A slider in plain units (dollars, %, counts) that starts at the real value.
    Returns the model-scale value; untouched sliders return the exact real value."""
    v = Y.to_plain(var, value)
    if var == "Test required":
        return 1.0 if st.toggle(f"Test required for fall {next_year}", value=bool(value), key=key) else 0.0
    if var in Y.SHARES:
        lo, hi = max(0.0, round(v) - 15.0), min(100.0, round(v) + 15.0)
        start = float(round(v * 2) / 2)
        out = st.slider(f"{var} (%)", lo, hi, start, 0.5, key=key, format="%.1f%%")
    elif var == Y.LAG:
        start = float(round(v * 2) / 2)
        out = st.slider(f"Fall {last_year} yield (%)", 10.0, 90.0, start, 0.5, key=key, format="%.1f%%")
    elif var == "SAT average":
        start = int(v)
        out = st.slider(var, start - 80, start + 80, start, 5, key=key)
    elif var in Y.DOLLARS and v >= 10000:   # show as $58.5k
        k = v / 1000
        step = 0.5 if k < 200 else 5.0
        lo, hi = round(k * 0.6 / step) * step, round(k * 1.4 / step) * step
        start = min(max(round(k / step) * step, lo), hi)
        out = st.slider(var, lo, hi, start, step, key=key, format="$%.1fk")
        if out == start:
            return value
        return Y.from_plain(var, out * 1000)
    else:
        step = int(10 ** max(int(np.log10(max(v, 1))) - 2, 0))
        lo, hi = int(v * 0.6 // step * step), int(v * 1.4 // step * step)
        start = min(max(int(round(v / step) * step), lo), hi)
        out = st.slider(var, lo, hi, start, step, key=key, format="$%d" if var in Y.DOLLARS else "%d")
    if out == start:
        return value
    return Y.from_plain(var, out)


chapter(5, "forecast", f"Forecast: fall {next_year} yield",
        f"The model predicts next fall's yield from this fall's numbers. The sliders start at each school's real "
        f"fall {last_year} values. Move them to test a scenario, or pick another school. Besides the factors above, "
        "the forecast also uses last fall's yield, since yield tends to carry over from one year to the next.")
schools = data.drop_duplicates("UNITID").set_index("UNITID")["school"].sort_values()
order = [Y.BC_ID] + [u for u in schools.index if u != Y.BC_ID]
uid = st.selectbox("School", order, format_func=lambda u: schools[u], key="fc_school")
name = schools[uid]
base_inputs = M["X"].loc[uid, M["fc_feats"]].to_dict()
if st.button("Reset sliders to real values"):
    for k in [k for k in st.session_state if str(k).startswith(f"fc_{uid}_")]:
        del st.session_state[k]
left, right = st.columns([2, 3])
with left:
    scen = {v: slider_for(v, base_inputs[v], f"fc_{uid}_{v}") for v in M["fc_feats"]}
base_fc = forecast_for(uid, base_inputs)
scen_fc = forecast_for(uid, scen)
rmse = M["rmse"]
hist = data[data["UNITID"] == uid].sort_values("fall")
bt = M["bt"][M["bt"]["UNITID"] == uid]
with right:
    fig = go.Figure()
    fig.add_scatter(x=hist["fall"], y=hist["target"], name=f"{name} actual", mode="lines+markers",
                    line=dict(color=MAROON if uid == Y.BC_ID else INK, width=3.5), marker=dict(size=7))
    fig.add_scatter(x=bt["fall"], y=bt["forecast"], name="Model's forecast, made a year ahead", mode="lines+markers",
                    line=dict(color=TEAL, width=2, dash="dash"), marker=dict(size=6))
    fig.add_scatter(x=[next_year, next_year], y=[scen_fc - rmse, scen_fc + rmse], mode="lines",
                    line=dict(color=GOLD, width=10), opacity=0.3, name=f"Typical miss (±{rmse:.1f} pts)")
    fig.add_scatter(x=[next_year], y=[base_fc], mode="markers", name="Forecast with real values",
                    marker=dict(size=12, color="white", line=dict(color=TEAL, width=2.5)))
    fig.add_scatter(x=[last_year, next_year], y=[float(hist["target"].iloc[-1]), scen_fc], mode="lines+markers",
                    name="Your scenario", line=dict(color=GOLD, width=3, dash="dot"),
                    marker=dict(size=[0, 20], symbol="star", color=GOLD))
    fig.update_yaxes(title="Yield", ticksuffix="%")
    fig.update_xaxes(range=[hist["fall"].min() - 0.5, next_year + 0.7])
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(style(fig, 470), width="stretch")
c1, c2, c3 = st.columns(3)
stat(c1, f"{scen_fc:.1f}%", f"{name}, predicted fall {next_year} yield")
stat(c2, f"{scen_fc - base_fc:+.1f} pts", "from your slider changes")
stat(c3, f"±{rmse:.1f} pts", "typical miss when we tested on past years")
takeaway(f"The dashed line shows how the model did on past years, each forecast made using only earlier years. "
         f"Big misses (like fall {int(bt.loc[(bt['target'] - bt['forecast']).abs().idxmax(), 'fall'])} here) are "
         "usually market-wide shocks the model can't see coming.")

# ---------------------------------------------------------------- 6. build a school
chapter(6, "build", "Build a school and predict its yield",
        "This uses the across-school model. It starts as Boston College. Change its traits and numbers to see "
        "what yield a school like that would usually get, and where it would rank.")
top_numeric = across[~across["variable"].isin(Y.TRAITS)].head(6)["variable"].tolist()
build = M["X"].loc[Y.BC_ID, M["numeric"]].to_dict()
left, right = st.columns([2, 3])
with left:
    bc_row = data[data["UNITID"] == Y.BC_ID].iloc[-1]
    for t in M["traits"]:
        build[t] = float(st.toggle(t, value=bool(bc_row[t]), key=f"b_{t}"))
    regions = sorted(data["Region"].dropna().unique())
    build["Region"] = st.selectbox("Region", regions, index=regions.index(bc_row["Region"]), key="b_region")
    for v in top_numeric:
        build[v] = slider_for(v, build[v], f"b_{v}")
    held = [v for v in M["numeric"] if v not in top_numeric]
    if held:
        st.markdown(f'<div class="source">Held at BC\'s values: {", ".join(held)}.</div>', unsafe_allow_html=True)
row = pd.DataFrame([{**build, "fall": last_year}])
row[M["numeric"]] = row[M["numeric"]].fillna(M["amed"])
built = float(M["afit"].predict(row).iloc[0])
row_bc = pd.DataFrame([{**M["X"].loc[Y.BC_ID, M["numeric"]].to_dict(), **{t: float(bc_row[t]) for t in M["traits"]},
                        "Region": bc_row["Region"], "fall": last_year}])
row_bc[M["numeric"]] = row_bc[M["numeric"]].fillna(M["amed"])
built_bc = float(M["afit"].predict(row_bc).iloc[0])
pool = latest.copy()
rank = int((pool["target"] > built).sum()) + 1
near = pool.iloc[(pool["target"] - built).abs().argsort()[:3]]
with right:
    fig = go.Figure()
    fig.add_scatter(x=pool["target"], y=np.zeros(len(pool)), mode="markers", name=f"Real schools, fall {last_year}",
                    marker=dict(size=12, color=[MAROON if u == Y.BC_ID else LIGHT for u in pool["UNITID"]],
                                line=dict(width=0)),
                    customdata=pool[["school"]], hovertemplate="%{customdata[0]}: %{x:.1f}%<extra></extra>")
    fig.add_scatter(x=[built], y=[0], mode="markers+text", name="Your school", text=[f"<b>Your school</b><br>{built:.1f}%"],
                    textposition="top center", textfont=dict(color=GOLD, size=14),
                    marker=dict(size=26, symbol="star", color=GOLD, line=dict(color=INK, width=1)))
    fig.update_yaxes(visible=False, range=[-1, 1.6])
    fig.update_xaxes(title="Yield", ticksuffix="%")
    st.plotly_chart(style(fig, 260, legend=False), width="stretch")
    st.markdown(f'<div class="source">Gold star = your school · maroon = Boston College · gray = real schools, '
                f'fall {last_year}</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    stat(c1, f"{built:.1f}%", "predicted yield for your school")
    stat(c2, f"#{rank} of {len(pool) + 1}", f"where it would rank in fall {last_year}")
    takeaway("Closest real schools: " + ", ".join(f"{r.school} ({r.target:.0f}%)" for r in near.itertuples())
             + f". A school with BC's exact profile would be expected at {built_bc:.1f}%; BC's actual was "
             f"{bc_last:.1f}%.")

# ---------------------------------------------------------------- 7. class planner
adm = F.admissions()
bc_adm = adm[(adm["unitid"] == Y.BC_ID)].sort_values("year")
bc_adm = bc_adm[bc_adm["year"] == bc_adm["year"].max()].iloc[0] if len(bc_adm) else None
default_class = int(round(bc_adm["number_enrolled"], -1)) if bc_adm is not None else 2400
default_apps = int(round(bc_adm["number_applied"], -2)) if bc_adm is not None else 36000
chapter(7, "planner", "How many offers does BC need to make?",
        "Yield decides how many students BC has to admit to fill a class. Set a class size, the number of "
        "applicants and an expected yield. The yield slider starts at our forecast.")
left, right = st.columns([2, 3])
with left:
    target_class = st.slider("Target first-year class", 1500, 3500, default_class, 10)
    apps_n = st.slider("Applications", 20000, 60000, default_apps, 500, format="%d")
    y_exp = st.slider("Expected yield (%)", 20.0, 60.0, float(round(bc_forecast * 2) / 2), 0.5, format="%.1f%%")
admits = target_class / (y_exp / 100)
with right:
    ys = np.linspace(20, 60, 81)
    fig = go.Figure()
    fig.add_scatter(x=ys, y=target_class / (ys / 100), mode="lines", name="Offers needed",
                    line=dict(color=INK, width=2.5), hovertemplate="Yield %{x:.1f}%: %{y:,.0f} offers<extra></extra>")
    fig.add_scatter(x=[y_exp], y=[admits], mode="markers", name="Your plan",
                    marker=dict(size=16, color=GOLD, symbol="star", line=dict(color=INK, width=1)))
    if bc_adm is not None:
        fig.add_scatter(x=[bc_last], y=[bc_adm["number_admitted"]], mode="markers", name=f"BC, fall {last_year}",
                        marker=dict(size=12, color=MAROON))
    fig.update_xaxes(title="Yield", ticksuffix="%")
    fig.update_yaxes(title="Students admitted", tickformat=",")
    st.plotly_chart(style(fig, 380), width="stretch")
c1, c2, c3 = st.columns(3)
stat(c1, f"{admits:,.0f}", "students to admit")
stat(c2, f"{admits / apps_n:.1%}", "admit rate that implies")
stat(c3, f"{target_class / ((y_exp - 1) / 100) - admits:+,.0f}", "extra offers if yield comes in 1 pt lower")
takeaway(f"This is why yield matters for planning: at {y_exp:.0f}% yield, missing by one point means about "
         f"{target_class / ((y_exp - 1) / 100) - admits:,.0f} extra offers for the same class, which pushes the admit "
         "rate up.")

st.markdown("<br><br>", unsafe_allow_html=True)
st.caption(f"Methods: yield = first-year enrollees / admitted students (IPEDS), fall {first_year}-{last_year}, "
           f"{n_schools} schools ({'private nonprofit 4-year colleges with admit rate under 50% and 2,000+ undergrads in fall 2019, plus BC and 14 peers' if big_sample else 'BC and 14 peers'}). "
           "Predictors from College Scorecard, using last year's values; dollar amounts logged; admit rate excluded. "
           "Within model: school + year fixed effects, variables chosen by forward selection with an expanding-window "
           "test. Across model: year fixed effects, region and school traits. Standard errors clustered by school "
           "(statsmodels). Effects are yield points per 1 SD.")
