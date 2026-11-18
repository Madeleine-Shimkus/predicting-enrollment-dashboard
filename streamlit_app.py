"""Why do admitted students choose Boston College? A scrolling data story about yield.

Run it locally with:  streamlit run streamlit_app.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import features as F
import tts_model as T
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
def get_tts(data_version):
    """The train/test-split model (Yield_TTS notebook): fit once, reused on every slider move."""
    tdata, traw, tfeats, _ = T.load()
    return tdata, traw, tfeats, T.evaluate(tdata, tfeats)


data, within, across, fitted, explained = get_data(Y.version())
tdata, traw, tfeats, TT = get_tts(T.version())
next_year = int(tdata["fall"].max()) + 1
mae = float(TT["scores"].loc["Blend", "MAE"])


def latest_raw(uid):
    """A school's real values for the latest fall (what next fall's prediction uses)."""
    r = traw[(traw["UNITID"] == uid) & (traw["fall"] == next_year - 1)]
    return r.iloc[0][tfeats].to_dict() if len(r) else {f: np.nan for f in tfeats}


def predict_raw(uid, raw_values):
    row = {f: (T.transform(f, v) if pd.notna(v) else np.nan) for f, v in raw_values.items()}
    row.update(fall=next_year - 1, UNITID=uid)   # next fall uses the latest year's effect
    return float(TT["model"].predict(pd.DataFrame([row]))[0])


bc_forecast = predict_raw(Y.BC_ID, latest_raw(Y.BC_ID))

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
stat(c4, f"{bc_forecast:.0f}%", f"Expected yield, fall {next_year}")

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
    st.plotly_chart(style(fig, 450), width="stretch", config={"displayModeBar": False})
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
st.plotly_chart(style(fig, 420, legend=False), width="stretch", config={"displayModeBar": False})
st.markdown('<div class="source">Maroon = Boston College · dark gray = the 14 original peers'
            + (" · light gray = other selective private colleges" if big_sample else "") + "</div>",
            unsafe_allow_html=True)

# ---------------------------------------------------------------- 3. which factors matter
imp = TT["importance"]
SHORT = {"6-yr grad rate": "graduation rate", "Avg faculty salary": "faculty salary", "Net price (all)": "cost after aid",
         "Net price $0-30k": "cost for low-income families", "Federal loan share": "share of students with loans",
         "Median debt": "student debt", "Instruction $/student": "instruction spending",
         "Student-faculty ratio": "students per faculty member"}
top_f = imp.index[0]
# each college's average (fall 2010 on) of each factor next to its average yield
yrs = traw[traw["fall"] >= T.MIN_YEAR - 1]
avg_f = yrs.groupby("UNITID")[tfeats].mean()
avg_y = tdata.groupby(["UNITID", "school"])["y"].mean().reset_index().set_index("UNITID")
cmp = avg_y.join(avg_f).reset_index()
cmp["group"] = np.where(cmp["UNITID"] == Y.BC_ID, "Boston College",
                        np.where(cmp["UNITID"].isin(Y.PEER_IDS), "Original peer", "Other selective private"))
top_r = cmp[top_f].corr(cmp["y"])
chapter(3, "factors", f"Colleges with a {'higher' if top_r > 0 else 'lower'} {SHORT[top_f]} keep more of the "
        "students they admit",
        "Of the factors we looked at, these do the most to set one college's yield apart from another's. "
        "Pick one to see how all the colleges compare.")
left, right = st.columns([2, 3])
with left:
    strength = imp.clip(lower=0)
    label = np.where(strength >= strength.max() * 0.5, "Strong",
                     np.where(strength >= strength.max() * 0.05, "Some", "Little"))
    i = strength.iloc[::-1]
    lab = label[::-1]
    fig = go.Figure(go.Bar(x=i.values, y=[SHORT[v].capitalize() for v in i.index], orientation="h",
                           marker_line_width=0, marker_color=[MAROON if l == "Strong" else (DARK_PEER if l == "Some" else LIGHT)
                                                              for l in lab],
                           text=list(lab), textposition="outside", cliponaxis=False, hoverinfo="skip"))
    fig.update_xaxes(visible=False, range=[0, i.max() * 1.35])
    fig.update_layout(title=dict(text="How much each factor matters", font=dict(size=14, color="#555")))
    st.plotly_chart(style(fig, 380, legend=False), width="stretch", config={"displayModeBar": False})
with right:
    pick = st.selectbox("Compare colleges by", list(imp.index), format_func=lambda v: SHORT[v].capitalize(), key="f_pick")
    d = cmp.dropna(subset=[pick, "y"]).copy()
    share = pick in T.SHARES
    d["x"] = d[pick] * 100 if share else d[pick]
    fig = go.Figure()
    for grp, color, size in [("Other selective private", LIGHT, 9), ("Original peer", DARK_PEER, 10),
                             ("Boston College", MAROON, 15)]:
        part = d[d["group"] == grp]
        fig.add_scatter(x=part["x"], y=part["y"], mode="markers+text" if grp == "Boston College" else "markers",
                        text=part["school"] if grp == "Boston College" else None, textposition="top center",
                        textfont=dict(color=MAROON, size=13), marker=dict(color=color, size=size, line=dict(width=0)),
                        customdata=part[["school"]],
                        hovertemplate="%{customdata[0]}<br>%{x:,.0f}" + ("%" if share else "") +
                                      "<br>Yield %{y:.0f}%<extra></extra>")
    bfit = np.polyfit(d["x"], d["y"], 1)
    xs = np.linspace(d["x"].min(), d["x"].max(), 50)
    fig.add_scatter(x=xs, y=np.polyval(bfit, xs), mode="lines", line=dict(color=INK, width=1.5, dash="dot"),
                    hoverinfo="skip")
    dollars = pick not in T.SHARES and pick != "Student-faculty ratio"
    fig.update_xaxes(title=T.FRIENDLY[pick] + (" (%)" if share else ""), tickprefix="$" if dollars else "",
                     ticksuffix="%" if share else "")
    fig.update_yaxes(title="Average yield", ticksuffix="%")
    st.plotly_chart(style(fig, 400, legend=False), width="stretch", config={"displayModeBar": False})
r = d["x"].corr(d["y"])
bcrow = d[d["UNITID"] == Y.BC_ID]
bc_txt = ""
if len(bcrow):
    bx = float(bcrow["x"].iloc[0])
    bc_txt = (f" BC's {SHORT[pick]} averages {'$' if dollars else ''}{bx:,.0f}{'%' if share else ''}, "
              f"{'above' if bx > d['x'].median() else 'below'} the typical college here.")
direction = "higher" if r > 0 else "lower"
takeaway(f"Colleges with a higher {SHORT[pick]} tend to have <b>{direction}</b> yield"
         + (" (a clear pattern)." if abs(r) >= 0.4 else " (a loose pattern)." if abs(r) >= 0.2 else ", but only barely.")
         + bc_txt + " Each dot is one college, averaged over fall 2010 to 2023.")
st.markdown('<div class="source">Maroon = Boston College · dark gray = the 14 original peers · light gray = other '
            'selective private colleges</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- 4. year to year
sig_w = within[within["p_value"] < 0.05]
top = within.iloc[0]
chapter(4, "within", "Within the same college, what moved yield from one year to the next",
        "This compares each college only with itself. When one of these was higher than usual, yield the "
        "following fall tended to rise (maroon) or fall (blue). Gray means no clear pattern.")


def effect_bars(e, height):
    e = e.iloc[::-1]
    sd_text = [Y.describe_sd(data, v) if v in data.columns and v not in Y.TRAITS else "yes vs. no" for v in e["variable"]]
    names_ = [Y.plain(v).capitalize() for v in e["variable"]]
    fig = go.Figure(go.Bar(
        x=e["effect"], y=names_, orientation="h", marker_line_width=0,
        marker_color=[(MAROON if v > 0 else POS) if p < 0.05 else LIGHT for v, p in zip(e["effect"], e["p_value"])],
        text=[f"{v:+.1f} pts" for v in e["effect"]], textposition="outside", cliponaxis=False,
        customdata=np.c_[sd_text],
        hovertemplate="<b>%{y}</b><br>Yield moved %{x:+.1f} pts when this was<br>%{customdata[0]} than usual"
                      "<extra></extra>"))
    lim = max(e["effect"].abs().max() * 1.4, 1)
    fig.add_vline(x=0, line=dict(color=INK, width=1))
    fig.update_xaxes(range=[-lim, lim], title="Change in next fall's yield (points)")
    fig.update_yaxes(categoryorder="array", categoryarray=names_)
    return style(fig, height, legend=False)


st.plotly_chart(effect_bars(within, 120 + 42 * len(within)), width="stretch", config={"displayModeBar": False})
ups = [Y.plain(r.variable) for r in sig_w.itertuples() if r.effect > 0]
downs = [Y.plain(r.variable) for r in sig_w.itertuples() if r.effect < 0]
takeaway(("When " + " or ".join(ups) + " went up, more admitted students enrolled the next year. " if ups else "")
         + ("When " + " or ".join(downs) + " went up, fewer did. " if downs else "")
         + "Hover a bar to see what a typical change looks like in real numbers.")

# ---------------------------------------------------------------- 5. predict BC
def raw_slider(var, v, fallback, key):
    """Slider in real units starting at the real value. Untouched -> returns the real value exactly
    (even if it's missing, so the model handles it the same way the notebook does)."""
    lab = T.FRIENDLY.get(var, var)
    start_val = v if pd.notna(v) else fallback
    if pd.isna(start_val):
        return v
    if var in T.SHARES:
        x = start_val * 100
        start = float(round(x * 2) / 2)
        out = st.slider(f"{lab} (%)", max(0.0, round(x) - 15.0), min(100.0, round(x) + 15.0), start, 0.5,
                        key=key, format="%.1f%%")
        return v if out == start else out / 100
    if var == "Student-faculty ratio":
        start = int(round(start_val))
        out = st.slider(lab, max(3, start - 6), start + 6, start, 1, key=key)
        return v if out == start else float(out)
    k = start_val / 1000          # dollars, shown in thousands
    step = 0.1 if k < 20 else 0.5
    lo, hi = round(k * 0.6 / step) * step, round(k * 1.4 / step) * step
    start = min(max(round(k / step) * step, lo), hi)
    out = st.slider(lab, lo, hi, start, step, key=key, format="$%.1fk")
    return v if out == start else out * 1000


LEVERS = ["Net price $0-30k", "Net price (all)", "Federal loan share", "6-yr grad rate"]
bc_real = latest_raw(Y.BC_ID)
bc_hist_raw = traw[traw["UNITID"] == Y.BC_ID].sort_values("fall")
chapter(5, "forecast", "What would change BC's yield next fall?",
        f"Based on how these factors line up with yield across {n_schools} colleges, we expect about "
        f"{bc_forecast:.0f}% of the students BC admits for fall {next_year} to enroll. The sliders start at BC's "
        f"real fall {next_year - 1} numbers. Move one to see how the expected yield shifts.")
left, right = st.columns([2, 3])
with left:
    st.markdown("**What if BC changed...**")
    scen = {}
    for v in [f for f in LEVERS if f in tfeats]:
        scen[v] = raw_slider(v, bc_real[v], bc_hist_raw[v].dropna().iloc[-1] if bc_hist_raw[v].notna().any() else np.nan,
                             f"tt_{v}")
    with st.expander("More factors the model uses"):
        for v in [f for f in tfeats if f not in LEVERS]:
            fb = bc_hist_raw[v].dropna().iloc[-1] if bc_hist_raw[v].notna().any() else np.nan
            scen[v] = raw_slider(v, bc_real[v], fb, f"tt_{v}")
    if st.button("Back to real numbers"):
        for k in [k for k in st.session_state if str(k).startswith("tt_")]:
            del st.session_state[k]
        st.rerun()
scen_fc = predict_raw(Y.BC_ID, scen)
changed = abs(scen_fc - bc_forecast) >= 0.05
hist = data[data["UNITID"] == Y.BC_ID].sort_values("fall")
last_val = float(hist["target"].iloc[-1])
with right:
    fig = go.Figure()
    fig.add_scatter(x=hist["fall"], y=hist["target"], mode="lines+markers", line=dict(color=MAROON, width=3.5),
                    marker=dict(size=7), hovertemplate="Fall %{x}: %{y:.1f}%<extra></extra>")
    fig.add_scatter(x=[last_year, next_year], y=[last_val, bc_forecast], mode="lines+markers",
                    line=dict(color=MAROON if not changed else MUTED, width=3, dash="dot"),
                    marker=dict(size=[0, 13], color="white", line=dict(color=MAROON if not changed else MUTED, width=3)),
                    hovertemplate="Expected: %{y:.1f}%<extra></extra>")
    fig.add_annotation(x=last_year, y=last_val, text=f"Fall {last_year}<br><b>{last_val:.0f}%</b>", showarrow=False,
                       xanchor="right", xshift=-10, font=dict(size=13, color=MAROON))
    up = scen_fc >= bc_forecast
    if changed:
        fig.add_scatter(x=[last_year, next_year], y=[last_val, scen_fc], mode="lines+markers",
                        line=dict(color=GOLD, width=3, dash="dot"), marker=dict(size=[0, 14], color=GOLD),
                        hovertemplate="With your changes: %{y:.1f}%<extra></extra>")
        fig.add_annotation(x=next_year, y=scen_fc, xanchor="left", xshift=12, yshift=16 if up else -16,
                           showarrow=False, align="left", text=f"Your changes<br><b>{scen_fc:.0f}%</b>",
                           font=dict(size=13, color="#9A7414"))
    fig.add_annotation(x=next_year, y=bc_forecast, xanchor="left", xshift=12, showarrow=False, align="left",
                       yshift=(-16 if up else 16) if changed else 0,
                       text=(f"Expected {next_year}<br><b>{bc_forecast:.0f}%</b>" if not changed
                             else f"No changes<br>{bc_forecast:.0f}%"),
                       font=dict(size=13, color=INK if not changed else "#888"))
    ticks = list(range(int(hist["fall"].min()), next_year + 1, 2))
    fig.update_xaxes(range=[hist["fall"].min() - 0.5, next_year + 5.5], tickvals=ticks, showline=False)
    fig.update_yaxes(title=None, ticksuffix="%",
                     range=[min(hist["target"].min(), scen_fc, bc_forecast) - 4,
                            max(hist["target"].max(), scen_fc, bc_forecast) + 6])
    fig.update_layout(title=dict(text="Share of admitted students who enrolled at BC", font=dict(size=14, color="#555")))
    st.plotly_chart(style(fig, 430, legend=False), width="stretch", config={"displayModeBar": False})
c1, c2, c3 = st.columns(3)
stat(c1, f"{bc_forecast:.0f}%", f"expected yield, fall {next_year}")
stat(c2, f"{bc_forecast - mae:.0f}–{bc_forecast + mae:.0f}%", "likely range")
stat(c3, f"{scen_fc:.0f}%" if changed else "–",
     f"with your changes ({scen_fc - bc_forecast:+.1f} pts)" if changed else "with your changes: move a slider")

# ---------------------------------------------------------------- 7. class planner
adm = F.admissions()
bc_adm = adm[(adm["unitid"] == Y.BC_ID)].sort_values("year")
bc_adm = bc_adm[bc_adm["year"] == bc_adm["year"].max()].iloc[0] if len(bc_adm) else None
default_class = int(round(bc_adm["number_enrolled"], -1)) if bc_adm is not None else 2400
default_apps = int(round(bc_adm["number_applied"], -2)) if bc_adm is not None else 36000
chapter(6, "planner", "How many offers does BC need to make?",
        "Yield decides how many students BC has to admit to fill a class. Set a class size, the number of "
        "applicants and an expected yield. The yield slider starts at our model's prediction.")
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
    st.plotly_chart(style(fig, 380), width="stretch", config={"displayModeBar": False})
c1, c2, c3 = st.columns(3)
stat(c1, f"{admits:,.0f}", "students to admit")
stat(c2, f"{admits / apps_n:.1%}", "admit rate that implies")
stat(c3, f"{target_class / ((y_exp - 1) / 100) - admits:+,.0f}", "extra offers if yield comes in 1 pt lower")
takeaway(f"This is why yield matters for planning: at {y_exp:.0f}% yield, missing by one point means about "
         f"{target_class / ((y_exp - 1) / 100) - admits:,.0f} extra offers for the same class, which pushes the admit "
         "rate up.")

# ---------------------------------------------------------------- about the prediction
bs = TT["by_school"].copy()
n_test = len(bs)
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("About the expected yield: how we checked it"):
    st.markdown(f'<div class="note">The expected yield in section 5 comes from a model that looks at last year\'s '
                f'numbers for each college. To check it, we hid {n_test} colleges (including BC) while it learned '
                f'from the rest, then asked it to estimate the hidden ones. It was off by about {mae:.0f} points on '
                f'average, compared with {TT["scores"].loc["Training mean", "MAE"]:.0f} points for just guessing the '
                'average college. Each dot below is one hidden college.</div>', unsafe_allow_html=True)
    fig = go.Figure()
    lo = min(bs["actual"].min(), bs["predicted"].min()) - 4
    hi = max(bs["actual"].max(), bs["predicted"].max()) + 4
    fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="#BBBBBB", width=1.5, dash="dash"),
                    hoverinfo="skip")
    fig.add_annotation(x=hi, y=hi, text="Exactly right", showarrow=False, xanchor="right", yshift=10,
                       font=dict(size=12, color="#888"))
    other = bs[bs["UNITID"] != Y.BC_ID]
    fig.add_scatter(x=other["actual"], y=other["predicted"], mode="markers",
                    marker=dict(size=11, color=DARK_PEER, opacity=0.8, line=dict(width=0)), customdata=other[["school"]],
                    hovertemplate="%{customdata[0]}<br>Real: %{x:.0f}%<br>Estimated: %{y:.0f}%<extra></extra>")
    b = bs[bs["UNITID"] == Y.BC_ID]
    fig.add_scatter(x=b["actual"], y=b["predicted"], mode="markers+text", text=["Boston College"],
                    textposition="middle left", textfont=dict(color=MAROON, size=13), marker=dict(size=15, color=MAROON),
                    hovertemplate="Boston College<br>Real: %{x:.0f}%<br>Estimated: %{y:.0f}%<extra></extra>")
    fig.update_xaxes(title="Real average yield", ticksuffix="%", range=[lo, hi])
    fig.update_yaxes(title="Estimated yield", ticksuffix="%", range=[lo, hi])
    st.plotly_chart(style(fig, 420, legend=False), width="stretch", config={"displayModeBar": False})

st.markdown("<br><br>", unsafe_allow_html=True)
st.caption(f"Methods: yield = first-year enrollees / admitted students (IPEDS), fall {first_year}-{last_year}, "
           f"{n_schools} private nonprofit 4-year colleges (admit rate under 50% and 2,000+ undergrads in fall 2019, "
           "plus BC and 14 peers). Factors from College Scorecard, using last year's values; admit rate left out. "
           "Section 3 and 5: Ridge regression + gradient boosting (scikit-learn), averaged, trained with a school-level "
           "80/20 train/test split; factor strength is permutation importance on the held-out colleges. Section 4: "
           "school + year fixed effects (statsmodels), effects per 1 standard deviation.")
