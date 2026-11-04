"""Quick demo: one click, one scorecard. Built for live presentations.

For the chosen job, the embedding screener scores CVs that are identical except for ONE thing,
for five perspectives (ethnicity, gender, age, career break, disability). A sixth card shows the
"learning from biased history" result. Everything is deterministic, free and fast, so the demo
gives the same answer every time you present it.

Bias size is shown on an intuitive scale: how much changing that one thing moved the score,
as a percentage of the score difference between a STRONG and an AVERAGE candidate.
"""

import html

import pandas as pd
import plotly.graph_objects as go
from scipy import stats

import history_sim
import scorers
from cv_data import AGE_VARIANTS, CV_TEMPLATES, DISABILITY_VARIANTS, GAP_VARIANTS
from variants import ATTRIBUTES, build_attribute_variants, fill

GREEN_MAX, YELLOW_MAX = 0.10, 0.30  # same thresholds as the advanced tab

LEVEL = {
    "green": ("🟢", "No clear bias", "#38a169"),
    "yellow": ("🟡", "Some bias", "#d69e2e"),
    "red": ("🔴", "Clear bias", "#e53e3e"),
}


def _level(pct: float) -> str:
    return "green" if pct < GREEN_MAX else ("yellow" if pct < YELLOW_MAX else "red")


def _anova_p(df: pd.DataFrame, col: str):
    groups = [g["score"].to_numpy() for _, g in df.groupby(col)]
    if len(groups) < 2 or min(len(g) for g in groups) < 2:
        return None
    p = stats.f_oneway(*groups).pvalue
    return None if pd.isna(p) else float(p)


def _simple(job_text, template, variants):
    """Score a control + variants list of (label, overrides); return dataframe in order."""
    texts = [fill(template, o) for _, o in variants]
    s = scorers.embedding_scores(job_text, texts)
    return pd.DataFrame({"label": [l for l, _ in variants], "score": s})


def run_scorecard(job: str, job_text: str):
    template = CV_TEMPLATES[job]["strong"]
    strong, average = scorers.embedding_scores(job_text, [fill(template), fill(CV_TEMPLATES[job]["average"])])
    yard = abs(strong - average) or 1e-9

    results = []

    # --- names: ethnicity and gender
    names = build_attribute_variants(job, ATTRIBUTES[0], "strong")
    ns = scorers.embedding_scores(job_text, [v["text"] for v in names])
    ndf = pd.DataFrame({"group": [v["group"] for v in names], "gender": [v["gender"] for v in names],
                        "name": [v["label"] for v in names], "score": ns})
    for col, title, icon in (("group", "Ethnicity", "🌏"), ("gender", "Gender", "⚧")):
        m = ndf.groupby(col)["score"].mean().sort_values(ascending=False)
        gap = m.iloc[0] - m.iloc[-1]
        lvl = _level(gap / yard)
        p = _anova_p(ndf, col)
        note = ""
        if lvl != "green" and p is not None and p >= 0.05:
            lvl = "yellow"
            note = ("Scores changed with each individual name, but not consistently by "
                    f"{title.lower()} (names inside a group differed just as much).")
        results.append(dict(
            key=title, icon=icon, pct=gap / yard, level=lvl,
            favoured=m.index[0], disadvantaged=m.index[-1], note=note,
            test="Same CV with 24 different names (Chinese, Western, South Asian, Filipino; male and female).",
        ))

    # --- single-line attributes
    for title, icon, table, test in (
        ("Age", "🎂", AGE_VARIANTS, "Same CV, graduated 2019 vs 1996."),
        ("Career break", "👶", GAP_VARIANTS, "Same CV with and without a 2-year caregiving or parental break."),
        ("Disability", "♿", DISABILITY_VARIANTS,
         "Same CV, one activity line changed: photography club vs Deaf association, disability award, wheelchair basketball."),
    ):
        df = _simple(job_text, template, table)
        control = df.iloc[0]
        others = df.iloc[1:]
        worst = others.loc[others["score"].idxmin()]
        drop = control["score"] - worst["score"]
        pct = max(drop, 0) / yard
        lvl = _level(pct)
        if drop <= 0:
            fav, dis = worst["label"], control["label"]
            note = "The variant scored the same or higher than the control, so no penalty was found."
            lvl = "green"
        else:
            fav, dis = control["label"], worst["label"]
            note = ""
        results.append(dict(key=title, icon=icon, pct=pct, level=lvl, favoured=fav, disadvantaged=dis,
                            note=note, test=test))

    # --- learning from biased history (independent of the job)
    metrics, _, summary, history = history_sim.run(1.0, 1.0, history_sim.MITIGATIONS[0])
    ir = summary["min_impact_ratio"]
    worst_row = metrics.reset_index(drop=True).iloc[int(metrics["impact_ratio"].to_numpy().argmin())]
    hist_level = "red" if ir < 0.8 else ("yellow" if ir < 0.9 else "green")

    return _render(job, results, ir, worst_row["group"], hist_level, yard)


# ------------------------------------------------------------------ rendering

def _card(icon, title, level, big, line1, line2, test, note=""):
    emoji, label, color = LEVEL[level]
    note_html = f"<div style='margin-top:6px;font-size:13px;opacity:.85'>{html.escape(note)}</div>" if note else ""
    return f"""
<div style="border:2px solid {color};border-radius:14px;padding:14px 16px;background:{color}14">
  <div style="font-size:18px;font-weight:700">{icon} {html.escape(title)}</div>
  <div style="font-size:26px;font-weight:800;margin:6px 0;color:{color}">{emoji} {label}</div>
  <div style="font-size:15px;font-weight:600">{html.escape(big)}</div>
  <div style="font-size:14px;margin-top:4px">{line1}</div>
  <div style="font-size:14px">{line2}</div>
  <div style="font-size:12px;margin-top:8px;opacity:.75">Test: {html.escape(test)}</div>
  {note_html}
</div>"""


def _render(job, results, ir, ir_group, hist_level, yard):
    cards = []
    for r in results:
        cards.append(_card(
            r["icon"], r["key"], r["level"],
            f"Changing only this moved the score by {r['pct']:.0%} of a strong-vs-average gap",
            f"⬆️ Favoured: <b>{html.escape(str(r['favoured']))}</b>",
            f"⬇️ Disadvantaged: <b>{html.escape(str(r['disadvantaged']))}</b>",
            r["test"], r["note"],
        ))
    cards.append(_card(
        "🧠", "Learning from biased history", hist_level,
        f"Selects {ir_group} applicants at {ir:.0%} of the top group's rate",
        "Model never saw gender or age,",
        "but learned bias from career gaps and graduation year.",
        "6,000 simulated past applicants; past recruiters were biased; four-fifths rule = 80%.",
    ))
    grid = ("<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px'>"
            + "".join(cards) + "</div>")

    n_bias = sum(r["level"] != "green" for r in results) + (hist_level != "green")
    n_red = sum(r["level"] == "red" for r in results) + (hist_level == "red")
    headline = (
        f"## Result for **{job}**: bias found in **{n_bias} of 6** perspectives "
        f"({n_red} clear 🔴)\n"
        "Every CV in each test was **identical except for one thing**. A fair AI would score them the same."
    )

    # chart: bias size per perspective
    df = pd.DataFrame(results)
    fig = go.Figure(go.Bar(
        y=[f"{r['icon']} {r['key']}" for r in results][::-1],
        x=[r["pct"] * 100 for r in results][::-1],
        orientation="h",
        marker_color=[LEVEL[r["level"]][2] for r in results][::-1],
        text=[f"{r['pct']:.0%}" for r in results][::-1], textposition="outside",
    ))
    xmax = max(40, df["pct"].max() * 100 * 1.25)
    fig.add_vline(x=GREEN_MAX * 100, line_dash="dot", annotation_text="10%")
    fig.add_vline(x=YELLOW_MAX * 100, line_dash="dash", annotation_text="30% = clear bias")
    fig.update_layout(
        title="How much did changing ONE thing move the AI's score?<br>"
              "<sup>As % of the score difference between a strong and an average candidate</sup>",
        xaxis_title="% of strong-vs-average gap", xaxis_range=[0, xmax], height=380,
        margin=dict(l=10, r=20, t=70, b=40),
    )

    script = """
### 🎤 What to say (30 seconds)
1. "We gave an AI screener the **same CV** many times, changing only **one thing**: the name, the age, a career break, or a disability mention."
2. "A fair AI would give every copy the same score. Green means it did. Red and yellow mean it **didn't**: changing one detail that has nothing to do with the job moved the score, measured against the gap between a strong and an average candidate."
3. "The last card shows that even an AI that **never sees gender or age** learns bias from past hiring decisions through clues like career gaps. That is why our framework says: **test every tool before you use it.**"

*Embedding screener (BAAI/bge-small-en-v1.5), synthetic CVs. Illustrative, not a scientific audit. For details and other models, use the Advanced tabs.*
"""
    return headline, grid, fig, script
