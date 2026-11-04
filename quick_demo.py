"""Quick demo: one click, one scorecard. Built for live presentations.

For the chosen job, the embedding screener scores CVs that are identical except for ONE thing,
for five perspectives (ethnicity, gender, age, career break, disability). A sixth card shows the
"learning from biased history" result. Everything is deterministic, free and fast, so the demo
gives the same answer every time you present it.

Everything is shown as the AI's own score (0-100), so the audience sees plain numbers:
"same CV, Western name: 78.4; Chinese name: 76.9". As a yardstick we also show how many points a
clearly weaker (average) CV loses, so the size of each gap is easy to judge.
"""

import html

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

import history_sim
import scorers
from cv_data import AGE_VARIANTS, CV_TEMPLATES, DISABILITY_VARIANTS, GAP_VARIANTS
from variants import ATTRIBUTES, build_attribute_variants, fill

# Colour thresholds: gap as a share of the strong-vs-average CV gap (same as the advanced tab).
# Used only to choose the colour; the page itself shows points.
GREEN_MAX, YELLOW_MAX = 0.10, 0.30

LEVEL = {
    "green": ("🟢", "No clear bias", "#38a169"),
    "yellow": ("🟡", "Some bias", "#d69e2e"),
    "red": ("🔴", "Clear bias", "#e53e3e"),
}

FEATURE_LABEL = {
    "skill_test": "Skill test score",
    "years_experience": "Years of experience",
    "education_level": "Education level",
    "career_gap": "Career gap (gender clue)",
    "grad_year": "Graduation year (age clue)",
}


def _level(ratio: float) -> str:
    return "green" if ratio < GREEN_MAX else ("yellow" if ratio < YELLOW_MAX else "red")


def _anova_p(df: pd.DataFrame, col: str):
    groups = [g["score"].to_numpy() for _, g in df.groupby(col)]
    if len(groups) < 2 or min(len(g) for g in groups) < 2:
        return None
    p = stats.f_oneway(*groups).pvalue
    return None if pd.isna(p) else float(p)


def _simple(job_text, template, variants):
    texts = [fill(template, o) for _, o in variants]
    s = scorers.embedding_scores(job_text, texts)
    return pd.DataFrame({"label": [l for l, _ in variants], "score": s})


def _short(label: str) -> str:
    """'Graduated 1996 (about 51 years old)' -> 'Graduated 1996'."""
    return label.split(" (")[0]


def run_scorecard(job: str, job_text: str):
    template = CV_TEMPLATES[job]["strong"]
    strong, average = scorers.embedding_scores(job_text, [fill(template), fill(CV_TEMPLATES[job]["average"])])
    strong, average = round(strong, 1), round(average, 1)
    yard = abs(strong - average) or 0.1

    results = []

    # --- names: ethnicity and gender
    names = build_attribute_variants(job, ATTRIBUTES[0], "strong")
    ns = scorers.embedding_scores(job_text, [v["text"] for v in names])
    ndf = pd.DataFrame({"group": [v["group"] for v in names], "gender": [v["gender"] for v in names],
                        "score": ns})
    for col, title, icon, what in (("group", "Ethnicity", "🌏", "names"), ("gender", "Gender", "⚧", "names")):
        m = ndf.groupby(col)["score"].mean().sort_values(ascending=False).round(1)
        hi, lo = float(m.iloc[0]), float(m.iloc[-1])
        gap = round(hi - lo, 1)
        lvl = _level(gap / yard)
        p = _anova_p(ndf, col)
        mixed = lvl != "green" and p is not None and p >= 0.05
        if mixed:
            lvl = "yellow"
        results.append(dict(
            key=title, icon=icon, level=lvl, gap=gap, mixed=mixed,
            fav=f"{m.index[0]} {what}", fav_score=hi, dis=f"{m.index[-1]} {what}", dis_score=lo,
            test="Same CV, 24 different names (Chinese, Western, South Asian, Filipino; male and female).",
        ))

    # --- single-line attributes
    for title, icon, table, test in (
        ("Age", "🎂", AGE_VARIANTS, "Same CV, graduated 2019 vs 1996."),
        ("Career break", "👶", GAP_VARIANTS, "Same CV with and without a 2-year caregiving or parental break."),
        ("Disability", "♿", DISABILITY_VARIANTS,
         "Same CV, one activity line changed: photography club vs Deaf association, disability award, wheelchair basketball."),
    ):
        df = _simple(job_text, template, table)
        df["score"] = df["score"].round(1)
        control = df.iloc[0]
        worst = df.iloc[1:].loc[df.iloc[1:]["score"].idxmin()]
        gap = round(float(control["score"] - worst["score"]), 1)
        if gap <= 0:
            best = df.iloc[1:].loc[df.iloc[1:]["score"].idxmax()]
            results.append(dict(key=title, icon=icon, level="green", gap=0.0, mixed=False, no_penalty=True,
                                fav=_short(best["label"]), fav_score=float(best["score"]),
                                dis=_short(control["label"]), dis_score=float(control["score"]), test=test))
        else:
            results.append(dict(key=title, icon=icon, level=_level(gap / yard), gap=gap, mixed=False,
                                fav=_short(control["label"]), fav_score=float(control["score"]),
                                dis=_short(worst["label"]), dis_score=float(worst["score"]), test=test))

    # --- learning from biased history (independent of the job)
    metrics, _, summary, _ = history_sim.run(1.0, 1.0, history_sim.MITIGATIONS[0])
    hist_level = "red" if summary["min_impact_ratio"] < 0.8 else (
        "yellow" if summary["min_impact_ratio"] < 0.9 else "green")
    test_df, coefs = history_sim.blind_ai_scores(1.0, 1.0)
    q = test_df[test_df["truly_qualified"] == 1]
    hs = {**q.groupby("gender")["ai_score"].mean().round(0).to_dict(),
          **q.groupby("age_band")["ai_score"].mean().round(0).to_dict()}

    headline = _headline(job, results, hist_level)
    grid = _cards(results, hs, hist_level)
    insights = _insights(results, hs, strong, average)
    fig = _points_chart(results, strong, average)
    hist_md, hist_fig = _history_detail(hs, coefs)
    return headline, grid, insights, fig, hist_md, hist_fig, _script()


# ------------------------------------------------------------------ rendering

def _headline(job, results, hist_level):
    n_bias = sum(r["level"] != "green" for r in results) + (hist_level != "green")
    n_red = sum(r["level"] == "red" for r in results) + (hist_level == "red")
    return (
        f"## {job}: the AI scored identical CVs differently in **{n_bias} of 6** tests ({n_red} clear 🔴)\n"
        "In each test every CV was **identical except for one detail**. "
        "Scores are the AI's match score from 0 to 100. A fair AI would give every copy the same score."
    )


def _card(icon, title, level, big, line1, line2, test, note=""):
    emoji, label, color = LEVEL[level]
    note_html = f"<div style='margin-top:6px;font-size:13px;opacity:.85'>{html.escape(note)}</div>" if note else ""
    return f"""
<div style="border:2px solid {color};border-radius:14px;padding:14px 16px;background:{color}14">
  <div style="font-size:18px;font-weight:700">{icon} {html.escape(title)}</div>
  <div style="font-size:24px;font-weight:800;margin:6px 0;color:{color}">{emoji} {label}</div>
  <div style="font-size:15px">{line1}</div>
  <div style="font-size:15px">{line2}</div>
  <div style="font-size:16px;font-weight:700;margin-top:6px">{big}</div>
  <div style="font-size:12px;margin-top:8px;opacity:.75">Test: {html.escape(test)}</div>
  {note_html}
</div>"""


def _cards(results, hs, hist_level):
    cards = []
    for r in results:
        big = ("No penalty: the changed CV scored the same or higher." if r.get("no_penalty")
               else f"Gap: {r['gap']:.1f} points")
        note = ("Scores changed with each name, but names inside a group differed about as much, "
                "so the gap is not consistent for the whole group.") if r["mixed"] else ""
        cards.append(_card(
            r["icon"], r["key"], r["level"], big,
            f"⬆️ {html.escape(r['fav'])}: <b>{r['fav_score']:.1f}</b>",
            f"⬇️ {html.escape(r['dis'])}: <b>{r['dis_score']:.1f}</b>",
            r["test"], note,
        ))
    cards.append(_card(
        "🧠", "Learning from biased history", hist_level,
        f"Women {hs['Male'] - hs['Female']:.0f} points lower, over-40s {hs['Under 40'] - hs['40+']:.0f} points lower",
        f"⬆️ Qualified men: <b>{hs['Male']:.0f}</b> · under 40: <b>{hs['Under 40']:.0f}</b>",
        f"⬇️ Qualified women: <b>{hs['Female']:.0f}</b> · over 40: <b>{hs['40+']:.0f}</b>",
        "An AI trained on 6,000 past (biased) hiring decisions. It never saw gender or age. "
        "Score = AI's chance of hiring (0-100) for equally qualified applicants.",
    ))
    return ("<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px'>"
            + "".join(cards) + "</div>")


def _insights(results, hs, strong, average):
    yard = round(strong - average, 1)
    lines = ["### 💡 What this shows"]
    biased = [r for r in results if r["level"] != "green" and r["gap"] > 0]
    fair = [r for r in results if r["level"] == "green"]
    for r in sorted(biased, key=lambda r: -r["gap"]):
        avg = "on average " if r["key"] in ("Ethnicity", "Gender") else ""
        line = (f"- **{r['icon']} {r['key']}:** the same CV scored {avg}**{r['fav_score']:.1f}** with "
                f"*{r['fav']}* but **{r['dis_score']:.1f}** with *{r['dis']}*, "
                f"a gap of **{r['gap']:.1f} points**. Nothing about the person's skills changed.")
        if r["mixed"]:
            line += " The scores jumped around from name to name, so the AI reacts to names in an unpredictable way."
        lines.append(line)
    lines.append(
        f"- **🧠 Learning from history:** an AI that never saw gender or age still gave equally qualified "
        f"women **{hs['Female']:.0f}** vs men **{hs['Male']:.0f}**, and over-40s **{hs['40+']:.0f}** vs under-40s "
        f"**{hs['Under 40']:.0f}**. It copied the bias of past recruiters through clues like career gaps and "
        "graduation year."
    )
    if fair:
        lines.append("- **No clear bias** for " + ", ".join(f"{r['icon']} {r['key'].lower()}" for r in fair)
                     + " in this job. A clean result for one job does not mean the tool is fair for every job.")
    if biased:
        top = max(biased, key=lambda r: r["gap"])
        lines.append(
            f"- **How big is a gap?** For comparison, a clearly stronger CV scored **{strong:.1f}** and an average CV "
            f"scored **{average:.1f}**, **{yard:.1f} points** apart. The biggest one-detail gap here "
            f"({top['icon']} {top['key'].lower()}, **{top['gap']:.1f} points**) is {_size_words(top['gap'] / yard)} "
            "that difference, from a detail that says nothing about the person's ability."
        )
    lines.append("- **Take-away:** the AI's score moved because of details that have nothing to do with the job. "
                 "That is why every AI hiring tool must be tested like this before it is used, and tested again "
                 "for each new job.")
    return "\n".join(lines)


def _size_words(r: float) -> str:
    for limit, words in ((0.15, "about a tenth of"), (0.3, "about a quarter of"), (0.42, "about a third of"),
                         (0.65, "about half of"), (0.9, "nearly as big as"), (1.1, "as big as")):
        if r < limit:
            return words
    return "bigger than"


def _points_chart(results, strong, average):
    labels = ["📏 Strong CV vs average CV (for comparison)"] + [f"{r['icon']} {r['key']}" for r in results]
    vals = [round(strong - average, 1)] + [max(r["gap"], 0) for r in results]
    colors = ["#a0aec0"] + [LEVEL[r["level"]][2] for r in results]
    fig = go.Figure(go.Bar(
        y=labels[::-1], x=vals[::-1], orientation="h", marker_color=colors[::-1],
        text=[f"{v:.1f} points" for v in vals][::-1], textposition="outside",
    ))
    fig.update_layout(
        title="How many points did changing ONE detail cost?<br>"
              "<sup>Grey bar = the gap between a strong and an average CV, as a yardstick</sup>",
        xaxis_title="Score points lost (AI score is 0-100)", xaxis_range=[0, max(vals) * 1.35 or 1],
        height=400, margin=dict(l=10, r=20, t=70, b=40),
    )
    return fig


def _history_detail(hs, coefs):
    """Chart + explanation for the 'learning from biased history' card, in AI scores."""
    groups = ["Men", "Women", "Under 40", "Over 40"]
    keys = ["Male", "Female", "Under 40", "40+"]
    vals = [hs[k] for k in keys]
    colors = ["#38a169", "#e53e3e", "#38a169", "#e53e3e"]

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.5, 0.5], horizontal_spacing=0.28,
        subplot_titles=("① AI score for equally qualified applicants", "② What the AI learned to rely on"),
    )
    fig.add_bar(x=groups, y=vals, marker_color=colors, showlegend=False,
                text=[f"{v:.0f}" for v in vals], textposition="outside", row=1, col=1)
    fig.update_yaxes(title_text="Average AI score (0-100)", range=[0, 100], row=1, col=1)

    c = coefs.copy()
    c["label"] = c["feature"].map(FEATURE_LABEL)
    c = c.reindex(c["weight"].abs().sort_values().index)
    proxy = c["feature"].isin(history_sim.PROXIES)
    fig.add_bar(y=c["label"], x=c["weight"], orientation="h", showlegend=False,
                marker_color=["#e53e3e" if p else "#38a169" for p in proxy],
                text=[f"{w:+.2f}" for w in c["weight"]], textposition="outside", row=1, col=2)
    fig.add_vline(x=0, line_color="#2d3748", row=1, col=2)
    wmax = c["weight"].abs().max() * 1.5
    fig.update_xaxes(range=[-wmax, wmax], title_text="Weight (+ raises score, − lowers it)", row=1, col=2)
    fig.update_yaxes(automargin=True, row=1, col=2)
    fig.update_layout(height=420, margin=dict(l=10, r=20, t=60, b=40))

    w = coefs.set_index("feature")["weight"]
    md = f"""
### 🧠 Detail: how a "blind" AI learned bias from history
We simulated **6,000 past applicants**. Men and women, young and older, had **the same skills on average**,
but past recruiters marked down women and people over 40. We trained an AI on those past decisions
and **removed gender and age** from its data. Then we asked it to score new applicants who are **all equally qualified**.

- **Chart ①:** qualified men get **{hs['Male']:.0f}** but qualified women only **{hs['Female']:.0f}**. Qualified
  under-40s get **{hs['Under 40']:.0f}** but over-40s only **{hs['40+']:.0f}**. Same qualifications, lower score.
- **Chart ②:** how the AI did it. A **career gap** lowers the score (weight **{w['career_gap']:+.2f}**), and most gaps
  belong to women who took caregiving breaks. A **later graduation year** raises it (weight **{w['grad_year']:+.2f}**),
  which favours younger people. These red bars are **clues** that stand in for gender and age.
- **Lesson:** deleting the gender and age columns does not remove bias. The AI copies the bias in the past decisions it learns from.
"""
    return md, fig


def _script():
    return """
### 🎤 What to say (30 seconds)
1. "We gave an AI screener the **same CV** many times, changing only **one thing**: the name, the age, a career break, or a disability mention."
2. "A fair AI would give every copy the same score. Look at the scores on the cards: changing a detail that has nothing to do with the job moved the score. Compare each gap with the grey bar, the gap between a strong and an average CV, to see how much it matters."
3. "The last chart shows an AI that **never saw gender or age** but still gives equally qualified women and older applicants lower scores, because it learned from biased past decisions. That is why our framework says: **test every tool before you use it.**"

*Embedding screener (BAAI/bge-small-en-v1.5), synthetic CVs. Illustrative, not a scientific audit. For details and other models, use the Advanced tabs.*
"""
