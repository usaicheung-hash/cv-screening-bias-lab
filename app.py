"""CV Screening Bias Lab: a teaching demo for IS5113 (CityU), Part 1 Topic 1.

Tab 1  Name-swap audit: identical CVs that differ in ONE attribute are scored by an AI.
Tab 2  Learning bias from history: a "blind" model learns bias through proxy features.
Tab 3  About and ethics.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import gradio as gr

import history_sim
import scorers
import quick_demo
import verdicts
from cv_data import CV_TEMPLATES, JOBS, NAMES
from variants import ATTRIBUTES, build_attribute_variants, build_pool, fill

SCORER_EMBED = "Embedding similarity (free, instant)"
SCORER_LLM = "LLM recruiter prompt (needs HF_TOKEN)"

FOUR_FIFTHS_NOTE = (
    "**Impact ratio** = a group's selection rate divided by the highest group's rate. "
    "Under the US EEOC *four-fifths rule*, a ratio below **0.80** is a warning sign of adverse impact."
)


# ------------------------------------------------------------------ Tab 1 helpers

def _score_variants(job_text, variants, scorer, model, repeats):
    """Return a long dataframe: one row per (variant, repeat)."""
    rows = []
    if scorer == SCORER_EMBED:
        s = scorers.embedding_scores(job_text, [v["text"] for v in variants])
        for v, score in zip(variants, s):
            rows.append({**{k: v[k] for k in ("label", "group", "gender")}, "score": score})
    else:
        # score in random order so that group order never lines up with call order
        for v in scorers.shuffled(variants):
            for score in scorers.llm_scores(job_text, v["text"], repeats=int(repeats), model=model):
                rows.append({**{k: v[k] for k in ("label", "group", "gender")}, "score": score})
    return pd.DataFrame(rows)


def run_audit(job, job_text, strength, attribute, scorer, model, repeats):
    variants = build_attribute_variants(job, attribute, strength)
    try:
        long = _score_variants(job_text, variants, scorer, model, repeats)
    except Exception as e:  # noqa: BLE001  show the error in the UI instead of crashing
        return None, None, f"⚠️ {e}", "", ""
    if long.empty:
        return None, None, "⚠️ The model returned no usable scores. Try again or pick another model.", "", ""

    if attribute.startswith("Name"):
        long["category"] = long["group"] + " · " + long["gender"]
    else:
        long["category"] = long["label"]

    agg = (
        long.groupby("category", sort=False)["score"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    agg["std"] = agg["std"].fillna(0)
    control = agg.iloc[0]["mean"] if not attribute.startswith("Name") else agg["mean"].max()
    agg["difference"] = agg["mean"] - control
    agg = agg.round(3)

    fig = go.Figure(go.Bar(
        x=agg["category"], y=agg["mean"],
        error_y=dict(type="data", array=agg["std"], visible=bool(agg["std"].any())),
        marker_color=["#2b6cb0" if i == 0 else "#dd6b20" for i in range(len(agg))]
        if not attribute.startswith("Name") else None,
        text=agg["mean"].map(lambda v: f"{v:.2f}"), textposition="outside",
    ))
    lo, hi = agg["mean"].min(), agg["mean"].max()
    pad = max(0.5, (hi - lo) * 0.6)
    fig.update_layout(
        title=f"Average AI score by {attribute.lower()} (same CV, only this attribute changes)",
        yaxis_title="Score", yaxis_range=[lo - pad, hi + pad], height=430, margin=dict(t=60, b=40),
    )

    spread = hi - lo
    best = agg.loc[agg["mean"].idxmax(), "category"]
    worst = agg.loc[agg["mean"].idxmin(), "category"]
    summary = (
        f"### What happened\n"
        f"Every CV was **identical except for the {attribute.lower()}**. "
        f"Highest average: **{best}**. Lowest: **{worst}**. Gap: **{spread:.2f} points**.\n\n"
        "A fair screener should give (almost) the same score to every version. "
        "Even a small gap matters when thousands of CVs are ranked and only the top few get interviews: "
        "try the shortlist simulation below to see the effect on a whole pool."
    )
    if attribute.startswith("Name"):
        summary += (
            "\n\n*Each bar averages the 3 names in that group; error bars show the spread between those names "
            "(and between repeated LLM answers, if you used the LLM).*"
        )
    elif scorer == SCORER_EMBED:
        summary += (
            "\n\n*Embedding scores are deterministic, so there are no error bars. "
            "Differences come purely from how the model represents the words that changed.*"
        )
    example = variants[-1]["text"]

    quality_gap = None
    if scorer == SCORER_EMBED:
        # yardstick: how much does the score change between a strong and an average CV?
        strong, average = scorers.embedding_scores(
            job_text, [fill(CV_TEMPLATES[job]["strong"]), fill(CV_TEMPLATES[job]["average"])])
        quality_gap = strong - average
    verdict = verdicts.audit_verdict(long, agg, attribute, scorer != SCORER_EMBED, quality_gap)
    return fig, agg, summary, example, verdict


def run_shortlist(job, job_text, top_pct):
    pool = build_pool(job)
    scores = scorers.embedding_scores(job_text, [p["text"] for p in pool])
    df = pd.DataFrame([{k: p[k] for k in ("label", "group", "gender", "strength")} for p in pool])
    df["score"] = scores
    k = max(1, int(round(len(df) * top_pct / 100)))
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["shortlisted"] = df.index < k

    rows = []
    for col in ("group", "gender"):
        g = df.groupby(col)["shortlisted"].mean().rename("selection_rate").reset_index()
        g["impact_ratio"] = g["selection_rate"] / g["selection_rate"].max()
        g = g.rename(columns={col: "group"})
        g.insert(0, "attribute", "Ethnicity" if col == "group" else "Gender")
        rows.append(g)
    table = pd.concat(rows).round(3)

    fig = px.bar(
        table, x="group", y="selection_rate", color="attribute", text="impact_ratio",
        title=f"Share of each group shortlisted (top {top_pct}% of {len(df)} CVs)",
        labels={"selection_rate": "Selection rate", "group": ""},
    )
    fig.update_traces(texttemplate="ratio %{text:.2f}", textposition="outside")
    fig.update_layout(height=430, yaxis_range=[0, min(1.0, table["selection_rate"].max() * 1.35 + 0.05)])

    flagged = table[table["impact_ratio"] < 0.8]
    if flagged.empty:
        verdict = "✅ No group falls below the four-fifths (0.80) threshold in this run."
    else:
        verdict = "🚩 Below the four-fifths threshold: " + ", ".join(
            f"**{r.group}** ({r.impact_ratio:.2f})" for r in flagged.itertuples()
        )
    note = (
        f"{verdict}\n\nThe pool has {len(df)} CVs: two quality levels (strong and average) × "
        f"{len(df)//2} names. Within each quality level the CVs are word-for-word identical apart from the name, "
        "so in a fair system every group would have exactly the same selection rate.\n\n" + FOUR_FIFTHS_NOTE
    )
    return fig, table, note, df[["label", "group", "gender", "strength", "score", "shortlisted"]].round(3)


def run_position_test(job, job_text, name_a, name_b, pairs, model):
    va = next(v for v in build_attribute_variants(job, ATTRIBUTES[0]) if v["label"] == name_a)
    vb = next(v for v in build_attribute_variants(job, ATTRIBUTES[0]) if v["label"] == name_b)
    picks = {name_a: 0, name_b: 0}
    first = 0
    total = 0
    try:
        for i in range(int(pairs)):
            # alternate the order so each name appears first half the time
            first_v, second_v = (va, vb) if i % 2 == 0 else (vb, va)
            ans = scorers.llm_pairwise(job_text, first_v["text"], second_v["text"], model=model)
            if ans is None:
                continue
            chosen = first_v if ans == "A" else second_v
            picks[chosen["label"]] += 1
            first += ans == "A"
            total += 1
    except Exception as e:  # noqa: BLE001
        return f"⚠️ {e}"
    if total == 0:
        return "⚠️ No usable answers."
    return (
        f"### Results from {total} comparisons\n"
        f"- **{name_a}** chosen {picks[name_a]} times ({picks[name_a]/total:.0%})\n"
        f"- **{name_b}** chosen {picks[name_b]} times ({picks[name_b]/total:.0%})\n"
        f"- The candidate listed **first** was chosen {first/total:.0%} of the time\n\n"
        "The two CVs are identical except for the name, and each name appeared first half the time. "
        "A fair recruiter would pick each about 50% of the time. Rozado (2025) found that most of 22 LLMs "
        "favoured female names **and** whichever candidate was listed first."
    )


# ------------------------------------------------------------------ Tab 2 helpers

def run_history(gender_bias, age_bias, mitigation):
    metrics, coefs, summary, history = history_sim.run(gender_bias, age_bias, mitigation)

    plot_df = pd.concat([
        history.assign(source="Past human decisions"),
        metrics.assign(source="AI model"),
    ])
    fig = px.bar(
        plot_df, x="group", y="selection_rate", color="source", barmode="group",
        facet_col="attribute", text="impact_ratio",
        title="Selection rate by group: past human decisions vs the AI model",
        labels={"selection_rate": "Selection rate", "group": ""},
        color_discrete_map={"Past human decisions": "#a0aec0", "AI model": "#dd6b20"},
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.for_each_annotation(lambda a: a.update(text=a.text.replace("attribute=", "")))
    fig.update_xaxes(matches=None)
    fig.update_layout(height=430, yaxis_range=[0, 0.8])

    cfig = px.bar(
        coefs.sort_values("weight"), x="weight", y="feature", orientation="h",
        title="What the model learned (logistic-regression weights, before any fairness fix)",
        color=coefs.sort_values("weight")["feature"].isin(history_sim.PROXIES).map(
            {True: "proxy (not job-relevant)", False: "job-relevant"}),
        color_discrete_map={"proxy (not job-relevant)": "#e53e3e", "job-relevant": "#2b6cb0"},
        labels={"color": ""},
    )
    cfig.update_layout(height=330)

    s = summary
    text = (
        f"### Results\n"
        f"| Measure | Value |\n|---|---|\n"
        f"| Lowest impact ratio (four-fifths rule: ≥ 0.80) | **{s['min_impact_ratio']:.2f}** |\n"
        f"| Equal-opportunity gap (difference in % of *truly qualified* people selected) | **{s['equal_opportunity_gap']:.2f}** |\n"
        f"| Accuracy against past (biased) decisions | {s['accuracy_vs_past_decisions']:.1%} |\n"
        f"| Accuracy against true qualification | {s['accuracy_vs_true_qualification']:.1%} |\n\n"
        "**How to read this.** The model never saw gender or age. It still learned to penalise "
        "**career gaps** (mostly women) and early **graduation years** (older applicants), because "
        "those features predicted the biased past decisions. Notice the dilemma: fixing the bias usually "
        "*lowers* accuracy against the old decisions, but can *raise* accuracy against true qualification. "
        "\"Accurate\" depends on what you treat as the truth."
    )
    table = metrics.round(3)
    verdict = verdicts.history_verdict(metrics, history, coefs, summary, mitigation, gender_bias, age_bias)
    return fig, cfig, text, table, verdict


def run_compare(gender_bias, age_bias):
    df = history_sim.compare_all(gender_bias, age_bias)
    fig = px.scatter(
        df, x="min_impact_ratio", y="accuracy_vs_past_decisions", color="mitigation",
        size=[14] * len(df), title="Fairness vs accuracy trade-off",
        labels={"min_impact_ratio": "Lowest impact ratio (fairer →)",
                "accuracy_vs_past_decisions": "Accuracy vs past decisions"},
    )
    fig.add_vline(x=0.8, line_dash="dash", annotation_text="four-fifths line")
    fig.update_layout(height=430, legend=dict(orientation="h", y=-0.3))
    return fig, df.round(3), verdicts.compare_verdict(df, gender_bias, age_bias)


# ------------------------------------------------------------------ UI

ALL_NAMES = [n for names in NAMES.values() for n in names]

ABOUT = """
## About this demo

This app was built for the **IS5113 AI Ethics group project (City University of Hong Kong), Part 1, Topic 1:
Algorithmic Bias in Hiring Platforms**. It turns the findings of recent studies into something you can try yourself.

**Tab 1: Name-swap audit.** We take one CV and change *only one thing*: the name, the graduation year (age),
a career break, or an activity that reveals a disability. A fair screener should give every version the same score.
- *Embedding similarity* measures how close the CV text is to the job ad, the way many CV-matching engines work
  (and the method of Wilson & Caliskan, AIES 2024).
- *LLM recruiter prompt* asks a chatbot to rate the CV from 0 to 100, as recruiters increasingly do
  (compare Bloomberg's 2024 GPT test and Glazko et al., FAccT 2024).
- *Position test* asks the LLM to choose between two identical CVs, to check name and order effects (Rozado, 2025).

**Tab 2: Learning bias from history.** A simulated company trains a model on its past hiring decisions,
which were biased against women and over-40s. The model is "blind" (gender and age are removed),
yet it learns the bias through **proxies** (career gaps and graduation year), echoing the employment-gap
allegations in *Mobley v. Workday*. You can then try fairness fixes from Microsoft's open-source **Fairlearn** library.

### Ethics of this demo
- **All CVs and data are synthetic.** No real person's information is used.
- This is an **illustrative experiment, not a scientific audit**: few CVs, few models, small samples.
  Results can change between runs and models. A null or reversed result is also a finding.
- It is designed as an **auditing tool**. It must not be used to find the "best-scoring" name or to optimise
  a CV to beat a screener.
- AI tools (Claude) helped write the code; the group designed the experiment, checked the results and
  interpreted them. See the AI-use declaration in our report.

### Key references
- Wilson, K. & Caliskan, A. (2024). *Gender, Race, and Intersectional Bias in Resume Screening via Language Model Retrieval*. AIES.
- Bloomberg (2024). *OpenAI's GPT Is a Recruiter's Dream Tool. Tests Show There's Racial Bias.*
- Glazko, K. et al. (2024). *Identifying and Improving Disability Bias in GPT-Based Resume Screening*. FAccT.
- Rozado, D. (2025). *Gender and Positional Biases in LLM-Based Hiring Decisions*. arXiv:2505.17049.
- *Mobley v. Workday, Inc.*, N.D. Cal. (collective certified May 2025).
- Bird, S. et al. (2020). *Fairlearn: A toolkit for assessing and improving fairness in AI*. Microsoft.
"""

with gr.Blocks(title="CV Screening Bias Lab") as demo:
    gr.Markdown(
        "# 🧪 CV Screening Bias Lab\n"
        "Does an AI screener give the same score to the same CV when only the **name, age, career break or "
        "disability** changes? And can a model that never sees gender or age still discriminate? "
        "*All data is synthetic. Built for the IS5113 AI Ethics project, CityU.*"
    )

    with gr.Tab("🎤 Quick demo"):
        gr.Markdown(
            "### One click: does the AI treat identical CVs the same?\n"
            "Pick a job and press the button. The app tests **6 perspectives** at once "
            "(ethnicity, gender, age, career break, disability, and learning from biased history) "
            "and shows a simple scorecard. Same result every time, no token needed."
        )
        with gr.Row():
            q_job = gr.Dropdown(list(JOBS), value="Marketing Executive (consumer brand, Hong Kong)", label="Job", scale=3)
            q_btn = gr.Button("▶ Run bias check", variant="primary", size="lg", scale=1)
        q_head = gr.Markdown()
        q_cards = gr.HTML()
        q_plot = gr.Plot(show_label=False)
        q_hist_md = gr.Markdown()
        q_hist_plot = gr.Plot(show_label=False)
        q_script = gr.Markdown()
        q_btn.click(lambda j: quick_demo.run_scorecard(j, JOBS[j]), q_job,
                    [q_head, q_cards, q_plot, q_hist_md, q_hist_plot, q_script])

    with gr.Tab("1 · Name-swap audit (advanced)"):
        with gr.Row():
            with gr.Column(scale=1):
                job = gr.Dropdown(list(JOBS), value=list(JOBS)[0], label="Job")
                job_text = gr.Textbox(JOBS[list(JOBS)[0]], label="Job description (editable)", lines=8)
                strength = gr.Radio(["strong", "average"], value="strong", label="CV quality")
                attribute = gr.Dropdown(ATTRIBUTES, value=ATTRIBUTES[0], label="Attribute to change")
                scorer = gr.Radio([SCORER_EMBED, SCORER_LLM], value=SCORER_EMBED, label="AI screener")
                with gr.Accordion("LLM settings", open=False):
                    model = gr.Textbox(scorers.DEFAULT_LLM, label="Hugging Face model id")
                    repeats = gr.Slider(1, 10, value=3, step=1, label="Repeats per CV (LLM answers vary)")
                run_btn = gr.Button("Run audit", variant="primary")
            with gr.Column(scale=2):
                plot = gr.Plot()
                verdict = gr.Markdown()
                summary = gr.Markdown()
                table = gr.Dataframe(label="Scores by variant", interactive=False)
                with gr.Accordion("Show one of the CVs used", open=False):
                    example = gr.Textbox(lines=18, show_label=False)

        job.change(lambda j: JOBS[j], job, job_text)
        run_btn.click(run_audit, [job, job_text, strength, attribute, scorer, model, repeats],
                      [plot, table, summary, example, verdict])

        gr.Markdown("---\n## Shortlist simulation (embedding screener)\n"
                    "Score a pool of CVs that differ only by name, keep the top X%, and check the four-fifths rule.")
        with gr.Row():
            top_pct = gr.Slider(10, 50, value=25, step=5, label="Shortlist the top X% of CVs")
            sl_btn = gr.Button("Run shortlist", variant="primary")
        sl_plot = gr.Plot()
        sl_note = gr.Markdown()
        sl_table = gr.Dataframe(label="Selection rate by group", interactive=False)
        with gr.Accordion("Full ranking", open=False):
            sl_rank = gr.Dataframe(interactive=False)
        sl_btn.click(run_shortlist, [job, job_text, top_pct], [sl_plot, sl_table, sl_note, sl_rank])

        gr.Markdown("---\n## Position test (LLM)\n"
                    "Give the LLM two CVs that are identical except for the name, and ask it to pick one. "
                    "Order is alternated so each name is listed first half the time.")
        with gr.Row():
            name_a = gr.Dropdown(ALL_NAMES, value="Chan Tai Man", label="Name A")
            name_b = gr.Dropdown(ALL_NAMES, value="Priya Sharma", label="Name B")
            pairs = gr.Slider(2, 30, value=10, step=2, label="Number of comparisons")
            pos_btn = gr.Button("Run position test", variant="primary")
        pos_out = gr.Markdown()
        pos_btn.click(run_position_test, [job, job_text, name_a, name_b, pairs, model], pos_out)

    with gr.Tab("2 · Learning bias from history (advanced)"):
        gr.Markdown(
            "A simulated company has 6,000 past applicants. Skills and experience are distributed **equally** "
            "across groups, but past recruiters marked down women and people aged 40+. "
            "We train a model on those past decisions **without** gender or age, then measure who it selects."
        )
        with gr.Row():
            with gr.Column(scale=1):
                gbias = gr.Slider(0, 2, value=1.0, step=0.1, label="Historical bias against women")
                abias = gr.Slider(0, 2, value=1.0, step=0.1, label="Historical bias against age 40+")
                mitig = gr.Radio(history_sim.MITIGATIONS, value=history_sim.MITIGATIONS[0], label="Fairness fix")
                h_btn = gr.Button("Train and evaluate", variant="primary")
                cmp_btn = gr.Button("Compare all fixes")
            with gr.Column(scale=2):
                h_plot = gr.Plot()
                h_verdict = gr.Markdown()
                h_text = gr.Markdown()
        h_coef = gr.Plot()
        h_table = gr.Dataframe(label="Metrics by group (test set)", interactive=False)
        gr.Markdown("### All fixes compared (click **Compare all fixes**)\n"
                    "Each dot is one fix. Right = fairer (impact ratio), up = closer to the past decisions.")
        cmp_plot = gr.Plot(show_label=False)
        cmp_verdict = gr.Markdown()
        cmp_table = gr.Dataframe(interactive=False, show_label=False)
        h_btn.click(run_history, [gbias, abias, mitig], [h_plot, h_coef, h_text, h_table, h_verdict])
        cmp_btn.click(run_compare, [gbias, abias], [cmp_plot, cmp_table, cmp_verdict])

    with gr.Tab("3 · About and ethics"):
        gr.Markdown(ABOUT)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
