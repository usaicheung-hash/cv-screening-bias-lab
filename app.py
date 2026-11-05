"""CV Screening Bias Lab: a teaching demo for IS5113 (CityU), Part 1 Topic 1.

Presentation demo (demo3.py): Step 1 job and CV, Step 2 LLM scores with reasons, Step 3 learning bias from
history, Step 4 fixing it with a better prompt. The helper functions below belong to the earlier advanced tabs,
which were removed from the UI but are kept for reference and run_experiment.py.
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
import demo3
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

The **🎤 Presentation demo** tells the story in four steps.

**Step 1: What are we testing?** One Hong Kong job post (fluent Cantonese and English, good written Chinese),
one standard CV, and copies that differ in *only one thing*: the name, the graduation year (age), a career
break, or an activity that reveals a disability.

**Step 2: Same CV, different score.** A large language model acts as the recruiter: it rates each copy from
0 to 100 and gives a reason (compare Bloomberg's 2024 GPT test and Glazko et al., FAccT 2024). Each CV is scored
3 times; a gap only counts as bias when it is bigger than the AI's own run-to-run variation.

**Step 3: How an AI learns bias from history.** A simulated company trains a model on its past hiring decisions,
which were biased against women and over-40s. The model is "blind" (gender and age are removed), yet it learns
the bias through **proxies** (career gaps and graduation year), echoing the employment-gap allegations in
*Mobley v. Workday*.

**Step 4: Fix it with a better prompt.** Add your own screening rules to the AI's prompt (for example: ignore
nationality, judge language only from what the CV states) and compare the scores before and after. A better
prompt can reduce a gap, but it is not a guarantee, so every fix must be tested again.

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

    DEFAULT_JOB = "Marketing Executive (consumer brand, Hong Kong)"

    with gr.Tab("🎤 Presentation demo"):
        d_job = gr.Dropdown(list(JOBS), value=DEFAULT_JOB, label="Job")
        with gr.Tabs():
            with gr.Tab("Step 1 · What are we testing?"):
                gr.Markdown("### What are we testing?\nOne Hong Kong job, one standard CV, and copies that differ in "
                            "**only ONE detail**. A fair recruiter would give every copy the same score.")
                d_s1 = gr.HTML(demo3.step1_html(DEFAULT_JOB))
            with gr.Tab("Step 2 · Same CV, different score"):
                gr.Markdown("### Same CV, one detail changed → different score?\n"
                            "An AI recruiter (large language model) scores each copy from 0 to 100 and explains why. "
                            "Pick a test.")
                with gr.Row():
                    d_test = gr.Radio(list(demo3.TESTS), value="🌏 Name", label="Test", scale=4)
                    d_live = gr.Button("🔄 Re-run live (about 1 minute)", size="sm", scale=1)
                d_s2 = gr.HTML()
            with gr.Tab("Step 3 · How an AI learns bias from history"):
                gr.Markdown("### How an AI learns bias from history, even when we hide gender and age")
                d_s3 = gr.HTML(demo3.step3_html())
            with gr.Tab("Step 4 · Fix it with a better prompt"):
                gr.Markdown("### Can we remove the bias by telling the AI what NOT to judge?\n"
                            "Write rules for the AI recruiter below (for example: ignore nationality, judge language "
                            "only from the language skills written on the CV). The same CVs are scored again with "
                            "your rules added to the prompt, and the result is compared with the normal prompt.")
                d_fix = gr.Textbox(demo3.DEFAULT_FIX, label="Your instruction to the AI (you can edit it)", lines=4)
                with gr.Row():
                    d_fix_test = gr.Radio(demo3.FIX_TESTS, value=demo3.FIX_TESTS[0], label="Test", scale=4)
                    d_fix_btn = gr.Button("▶ Apply instruction", variant="primary", scale=1)
                    d_fix_live = gr.Button("🔄 Run again (about 1 minute)", size="sm", scale=1)
                d_s4 = gr.HTML()
        d_job.change(demo3.step1_html, d_job, d_s1)
        d_job.change(demo3.step2, [d_job, d_test], d_s2)
        d_test.change(demo3.step2, [d_job, d_test], d_s2)
        d_live.click(lambda j, t: demo3.step2(j, t, live=True), [d_job, d_test], d_s2)
        demo.load(demo3.step2, [d_job, d_test], d_s2)
        d_fix_btn.click(demo3.step4, [d_job, d_fix_test, d_fix], d_s4)
        d_fix_test.change(demo3.step4, [d_job, d_fix_test, d_fix], d_s4)
        d_fix_live.click(lambda j, t, i: demo3.step4(j, t, i, live=True), [d_job, d_fix_test, d_fix], d_s4)
        demo.load(demo3.step4, [d_job, d_fix_test, d_fix], d_s4)
        d_export = gr.Button("export", visible=False)
        d_export_out = gr.Textbox(visible=False)
        d_export.click(demo3.export_cache, None, d_export_out, api_name="export_cache")
        d_pre_job = gr.Textbox(visible=False)
        d_pre = gr.Button("precompute", visible=False)
        d_pre.click(demo3.precompute, d_pre_job, d_export_out, api_name="precompute")
        d_fix_pre = gr.Button("precompute_fix", visible=False)
        d_fix_pre_job = gr.Textbox(visible=False)
        d_fix_pre.click(demo3.precompute_fix, d_fix_pre_job, d_export_out, api_name="precompute_fix")

    with gr.Tab("ℹ️ About and ethics"):
        gr.Markdown(ABOUT)

demo3.prewarm("Marketing Executive (consumer brand, Hong Kong)")

if __name__ == "__main__":
    # The demo pages are designed for a light background: switch off Gradio's dark mode.
    FORCE_LIGHT = """() => {
      const light = () => { document.body.classList.remove('dark');
        document.querySelectorAll('.dark').forEach(e => e.classList.remove('dark')); };
      light(); new MutationObserver(light).observe(document.body, {attributes: true, subtree: false});
    }"""
    demo.launch(theme=gr.themes.Soft(), js=FORCE_LIGHT)
