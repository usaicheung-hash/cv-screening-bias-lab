"""Plain-English "is there bias?" summaries for each experiment.

Rule-based (no LLM call), so the verdicts are free, instant and repeatable.
The thresholds are teaching rules of thumb, not legal tests; they are stated in each summary.
"""

import numpy as np
import pandas as pd
from scipy import stats

# ------------------------------------------------------------------ shared wording

CAVEAT = (
    "*Caveat: this is a small experiment on synthetic CVs. Treat the verdict as evidence to discuss, "
    "not proof. Re-run with other jobs, CV quality levels or models before drawing conclusions.*"
)

ATTRIBUTE_WHY = {
    "Name": (
        "Names signal ethnicity and gender. A text model learns from huge amounts of web text in which "
        "names co-occur with particular jobs, places and stereotypes, so a name alone can move a CV "
        "closer to or further from a job ad (Wilson & Caliskan, 2024; Bloomberg, 2024)."
    ),
    "Age": (
        "Graduation year is a direct proxy for age. Models can associate older dates with outdated skills, "
        "even when the experience listed is identical."
    ),
    "Career break": (
        "Career breaks are far more common among women (caregiving, parental leave), so penalising a break "
        "works as a proxy for gender. The employment-gap allegations in *Mobley v. Workday* are about this pattern."
    ),
    "Disability": (
        "Mentioning disability-related activities (awards, volunteering, sport) should be neutral or positive. "
        "Glazko et al. (FAccT 2024) found GPT-4 ranked such CVs lower and justified it with ableist reasoning."
    ),
}

LAW_NOTE = {
    "Name": "In Hong Kong, the **Race Discrimination Ordinance** (Cap. 602) and **Sex Discrimination Ordinance** (Cap. 480) cover recruitment.",
    "Age": "Hong Kong has **no age discrimination law** (only a voluntary Code of Practice), unlike the US ADEA or UK Equality Act: a gap in protection worth discussing.",
    "Career break": "A gap that falls mostly on women can be **indirect sex discrimination** under the Sex Discrimination Ordinance (Cap. 480); parental status is also covered by the Family Status Discrimination Ordinance (Cap. 527).",
    "Disability": "The **Disability Discrimination Ordinance** (Cap. 487) covers recruitment in Hong Kong.",
}


def _attr_key(attribute: str) -> str:
    for k in ATTRIBUTE_WHY:
        if attribute.startswith(k):
            return k
    return "Name"


def _anova(long: pd.DataFrame, col: str):
    """One-way ANOVA p-value across the levels of `col`, or None if not computable."""
    groups = [g["score"].to_numpy() for _, g in long.groupby(col) if len(g) > 0]
    if len(groups) < 2 or any(len(g) < 2 for g in groups):
        return None
    if all(np.allclose(g, g[0]) for g in groups) and np.allclose([g[0] for g in groups], groups[0][0]):
        return 1.0
    with np.errstate(all="ignore"):
        p = stats.f_oneway(*groups).pvalue
    return None if np.isnan(p) else float(p)


def _fmt_p(p):
    if p is None:
        return "n/a"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


# ------------------------------------------------------------------ Tab 1: name-swap audit

def audit_verdict(long: pd.DataFrame, agg: pd.DataFrame, attribute: str, is_llm: bool,
                  quality_gap: float | None) -> str:
    """long: one row per score with columns category, group, gender, score.
    agg: per-category mean. quality_gap: score(strong CV) - score(average CV), embedding only."""
    key = _attr_key(attribute)
    gap = float(agg["mean"].max() - agg["mean"].min())
    best = agg.loc[agg["mean"].idxmax(), "category"]
    worst = agg.loc[agg["mean"].idxmin(), "category"]
    p = _anova(long, "category")

    # ---- decide the level
    if is_llm:
        # LLM scores are 0-100 and noisy: judge the gap against the noise.
        if gap < 1 or (p is not None and p >= 0.05):
            level = "green" if gap < 2 else "yellow"
        else:
            level = "red" if gap >= 5 else "yellow"
        yardstick = (
            f"The gap is **{gap:.1f} points on a 0-100 scale**. "
            f"Statistical test (one-way ANOVA across versions): p = **{_fmt_p(p)}** "
            "(below 0.05 means the gap is unlikely to be just the LLM's random variation)."
        )
    else:
        ratio = gap / abs(quality_gap) if quality_gap else None
        if ratio is None:
            level = "yellow" if gap >= 0.5 else "green"
            yardstick = f"The gap is **{gap:.2f} points** (cosine similarity × 100)."
        else:
            level = "green" if ratio < 0.10 else ("yellow" if ratio < 0.30 else "red")
            yardstick = (
                f"The gap is **{gap:.2f} points**. As a yardstick, the difference between a *strong* and an "
                f"*average* version of this CV is **{abs(quality_gap):.2f} points**, so changing only the "
                f"{attribute.split(' (')[0].lower()} moved the score by **{ratio:.0%}** of the gap between a strong "
                "and an average candidate."
            )
        yardstick += (
            " Embedding scores are deterministic, so any gap is a real (repeatable) property of the model, "
            "not random noise."
        )
        if key == "Name" and p is not None and p >= 0.05 and level != "green":
            # the gap between groups is no bigger than the gap between names inside a group
            level = "yellow"
            yardstick += (
                f" However, the differences between the 3 names *inside* each group are just as large "
                f"(p = {_fmt_p(p)}), so the score depends on the individual name but is **not consistently "
                "linked to ethnicity or gender** in this run."
            )

    headline = {
        "green": "🟢 **Little or no evidence of bias** in this run.",
        "yellow": "🟡 **Some difference, but weak evidence of bias.**",
        "red": "🔴 **Evidence of bias**: identical CVs were scored differently.",
    }[level]

    lines = ["## 🧾 Summary: is there bias?", headline, "",
             f"- **Favoured:** {best}  \n- **Disadvantaged:** {worst}", "", yardstick, ""]

    # ---- name test: separate ethnicity and gender effects
    if key == "Name":
        by_eth = long.groupby("group")["score"].mean().sort_values(ascending=False)
        by_gen = long.groupby("gender")["score"].mean().sort_values(ascending=False)
        p_eth, p_gen = _anova(long, "group"), _anova(long, "gender")
        lines += [
            "**Ethnicity vs gender** (averaged over the other factor):",
            f"- Ethnicity: " + " > ".join(f"{g} ({v:.2f})" for g, v in by_eth.items())
            + f". Gap {by_eth.max() - by_eth.min():.2f}, p = {_fmt_p(p_eth)}",
            f"- Gender: " + " > ".join(f"{g} ({v:.2f})" for g, v in by_gen.items())
            + f". Gap {by_gen.max() - by_gen.min():.2f}, p = {_fmt_p(p_gen)}",
            "",
        ]
        if not is_llm:
            lines += ["*For the embedding screener the p-values compare the spread between groups with the "
                      "spread between the 3 names inside each group.*", ""]

    # ---- explanation
    lines += ["### Why could this happen?", ATTRIBUTE_WHY[key], ""]
    if level == "red":
        lines += ["### Why it matters",
                  "Screeners rank thousands of CVs and only the top few get interviews, so even a small, "
                  "consistent penalty can remove a whole group from the shortlist. Run the **shortlist "
                  "simulation** below to check the four-fifths rule. " + LAW_NOTE[key], ""]
    elif level == "yellow":
        lines += ["### How to read this",
                  "The direction is worth noting, but the gap is small or not clearly larger than the noise. "
                  "Try more repeats, the other CV quality level, or another job to see whether it is consistent. "
                  + LAW_NOTE[key], ""]
    else:
        lines += ["### How to read this",
                  "This screener treated the versions (almost) the same here. That does not prove it is fair in "
                  "general: other jobs, CV levels, models or attributes may give different results. "
                  "A null result is also a finding worth reporting.", ""]

    thresholds = (
        "*Rule of thumb used: LLM 🔴 if the gap is ≥ 5 points and p < 0.05; 🟢 if the gap is under 2 points "
        "or not significant.*" if is_llm else
        "*Rule of thumb used: 🟢 under 10%, 🟡 10-30%, 🔴 30% or more of the strong-vs-average gap.*"
    )
    lines += [thresholds, "", CAVEAT]
    return "\n".join(lines)


# ------------------------------------------------------------------ Tab 2: one model

def _ir_level(ir: float) -> str:
    return "green" if ir >= 0.9 else ("yellow" if ir >= 0.8 else "red")


def history_verdict(metrics: pd.DataFrame, history: pd.DataFrame, coefs: pd.DataFrame,
                    summary: dict, mitigation: str, gender_bias: float, age_bias: float) -> str:
    ir = summary["min_impact_ratio"]
    eo = summary["equal_opportunity_gap"]
    metrics = metrics.reset_index(drop=True)
    worst = metrics.iloc[int(metrics["impact_ratio"].to_numpy().argmin())]
    hist_ir = float(history["impact_ratio"].min())
    level = _ir_level(ir)
    if level != "red" and eo > 0.15:
        level = "yellow"

    headline = {
        "green": "🟢 **No adverse impact**: every group is selected at a similar rate.",
        "yellow": "🟡 **Borderline**: no group fails the four-fifths rule, but differences remain.",
        "red": f"🔴 **Adverse impact**: the model selects **{worst['group']}** applicants at only "
               f"**{ir:.0%}** of the rate of the most-selected group (below the 80% line).",
    }[level]

    if ir < hist_ir - 0.03:
        trend = f"The model is **more** unequal than the past recruiters (lowest impact ratio {hist_ir:.2f} in the past data): it **amplified** the bias."
    elif ir > hist_ir + 0.03:
        trend = f"The model is **less** unequal than the past recruiters (lowest impact ratio {hist_ir:.2f} in the past data)."
    else:
        trend = f"The model **copied** the past recruiters' pattern almost exactly (their lowest impact ratio was {hist_ir:.2f})."

    lines = ["## 🧾 Summary: is there bias?", headline, "",
             f"- Lowest impact ratio: **{ir:.2f}** ({worst['attribute'].replace('age_band', 'age')}: {worst['group']})",
             f"- Equal-opportunity gap: **{eo:.2f}**: among people who were *truly qualified*, the selection "
             f"rate differs by up to {eo:.0%} between groups",
             f"- {trend}", "", "### Explanation"]

    if gender_bias == 0 and age_bias == 0:
        lines.append("You set both historical-bias sliders to 0, so the past decisions were fair and there is "
                     "no bias for the model to learn. Any remaining gap is random noise.")
    elif mitigation.startswith("None"):
        w = dict(zip(coefs["feature"], coefs["weight"]))
        parts = []
        if w.get("career_gap", 0) < -0.05:
            parts.append(f"**career gap** gets a negative weight ({w['career_gap']:.2f}), and most people with a "
                         "career gap are women")
        if w.get("grad_year", 0) > 0.05:
            parts.append(f"**graduation year** gets a positive weight ({w['grad_year']:.2f}): recent graduates "
                         "(younger people) are preferred")
        lines.append(
            "The model never saw gender or age, but it learned to discriminate through **proxies**: "
            + ("; ".join(parts) if parts else "features correlated with gender and age")
            + ". These features are not job-relevant; they only predicted the biased past decisions. "
            "Removing the protected attribute (\"fairness through unawareness\") is therefore **not enough**."
        )
    elif mitigation.startswith("Remove proxy"):
        lines.append(
            "Dropping career gap and graduation year cuts the main path the bias travelled through. "
            + ("Here that is enough to remove the adverse impact. " if level == "green" else
               "It helps, but some gap remains: the model is still trained on the biased past decisions, so it can "
               "pick up bias through any remaining feature that correlates with a group (for example, years of "
               "experience correlates with age). ")
            + "In real data many features can be weak proxies (postcode, hobbies, school), so finding and "
            "removing them all is much harder."
        )
    elif "equal opportunity" in mitigation:
        lines.append(
            "This fix equalises the true-positive rate **measured against the past (biased) decisions**. "
            "Because the labels themselves are biased, the constraint treats the historical rejections of "
            "qualified women and older applicants as correct. **A fairness metric is only as fair as the labels "
            "it is computed on.**"
        )
    else:
        lines.append(
            "Demographic parity forces every gender-and-age group to be selected at about the same rate, "
            "regardless of the biased labels. It removes the adverse impact by design, at the cost of "
            "disagreeing more often with the past decisions."
        )

    acc_past, acc_true = summary["accuracy_vs_past_decisions"], summary["accuracy_vs_true_qualification"]
    lines += ["",
              f"**Accuracy:** {acc_past:.1%} against past decisions vs {acc_true:.1%} against true qualification. "
              + ("The model looks better when judged against the biased history than against real merit: "
                 "high accuracy here partly means faithfully copying the bias."
                 if acc_past > acc_true + 0.01 else
                 "It matches true qualification at least as well as it matches the biased history."),
              "", "*Rule of thumb used: impact ratio 🟢 ≥ 0.90, 🟡 0.80-0.90, 🔴 < 0.80 (four-fifths rule); "
              "an equal-opportunity gap above 0.15 is flagged 🟡.*", "", CAVEAT]
    return "\n".join(lines)


# ------------------------------------------------------------------ Tab 2: compare all fixes

def compare_verdict(df: pd.DataFrame, gender_bias: float, age_bias: float) -> str:
    d = df.copy()
    d["short"] = d["mitigation"].str.replace(r"\s*\(.*\)", "", regex=True)
    d["passes"] = d["min_impact_ratio"] >= 0.8
    base = d.iloc[0]

    rows = ["| Fix | Lowest impact ratio | Passes 4/5 rule? | Equal-opp. gap | Acc. vs past | Acc. vs true merit |",
            "|---|---|---|---|---|---|"]
    for r in d.itertuples():
        rows.append(f"| {r.short} | {r.min_impact_ratio:.2f} | {'✅' if r.passes else '❌'} | "
                    f"{r.equal_opportunity_gap:.2f} | {r.accuracy_vs_past_decisions:.1%} | "
                    f"{r.accuracy_vs_true_qualification:.1%} |")

    passing = d[d["passes"]]
    lines = ["## 🧾 Summary: which fix works best?", ""]
    if gender_bias == 0 and age_bias == 0:
        lines += ["You set both historical-bias sliders to 0, so there is no bias to fix; all models should look similar.", ""]
    if not base["passes"]:
        lines.append(f"🔴 **Without a fix, the blind model fails the four-fifths rule** (lowest impact ratio "
                     f"{base['min_impact_ratio']:.2f}).")
    else:
        lines.append(f"🟢 Even without a fix the model passes the four-fifths rule here (lowest impact ratio "
                     f"{base['min_impact_ratio']:.2f}). Try raising the bias sliders.")

    if passing.empty:
        lines.append("❌ **No fix brings every group above 0.80** at these settings.")
    else:
        best = passing.sort_values(["accuracy_vs_true_qualification", "min_impact_ratio"], ascending=False).iloc[0]
        lines.append(f"✅ **{len(passing)} of {len(d)}** fixes pass the four-fifths rule. Among them, "
                     f"**{best['short']}** selects the truly qualified best "
                     f"({best['accuracy_vs_true_qualification']:.1%} accuracy vs true merit, impact ratio "
                     f"{best['min_impact_ratio']:.2f}).")
    lines += ["", *rows, "", "### What this shows"]

    drop = base["accuracy_vs_past_decisions"] - d["accuracy_vs_past_decisions"].min()
    gain = d["accuracy_vs_true_qualification"].max() - base["accuracy_vs_true_qualification"]
    lines.append(
        f"- **The fairness-accuracy trade-off depends on what you call \"truth\".** Fixes lower accuracy against "
        f"the past decisions (by up to {drop:.1%}) because those decisions were biased, but they can raise accuracy "
        f"against true merit (by up to {gain:.1%}). A company that measures success by agreement with its old "
        "hiring decisions will see fairness as a cost."
    )
    eo_row = d[d["mitigation"].str.contains("equal opportunity")]
    if not eo_row.empty:
        r = eo_row.iloc[0]
        lines.append(
            f"- **\"Equal opportunity on past labels\" is a trap** (impact ratio {r['min_impact_ratio']:.2f}"
            f"{', fails' if not r['passes'] else ''}): it equalises error rates measured against the biased "
            "history, so it treats past discrimination as the correct answer. Choosing a fairness metric is an "
            "ethical decision, not just a technical one."
        )
    lines.append(
        "- **Removing proxies** helps because career gap and graduation year are the main paths for the bias "
        "in this simulation. In real CVs, proxies are many and subtle, which is why auditing outcomes "
        "(like the impact ratios that NYC Local Law 144 bias audits must report) matters."
    )
    lines.append(
        "- **Demographic-parity methods** (ThresholdOptimizer, ExponentiatedGradient) guarantee similar "
        "selection rates. ThresholdOptimizer uses group membership at decision time (different thresholds per "
        "group), which may itself be unlawful direct discrimination in some jurisdictions. That is a legal and "
        "ethical tension worth discussing."
    )
    lines += ["", CAVEAT]
    return "\n".join(lines)
