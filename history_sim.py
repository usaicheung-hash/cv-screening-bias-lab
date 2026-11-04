"""Tab 2: a model learns bias from biased past hiring decisions.

We simulate a company's last few years of applicants. Each applicant has a true
(job-relevant) qualification. Past human recruiters hired the qualified ones, but also
marked down women and people over 40. We then train a "blind" model that never sees
gender or age, and show that it still discriminates through proxies:
  - career_gap  (much more common for women, who more often take caregiving breaks)
  - grad_year   (directly reveals age)
Finally we apply fixes and show the fairness vs accuracy trade-off.
All data is synthetic.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ALL_FEATURES = ["skill_test", "years_experience", "education_level", "career_gap", "grad_year"]
PROXIES = ["career_gap", "grad_year"]

MITIGATIONS = [
    "None (blind model: gender and age removed)",
    "Remove proxy features (career gap, graduation year)",
    "Fairlearn ThresholdOptimizer: demographic parity",
    "Fairlearn ExponentiatedGradient: demographic parity",
    "Fairlearn ThresholdOptimizer: equal opportunity on past labels",
]


def simulate(n: int = 6000, gender_bias: float = 1.0, age_bias: float = 1.0, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    gender = rng.choice(["Female", "Male"], size=n)
    female = gender == "Female"
    age = rng.integers(22, 61, size=n)
    over40 = age >= 40
    grad_year = 2026 - (age - 22) + rng.integers(-1, 2, size=n)

    # Job-relevant features have the SAME distribution for every group.
    skill = rng.normal(60, 15, size=n).clip(0, 100)
    years_exp = (np.minimum(age - 22, 25) * rng.uniform(0.4, 1.0, size=n)).round(1)
    education = rng.choice([0, 1, 2], size=n, p=[0.3, 0.5, 0.2])  # diploma / degree / master

    # Career gaps: much more common for women (caregiving). Not job-relevant.
    gap_prob = np.where(female, 0.55, 0.05)
    career_gap = (rng.uniform(size=n) < gap_prob).astype(int)

    # True qualification depends only on job-relevant features.
    merit = (
        0.06 * (skill - 60)
        + 0.12 * np.minimum(years_exp, 10)
        + 0.5 * education
        + rng.normal(0, 0.6, size=n)
    )
    truly_qualified = (merit > np.quantile(merit, 0.6)).astype(int)

    # Past human decisions: merit MINUS bias against women and over-40s.
    biased_score = merit - gender_bias * female - age_bias * over40 + rng.normal(0, 0.3, size=n)
    hired_historically = (biased_score > np.quantile(biased_score, 0.6)).astype(int)

    return pd.DataFrame({
        "gender": gender,
        "age_band": np.where(over40, "40+", "Under 40"),
        "skill_test": skill.round(1),
        "years_experience": years_exp,
        "education_level": education,
        "career_gap": career_gap,
        "grad_year": grad_year,
        "truly_qualified": truly_qualified,
        "hired_historically": hired_historically,
    })


def _group_metrics(df: pd.DataFrame, pred: np.ndarray, col: str) -> pd.DataFrame:
    rows = []
    for g, sub in df.groupby(col):
        p = pred[sub.index]
        qualified = sub["truly_qualified"].to_numpy() == 1
        rows.append({
            "attribute": col,
            "group": g,
            "applicants": len(sub),
            "selection_rate": p.mean(),
            # equal opportunity: of the truly qualified, how many get selected?
            "qualified_selected": p[qualified].mean() if qualified.any() else np.nan,
        })
    out = pd.DataFrame(rows)
    out["impact_ratio"] = out["selection_rate"] / out["selection_rate"].max()
    return out


def run(gender_bias: float, age_bias: float, mitigation: str, seed: int = 7):
    df = simulate(gender_bias=gender_bias, age_bias=age_bias, seed=seed)
    train, test = train_test_split(df, test_size=0.4, random_state=seed, stratify=df["hired_historically"])
    test = test.reset_index(drop=True)

    features = ALL_FEATURES if not mitigation.startswith("Remove proxy") else [
        f for f in ALL_FEATURES if f not in PROXIES
    ]
    scaler = StandardScaler().fit(train[features])
    Xtr, Xte = scaler.transform(train[features]), scaler.transform(test[features])
    ytr = train["hired_historically"].to_numpy()

    # The intersection of gender and age is the sensitive feature for mitigation.
    sens_tr = train["gender"] + " / " + train["age_band"]
    sens_te = test["gender"] + " / " + test["age_band"]

    base = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    coefs = pd.DataFrame({"feature": features, "weight": base.coef_[0]})

    if mitigation.startswith("Fairlearn ThresholdOptimizer"):
        from fairlearn.postprocessing import ThresholdOptimizer

        # "equal opportunity on past labels" equalises true-positive rates measured against
        # the BIASED historical decisions: a deliberate trap that shows a fairness metric
        # is only as good as the labels it is computed on.
        constraint = "true_positive_rate_parity" if "equal opportunity" in mitigation else "demographic_parity"
        to = ThresholdOptimizer(
            estimator=base, constraints=constraint,
            objective="accuracy_score", prefit=True, predict_method="predict_proba",
        )
        to.fit(Xtr, ytr, sensitive_features=sens_tr)
        pred = np.asarray(to.predict(Xte, sensitive_features=sens_te, random_state=seed))
    elif mitigation.startswith("Fairlearn ExponentiatedGradient"):
        from fairlearn.reductions import DemographicParity, ExponentiatedGradient

        eg = ExponentiatedGradient(LogisticRegression(max_iter=1000), constraints=DemographicParity())
        eg.fit(Xtr, ytr, sensitive_features=sens_tr)
        pred = np.asarray(eg.predict(Xte, random_state=seed))
    else:
        pred = base.predict(Xte)

    metrics = pd.concat([_group_metrics(test, pred, "gender"), _group_metrics(test, pred, "age_band")])

    summary = {
        "accuracy_vs_past_decisions": float((pred == test["hired_historically"]).mean()),
        "accuracy_vs_true_qualification": float((pred == test["truly_qualified"]).mean()),
        "min_impact_ratio": float(metrics["impact_ratio"].min()),
        "equal_opportunity_gap": float(
            metrics.groupby("attribute")["qualified_selected"].agg(lambda s: s.max() - s.min()).max()
        ),
        "share_selected": float(pred.mean()),
    }
    history = pd.concat([
        _group_metrics(test, test["hired_historically"].to_numpy(), "gender"),
        _group_metrics(test, test["hired_historically"].to_numpy(), "age_band"),
    ])
    return metrics, coefs, summary, history


def compare_all(gender_bias: float, age_bias: float, seed: int = 7) -> pd.DataFrame:
    rows = []
    for m in MITIGATIONS:
        _, _, s, _ = run(gender_bias, age_bias, m, seed)
        rows.append({"mitigation": m, **s})
    return pd.DataFrame(rows)
