"""Step 3: a model learns bias from biased past hiring decisions.

We simulate a company's last few years of applicants. Each applicant has a true
(job-relevant) qualification. Past human recruiters hired the qualified ones, but also
marked down women and people over 40. We then train a "blind" model that never sees
gender or age, and show that it still discriminates through proxies:
  - career_gap  (much more common for women, who more often take caregiving breaks)
  - grad_year   (directly reveals age)
All data is synthetic.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ALL_FEATURES = ["skill_test", "years_experience", "education_level", "career_gap", "grad_year"]
PROXIES = ["career_gap", "grad_year"]


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


def blind_ai_scores(gender_bias: float = 1.0, age_bias: float = 1.0, seed: int = 7):
    """Train the 'blind' model (no gender, no age) and return each test applicant's
    AI score (predicted chance of being hired, 0-100) plus the model's feature weights."""
    df = simulate(gender_bias=gender_bias, age_bias=age_bias, seed=seed)
    train, test = train_test_split(df, test_size=0.4, random_state=seed, stratify=df["hired_historically"])
    test = test.reset_index(drop=True)
    scaler = StandardScaler().fit(train[ALL_FEATURES])
    model = LogisticRegression(max_iter=1000).fit(scaler.transform(train[ALL_FEATURES]), train["hired_historically"])
    test["ai_score"] = model.predict_proba(scaler.transform(test[ALL_FEATURES]))[:, 1] * 100
    coefs = pd.DataFrame({"feature": ALL_FEATURES, "weight": model.coef_[0]})
    return test, coefs
