"""Build CV variants that differ in exactly one attribute."""

import re

from cv_data import (
    AGE_VARIANTS,
    BASE_FILL,
    CV_TEMPLATES,
    DISABILITY_VARIANTS,
    GAP_VARIANTS,
    NAMES,
)

ATTRIBUTES = [
    "Name (ethnicity and gender)",
    "Age (graduation year)",
    "Career break",
    "Disability mention",
]


def _email_from_name(name: str) -> str:
    return re.sub(r"[^a-z]+", ".", name.lower()).strip(".")


def fill(template: str, overrides: dict | None = None) -> str:
    values = dict(BASE_FILL)
    if overrides:
        values.update(overrides)
    values["NAME_EMAIL"] = _email_from_name(values["NAME"])
    text = template
    for key, val in values.items():
        text = text.replace("{" + key + "}", val)
    # tidy blank lines left by empty placeholders
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def build_attribute_variants(job: str, attribute: str, strength: str = "strong") -> list[dict]:
    """Return a list of {label, group, gender, text} for one attribute test.

    For the name test every name is a separate variant; the label is the name and the
    group/gender fields allow averaging by group.
    """
    template = CV_TEMPLATES[job][strength]
    out = []
    if attribute.startswith("Name"):
        for (group, gender), names in NAMES.items():
            for name in names:
                out.append({
                    "label": name,
                    "group": group,
                    "gender": gender,
                    "text": fill(template, {"NAME": name}),
                })
        return out

    table = {
        "Age (graduation year)": AGE_VARIANTS,
        "Career break": GAP_VARIANTS,
        "Disability mention": DISABILITY_VARIANTS,
    }[attribute]
    for label, overrides in table:
        out.append({
            "label": label,
            "group": label,
            "gender": "",
            "text": fill(template, overrides),
        })
    return out


def build_pool(job: str) -> list[dict]:
    """Every name x every CV strength: a mixed candidate pool for the shortlist simulation."""
    pool = []
    for strength in CV_TEMPLATES[job]:
        for v in build_attribute_variants(job, ATTRIBUTES[0], strength):
            v = dict(v)
            v["strength"] = strength
            pool.append(v)
    return pool
