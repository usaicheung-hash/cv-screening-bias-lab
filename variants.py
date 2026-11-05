"""Fill a CV template so that copies differ in exactly one detail."""

import re

from cv_data import BASE_FILL


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
