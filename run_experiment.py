"""Run every audit offline and save CSVs + charts for the report and presentation.

Usage (on your own computer, after `pip install -r requirements.txt`):
    python run_experiment.py               # embedding screener only (free)
    HF_TOKEN=hf_xxx python run_experiment.py --llm --repeats 5

Results go to ./results/. Use these cached results in the presentation so the
demo never depends on live Wi-Fi or API limits.
"""

import argparse
import os

import pandas as pd

import history_sim
import scorers
from cv_data import JOBS
from variants import ATTRIBUTES, build_attribute_variants, build_pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="also run the LLM scorer (needs HF_TOKEN)")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--model", default=scorers.DEFAULT_LLM)
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True)

    rows = []
    for job, jd in JOBS.items():
        for strength in ("strong", "average"):
            for attr in ATTRIBUTES:
                vs = build_attribute_variants(job, attr, strength)
                emb = scorers.embedding_scores(jd, [v["text"] for v in vs])
                for v, s in zip(vs, emb):
                    rows.append({"job": job, "strength": strength, "attribute": attr, "scorer": "embedding",
                                 "label": v["label"], "group": v["group"], "gender": v["gender"], "score": s})
                if args.llm:
                    for v in scorers.shuffled(vs):
                        for s in scorers.llm_scores(jd, v["text"], args.repeats, args.model):
                            rows.append({"job": job, "strength": strength, "attribute": attr,
                                         "scorer": args.model, "label": v["label"], "group": v["group"],
                                         "gender": v["gender"], "score": s})
                print("done:", job, strength, attr)
    df = pd.DataFrame(rows)
    df.to_csv("results/audit_scores.csv", index=False)

    # summary: mean score per group/label, per job and scorer
    names = df[df.attribute == ATTRIBUTES[0]]
    names.groupby(["scorer", "job", "group", "gender"])["score"].mean().round(3).to_csv("results/summary_names.csv")
    other = df[df.attribute != ATTRIBUTES[0]]
    other.groupby(["scorer", "job", "attribute", "label"], sort=False)["score"].mean().round(3).to_csv(
        "results/summary_attributes.csv")

    # shortlist simulation with the embedding scorer
    sl = []
    for job, jd in JOBS.items():
        pool = build_pool(job)
        s = scorers.embedding_scores(jd, [p["text"] for p in pool])
        d = pd.DataFrame(pool).drop(columns="text")
        d["score"] = s
        k = round(len(d) * 0.25)
        d = d.sort_values("score", ascending=False).reset_index(drop=True)
        d["shortlisted"] = d.index < k
        d["job"] = job
        sl.append(d)
    pd.concat(sl).to_csv("results/shortlist_top25.csv", index=False)

    history_sim.compare_all(1.0, 1.0).to_csv("results/history_mitigations.csv", index=False)
    print("Saved results to ./results/")


if __name__ == "__main__":
    main()
