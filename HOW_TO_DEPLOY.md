# How to put the CV Screening Bias Lab online (Hugging Face Spaces)

Time needed: about 15 minutes. Cost: free.

## Option A: upload in the browser (easiest)

1. Create a free account at https://huggingface.co/join.
2. Go to https://huggingface.co/new-space.
   - **Space name:** `cv-screening-bias-lab`
   - **License:** MIT
   - **SDK:** Gradio (choose the "Blank" template)
   - **Hardware:** CPU basic (free)
   - **Visibility:** Public (so classmates and the lecturer can open it), or Private while testing.
3. Click **Create Space**.
4. Open the **Files** tab, click **Add file > Upload files**, and upload everything in this folder:
   `app.py`, `cv_data.py`, `variants.py`, `scorers.py`, `history_sim.py`, `run_experiment.py`,
   `requirements.txt`, `README.md`.
   (Replace the README the Space created; ours has the settings header the Space needs.)
5. Click **Commit**. The Space builds for 3 to 5 minutes (watch the **Logs** button). When it says
   **Running**, the app is live at `https://huggingface.co/spaces/<your-username>/cv-screening-bias-lab`.

The first click on **Run audit** takes about 20 seconds because the embedding model downloads once.

## Turn on the LLM scorer (optional, recommended)

The embedding screener works without this. The LLM recruiter and the position test need a token:

1. Go to https://huggingface.co/settings/tokens > **Create new token** > type **Fine-grained**,
   tick **"Make calls to Inference Providers"**, create, and copy it (starts with `hf_`).
2. In your Space: **Settings > Variables and secrets > New secret**.
   Name: `HF_TOKEN`, Value: the token. Save. The Space restarts.
3. Free accounts get a small monthly inference allowance, so keep **Repeats** low (3) during
   testing. If the default model gives an error, open **LLM settings** in the app and try another chat
   model id, for example `meta-llama/Llama-3.1-8B-Instruct` (accept its licence on its model page first)
   (any model with an "Inference Providers" widget on its Hugging Face page works).

**Never paste the token into the code or the report.** Only put it in the Space secret.

## Option B: run on your own computer first

```
cd cv-bias-demo
python -m venv .venv
.venv\Scripts\activate          (Windows)  or  source .venv/bin/activate  (Mac)
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:7860. For the LLM scorer, set the token first:
`set HF_TOKEN=hf_xxx` (Windows cmd) or `$env:HF_TOKEN="hf_xxx"` (PowerShell).

## Get results for the report

```
python run_experiment.py            # embedding screener, all jobs and attributes
python run_experiment.py --llm      # also the LLM (uses your inference allowance)
```
CSV files appear in `results/`. Put one chart in the report, and use these saved results in the
presentation as a backup in case the Wi-Fi or the API fails.

## Optional: also keep the code on GitHub

Create a GitHub repository, upload the same files, and link it in the Space's README.
You can also connect them so GitHub pushes update the Space (see "Managing Spaces with GitHub Actions"
in the Hugging Face docs). Not required for the project.

## Suggested 60-second live demo script

1. Tab 1, Data Analyst job, attribute **Name**, embedding screener: "Same CV, only the name changes. Look at the gap."
2. **Run shortlist** at 25%: "Small score gaps become big selection gaps. Here is the four-fifths check."
3. Tab 2, **Train and evaluate**: "The model never saw gender or age, yet it learned to penalise career gaps and graduation year."
4. Pick **Equal opportunity on past labels**: "A fairness fix built on biased labels can make things worse." Then **Compare all fixes**.
5. Close: "This test is exactly stage 3 of our FAIR-HIRE framework: test before use."

## What I could and could not test

The app was tested end to end on a local machine (all three tabs, the shortlist, the history simulation
and all five fixes) with a stand-in text model, because the build environment could not download models
from Hugging Face. The real embedding model and the LLM scorer will run for the first time on your Space,
so check the numbers there before quoting any in the report.
