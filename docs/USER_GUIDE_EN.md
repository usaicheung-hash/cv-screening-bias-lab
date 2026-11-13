# CV Screening Bias Lab: User Guide (English)

**Live app:** https://huggingface.co/spaces/edwardwww/cv-screening-bias-lab
**Chinese version:** [USER_GUIDE_ZH.md](USER_GUIDE_ZH.md)

---

## 1. What is this app?

CV Screening Bias Lab is an interactive teaching demo. It answers two questions:

1. **Does an AI CV screener give the same score to the same CV when only the candidate's name, age, career break or disability changes?**
2. **Can a hiring model that never sees gender or age still discriminate?**

It was built for the IS5113 *AI Ethics* group project at City University of Hong Kong (Part 1, Topic 1: *Algorithmic Bias in Hiring Platforms*). It turns recent research and court cases into experiments anyone can run in a browser.

**All CVs and data are synthetic.** No real person's information is used.

---

## 2. Background: why this matters

More and more employers use AI to sort, score and reject job applicants before a human reads the CV. Recent evidence shows these tools can treat equal candidates unequally:

| Source | What it found |
|---|---|
| Wilson & Caliskan, University of Washington (AIES 2024) | In 3 million CV-to-job comparisons, AI models preferred white-associated names 85% of the time and never preferred Black male names over white male names. |
| Bloomberg (2024) | GPT ranked identical CVs differently depending on the name's implied race and gender. |
| Glazko et al. (FAccT 2024) | GPT-4 ranked CVs lower when they mentioned disability-related awards or activities. |
| Rozado (2025) | 22 LLMs favoured female names and, more strikingly, whichever candidate was listed **first**. |
| *Mobley v. Workday* (US, 2023 to now) | Class action alleging an AI screening platform rejected applicants aged 40+, Black and disabled applicants, partly through proxies such as employment gaps. |
| NY State Comptroller audit (Dec 2025) | New York City's AI hiring-audit law (Local Law 144) was barely enforced. |

The app lets you reproduce these kinds of tests on a small scale and see the mechanisms for yourself.

---

## 3. Presentation demo (about 4 minutes): the 🎤 Presentation demo tab

The first tab tells the story in **four steps**. Pick a job at the top (default: Marketing Executive, Hong Kong).

**Step 1 · What are we testing?** Shows the Hong Kong job post (the requirement for fluent Cantonese and English and good written Chinese is highlighted), the standard CV (the yellow parts are the only things that change), and the 10 names used: Hong Kong, British, Black African and Japanese (one man and one woman each), South Asian and Filipino. It also lists the other tests: age (graduated 2019 vs 1996), gender, career break and disability.

**Step 2 · Same CV, different score.** An AI recruiter (the large language model Qwen2.5-72B-Instruct) gives each copy a score from 0 to 100 and a short reason. Pick a test (🌏 Name, 🎂 Age, ⚧ Gender, 👶 Career break, ♿ Disability):
- the bars show each copy's score (average of 3 runs), highest in green and lowest in red;
- the box under the bars gives the gap in points, and how much lower a clearly weaker CV scored, for scale;
- on the right, the AI's own reasons for the **highest** and **lowest** score are shown side by side;
- the 💡 **Insight** box says in plain words what this means. It checks whether the AI's reason openly mentions the changed detail, or hides it behind reasonable-sounding words. If the gap is within the AI's normal run-to-run variation, it says so honestly.
- For ⚧ Gender, the box shows the average score of the 5 men and the 5 women, and the two reasons come from the same-background pair with the biggest gap.

Results are saved, so the presentation shows the same numbers every time and does not wait for the model. **🔄 Re-run live** asks the AI again (about 1 minute) and replaces the saved result. Step 2 needs the `HF_TOKEN` Space secret.

**Step 3 · How an AI learns bias from history.** Three cards: (1) past recruiters hired 55 of every 100 men but only 25 of every 100 women with the same skills; (2) we train an AI on those decisions and delete gender and age, but career gap and graduation year remain; (3) the AI gives equally qualified women and over-40s about 11 to 13 points less. Below, **"Why does the AI learn the bias?"** explains it in three points with the data: the AI's "right answer" is the past decisions; 55 of every 100 women but only 5 of every 100 men have a career gap; in the past only 27 of every 100 people with a gap were hired, versus 46 without, so the AI learns "career gap = lower score".

**Step 4 · Fix it with a better prompt.** A text box holds rules for the AI recruiter, for example "do not consider nationality; judge Cantonese, English and Chinese only from what the CV states; a caregiving break is not a weakness". You can edit it. Pick a test and click **▶ Apply instruction**: the same CVs are scored again with your rules added to the prompt, and the page shows **Before (normal prompt)** and **After (prompt + your instruction)** side by side, with the gap and the AI's reason for the lowest score. The 💡 Result box says whether the clear gap disappeared, stayed, or was never there. The default instruction is pre-computed for Marketing Executive; editing it or clicking **🔄 Run again** asks the AI live (about 1 minute). Lesson: a better prompt can reduce bias, but it is not a guarantee, so the fix must be tested again.

![Step 2](screenshot_step2.png)

![Step 3](screenshot_step3.png)

![Step 4](screenshot_step4.png)

### How the Step 3 data is simulated
| Feature | Job-relevant? | Notes |
|---|---|---|
| `skill_test`, `years_experience`, `education_level` | Yes | Same distribution for every group. |
| `career_gap` | **No** | 55% of women vs 5% of men have one (caregiving). Works as a hidden **gender** signal. |
| `grad_year` | **No** | Directly reveals age. Works as a hidden **age** signal. |

- **True qualification** depends only on the job-relevant features.
- **Past decision** = true qualification − (bias × being female) − (bias × being 40+) + noise.
- The AI is a logistic regression trained on the past decisions, without gender or age, and scored on a held-out 40% test set.

---

## 4. Suggested presentation script (about 4 minutes)

1. **Step 1** (30 s): "One Hong Kong job, one standard CV. We make copies where only one detail changes. A fair recruiter gives every copy the same score."
2. **Step 2, 🌏 Name** (45 s): "The scores are close and the gap could be chance. No clear bias here, and that is a finding too."
3. **Step 2, 👶 Career break** (45 s): "Same CV, but a 2-year caregiving break costs 5 points, and the AI's own reason says the break may affect recent knowledge."
4. **Step 3** (60 s): "Even when we delete gender and age, an AI trained on biased past decisions finds clues, career gap and graduation year, and repeats the bias."
5. **Step 4, 👶 Career break** (45 s): "We add fairness rules to the prompt. The gap falls from 5.0 to 3.3 points and is no longer clear. But a prompt is not a guarantee, so every fix must be tested again."

---

## 5. View and download the data

All the data behind the demo is synthetic and public as a Hugging Face dataset:
**[edwardwww/cv-screening-bias-lab-data](https://huggingface.co/datasets/edwardwww/cv-screening-bias-lab-data)**. Anyone can view or download it; no account is needed.

| Subset (config) | Rows | What it is |
|---|---|---|
| `hiring_history` | 6,000 | Step 3: simulated past applicants and the biased human hiring decisions |
| `cv_variants` | 80 | Steps 1 and 2: the 4 job ads and the CV copies that differ in one detail |
| `llm_scores` | 177 | Steps 2 and 4: the AI recruiter's saved scores and reasons (3 runs per CV) |

**View it in the browser.** Open the link above. The **Dataset Viewer** at the top shows the table; pick a
table in the **Subset** box. You can sort columns, search, and click **Data Studio** to filter with SQL.
(Just after an upload the viewer can say "should be available soon" for a while; reload later.)

**Download in the browser.** Open the **Files and versions** tab, go into `data/`, click a CSV file
(for example `hiring_history.csv`) and click the **download** icon. The CSVs open in Excel or Google Sheets.

**Load it in Python.**
```python
# pip install datasets pandas
from datasets import load_dataset

history = load_dataset("edwardwww/cv-screening-bias-lab-data", "hiring_history", split="train").to_pandas()
cvs     = load_dataset("edwardwww/cv-screening-bias-lab-data", "cv_variants",    split="train").to_pandas()
scores  = load_dataset("edwardwww/cv-screening-bias-lab-data", "llm_scores",     split="train").to_pandas()

print(history.groupby("gender")[["career_gap", "truly_qualified", "hired_historically"]].mean())
```
Or read a CSV straight into pandas:
```python
import pandas as pd
url = "https://huggingface.co/datasets/edwardwww/cv-screening-bias-lab-data/resolve/main/data/hiring_history.csv"
history = pd.read_csv(url)
```
The same files are also in this repository under [`dataset/`](../dataset), and the dataset card
([`dataset/README.md`](../dataset/README.md)) explains every column and how the data was generated (seed 7).
The app itself does not read this dataset: it re-creates the Step 3 data in memory each time it starts.

---

## 6. Run it yourself

**On your computer**
```
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:7860. Steps 1 and 3 work offline. Steps 2 and 4 show saved results from `demo_cache.json`; to ask the AI again, set an environment variable `HF_TOKEN` to a Hugging Face token with "Make calls to Inference Providers" permission.

**On the public Space,** saved results are free to view, but asking the AI again (Re-run live, Run again, a new Step 4 instruction) needs the owner password, because each call spends the owner's credits.

**Deploy your own copy on Hugging Face Spaces:** see [`HOW_TO_DEPLOY.md`](../HOW_TO_DEPLOY.md).

### Files
| File | Purpose |
|---|---|
| `app.py` | Gradio interface: 🎤 Presentation demo and ℹ️ About and ethics |
| `demo3.py` | The four steps: page layout, LLM scoring with reasons, the bias check and the prompt fix |
| `demo_cache.json` | Saved AI scores, so the presentation shows the same numbers every time |
| `cv_data.py` | Synthetic Hong Kong job ads, CV templates and the details that change |
| `variants.py` | Fills a CV template so copies differ in exactly one detail |
| `scorers.py` | Connects to the AI recruiter (default `Qwen/Qwen2.5-72B-Instruct`) |
| `history_sim.py` | Step 3: biased-history simulation and the "blind" AI |

---

## 7. Limitations and ethics

- **Illustrative, not a scientific audit:** few CVs, four jobs, one model, 3 runs per CV. Results can change between jobs, CVs, models and runs. A "no bias" result is also a finding.
- **Synthetic data only.** Names are typical of groups in Hong Kong's labour market but represent no real person.
- **Simplified simulation:** Step 3 is a model of how bias is learned, not real company data.
- **Use it as an auditing tool**, not to find a "best-scoring" name or to optimise a CV to beat a screener.
- **AI-use declaration:** AI tools (Claude) helped write the code and documentation. The project group designed the experiment, checked and interpreted the results.

---

## 8. References

- Wilson, K. & Caliskan, A. (2024). *Gender, Race, and Intersectional Bias in Resume Screening via Language Model Retrieval*. AAAI/ACM AIES.
- Bloomberg (2024). *OpenAI's GPT Is a Recruiter's Dream Tool. Tests Show There's Racial Bias.*
- Glazko, K. et al. (2024). *Identifying and Improving Disability Bias in GPT-Based Resume Screening*. ACM FAccT.
- Rozado, D. (2025). *Gender and Positional Biases in LLM-Based Hiring Decisions*. arXiv:2505.17049.
- *Mobley v. Workday, Inc.*, U.S. District Court, N.D. California (collective certified 16 May 2025).
- Office of the New York State Comptroller (2025). *Enforcement of Local Law 144: Automated Employment Decision Tools*.
- International Labour Organization (2018). *Care Work and Care Jobs for the Future of Decent Work*.
