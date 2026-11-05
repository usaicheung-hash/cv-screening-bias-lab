---
title: CV Screening Bias Lab
emoji: 🧪
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
pinned: false
license: mit
short_description: Do AI CV screeners treat identical CVs the same?
---

# 🧪 CV Screening Bias Lab

**Does an AI CV screener give the same score to the same CV when only the name, age, career break or disability changes? And can a hiring model that never sees gender or age still discriminate?**

▶️ **Live demo:** https://huggingface.co/spaces/edwardwww/cv-screening-bias-lab

📖 **User guide:** [English](docs/USER_GUIDE_EN.md) · [中文（繁體）](docs/USER_GUIDE_ZH.md)

A teaching demo built for the IS5113 *AI Ethics* group project at City University of Hong Kong (Part 1, Topic 1: *Algorithmic Bias in Hiring Platforms*). **All CVs and data are synthetic.**

![Step 2: same CV, different score](docs/screenshot_step2.png)

## What you can do

| Tab | Experiment |
|---|---|
| **🎤 Presentation demo** | Four steps for a talk: (1) the Hong Kong job post, standard CV and 10 names; (2) an LLM recruiter scores identical CVs that differ in one detail and explains the highest and lowest score; (3) how a "blind" AI learns bias from history, and why; (4) add your own fairness instruction to the prompt and compare the scores before and after. |
| **ℹ️ About and ethics** | Background, ethics of the demo and references. |

More screenshots: [Step 1](docs/screenshot_step1.png) · [Step 3](docs/screenshot_step3.png) · [Step 4](docs/screenshot_step4.png)

## Run locally

```bash
pip install -r requirements.txt
python app.py            # open http://127.0.0.1:7860
```

Steps 1 and 3 work offline. Steps 2 and 4 show saved results from `demo_cache.json`; to ask the AI again, set an environment variable `HF_TOKEN` (a Hugging Face token with "Make calls to Inference Providers" permission).

To deploy your own copy on Hugging Face Spaces, see [HOW_TO_DEPLOY.md](HOW_TO_DEPLOY.md). The YAML block at the top of this README is the Space configuration.

## Files

| File | Purpose |
|---|---|
| `app.py` | Gradio interface: 🎤 Presentation demo and ℹ️ About and ethics |
| `demo3.py` | The four steps: page layout, LLM scoring with reasons, the bias check and the prompt fix |
| `demo_cache.json` | Saved AI scores, so the presentation shows the same numbers every time |
| `cv_data.py` | Synthetic Hong Kong job ads, CV templates and the details that change |
| `variants.py` | Fills a CV template so copies differ in exactly one detail |
| `scorers.py` | Connects to the AI recruiter (default `Qwen/Qwen2.5-72B-Instruct`) |
| `history_sim.py` | Step 3: biased-history simulation and the "blind" AI |

## Limitations and ethics

This is an illustrative experiment, not a scientific audit: few CVs, four jobs, one model. Use it as an auditing and teaching tool, not to optimise a CV against a screener. AI tools (Claude) helped write the code and documentation; the project group designed the experiment and interpreted the results.

## Key references

- Wilson & Caliskan (2024), *Gender, Race, and Intersectional Bias in Resume Screening via Language Model Retrieval*, AIES.
- Bloomberg (2024), *OpenAI's GPT Is a Recruiter's Dream Tool. Tests Show There's Racial Bias.*
- Glazko et al. (2024), *Identifying and Improving Disability Bias in GPT-Based Resume Screening*, FAccT.
- Rozado (2025), *Gender and Positional Biases in LLM-Based Hiring Decisions*, arXiv:2505.17049.
- *Mobley v. Workday, Inc.*, N.D. Cal.
- ILO (2018), *Care Work and Care Jobs for the Future of Decent Work*.

Licensed under the [MIT License](LICENSE).
