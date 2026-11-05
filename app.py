"""CV Screening Bias Lab: a Gradio app for the IS5113 AI Ethics project (CityU).

Two tabs: a four-step Presentation demo (see demo3.py) and an About page.
All CVs and data are synthetic.
"""

import gradio as gr

import demo3
from cv_data import JOBS

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
