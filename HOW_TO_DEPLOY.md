# How to put the CV Screening Bias Lab online (Hugging Face Spaces)

Time needed: about 15 minutes.

## 1. Create the Space

1. Create an account at https://huggingface.co/join.
2. Go to https://huggingface.co/new-space.
   - **Space name:** `cv-screening-bias-lab`
   - **License:** MIT
   - **SDK:** Gradio (choose the "Blank" template)
   - **Hardware:** CPU basic
   - **Visibility:** Public (so classmates and the lecturer can open it), or Private while testing.
3. Click **Create Space**.

## 2. Upload the files

Open the **Files** tab, click **Add file > Upload files**, and upload:
`app.py`, `demo3.py`, `demo_cache.json`, `cv_data.py`, `variants.py`, `scorers.py`, `history_sim.py`,
`requirements.txt`, `README.md`.
(Replace the README the Space created; ours has the settings header the Space needs.)

Click **Commit**. The Space builds for 2 to 4 minutes (watch the **Logs** button). When it says
**Running**, the app is live at `https://huggingface.co/spaces/<your-username>/cv-screening-bias-lab`.

`demo_cache.json` holds the saved AI scores, so Steps 2 and 4 show results straight away and the
presentation shows the same numbers every time.

## 3. Turn on the AI recruiter (needed for live re-runs)

Steps 1 and 3 need nothing. Steps 2 and 4 show the saved results without a token, but **🔄 Re-run live**,
**🔄 Run again** and editing the Step 4 instruction ask the AI again, which needs a token:

1. Go to https://huggingface.co/settings/tokens > **Create new token** > type **Fine-grained**,
   tick **"Make calls to Inference Providers"**, create, and copy it.
2. In your Space: **Settings > Variables and secrets > New secret**.
   Name: `HF_TOKEN`, Value: the token. Save. The Space restarts.
3. To use another chat model, add a Space variable `LLM_MODEL` with its model id (any model with an
   "Inference Providers" widget on its Hugging Face page works). Saved results are per model run, so
   re-run Steps 2 and 4 afterwards.

**Never paste the token into the code, a chat or the report.** Only put it in the Space secret.

### Lock the paid AI calls with a password

Because the Space is public, anyone could press **Re-run live** and spend your credits. The app only lets
someone ask the AI again (Re-run live, Run again, a new Step 4 instruction, or a job with no saved results)
if they type the owner password in the **🔒 Owner password** box. Saved results stay free for everyone.

1. In your Space: **Settings > Variables and secrets > New secret**.
   Name: `APP_PASSWORD`, Value: a password only you know. Save. The Space restarts.
2. If `APP_PASSWORD` is not set, nobody (not even you) can trigger new AI calls from the page.

The password is checked on the server, and the app's "Use via API" endpoints are switched off, so it
cannot be skipped by calling the API. **Never put the password in the code, GitHub, a chat or the report.**

To refresh `demo_cache.json` on your own computer, set `HF_TOKEN` and `APP_PASSWORD` in your terminal and run
`python -c "import demo3; open('demo_cache.json','w',encoding='utf-8').write(demo3.precompute('<job>', password='<your password>'))"`.

## 4. Run on your own computer (optional)

```
python -m venv .venv
.venv\Scripts\activate          (Windows)  or  source .venv/bin/activate  (Mac)
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:7860. For live re-runs, set the `HF_TOKEN` environment variable in your own
terminal first.

## 5. Keep the code on GitHub (optional)

The same files live in the GitHub repository. You can connect it so GitHub pushes update the Space
(see "Managing Spaces with GitHub Actions" in the Hugging Face docs).

For a suggested presentation script, see the user guide: [English](docs/USER_GUIDE_EN.md) ·
[中文](docs/USER_GUIDE_ZH.md).
