---
title: CV Screening Bias Lab (ZeroGPU test)
emoji: 🧪
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
pinned: false
license: mit
short_description: Free ZeroGPU version with a smaller open model
---

Test copy of [CV Screening Bias Lab](https://huggingface.co/spaces/edwardwww/cv-screening-bias-lab) that runs
`Qwen/Qwen2.5-7B-Instruct` on Hugging Face's free ZeroGPU instead of the paid Inference API.
Space settings: hardware **ZeroGPU**, variables `SCORER_BACKEND=zerogpu` and `CACHE_FILE=zerogpu_cache.json`.
