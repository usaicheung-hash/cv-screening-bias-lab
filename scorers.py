"""The AI recruiter: an instruction-tuned LLM called through the Hugging Face Inference API.

Needs an HF_TOKEN secret (set it as a Space secret, never in code).
The prompt itself lives in demo3.SCORE_REASON_PROMPT.
"""

import os

DEFAULT_LLM = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-72B-Instruct")


def _client():
    from huggingface_hub import InferenceClient

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "No HF_TOKEN found. Add a Hugging Face access token as a Space secret "
            "named HF_TOKEN (Settings > Variables and secrets) to use the AI recruiter."
        )
    return InferenceClient(token=token)


def _chat(client, model: str, prompt: str, temperature: float = 0.7, max_tokens: int = 8) -> str:
    resp = client.chat_completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------- free option: ZeroGPU
# Set the Space variable SCORER_BACKEND=zerogpu (and Space hardware "ZeroGPU") to run a
# smaller open model on Hugging Face's free shared GPU instead of the paid Inference API.

LOCAL = os.environ.get("SCORER_BACKEND", "").lower() == "zerogpu"

if LOCAL:
    import spaces
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    DEFAULT_LLM = os.environ.get("ZERO_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    _tok = AutoTokenizer.from_pretrained(DEFAULT_LLM, padding_side="left")
    _model = AutoModelForCausalLM.from_pretrained(DEFAULT_LLM, torch_dtype=torch.bfloat16)
    _model.to("cuda")

    @spaces.GPU(duration=120)
    def batch_chat(prompts: list[str], temperature: float = 0.7, max_tokens: int = 120) -> list[str]:
        texts = [_tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                          add_generation_prompt=True) for p in prompts]
        enc = _tok(texts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = _model.generate(**enc, max_new_tokens=max_tokens, do_sample=True,
                                  temperature=temperature, top_p=0.95, pad_token_id=_tok.eos_token_id)
        new = out[:, enc["input_ids"].shape[1]:]
        return [t.strip() for t in _tok.batch_decode(new, skip_special_tokens=True)]
