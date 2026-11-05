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
