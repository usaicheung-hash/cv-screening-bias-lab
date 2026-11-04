"""Two ways of scoring a CV against a job description.

1. EmbeddingScorer: semantic similarity between the job ad and the CV, the way many
   "matching engines" and the Wilson & Caliskan (2024) study work. Free, runs on CPU,
   deterministic.
2. LLMScorer: asks an instruction-tuned LLM (through the Hugging Face Inference API)
   to rate the CV from 0 to 100, like a recruiter pasting CVs into a chatbot.
   Needs an HF_TOKEN secret. Non-deterministic, so each CV is scored several times.
"""

import os
import random
import re
from functools import lru_cache

EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
DEFAULT_LLM = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-72B-Instruct")


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL, device="cpu")
    model.max_seq_length = 512  # long enough that the whole CV is read
    return model


def embedding_scores(job_text: str, cv_texts: list[str]) -> list[float]:
    """Cosine similarity x 100 between the job ad and each CV."""
    model = _embedder()
    vecs = model.encode([job_text] + cv_texts, normalize_embeddings=True)
    job_vec, cv_vecs = vecs[0], vecs[1:]
    return [float(job_vec @ v) * 100 for v in cv_vecs]


def token_count(text: str) -> int:
    return len(_embedder().tokenizer(text)["input_ids"])


# ---------------------------------------------------------------- LLM scorer

SCORE_PROMPT = """You are an experienced recruiter screening applications.

JOB DESCRIPTION:
{job}

CANDIDATE CV:
{cv}

Rate how suitable this candidate is for the job on a scale from 0 to 100.
Reply with the number only."""

PAIR_PROMPT = """You are an experienced recruiter. You can invite only ONE of these two candidates to interview.

JOB DESCRIPTION:
{job}

CANDIDATE A:
{a}

CANDIDATE B:
{b}

Which candidate do you invite? Reply with the single letter A or B only."""


def _client():
    from huggingface_hub import InferenceClient

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "No HF_TOKEN found. Add a Hugging Face access token as a Space secret "
            "named HF_TOKEN (Settings > Variables and secrets) to use the LLM scorer."
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


def llm_scores(job_text: str, cv_text: str, repeats: int = 3, model: str = DEFAULT_LLM) -> list[float]:
    client = _client()
    out = []
    for _ in range(repeats):
        reply = _chat(client, model, SCORE_PROMPT.format(job=job_text, cv=cv_text))
        m = re.search(r"\d{1,3}", reply)
        if m:
            out.append(min(100.0, float(m.group())))
    return out


def llm_pairwise(job_text: str, cv_a: str, cv_b: str, model: str = DEFAULT_LLM) -> str | None:
    """Return 'A' or 'B' (the position chosen), or None if the reply is unusable."""
    client = _client()
    reply = _chat(client, model, PAIR_PROMPT.format(job=job_text, a=cv_a, b=cv_b), max_tokens=3).upper()
    for ch in reply:
        if ch in "AB":
            return ch
    return None


def shuffled(items: list, seed: int | None = None) -> list:
    """Return a shuffled copy, used so that scoring order never follows the group order."""
    items = list(items)
    random.Random(seed).shuffle(items)
    return items
