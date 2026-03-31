"""
Constrained single-token MCQ scoring for HuggingFace models.
==============================================================
Instead of generating free text and parsing it for A/B/C/D,
we request max_tokens=1 and read the single output token directly.

When the provider supports logprobs, we also extract the full
probability distribution over A/B/C/D.

This **eliminates parse failures entirely** for HF models:
  - max_tokens=1 prevents rambling/multi-line responses
  - logprobs (where available) give confidence scores as a bonus

For non-HF models (OpenAI, Gemini, Claude, Ollama), the eval
scripts fall back to the standard generate-then-parse approach.

Provider compatibility (tested March 2026):
  - logprobs=True works:   Llama-3.2-1B, Llama-3.1-8B, Qwen3-8B,
                           Qwen3-32B, DS-R1-Distill-32B, Qwen2.5-72B,
                           Mixtral-8x22B
  - logprobs silently ignored: Qwen2.5-7B/14B/32B, Mistral-7B,
                                Gemma-2-9b, Gemma-3-27b, DeepSeek-R1
  - logprobs causes error:  Phi-3-mini, Phi-3-medium, Cohere Aya,
                             Llama-3.3-70B
"""

import math
import os
import re
import time
from typing import Dict, Optional, Tuple

VALID_LETTERS = {"A", "B", "C", "D"}

# Models that use <think> blocks and support /no_think to disable them
_THINKING_MODELS = {"Qwen/Qwen3-32B", "Qwen/Qwen3-8B"}


def _get_hf_client():
    """Lazily create a HuggingFace InferenceClient."""
    from huggingface_hub import InferenceClient

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN not set — cannot use logprob scoring")
    return InferenceClient(api_key=token)


def _model_id_from_llm_name(llm_name: str) -> str:
    """Extract HF model ID from our naming convention.

    'huggingface:Qwen/Qwen2.5-7B-Instruct' -> 'Qwen/Qwen2.5-7B-Instruct'
    """
    return llm_name.replace("huggingface:", "")


def _token_to_letter(token_text: str) -> Optional[str]:
    """Map a token string to A/B/C/D if it matches."""
    token_stripped = token_text.strip().upper().rstrip(":.)").lstrip("(")
    if token_stripped in VALID_LETTERS:
        return token_stripped
    return None


def _extract_letter_from_text(text: str) -> Optional[str]:
    """Extract A/B/C/D from free-text response using robust patterns.

    Only matches STANDALONE letters — won't grab 'a' from 'a complicated'.
    Priority order: most specific patterns first.

    Examples that SHOULD match:
        "B"                    -> B
        "B: Paris"             -> B
        "The answer is B"      -> B
        "(B)"                  -> B
        "**B**"                -> B
        "Based on ... B."      -> B

    Examples that should NOT match:
        "a complicated one"    -> None  (lowercase 'a' in a word)
        "Because..."           -> None  ('B' starts a word)
    """
    import re
    if not text:
        return None

    # 1. Entire response is just the letter (possibly with punctuation)
    stripped = text.strip().upper().rstrip(":.)").lstrip("(")
    if stripped in VALID_LETTERS:
        return stripped

    # 2. Response starts with the letter followed by non-alpha
    #    e.g. "B", "B:", "B.", "B) Paris"
    m = re.match(r"^([A-D])\s*[^a-zA-Z]", text.strip())
    if m:
        return m.group(1).upper()

    # 3. Priority patterns — checked in order, first match wins
    patterns = [
        r"\(([A-D])\)",                          # (B)
        r"\[([A-D])\]",                          # [B]
        r"\*\*([A-D])\*\*",                      # **B**
        r"[Aa]nswer\s*(?:is|:)\s*([A-D])\b",    # "answer is B" / "answer: B"
        r"[Oo]ption\s+([A-D])\b",               # "option B"
        r"(?:choose|select|pick)\s+([A-D])\b",   # "choose B"
        r"correct\s+(?:answer|option)\s+is\s+([A-D])\b",  # "correct answer is B"
        r"(?:^|\.\s+)([A-D])\.",                 # "B." at start or after sentence
        r"\b(?:is|be)\s+([A-D])\b",             # "is B" / "would be B"
        r"(?:it'?s|its)\s+([A-D])\b",           # "it's B" / "its B"
        r",\s*([A-D])\.?\s*$",                   # ", B" or ", B." at end
        r"\s([A-D])\.?\s*$",                     # trailing " B" or " B." at end of text
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()

    return None


def _extract_logprobs(result) -> Tuple[Dict[str, float], bool]:
    """Extract letter probabilities from chat_completion result.

    Returns (letter_probs, has_logprobs).
    """
    letter_probs: Dict[str, float] = {}
    has_logprobs = False

    try:
        if (
            result.choices[0].logprobs
            and result.choices[0].logprobs.content
        ):
            has_logprobs = True
            for lp_entry in result.choices[0].logprobs.content:
                # Check the generated token itself
                letter = _token_to_letter(lp_entry.token)
                if letter and letter not in letter_probs:
                    letter_probs[letter] = math.exp(lp_entry.logprob)

                # Check all top tokens
                if lp_entry.top_logprobs:
                    for tlp in lp_entry.top_logprobs:
                        letter = _token_to_letter(tlp.token)
                        if letter and letter not in letter_probs:
                            letter_probs[letter] = math.exp(tlp.logprob)
    except Exception:
        pass

    return letter_probs, has_logprobs


def call_llm_logprob(
    llm_name: str,
    prompt_text: str,
    max_retries: int = 2,
) -> Tuple[Optional[str], Dict[str, float], str]:
    """
    Call an HF model with max_tokens=1 and extract the answer letter.

    Strategy:
      1. Try with logprobs=True, top_logprobs=5
      2. If provider rejects logprobs, retry WITHOUT logprobs
      3. In either case, read the single output token
      4. If logprobs available, pick letter by highest probability
      5. Otherwise, parse the single token directly

    Returns:
        (answer_letter, letter_probs, raw_token)
        - answer_letter: 'A'/'B'/'C'/'D' or None
        - letter_probs: {letter: probability} (empty if no logprobs)
        - raw_token: the actual generated token text (for logging)
    """
    client = _get_hf_client()
    model_id = _model_id_from_llm_name(llm_name)
    is_thinking_model = model_id in _THINKING_MODELS

    # For thinking models (Qwen3), append /no_think to disable <think> blocks
    effective_prompt = prompt_text
    if is_thinking_model:
        effective_prompt = prompt_text + "\n/no_think"

    for attempt in range(max_retries + 1):
        try:
            # --- Attempt 1: with logprobs ---
            try:
                result = client.chat_completion(
                    model=model_id,
                    messages=[{"role": "user", "content": effective_prompt}],
                    max_tokens=1,
                    temperature=0.01,
                    logprobs=True,
                    top_logprobs=5,
                )
            except Exception as logprob_err:
                # Provider doesn't support logprobs — retry without
                err_msg = str(logprob_err).lower()
                if any(t in err_msg for t in (
                    "logprob", "unprocessable",
                    "bad request", "422", "400",
                )) and "stopiteration" not in err_msg:
                    result = client.chat_completion(
                        model=model_id,
                        messages=[{"role": "user", "content": effective_prompt}],
                        max_tokens=1,
                        temperature=0.01,
                    )
                else:
                    raise  # Re-raise transient/auth errors

            raw_token = result.choices[0].message.content or ""

            # For thinking models, strip any residual <think> wrapper from token
            if is_thinking_model:
                raw_token = re.sub(
                    r"<think>.*?</think>\s*", "", raw_token, flags=re.DOTALL
                ).strip()

            letter_probs, has_logprobs = _extract_logprobs(result)

            # If logprobs found letters, pick highest probability
            if letter_probs:
                answer = max(letter_probs, key=letter_probs.get)
                return answer, letter_probs, raw_token

            # No logprobs — parse the single token directly
            fallback_letter = _token_to_letter(raw_token)
            if fallback_letter:
                return fallback_letter, {fallback_letter: 1.0}, raw_token

            # First token wasn't a letter (e.g. " Based", " The", "<think>", "")
            # Tier 1: give the model more tokens to finish its thought
            # Use 500 tokens for thinking models (need room for think block),
            # 200 for others (up from 100, helps Gemma verbose responses)
            tier1_tokens = 500 if is_thinking_model else 200
            result2 = client.chat_completion(
                model=model_id,
                messages=[{"role": "user", "content": effective_prompt}],
                max_tokens=tier1_tokens,
                temperature=0.01,
            )
            full_text = result2.choices[0].message.content or ""
            # Strip <think>...</think> blocks (may be present even with /no_think
            # on some providers, or for non-thinking models that still emit them)
            clean = re.sub(r"<think>.*?</think>", "", full_text, flags=re.DOTALL).strip()
            if clean:
                letter = _extract_letter_from_text(clean)
                if letter:
                    return letter, {}, full_text

            # Tier 2: use chat history — feed the model's own response back
            # as an assistant turn, then ask for just the letter
            tier2_prompt = effective_prompt
            result3 = client.chat_completion(
                model=model_id,
                messages=[
                    {"role": "user", "content": tier2_prompt},
                    {"role": "assistant", "content": full_text},
                    {"role": "user", "content": (
                        "Now respond with ONLY the letter of your answer. "
                        "Just one letter: A, B, C, or D.\n/no_think"
                        if is_thinking_model else
                        "Now respond with ONLY the letter of your answer. "
                        "Just one letter: A, B, C, or D."
                    )},
                ],
                max_tokens=5 if is_thinking_model else 1,
                temperature=0.01,
            )
            forced_token = (result3.choices[0].message.content or "").strip()
            # Strip think blocks from forced response too
            forced_token = re.sub(
                r"<think>.*?</think>\s*", "", forced_token, flags=re.DOTALL
            ).strip()
            forced_letter = _token_to_letter(forced_token)
            if not forced_letter:
                forced_letter = _extract_letter_from_text(forced_token)
            if forced_letter:
                return forced_letter, {}, full_text

            return None, letter_probs, raw_token

        except Exception as e:
            err_str = str(e).lower()
            is_transient = any(
                t in err_str
                for t in ("rate", "timeout", "429", "503", "overloaded", "502")
            )
            if is_transient and attempt < max_retries:
                wait = 5 * (attempt + 1)
                print(f"    [retry {attempt + 1}/{max_retries} in {wait}s: {e}]")
                time.sleep(wait)
                continue
            print(f"    ! logprob call failed ({model_id}): {type(e).__name__}: {e}")
            return None, {}, ""

    return None, {}, ""


def is_hf_model(llm_name: str) -> bool:
    """Check if model uses HuggingFace Inference API."""
    return llm_name.startswith("huggingface:")


def format_mcq_prompt(
    question: str, options: Dict[str, str], context: str = "", mode: str = "plain"
) -> str:
    """Build the MCQ prompt — identical to the original but shared here."""
    options_text = "\n".join(f"{k}: {v}" for k, v in options.items())

    if mode == "informed" and context:
        return (
            f"You may find the following reference material helpful. "
            f"Use it if relevant, otherwise rely on your own knowledge.\n\n"
            f"REFERENCE:\n{context}\n\n"
            f"QUESTION:\n{question}\n\n"
            f"OPTIONS:\n{options_text}\n\n"
            f"Respond with ONLY the letter (A, B, C, or D)."
        )
    elif context:  # rag_only
        return (
            f"Use the following context to answer the question.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION:\n{question}\n\n"
            f"OPTIONS:\n{options_text}\n\n"
            f"Respond with ONLY the letter (A, B, C, or D)."
        )
    else:
        return (
            f"QUESTION:\n{question}\n\n"
            f"OPTIONS:\n{options_text}\n\n"
            f"Respond with ONLY the letter (A, B, C, or D)."
        )


def evaluate_answer_logprob(
    llm_name: str,
    question: str,
    options: Dict[str, str],
    context: str = "",
    mode: str = "plain",
) -> Tuple[Optional[str], str, Dict[str, float]]:
    """
    Evaluate a single MCQ answer using constrained single-token scoring.

    Returns:
        (letter, raw_token, letter_probs)
    """
    prompt = format_mcq_prompt(question, options, context=context, mode=mode)
    letter, probs, raw = call_llm_logprob(llm_name, prompt)
    return letter, raw, probs
