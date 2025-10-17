# qa_engine.py
import os
import json
import time
from typing import List, Dict, Any, Optional

# google-genai SDK
from google import genai
from google.genai import types

# Configure client (uses env var GOOGLE_API_KEY or application default credentials)
client = genai.Client(api_key="")

# Default model — change to whatever is available to you
DEFAULT_MODEL = "gemini-2.0-flash"


def _build_prompt(question: str, options: List[str]) -> str:
    """
    Build a concise instruction prompt asking the model to respond with JSON.
    """
    # Make numbered options if not already
    formatted_opts = []
    labels = []
    for i, opt in enumerate(options):
        label = chr(ord("A") + i)
        labels.append(label)
        formatted_opts.append(f"{label}. {opt.strip()}")
    opts_block = "\n".join(formatted_opts)

    prompt = f"""
You are an assistant that answers multiple-choice questions.
Respond ONLY with a single valid JSON object (no extra text) with keys:
 - choice: the single letter of the chosen option (e.g. \"A\") or the option text.
 - confidence: a numeric confidence from 0.0 to 1.0 (estimate).
 - explanation: a short (1-2 sentences) rationale for the choice.

Question: {question.strip()}
Options:
{opts_block}

Return only JSON. Example:
{{"choice":"A","confidence":0.85,"explanation":"Because ..."}}
"""
    return prompt.strip()


def get_answer_from_gemini(question: str, options: List[str],
                           model: str = DEFAULT_MODEL,
                           max_retries: int = 2,
                           timeout_secs: int = 30) -> Dict[str, Any]:
    """
    Query Gemini and return parsed JSON result.
    Returns a dict: {"choice":..., "confidence":..., "explanation":...}
    Raises ValueError on parse failure or RuntimeError on API errors.
    """
    prompt = _build_prompt(question, options)

    attempt = 0
    backoff = 1.0
    while attempt <= max_retries:
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt
            )
            # google-genai returns response object(s); extract text
            text = ""
            # response may have different structure; handle common case:
            if hasattr(response, "text") and response.text:
                text = response.text
            else:
                # try to join partials
                parts = []
                for msg in getattr(response, "candidates", []) or []:
                    parts.append(getattr(msg, "content", "") or getattr(msg, "text", ""))
                text = "\n".join(parts).strip()

            # Safety: trim surrounding whitespace
            text = text.strip()

            # Try to parse JSON from the start of the model output
            # Some models might add backticks or triple quotes; sanitize:
            # find first '{' and last '}'
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end < start:
                raise ValueError(f"Could not find JSON in model output: {text!r}")

            json_str = text[start:end+1]
            parsed = json.loads(json_str)

            # Basic validation
            if "choice" not in parsed:
                raise ValueError("Parsed JSON missing 'choice' key.")
            # Normalize choice to letter if possible
            choice = parsed["choice"]
            # if user provided options, convert option-text to letter if needed
            if isinstance(choice, str) and len(choice.strip()) > 1:
                # map option text -> letter (best-effort)
                normalized = None
                lc_choice = choice.strip().lower()
                for i, opt in enumerate(options):
                    if lc_choice == opt.strip().lower():
                        normalized = chr(ord("A") + i)
                        break
                if normalized:
                    parsed["choice"] = normalized

            # cast confidence
            conf = parsed.get("confidence", None)
            if conf is None:
                parsed["confidence"] = 0.0
            else:
                try:
                    parsed["confidence"] = float(conf)
                except Exception:
                    parsed["confidence"] = 0.0

            return parsed

        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                raise RuntimeError(f"Gemini API failed after {attempt} attempts: {e}") from e
            time.sleep(backoff)
            backoff *= 2

# For quick local test
if __name__ == "__main__":
    q = "Who is the father of the C language?"
    opts = ["Steve Jobs", "James Gosling", "Dennis Ritchie", "Rasmus Lerdorf"]
    out = get_answer_from_gemini(q, opts)
    print("Result:", out)

