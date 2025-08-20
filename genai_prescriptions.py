# genai_prescriptions.py
import re
from typing import Dict, List

import streamlit as st
import google.generativeai as genai
from openai import OpenAI

# ---------------------------------------------------------------------
# Prompt + parser
# ---------------------------------------------------------------------

SECTION_PROMPT = """
You are a SOC analyst. Based on these URL indicators, produce:
1) RECOMMENDED ACTIONS: concise incident-response steps (5–8 bullets, technical).
2) COMMUNICATION DRAFT: a short, plain-language message (<=120 words) for execs/users.

STRICT OUTPUT FORMAT (no extra text outside these sections):

===RECOMMENDED ACTIONS===
- bullet 1
- bullet 2
- ...
===COMMUNICATION DRAFT===
<one short paragraph>
===END===
""".strip()


def _build_prompt(alert_details: Dict) -> str:
    return f"{SECTION_PROMPT}\n\nIndicators:\n{alert_details}\n"


def _parse_sections(text: str) -> Dict[str, object]:
    """
    Parse model response into:
      { "recommended_actions": [str, ...], "communication_draft": str }
    Always returns both keys (possibly empty).
    """
    text = text or ""

    actions_match = re.search(
        r"===RECOMMENDED ACTIONS===\s*(.*?)\s*===COMMUNICATION DRAFT===",
        text,
        re.S | re.I,
    )
    draft_match = re.search(
        r"===COMMUNICATION DRAFT===\s*(.*?)\s*===END===",
        text,
        re.S | re.I,
    )

    actions_block = (actions_match.group(1) if actions_match else "").strip()
    draft_block = (draft_match.group(1) if draft_match else "").strip()

    actions: List[str] = []
    for line in actions_block.splitlines():
        line = line.strip()
        if not line:
            continue
        # remove common bullet prefixes
        line = line.lstrip("-•\t ").strip()
        if line:
            actions.append(line)

    return {
        "recommended_actions": actions,
        "communication_draft": draft_block,
    }


# ---------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------

def _openai_call(prompt: str) -> Dict[str, object]:
    # Ensure key exists
    if "OPENAI_API_KEY" not in st.secrets:
        return {
            "recommended_actions": [],
            "communication_draft": "",
            "error": "Missing OPENAI_API_KEY in .streamlit/secrets.toml",
        }
    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        resp = client.chat.completions.create(
            model="gpt-4o-mini",           # affordable + capable
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=600,
        )
        text = (resp.choices[0].message.content or "").strip()
        parsed = _parse_sections(text)
        if not parsed["recommended_actions"] and not parsed["communication_draft"]:
            parsed["communication_draft"] = (
                "The AI returned an empty response. Please retry or switch provider."
            )
        return parsed
    except Exception as e:
        return {
            "recommended_actions": [],
            "communication_draft": "",
            "error": f"OpenAI error: {e}",
        }


def _gemini_call(prompt: str) -> Dict[str, object]:
    # Ensure key exists
    if "GEMINI_API_KEY" not in st.secrets:
        return {
            "recommended_actions": [],
            "communication_draft": "",
            "error": "Missing GEMINI_API_KEY in .streamlit/secrets.toml",
        }
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel("gemini-1.5-flash")
        resp = model.generate_content(prompt)
        text = getattr(resp, "text", "") or ""
        parsed = _parse_sections(text)
        if not parsed["recommended_actions"] and not parsed["communication_draft"]:
            parsed["communication_draft"] = (
                "The AI returned an empty response. Please retry or switch provider."
            )
        return parsed
    except Exception as e:
        return {
            "recommended_actions": [],
            "communication_draft": "",
            "error": f"Gemini error: {e}",
        }


# ---------------------------------------------------------------------
# Public entry point used by app.py
# ---------------------------------------------------------------------

def generate_prescription(provider: str, alert_details: Dict) -> Dict[str, object]:
    """
    Returns:
      {
        "recommended_actions": [str, ...],
        "communication_draft": str,
        # optional:
        "error": str
      }
    """
    p = (provider or "").strip().lower()
    prompt = _build_prompt(alert_details)

    if p == "openai":
        return _openai_call(prompt)
    if p == "gemini":
        return _gemini_call(prompt)

    # Not implemented (e.g., Grok)
    return {
        "recommended_actions": ["Selected provider is not supported. Choose OpenAI or Gemini."],
        "communication_draft": "Automated draft unavailable for the selected provider.",
        "error": f"Provider '{provider}' not supported in this build.",
    }
