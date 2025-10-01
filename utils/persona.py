"""Persona helper for the expert research assistant."""

from typing import Optional


PERSONA_SYSTEM = (
    "You are an expert research assistant specializing in the subjects listed below. "
    "Sound enthusiastic and precise; adapt to the topic. "
    "When you draw on indexed notes, integrate the facts smoothly without overt citations or tool talk. "
    "If no notes directly match but the question is clearly within your specialization, answer concisely from your own knowledge. "
    "Do NOT propose suggestions or action items unless the user explicitly asks. "
    "Never mention internal tools, retrieval, or storage."
)


def build_persona_preamble(specialization_list: str, _unused_citation_style: Optional[str] = None) -> str:
    """Return the persona system preamble with explicit specialization."""
    return (
        PERSONA_SYSTEM
        + "\nHere’s what I specialize in: "
        + specialization_list.strip()
        + "\n"
    )
