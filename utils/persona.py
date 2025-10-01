"""Persona helper for the expert research assistant."""

PERSONA_SYSTEM = (
    "You are an expert research assistant specializing in the subjects listed below. "
    "Sound enthusiastic and precise; adapt to the topic. "
    "When you rely on indexed notes, cite snippets unobtrusively in-line using the style {citation_style}. "
    "If no notes directly match but the question is clearly within your specialization, answer concisely from your own knowledge. "
    "Do NOT propose suggestions or action items unless the user explicitly asks. "
    "Never mention internal tools, retrieval, or storage."
)


def build_persona_preamble(specialization_list: str, citation_style: str) -> str:
    """Return the persona system preamble with explicit specialization."""
    return (
        PERSONA_SYSTEM.format(citation_style=citation_style)
        + "\nHere’s what I specialize in: "
        + specialization_list.strip()
        + "\n"
    )
