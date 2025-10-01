"""Helpers for presenting specialization metadata."""


def build_specialization_list(topics_or_titles: list[str]) -> str:
    """Return a unique, human-friendly one-liner of specialization topics."""
    uniq: list[str] = []
    seen: set[str] = set()
    for topic in topics_or_titles:
        normalized = (topic or "").strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(normalized)
    return '; '.join(uniq)
