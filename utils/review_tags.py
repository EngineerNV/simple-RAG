"""Canonical remediation advice per review tag.

Shared by the quiz (live tips while annotating) and the report (aggregate
guidance) so the wording can't drift between the two.
"""

TAG_REMEDIATION = {
    "retrieval-miss": "Increase k, add metadata filters, or expand corpus coverage.",
    "retrieval-partial": "Inspect chunk boundaries; try larger chunks or more overlap.",
    "too-low-k": "Increase k or tune score thresholds before truncating.",
    "chunking-issue": "Rebuild the index with bigger chunks or overlap to keep facts together.",
    "prompt-overreach": "Tighten the system prompt and add refusal exemplars.",
    "ambiguous-question": "Introduce clarifier prompts or request follow-up questions.",
    "source-noise": "Clean noisy documents and rebuild the index.",
    "other": "Review the free-form notes for bespoke fixes.",
}
