"""Output sanitizer utilities."""

import re


def validate_output(text: str, allow_suggestions: bool) -> str:
    """Remove suggestion-style imperatives unless explicitly allowed."""
    if allow_suggestions:
        return text
    pattern = re.compile(r'^\s*(you should|try|consider|here are some suggestions)\b.*$', re.IGNORECASE | re.MULTILINE)
    return re.sub(pattern, '', text).strip()
