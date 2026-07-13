"""Rolling summary-buffer chat history.

Replaces ``langchain.memory.ConversationSummaryBufferMemory``, which was
deprecated and removed in langchain 1.x. Recent turns are kept verbatim;
once the buffer exceeds the token budget, the oldest turns are folded into a
running summary produced by the LLM (with a deterministic truncation fallback
so a summarizer failure never loses the whole history).
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, List, Optional

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

SUMMARY_SYSTEM = (
    "You maintain a running summary of a conversation. Update the summary with the new lines: "
    "keep decisions, facts, names, numbers, and open questions; drop pleasantries and narrative. "
    "Return only the updated summary text."
)

SUMMARY_TEMPLATE = """\
Current summary:
{summary}

New conversation lines:
{new_lines}
"""

# Cap on the fallback summary when the LLM summarizer is unavailable.
_FALLBACK_SUMMARY_CHARS = 4000


class SummaryBufferHistory(BaseChatMessageHistory):
    """Chat history with a token budget and an LLM-maintained rolling summary."""

    def __init__(self, llm: Optional[Any] = None, max_token_limit: int = 1200) -> None:
        self.llm = llm
        self.max_token_limit = max_token_limit
        self.moving_summary_buffer = ""
        self._messages: List[BaseMessage] = []

    @property
    def raw_messages(self) -> List[BaseMessage]:
        """The retained verbatim turns, without the summary preamble."""
        return list(self._messages)

    @property
    def messages(self) -> List[BaseMessage]:
        if self.moving_summary_buffer:
            preamble = SystemMessage(
                content="Conversation summary so far:\n" + self.moving_summary_buffer
            )
            return [preamble, *self._messages]
        return list(self._messages)

    def add_message(self, message: BaseMessage) -> None:
        self._messages.append(message)
        self.prune()

    def add_messages(self, messages: Iterable[BaseMessage]) -> None:
        self._messages.extend(messages)
        self.prune()

    def clear(self) -> None:
        self._messages.clear()
        self.moving_summary_buffer = ""

    def prune(self) -> None:
        """Fold the oldest turns into the summary once over the token budget."""
        if self._count_tokens(self._messages) <= self.max_token_limit:
            return
        pruned: List[BaseMessage] = []
        # Always keep at least the latest exchange verbatim.
        while len(self._messages) > 2 and self._count_tokens(self._messages) > self.max_token_limit:
            pruned.append(self._messages.pop(0))
        # Retained history must start with a human turn — Anthropic and Gemini
        # reject conversations whose first non-system message is an AI turn.
        while self._messages and not isinstance(self._messages[0], HumanMessage):
            pruned.append(self._messages.pop(0))
        if pruned:
            self._update_summary(pruned)

    def _count_tokens(self, messages: List[BaseMessage]) -> int:
        if self.llm is not None:
            try:
                return self.llm.get_num_tokens_from_messages(messages)
            except Exception:  # pragma: no cover - provider-specific counters vary
                pass
        # ~4 characters per token is close enough for a budget check.
        return sum(len(str(getattr(m, "content", "") or "")) for m in messages) // 4

    def _update_summary(self, pruned: List[BaseMessage]) -> None:
        new_lines = "\n".join(
            f"{getattr(m, 'type', m.__class__.__name__)}: {getattr(m, 'content', '')}"
            for m in pruned
        )
        if self.llm is not None:
            try:
                response = self.llm.invoke(
                    [
                        {"role": "system", "content": SUMMARY_SYSTEM},
                        {
                            "role": "user",
                            "content": SUMMARY_TEMPLATE.format(
                                summary=self.moving_summary_buffer or "(empty)",
                                new_lines=new_lines,
                            ),
                        },
                    ]
                )
                content = getattr(response, "content", response)
                self.moving_summary_buffer = str(content).strip()
                return
            except Exception as exc:
                logger.warning("Summary LLM call failed (%s); keeping truncated raw lines.", exc)
        combined = (self.moving_summary_buffer + "\n" + new_lines).strip()
        self.moving_summary_buffer = combined[-_FALLBACK_SUMMARY_CHARS:]
