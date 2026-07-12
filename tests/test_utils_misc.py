from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

import agent_orchestration_helper as aoh
from utils.chat_history import SummaryBufferHistory
from utils.inventory_view import build_specialization_list
from utils.textproc import compute_overlap_ratio


def test_build_specialization_list_dedupes_and_trims() -> None:
    assert build_specialization_list([" Pikachu ", "pikachu", "", "Raichu"]) == "Pikachu; Raichu"


def test_format_context_documents_empty() -> None:
    assert aoh.format_context_documents([]) == ""


def test_format_context_documents_structure() -> None:
    doc = Document(page_content="  spaced   text ", metadata={"source": "notes.md"})
    block = aoh.format_context_documents([(doc, 0.5)])
    assert block.startswith("<documents>")
    assert block.endswith("</documents>")
    assert '<document index="0" source="notes.md" score="0.500">' in block
    assert "spaced text" in block


def test_build_system_prompt_is_byte_stable() -> None:
    # The system prompt doubles as the provider prompt-cache prefix, so two
    # calls must produce identical bytes.
    assert aoh.build_system_prompt() == aoh.build_system_prompt()
    assert "<documents>" in aoh.build_system_prompt()


def test_compute_overlap_ratio_bounds() -> None:
    assert compute_overlap_ratio("", ["context"]) == 0.0
    assert compute_overlap_ratio("word", []) == 0.0
    assert compute_overlap_ratio("pikachu evolves", ["pikachu evolves into raichu"]) == 1.0


def test_summary_buffer_history_prunes_into_summary() -> None:
    history = SummaryBufferHistory(llm=None, max_token_limit=20)
    history.add_messages(
        [
            HumanMessage(content="first question about pikachu " * 10),
            AIMessage(content="first answer about thunderbolt " * 10),
            HumanMessage(content="second question"),
            AIMessage(content="second answer"),
        ]
    )
    # Oldest turns were folded into the summary; latest exchange kept verbatim.
    assert history.moving_summary_buffer
    assert len(history.raw_messages) >= 2
    assert history.raw_messages[-1].content == "second answer"
    # The exposed messages start with the summary preamble.
    assert history.messages[0].content.startswith("Conversation summary so far:")

    history.clear()
    assert history.raw_messages == []
    assert history.moving_summary_buffer == ""
