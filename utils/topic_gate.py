"""Topic gate helper to detect off-topic requests before retrieval."""

from pydantic import BaseModel, Field


class TopicGateDecision(BaseModel):
    on_topic: bool = Field(..., description="True if request overlaps assistant specialization.")
    reason: str = Field(..., description="Short rationale.")
    confidence: float = Field(0.0, ge=0.0, le=1.0)


TOPIC_GUARD_SYSTEM = (
    "You are a scope classifier for an expert research assistant. "
    "Decide if the user's request falls within the assistant's specialization. "
    "Return on_topic=True ONLY if the core intent clearly overlaps the listed subjects. "
    "If ambiguous but plausible, prefer True."
)

TOPIC_GUARD_TEMPLATE = '''\
Assistant specialization subjects:
{specialization_list}

User message:
{user_msg}

Return a compact decision.'''


def build_topic_guard(llm):
    """Bind the provided LLM to the topic guard schema."""
    return llm.bind(max_tokens=800, max_completion_tokens=800).with_structured_output(TopicGateDecision)


def topic_gate(guard_llm, user_message: str, specialization_list: str, min_conf: float = 0.5) -> TopicGateDecision:
    """Run the request through the classifier and enforce a minimum confidence."""
    prompt = TOPIC_GUARD_TEMPLATE.format(
        specialization_list=specialization_list.strip(),
        user_msg=user_message,
    )
    decision: TopicGateDecision = guard_llm.invoke(
        [
            {"role": "system", "content": TOPIC_GUARD_SYSTEM},
            {"role": "user", "content": prompt},
        ]
    )
    if decision.confidence < min_conf:
        return TopicGateDecision(
            on_topic=True,
            reason=f"below min_conf; {decision.reason}",
            confidence=decision.confidence,
        )
    return decision
