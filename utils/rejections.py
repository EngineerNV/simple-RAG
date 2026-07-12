"""Dynamic off-topic refusal helper."""

from pydantic import BaseModel, Field


class RejectionOut(BaseModel):
    text: str = Field(..., description="2–4 sentence refusal; includes specialization list; no suggestions.")


REJECTION_SYSTEM = (
    "You are an expert research assistant. "
    "You do not handle questions outside your specialization. "
    "Write a brief refusal (2-4 sentences), professional and calm. "
    "Identify yourself as an expert research assistant specializing in the listed subjects. "
    "List the subjects succinctly with a 'Here’s what I specialize in:' lead-in. "
    "Do NOT offer suggestions or alternatives unless explicitly asked. "
    "If the user's language is rude or unsafe, keep your tone firm but civil."
)

REJECTION_TEMPLATE = '''\
Specialization subjects:
{specialization_list}

User message:
{user_msg}

Write only the refusal text.'''


def build_rejection_writer(llm):
    """Bind the provided LLM to the rejection schema."""
    return llm.with_structured_output(RejectionOut)


def generate_rejection(llm, specialization_list: str, user_message: str) -> str:
    """Produce a specialization-forward refusal."""
    prompt = REJECTION_TEMPLATE.format(
        specialization_list=specialization_list.strip(),
        user_msg=user_message,
    )
    out: RejectionOut = llm.invoke(
        [
            {"role": "system", "content": REJECTION_SYSTEM},
            {"role": "user", "content": prompt},
        ]
    )
    return out.text.strip()
