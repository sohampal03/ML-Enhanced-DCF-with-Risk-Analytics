"""
Education content module — concepts and their structured content.
Used by the Learning Center dashboard page.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class FinanceConcept:
    """A structured finance or ML concept for the Learning Center."""
    name: str
    icon: str
    definition: str
    formula: str
    intuition: str
    example: str
    pitfalls: list[str]
    interview_questions: list[str]
    difficulty: str = "Intermediate"  # Beginner / Intermediate / Advanced


CONCEPTS: dict[str, FinanceConcept] = {
    "DCF": FinanceConcept(
        name="Discounted Cash Flow",
        icon="📊",
        definition=(
            "DCF is a valuation method that estimates intrinsic value by discounting "
            "expected future free cash flows to present value using the cost of capital (WACC)."
        ),
        formula="Value = Σ FCFF_t / (1+WACC)^t + TV / (1+WACC)^n",
        intuition=(
            "Money today is worth more than money in the future. "
            "DCF quantifies exactly how much future cash flows are worth right now."
        ),
        example=(
            "If a company generates $10B FCFF annually for 10 years, growing at 5%, "
            "with WACC=9%, the present value is approximately $65B — not $100B."
        ),
        pitfalls=[
            "Small changes in WACC or terminal growth rate cause large value changes",
            "Terminal value often represents 70–80% of total value",
            "Garbage-in, garbage-out: quality of inputs determines quality of output",
        ],
        interview_questions=[
            "Walk me through a DCF",
            "What are the limitations of DCF valuation?",
            "How sensitive is DCF to the terminal growth rate?",
        ],
        difficulty="Intermediate",
    ),
}


def get_all_concepts() -> dict[str, FinanceConcept]:
    return CONCEPTS


__all__ = ["FinanceConcept", "CONCEPTS", "get_all_concepts"]
