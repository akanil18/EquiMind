"""
Investment Committee & Adversarial Debate Engine for EquiMind.
"""

from .schema import (
    InvestmentRating,
    BullCase,
    BearCase,
    DebateSynthesis,
    InvestmentRecommendation,
)
from .bull_agent import BullAgent
from .bear_agent import BearAgent
from .judge_agent import JudgeAgent

__all__ = [
    "InvestmentRating",
    "BullCase",
    "BearCase",
    "DebateSynthesis",
    "InvestmentRecommendation",
    "BullAgent",
    "BearAgent",
    "JudgeAgent",
]
