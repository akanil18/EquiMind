"""
Alpha Research Laboratory & Factor Evaluation Engine for EquiMind.
Discovers, evaluates, statistically validates, and ranks predictive alpha signals.
"""

import numpy as np
from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class FactorCategory(str, Enum):
    MOMENTUM = "MOMENTUM"
    VALUE = "VALUE"
    QUALITY = "QUALITY"
    ALTERNATIVE = "ALTERNATIVE"
    DEVELOPER_VELOCITY = "DEVELOPER_VELOCITY"
    MACRO_SENSITIVITY = "MACRO_SENSITIVITY"


class AlphaFactor(BaseModel):
    """Structured statistical alpha factor definition."""
    name: str
    category: FactorCategory
    description: str
    information_coefficient: float     # IC: Pearson corr with forward return
    rank_ic: float                     # Rank IC: Spearman rank corr with forward return
    sharpe_ratio: float                # Factor Sharpe ratio
    decay_half_life_days: float        # Signal decay half life
    is_statistically_significant: bool # True if |Rank IC| >= 0.05 and |Sharpe| >= 1.0


class AlphaResearchLab:
    """Institutional Alpha Signal Discovery & Factor Evaluation Engine."""

    @classmethod
    def calculate_ic(cls, factor_values: List[float], forward_returns: List[float]) -> float:
        """Calculates Information Coefficient (Pearson Correlation)."""
        if len(factor_values) != len(forward_returns) or len(factor_values) < 3:
            return 0.0

        f = np.array(factor_values, dtype=float)
        r = np.array(forward_returns, dtype=float)

        if np.std(f) == 0 or np.std(r) == 0:
            return 0.0

        corr = np.corrcoef(f, r)[0, 1]
        return round(float(corr), 4) if not np.isnan(corr) else 0.0

    @classmethod
    def calculate_rank_ic(cls, factor_values: List[float], forward_returns: List[float]) -> float:
        """Calculates Rank Information Coefficient (Spearman Rank Correlation)."""
        if len(factor_values) != len(forward_returns) or len(factor_values) < 3:
            return 0.0

        f_ranks = np.argsort(np.argsort(factor_values))
        r_ranks = np.argsort(np.argsort(forward_returns))

        if np.std(f_ranks) == 0 or np.std(r_ranks) == 0:
            return 0.0

        corr = np.corrcoef(f_ranks, r_ranks)[0, 1]
        return round(float(corr), 4) if not np.isnan(corr) else 0.0

    @classmethod
    def evaluate_alpha_factor(
        cls,
        factor_name: str,
        category: FactorCategory,
        description: str,
        factor_values: List[float],
        forward_returns: List[float],
        half_life_days: float = 14.0,
    ) -> AlphaFactor:
        """Evaluates an alpha factor signal against historical forward returns."""
        ic = cls.calculate_ic(factor_values, forward_returns)
        rank_ic = cls.calculate_rank_ic(factor_values, forward_returns)

        # Approximate Sharpe from IC * sqrt(N)
        n = len(factor_values)
        sharpe = round(ic * np.sqrt(n), 2) if n > 0 else 0.0
        is_sig = abs(rank_ic) >= 0.05 and abs(sharpe) >= 1.0

        return AlphaFactor(
            name=factor_name,
            category=category,
            description=description,
            information_coefficient=ic,
            rank_ic=rank_ic,
            sharpe_ratio=sharpe,
            decay_half_life_days=half_life_days,
            is_statistically_significant=is_sig,
        )

    @classmethod
    def rank_alpha_factors(cls, factors: List[AlphaFactor]) -> List[AlphaFactor]:
        """Ranks alpha factors by absolute Rank IC and Sharpe ratio."""
        return sorted(
            factors,
            key=lambda f: (abs(f.rank_ic), abs(f.sharpe_ratio)),
            reverse=True,
        )
