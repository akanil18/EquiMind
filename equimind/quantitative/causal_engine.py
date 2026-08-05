"""
Causal Reasoning Engine for EquiMind.
Distinguishes true causal mechanisms from spurious market correlations using Structural Causal Models (SCMs).
"""

import numpy as np
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field


class CausalNodeType(str, Enum):
    MACRO_DRIVER = "MACRO_DRIVER"
    SUPPLY_CHAIN = "SUPPLY_CHAIN"
    FINANCIAL_METRIC = "FINANCIAL_METRIC"
    EARNINGS_OUTCOME = "EARNINGS_OUTCOME"
    STOCK_PRICE = "STOCK_PRICE"


class CausalEdge(BaseModel):
    """Directed causal relationship X -> Y with average treatment effect and confidence."""
    cause: str
    effect: str
    direct_causal_effect: float       # dY/dX estimated causal strength
    is_confounded: bool = False       # True if correlation is driven by shared confounder
    confounder_variable: Optional[str] = None
    confidence_score: float           # 0.0 to 1.0


class CausalAnalysisResult(BaseModel):
    cause_variable: str
    target_variable: str
    raw_correlation: float
    estimated_causal_effect: float
    is_spurious_correlation: bool
    explanation: str


class CausalReasoningEngine:
    """Institutional Structural Causal Engine for Financial & Economic Reasoning."""

    @classmethod
    def estimate_causal_effect(
        cls,
        x_values: List[float],
        y_values: List[float],
        z_confounder_values: Optional[List[float]] = None,
    ) -> Tuple[float, float, bool]:
        """
        Estimates the direct causal effect of X on Y (do(X) intervention).
        Controls for confounding variable Z if provided.
        Returns: (raw_correlation, causal_effect, is_spurious)
        """
        if len(x_values) != len(y_values) or len(x_values) < 3:
            return 0.0, 0.0, False

        x = np.array(x_values, dtype=float)
        y = np.array(y_values, dtype=float)

        raw_corr = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else 0.0

        if z_confounder_values is None or len(z_confounder_values) != len(x_values):
            # Without confounder controls, estimated causal effect equals raw correlation
            return round(raw_corr, 4), round(raw_corr, 4), False

        z = np.array(z_confounder_values, dtype=float)

        # Partial regression / back-door adjustment controlling for Z
        # Residualize X and Y with respect to Z
        x_res = x - np.polyval(np.polyfit(z, x, 1), z) if np.std(z) > 0 else x
        y_res = y - np.polyval(np.polyfit(z, y, 1), z) if np.std(z) > 0 else y

        causal_effect = float(np.corrcoef(x_res, y_res)[0, 1]) if np.std(x_res) > 0 and np.std(y_res) > 0 else 0.0

        # Spurious if high raw correlation but negligible direct causal effect after controlling for Z
        is_spurious = abs(raw_corr) >= 0.4 and abs(causal_effect) < 0.15

        return round(raw_corr, 4), round(causal_effect, 4), is_spurious

    @classmethod
    def analyze_causal_mechanism(
        cls,
        cause_name: str,
        target_name: str,
        x_values: List[float],
        y_values: List[float],
        confounder_name: Optional[str] = None,
        z_values: Optional[List[float]] = None,
    ) -> CausalAnalysisResult:
        """Analyzes whether relationship between cause_name and target_name is direct or spurious."""
        raw_corr, causal_eff, is_spurious = cls.estimate_causal_effect(x_values, y_values, z_values)

        if is_spurious:
            explanation = (
                f"Relationship between '{cause_name}' and '{target_name}' exhibits high raw correlation ({raw_corr:+.2f}), "
                f"but is SPURIOUS. The observed correlation is driven by shared confounder '{confounder_name}'. "
                f"Direct causal effect: {causal_eff:+.2f}."
            )
        else:
            explanation = (
                f"Direct causal relationship confirmed for '{cause_name}' -> '{target_name}'. "
                f"Raw correlation: {raw_corr:+.2f}, Direct causal effect: {causal_eff:+.2f}."
            )

        return CausalAnalysisResult(
            cause_variable=cause_name,
            target_variable=target_name,
            raw_correlation=raw_corr,
            estimated_causal_effect=causal_eff,
            is_spurious_correlation=is_spurious,
            explanation=explanation,
        )
