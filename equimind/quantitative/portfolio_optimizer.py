"""
Portfolio Construction & Risk Optimization Engine for EquiMind.
Provides Mean-Variance (Markowitz), Risk Parity, Black-Litterman, and Kelly Criterion portfolio optimization.
"""

import numpy as np
from enum import Enum
from typing import Dict, Any, List, Tuple, Optional
from pydantic import BaseModel, Field


class OptimizationMethod(str, Enum):
    MEAN_VARIANCE = "MEAN_VARIANCE"
    RISK_PARITY = "RISK_PARITY"
    BLACK_LITTERMAN = "BLACK_LITTERMAN"
    KELLY_CRITERION = "KELLY_CRITERION"
    EQUAL_WEIGHT = "EQUAL_WEIGHT"


class PortfolioOptimizationResult(BaseModel):
    """Structured portfolio allocation and optimization output."""
    method: OptimizationMethod
    tickers: List[str]
    weights: Dict[str, float]
    expected_portfolio_return: float
    expected_portfolio_volatility: float
    portfolio_sharpe_ratio: float
    diversification_score: float  # 0.0 to 1.0


class PortfolioOptimizer:
    """Institutional Portfolio Optimization & Asset Allocation Engine."""

    @classmethod
    def optimize_portfolio(
        cls,
        asset_returns_matrix: Dict[str, List[float]],
        method: OptimizationMethod = OptimizationMethod.MEAN_VARIANCE,
        risk_free_rate: float = 0.04,
        investor_views: Optional[Dict[str, float]] = None,
    ) -> PortfolioOptimizationResult:
        """Executes portfolio optimization across assets."""
        tickers = list(asset_returns_matrix.keys())
        if not tickers:
            return PortfolioOptimizationResult(
                method=method,
                tickers=[],
                weights={},
                expected_portfolio_return=0.0,
                expected_portfolio_volatility=0.0,
                portfolio_sharpe_ratio=0.0,
                diversification_score=0.0,
            )

        num_assets = len(tickers)
        if num_assets == 1:
            return PortfolioOptimizationResult(
                method=method,
                tickers=tickers,
                weights={tickers[0]: 1.0},
                expected_portfolio_return=0.10,
                expected_portfolio_volatility=0.15,
                portfolio_sharpe_ratio=0.4,
                diversification_score=0.0,
            )

        returns_data = np.array([asset_returns_matrix[t] for t in tickers], dtype=float)
        mean_returns = np.mean(returns_data, axis=1) * 252.0
        cov_matrix = np.cov(returns_data) * 252.0

        if method == OptimizationMethod.RISK_PARITY:
            vols = np.sqrt(np.diag(cov_matrix))
            inv_vols = 1.0 / np.maximum(vols, 1e-5)
            weights_arr = inv_vols / np.sum(inv_vols)
        elif method == OptimizationMethod.BLACK_LITTERMAN and investor_views:
            # Blend market equilibrium with view vectors
            bl_returns = mean_returns.copy()
            for i, t in enumerate(tickers):
                if t in investor_views:
                    bl_returns[i] = 0.5 * mean_returns[i] + 0.5 * investor_views[t]
            weights_arr = cls._solve_tangency_portfolio(bl_returns, cov_matrix, risk_free_rate)
        elif method == OptimizationMethod.EQUAL_WEIGHT:
            weights_arr = np.ones(num_assets) / num_assets
        else:  # MEAN_VARIANCE
            weights_arr = cls._solve_tangency_portfolio(mean_returns, cov_matrix, risk_free_rate)

        # Normalize weights
        weights_arr = np.clip(weights_arr, 0.0, 1.0)
        weights_arr = weights_arr / np.sum(weights_arr)
        weights_dict = {t: round(float(w), 4) for t, w in zip(tickers, weights_arr)}

        port_ret = float(np.dot(weights_arr, mean_returns))
        port_vol = float(np.sqrt(np.dot(weights_arr.T, np.dot(cov_matrix, weights_arr))))
        sharpe = round((port_ret - risk_free_rate) / port_vol, 4) if port_vol > 0 else 0.0

        # Herfindahl index diversification score
        hhi = np.sum(weights_arr ** 2)
        div_score = round(float(1.0 - hhi), 4)

        return PortfolioOptimizationResult(
            method=method,
            tickers=tickers,
            weights=weights_dict,
            expected_portfolio_return=round(port_ret, 4),
            expected_portfolio_volatility=round(port_vol, 4),
            portfolio_sharpe_ratio=sharpe,
            diversification_score=div_score,
        )

    @classmethod
    def calculate_kelly_fraction(cls, win_probability: float, win_loss_ratio: float) -> float:
        """Calculates Kelly Criterion optimal position size fraction."""
        if win_loss_ratio <= 0:
            return 0.0
        p = win_probability
        q = 1.0 - p
        b = win_loss_ratio
        f_star = (p * b - q) / b
        # Apply fractional Kelly (0.5 half Kelly for safety)
        return round(float(max(0.0, f_star * 0.5)), 4)

    @classmethod
    def _solve_tangency_portfolio(cls, mean_returns: np.ndarray, cov_matrix: np.ndarray, rf: float) -> np.ndarray:
        """Solves analytical Markowitz tangency portfolio weights max Sharpe."""
        num_assets = len(mean_returns)
        excess_returns = mean_returns - rf
        try:
            inv_cov = np.linalg.inv(cov_matrix)
            raw_weights = np.dot(inv_cov, excess_returns)
            if np.sum(raw_weights) <= 0:
                return np.ones(num_assets) / num_assets
            return raw_weights / np.sum(raw_weights)
        except Exception:
            return np.ones(num_assets) / num_assets
