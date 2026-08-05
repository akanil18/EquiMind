from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


class RiskEngine:
    """Pure mathematical Risk & Return Engine."""

    @staticmethod
    def calculate_risk_metrics(
        returns: pd.Series,
        risk_free_rate: float = 0.04,  # Annualized 4%
        benchmark_returns: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """Calculates Volatility, Sharpe Ratio, Sortino Ratio, Max Drawdown, VaR 95%, CVaR 95%."""
        if len(returns) < 5:
            return {}

        clean_returns = returns.dropna()

        # Volatility (Annualized)
        daily_vol = float(clean_returns.std())
        annualized_vol = round(daily_vol * np.sqrt(252) * 100.0, 2)

        # Annualized Return
        mean_daily = float(clean_returns.mean())
        annualized_return = round(mean_daily * 252 * 100.0, 2)

        # Sharpe Ratio
        excess_return = (annualized_return / 100.0) - risk_free_rate
        sharpe_ratio = round(excess_return / (annualized_vol / 100.0), 2) if annualized_vol > 0 else 0.0

        # Sortino Ratio (Downside deviation)
        downside = clean_returns[clean_returns < 0]
        downside_vol = float(downside.std()) * np.sqrt(252) if len(downside) > 0 else 0.0001
        sortino_ratio = round(excess_return / downside_vol, 2) if downside_vol > 0 else 0.0

        # Max Drawdown
        cum_returns = (1.0 + clean_returns).cumprod()
        peak = cum_returns.cummax()
        drawdown = (cum_returns - peak) / peak
        max_drawdown = round(float(drawdown.min()) * 100.0, 2)

        # Value at Risk (VaR 95% & 99%)
        var_95 = round(float(np.percentile(clean_returns, 5)) * 100.0, 2)
        var_99 = round(float(np.percentile(clean_returns, 1)) * 100.0, 2)

        # Conditional VaR / Expected Shortfall (CVaR 95%)
        tail_returns = clean_returns[clean_returns <= (var_95 / 100.0)]
        cvar_95 = round(float(tail_returns.mean()) * 100.0, 2) if len(tail_returns) > 0 else var_95

        # Beta & Alpha relative to Benchmark
        beta = 1.0
        alpha = 0.0
        if benchmark_returns is not None and len(benchmark_returns) == len(clean_returns):
            cov_matrix = np.cov(clean_returns, benchmark_returns.dropna())
            cov = cov_matrix[0, 1]
            bench_var = cov_matrix[1, 1]
            if bench_var > 0:
                beta = round(float(cov / bench_var), 2)
                alpha = round(annualized_return - (risk_free_rate * 100) - beta * ((benchmark_returns.mean() * 252 * 100) - (risk_free_rate * 100)), 2)

        return {
            "annualized_return_pct": annualized_return,
            "annualized_volatility_pct": annualized_vol,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown_pct": max_drawdown,
            "var_95_daily_pct": var_95,
            "var_99_daily_pct": var_99,
            "cvar_95_daily_pct": cvar_95,
            "beta": beta,
            "alpha_pct": alpha,
        }

    @staticmethod
    def calculate_expected_return_distribution(
        returns: pd.Series, horizon_days: int = 30
    ) -> Dict[str, float]:
        """Calculates projected return distribution and confidence interval over horizon days."""
        if len(returns) < 5:
            return {"expected_return_pct": 0.0, "ci_95_lower_pct": 0.0, "ci_95_upper_pct": 0.0}

        clean = returns.dropna()
        mean_daily = float(clean.mean())
        std_daily = float(clean.std())

        horizon_mean = mean_daily * horizon_days
        horizon_std = std_daily * np.sqrt(horizon_days)

        expected_return_pct = round(horizon_mean * 100.0, 2)
        ci_95_lower = round((horizon_mean - 1.96 * horizon_std) * 100.0, 2)
        ci_95_upper = round((horizon_mean + 1.96 * horizon_std) * 100.0, 2)

        # Risk-reward ratio: expected return vs potential downside
        downside_risk = abs(ci_95_lower) if ci_95_lower < 0 else 0.01
        risk_reward_ratio = round(expected_return_pct / downside_risk, 2) if expected_return_pct > 0 else 0.0

        return {
            "horizon_days": horizon_days,
            "expected_return_pct": expected_return_pct,
            "ci_95_lower_pct": ci_95_lower,
            "ci_95_upper_pct": ci_95_upper,
            "risk_reward_ratio": risk_reward_ratio,
        }
