"""
Monte Carlo Stochastic Simulator for EquiMind.
Generates 1,000+ stochastic price trajectories using Geometric Brownian Motion (GBM)
and Jump Diffusion processes.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pydantic import BaseModel, Field


class MonteCarloSimulationResult(BaseModel):
    """Statistical summary of stochastic Monte Carlo price path simulations."""
    ticker: str
    num_simulations: int
    horizon_days: int
    initial_price: float
    expected_final_price: float
    median_final_price: float
    percentile_5: float             # P05 downside risk threshold
    percentile_95: float            # P95 upside reward threshold
    probability_of_profit: float    # Probability that S_T > S_0
    simulated_max_drawdown_avg: float


class MonteCarloSimulator:
    """Stochastic Monte Carlo Simulation Engine."""

    @classmethod
    def run_simulation(
        cls,
        prices: List[float],
        ticker: str = "TICKER",
        num_simulations: int = 1000,
        horizon_days: int = 30,
        drift_override: Optional[float] = None,
        volatility_override: Optional[float] = None,
        jump_probability: float = 0.02,
    ) -> MonteCarloSimulationResult:
        """Executes stochastic Geometric Brownian Motion with optional jump diffusion."""
        if not prices:
            prices = [100.0]

        curr_price = float(prices[-1])

        if len(prices) > 1:
            arr = np.array(prices, dtype=float)
            returns = np.diff(arr) / arr[:-1]
            mu = drift_override if drift_override is not None else float(np.mean(returns)) * 252.0
            sigma = volatility_override if volatility_override is not None else float(np.std(returns)) * np.sqrt(252.0)
        else:
            mu = drift_override if drift_override is not None else 0.08
            sigma = volatility_override if volatility_override is not None else 0.25

        dt = 1.0 / 252.0  # Daily time step

        # Generate stochastic paths matrix (num_simulations x horizon_days)
        # S_{t+1} = S_t * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
        rand_shocks = np.random.normal(0, 1, (num_simulations, horizon_days))
        
        drift = (mu - 0.5 * (sigma ** 2)) * dt
        diffusion = sigma * np.sqrt(dt) * rand_shocks

        # Jump diffusion component
        jump_shocks = np.random.binomial(1, jump_probability, (num_simulations, horizon_days))
        jumps = jump_shocks * np.random.normal(-0.02, 0.05, (num_simulations, horizon_days))

        daily_returns = np.exp(drift + diffusion + jumps)
        
        # Cumulative price trajectories
        paths = np.zeros((num_simulations, horizon_days + 1))
        paths[:, 0] = curr_price
        paths[:, 1:] = curr_price * np.cumprod(daily_returns, axis=1)

        final_prices = paths[:, -1]
        
        expected_final = float(np.mean(final_prices))
        median_final = float(np.median(final_prices))
        p05 = float(np.percentile(final_prices, 5))
        p95 = float(np.percentile(final_prices, 95))
        prob_profit = float(np.mean(final_prices > curr_price))

        # Max drawdown computation across paths
        peak_prices = np.maximum.accumulate(paths, axis=1)
        drawdowns = (paths - peak_prices) / peak_prices
        max_drawdowns = np.min(drawdowns, axis=1)
        avg_max_drawdown = float(np.mean(max_drawdowns))

        return MonteCarloSimulationResult(
            ticker=ticker,
            num_simulations=num_simulations,
            horizon_days=horizon_days,
            initial_price=round(curr_price, 2),
            expected_final_price=round(expected_final, 2),
            median_final_price=round(median_final, 2),
            percentile_5=round(p05, 2),
            percentile_95=round(p95, 2),
            probability_of_profit=round(prob_profit, 4),
            simulated_max_drawdown_avg=round(avg_max_drawdown, 4),
        )
