from typing import Dict, Any, Optional


class FundamentalEngine:
    """Pure mathematical Fundamental Analysis engine."""

    @staticmethod
    def calculate_valuation_ratios(
        market_cap: float,
        price: float,
        eps: float,
        book_value_per_share: float,
        free_cash_flow: float,
        earnings_growth_rate: float,  # Percentage, e.g. 15.0 for 15%
    ) -> Dict[str, float]:
        """Calculates PE, PB, PEG, FCF Yield."""
        pe = round(price / eps, 2) if eps > 0 else 0.0
        pb = round(price / book_value_per_share, 2) if book_value_per_share > 0 else 0.0
        peg = round(pe / earnings_growth_rate, 2) if (pe > 0 and earnings_growth_rate > 0) else 0.0
        fcf_yield = round((free_cash_flow / market_cap) * 100.0, 2) if market_cap > 0 else 0.0

        return {
            "pe_ratio": pe,
            "pb_ratio": pb,
            "peg_ratio": peg,
            "fcf_yield_pct": fcf_yield,
        }

    @staticmethod
    def calculate_profitability_metrics(
        net_income: float,
        revenue: float,
        total_assets: float,
        shareholder_equity: float,
        operating_income: float,
    ) -> Dict[str, float]:
        """Calculates ROE, ROA, Operating Margin, Net Margin."""
        roe = round((net_income / shareholder_equity) * 100.0, 2) if shareholder_equity > 0 else 0.0
        roa = round((net_income / total_assets) * 100.0, 2) if total_assets > 0 else 0.0
        net_margin = round((net_income / revenue) * 100.0, 2) if revenue > 0 else 0.0
        op_margin = round((operating_income / revenue) * 100.0, 2) if revenue > 0 else 0.0

        return {
            "roe_pct": roe,
            "roa_pct": roa,
            "net_margin_pct": net_margin,
            "operating_margin_pct": op_margin,
        }

    @staticmethod
    def calculate_financial_health(
        current_assets: float,
        current_liabilities: float,
        total_debt: float,
        shareholder_equity: float,
    ) -> Dict[str, float]:
        """Calculates Current Ratio, Debt-to-Equity ratio."""
        current_ratio = round(current_assets / current_liabilities, 2) if current_liabilities > 0 else 0.0
        debt_to_equity = round(total_debt / shareholder_equity, 2) if shareholder_equity > 0 else 0.0

        return {
            "current_ratio": current_ratio,
            "debt_to_equity": debt_to_equity,
        }

    @staticmethod
    def calculate_piotroski_f_score(fin: Dict[str, float]) -> Dict[str, Any]:
        """Calculates Piotroski F-Score (0 to 9) measuring financial strength.
        
        Input dictionary keys expected:
        - net_income, roa, operating_cash_flow, ocf_gt_net_income
        - long_term_debt_current, long_term_debt_prior
        - current_ratio_current, current_ratio_prior
        - shares_outstanding_current, shares_outstanding_prior
        - gross_margin_current, gross_margin_prior
        - asset_turnover_current, asset_turnover_prior
        """
        score = 0
        checks = {}

        # Profitability
        checks["positive_net_income"] = 1 if fin.get("net_income", 0) > 0 else 0
        checks["positive_roa"] = 1 if fin.get("roa", 0) > 0 else 0
        checks["positive_ocf"] = 1 if fin.get("operating_cash_flow", 0) > 0 else 0
        checks["ocf_gt_net_income"] = 1 if fin.get("operating_cash_flow", 0) > fin.get("net_income", 0) else 0

        # Leverage & Liquidity
        checks["lower_debt"] = 1 if fin.get("long_term_debt_current", 0) < fin.get("long_term_debt_prior", 0) else 0
        checks["higher_current_ratio"] = 1 if fin.get("current_ratio_current", 0) > fin.get("current_ratio_prior", 0) else 0
        checks["no_dilution"] = 1 if fin.get("shares_outstanding_current", 0) <= fin.get("shares_outstanding_prior", 0) else 0

        # Operating Efficiency
        checks["higher_gross_margin"] = 1 if fin.get("gross_margin_current", 0) > fin.get("gross_margin_prior", 0) else 0
        checks["higher_asset_turnover"] = 1 if fin.get("asset_turnover_current", 0) > fin.get("asset_turnover_prior", 0) else 0

        score = sum(checks.values())

        if score >= 8:
            rating = "Strong Financial Health"
        elif score >= 5:
            rating = "Moderate Financial Health"
        else:
            rating = "Weak Financial Health"

        return {
            "piotroski_f_score": score,
            "max_score": 9,
            "rating": rating,
            "breakdown": checks,
        }

    @staticmethod
    def calculate_altman_z_score(
        working_capital: float,
        retained_earnings: float,
        ebit: float,
        market_cap: float,
        revenue: float,
        total_assets: float,
        total_liabilities: float,
    ) -> Dict[str, Any]:
        """Calculates Altman Z-Score predicting bankruptcy risk.
        
        Formula: Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 0.999*X5
        X1 = Working Capital / Total Assets
        X2 = Retained Earnings / Total Assets
        X3 = EBIT / Total Assets
        X4 = Market Value of Equity / Total Liabilities
        X5 = Revenue / Total Assets
        """
        if total_assets <= 0 or total_liabilities <= 0:
            return {"z_score": 0.0, "zone": "Unknown"}

        x1 = working_capital / total_assets
        x2 = retained_earnings / total_assets
        x3 = ebit / total_assets
        x4 = market_cap / total_liabilities
        x5 = revenue / total_assets

        z_score = round((1.2 * x1) + (1.4 * x2) + (3.3 * x3) + (0.6 * x4) + (0.999 * x5), 2)

        if z_score > 2.99:
            zone = "Safe Zone (Low Bankruptcy Risk)"
        elif z_score >= 1.81:
            zone = "Grey Zone (Moderate Risk)"
        else:
            zone = "Distress Zone (High Bankruptcy Risk)"

        return {
            "z_score": z_score,
            "zone": zone,
            "components": {
                "x1_working_capital_to_assets": round(x1, 4),
                "x2_retained_earnings_to_assets": round(x2, 4),
                "x3_ebit_to_assets": round(x3, 4),
                "x4_market_cap_to_liabilities": round(x4, 4),
                "x5_asset_turnover": round(x5, 4),
            },
        }
