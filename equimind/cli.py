import argparse
import json
import sys
from equimind.orchestrator.engine import EquiMindEngine


def main():
    parser = argparse.ArgumentParser(description="EquiMind CLI: Autonomous AI Equity Research Firm")
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker symbol (e.g. NVDA, AAPL, JPM)")
    parser.add_argument("--query", type=str, default="Complete institutional equity research analysis", help="Investment question or research goal")
    parser.add_argument("--provider", type=str, default="openai", help="LLM Provider (openai, anthropic, gemini, deepseek, qwen, ollama, mock)")
    parser.add_argument("--model", type=str, default=None, help="LLM Model name override")
    parser.add_argument("--as-of-date", type=str, default=None, help="Historical backtesting cutoff date (YYYY-MM-DD)")

    args = parser.parse_args()

    print(f"\n🚀 Launching EquiMind Research Engine for ${args.ticker.upper()}...")
    print(f"📋 Query: '{args.query}' | Provider: {args.provider.upper()}\n")

    engine = EquiMindEngine()
    results = engine.analyze_equity(
        ticker=args.ticker,
        query=args.query,
        provider_name=args.provider,
        model_name=args.model,
        as_of_date_str=args.as_of_date,
    )

    rec = results["recommendation"]
    print("=" * 70)
    print(f"INSTITUTIONAL RECOMMENDATION FOR ${results['ticker']}")
    print("=" * 70)
    print(f"RATING: {rec['rating']} (Conviction: {rec['conviction_score'] * 100:.0f}%)")
    print(f"CURRENT PRICE: ${rec['current_price']:.2f}")
    print(f"TARGET ENTRY RANGE: ${rec['target_entry_range'][0]:.2f} - ${rec['target_entry_range'][1]:.2f}")
    print(f"PORTFOLIO ALLOCATION: {rec['recommended_portfolio_allocation']}")
    print(f"RISK/REWARD RATIO: {rec['expected_risk_reward_ratio']}:1")
    print("-" * 70)
    print("\n🐂 BULL CASE:")
    print(f"  {rec['bull_case']['thesis']}")
    print("  Key Catalysts:")
    for cat in rec['bull_case']['key_catalysts']:
        print(f"    - {cat}")

    print("\n🐻 BEAR CASE:")
    print(f"  {rec['bear_case']['thesis']}")
    print("  Key Headwinds:")
    for hw in rec['bear_case']['key_headwinds']:
        print(f"    - {hw}")

    print("-" * 70)
    print(f"⚖️ DEBATE SYNTHESIS: {rec['debate_synthesis']['winning_thesis_summary']}")
    print("\n🔗 PROVENANCE CITATIONS:")
    for c in rec['provenance_citations'][:5]:
        print(f"  - [{c['source'].upper()}] {c['title']} (Credibility: {c['credibility'].upper()})")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
