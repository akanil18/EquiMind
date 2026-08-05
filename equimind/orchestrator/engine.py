import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from equimind.config import settings
from equimind.providers.factory import ProviderFactory
from equimind.planner.reasoning_planner import ReasoningPlanner, ResearchPlan
from equimind.evidence.graph import EvidenceGraph
from equimind.context.compressor import ContextCompressor
from equimind.teams.market_data_team import MarketDataTeam
from equimind.teams.fundamental_team import FundamentalTeam
from equimind.teams.macro_team import MacroTeam
from equimind.teams.web_intelligence_team import WebIntelligenceTeam
from equimind.time_machine.temporal_guard import TemporalGuard
from equimind.committee.bull_agent import BullAgent
from equimind.committee.bear_agent import BearAgent
from equimind.committee.judge_agent import JudgeAgent
from equimind.committee.schema import InvestmentRecommendation
from equimind.memory.hierarchical_store import HierarchicalMemoryStore
from equimind.memory.delta_engine import DeltaResearchEngine

logger = logging.getLogger(__name__)


class EquiMindEngine:
    """Master Orchestrator Engine for the EquiMind Financial Research Framework."""

    def __init__(self, memory_store: Optional[HierarchicalMemoryStore] = None):
        self.memory_store = memory_store or HierarchicalMemoryStore()
        self.teams = {
            "market_data": MarketDataTeam(),
            "fundamentals": FundamentalTeam(),
            "macro": MacroTeam(),
            "web_intelligence": WebIntelligenceTeam(),
        }

    def analyze_equity(
        self,
        ticker: str,
        query: str,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        as_of_date_str: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Runs complete end-to-end equity research pipeline."""
        ticker_upper = ticker.upper().strip()
        as_of_dt = TemporalGuard.parse_as_of_date(as_of_date_str)

        # 1. Instantiate Model-Agnostic LLM Provider
        provider = ProviderFactory.create(provider_name=provider_name, model_name=model_name)
        logger.info(f"Initialized LLM Provider: {provider.provider_name} (Model: {provider.model_name})")

        # 2. Delta Research & Memory Lookup
        has_prev, last_time, cached_evidence = DeltaResearchEngine.compute_delta_research_plan(
            ticker_upper, self.memory_store
        )

        # 3. Dynamic Reasoning Planner Agent
        plan: ResearchPlan = ReasoningPlanner.plan(
            query=query, ticker=ticker_upper, provider=provider, as_of_date=as_of_date_str
        )
        logger.info(f"Reasoning Planner generated plan for sector: {plan.sector.value}")

        # 4. Execute Specialized Research Subagent Teams
        raw_evidence_nodes = list(cached_evidence)
        with TemporalGuard(as_of_date=as_of_dt) as guard:
            for team_name in plan.active_teams:
                if team_name in self.teams:
                    team_nodes = self.teams[team_name].research(
                        ticker=ticker_upper,
                        query=query,
                        context={"active_adapters": plan.active_adapters},
                        provider=provider,
                        as_of_date=as_of_dt,
                    )
                    raw_evidence_nodes.extend(team_nodes)

            # Apply temporal guard filtering
            filtered_nodes = guard.filter_evidence(raw_evidence_nodes)

        # 5. Build Evidence Graph
        graph = EvidenceGraph()
        for node in filtered_nodes:
            graph.add_node(node)

        # 6. Context Optimization & Non-LLM Compression Engine
        compressed_nodes = ContextCompressor.compress(
            nodes=filtered_nodes,
            query_context=query,
            max_token_budget=settings.max_context_tokens // 8,
            as_of_date=as_of_dt,
        )

        # Extract Quantitative Summaries
        quant_summary = {}
        for n in compressed_nodes:
            if n.source_type == "market_prices" and "last_price" in n.metadata:
                quant_summary = n.metadata
                break

        if not quant_summary:
            quant_summary = {"last_price": 100.0}

        risk_summary = {"annualized_volatility_pct": 22.5}

        # 7. Investment Committee Adversarial Debate (Bull vs Bear vs Judge)
        bull_case = BullAgent.evaluate(
            ticker=ticker_upper, nodes=compressed_nodes, quant_summary=quant_summary, provider=provider
        )
        bear_case = BearAgent.evaluate(
            ticker=ticker_upper, nodes=compressed_nodes, quant_summary=quant_summary, provider=provider
        )

        recommendation: InvestmentRecommendation = JudgeAgent.evaluate_debate(
            ticker=ticker_upper,
            bull_case=bull_case,
            bear_case=bear_case,
            nodes=compressed_nodes,
            quant_summary=quant_summary,
            risk_summary=risk_summary,
            provider=provider,
        )

        # 8. Store in Hierarchical Persistent Memory
        report_record = self.memory_store.store_research_report(
            ticker=ticker_upper,
            user_query=query,
            rating=recommendation.rating.value,
            conviction_score=recommendation.conviction_score,
            summary=recommendation.debate_synthesis.winning_thesis_summary,
            evidence_nodes=compressed_nodes,
        )

        return {
            "ticker": ticker_upper,
            "query": query,
            "timestamp": report_record.timestamp.isoformat(),
            "provider_used": f"{provider.provider_name} ({provider.model_name})",
            "research_plan": plan.model_dump(),
            "evidence_graph_nodes": len(graph.nodes),
            "compressed_evidence_count": len(compressed_nodes),
            "recommendation": recommendation.model_dump(),
        }
