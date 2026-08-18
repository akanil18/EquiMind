import json
import logging
import os
from typing import Dict, Any, Optional, List
from pydantic import BaseModel

try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse, FileResponse
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from equimind.orchestrator.engine import EquiMindEngine
from equimind.evaluation import AccuracyTracker, WalkForwardBacktester
from equimind.api.websocket_manager import ws_manager
from equimind.adapters import YFinanceAdapter
from equimind.adapters.sec_edgar_adapter import SECEdgarAdapter

logger = logging.getLogger(__name__)

if HAS_FASTAPI:
    app = FastAPI(
        title="EquiMind API",
        description="Autonomous AI Equity Research Firm & Financial Research Orchestration Framework",
        version="0.2.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    engine = EquiMindEngine()
    accuracy_tracker = AccuracyTracker()

    class ResearchRequest(BaseModel):
        ticker: str
        query: str = "Should I invest in this stock for long-term?"
        provider: str = "openai"
        model: Optional[str] = None
        as_of_date: Optional[str] = None

    class BacktestRequest(BaseModel):
        ticker: str
        start_date: str = "2023-01-01"
        end_date: str = "2024-01-01"
        step_days: int = 30
        eval_window_days: int = 30

    @app.get("/api/v1/health")
    def health_check():
        return {
            "status": "ok",
            "service": "EquiMind Agentic RAG Engine",
            "version": "0.3.0",
            "agentic_rag": True,
            "hybrid_search": "HNSW + BM25 + RRF",
            "real_data_enabled": True,
        }

    @app.post("/api/v1/research")
    def run_research(req: ResearchRequest):
        try:
            results = engine.analyze_equity(
                ticker=req.ticker,
                query=req.query,
                provider_name=req.provider,
                model_name=req.model,
                as_of_date_str=req.as_of_date,
            )
            
            # Record in Accuracy Tracker
            rec = results.get("recommendation", {})
            if rec:
                price = rec.get("quant_summary", {}).get("last_price", 100.0)
                accuracy_tracker.record_recommendation(
                    ticker=req.ticker,
                    rating=rec.get("rating", "HOLD"),
                    conviction_score=rec.get("conviction_score", 0.5),
                    current_price=price,
                    as_of_date=None,
                )

            return results
        except Exception as e:
            logger.error(f"Research API error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/v1/memory/{ticker}")
    def get_ticker_memory(ticker: str):
        entity = engine.memory_store.ticker_knowledge.get(ticker.upper())
        if not entity:
            raise HTTPException(status_code=404, detail=f"No persistent knowledge found for {ticker}")
        return entity.model_dump()

    @app.get("/api/v1/accuracy")
    def get_accuracy_metrics():
        """Get aggregate accuracy, calibration Brier score, and Sharpe metrics."""
        accuracy_tracker.evaluate_outcomes(force_evaluate=True)
        return accuracy_tracker.compute_metrics().model_dump()

    @app.get("/api/v1/predictions")
    def list_predictions():
        """Get history of recorded predictions."""
        return [p.model_dump(mode="json") for p in accuracy_tracker.predictions]

    @app.post("/api/v1/backtest")
    def run_backtest(req: BacktestRequest):
        """Run historical walk-forward backtest."""
        try:
            backtester = WalkForwardBacktester(engine=engine)
            summary = backtester.run_backtest(
                ticker=req.ticker,
                start_date_str=req.start_date,
                end_date_str=req.end_date,
                step_days=req.step_days,
                eval_window_days=req.eval_window_days,
            )
            return summary.model_dump()
        except Exception as e:
            logger.error(f"Backtest error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/v1/market/{ticker}")
    def get_market_info(ticker: str):
        """Get real-time price history & company profile."""
        info = YFinanceAdapter.get_company_info(ticker)
        sec = SECEdgarAdapter.get_financial_summary(ticker)
        return {"company": info, "sec": sec}

    # ── WebSocket Real-Time Research Stream ────────────────────

    @app.websocket("/ws/research/{session_id}")
    async def websocket_research_endpoint(websocket: WebSocket, session_id: str):
        await ws_manager.connect(websocket, session_id)
        try:
            while True:
                data = await websocket.receive_text()
                # Parse search query request from web client
                try:
                    payload = json.loads(data)
                    ticker = payload.get("ticker", "NVDA").upper()
                    query = payload.get("query", "Analyze stock")

                    # Stream step 1: Planner
                    await ws_manager.send_event(
                        session_id, "planner_start", "ReasoningPlanner", "running",
                        {"message": f"Planner decomposing research objectives for {ticker}"}
                    )

                    # Stream step 2: Teams
                    results = engine.analyze_equity(ticker=ticker, query=query)

                    await ws_manager.send_event(
                        session_id, "research_complete", "EquiMindEngine", "completed",
                        results
                    )

                except json.JSONDecodeError:
                    await websocket.send_text(json.dumps({"error": "Invalid JSON"}))

        except WebSocketDisconnect:
            ws_manager.disconnect(websocket, session_id)

    # ── Static Files Serving (Web App) ─────────────────────────

    web_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "web")
    if os.path.exists(web_dir):
        app.mount("/", StaticFiles(directory=web_dir, html=True), name="static_web")
