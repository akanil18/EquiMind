import json
import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel

try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse, FileResponse
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from equimind.orchestrator.engine import EquiMindEngine

logger = logging.getLogger(__name__)

if HAS_FASTAPI:
    app = FastAPI(
        title="EquiMind API",
        description="Autonomous AI Equity Research Firm & Financial Research Orchestration Framework",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    engine = EquiMindEngine()

    class ResearchRequest(BaseModel):
        ticker: str
        query: str = "Should I invest in this stock for long-term?"
        provider: str = "openai"
        model: Optional[str] = None
        as_of_date: Optional[str] = None

    @app.get("/api/v1/health")
    def health_check():
        return {"status": "ok", "service": "EquiMind Orchestration Engine", "version": "0.1.0"}

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
