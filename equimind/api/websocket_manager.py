"""
EquiMind Real-Time Research WebSocket Stream Manager
===================================================
Streams live research events, DAG state transitions, team execution logs,
and evidence collection over WebSocket to interactive web UI clients.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Set
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages active WebSocket client connections and event broadcasting."""

    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = set()
        self.active_connections[session_id].add(websocket)
        logger.info(f"WebSocket client connected for session {session_id}")

    def disconnect(self, websocket: WebSocket, session_id: str):
        if session_id in self.active_connections:
            self.active_connections[session_id].discard(websocket)
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]
        logger.info(f"WebSocket client disconnected for session {session_id}")

    async def send_event(self, session_id: str, event_type: str, agent: str,
                         status: str, payload: Dict[str, Any]):
        """Send structured research execution event to session clients."""
        if session_id not in self.active_connections:
            return

        message = {
            "type": event_type,
            "agent": agent,
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        }

        dead_sockets = set()
        for websocket in self.active_connections[session_id]:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.debug(f"Error sending event to websocket: {e}")
                dead_sockets.add(websocket)

        for ws in dead_sockets:
            self.disconnect(ws, session_id)


ws_manager = ConnectionManager()
