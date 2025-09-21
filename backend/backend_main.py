import os
import json
import copy
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Query, Body, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

load_dotenv()

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
OPENAI_KEY = os.getenv("OPENAI_KEY")
FIRECRAWL_KEY = os.getenv("FIRECRAWL_KEY")
GH_PAT = os.getenv("GH_PAT")

CORAL_SERVER_HOST = os.getenv("CORAL_SERVER_HOST", "http://localhost:5555")
THIS_HOST = os.getenv("THIS_HOST", "http://localhost:8000")

# For local dev you can set: CORS_ALLOWED_ORIGINS="http://localhost:5173,http://localhost:3000"
CORS_ALLOWED_ORIGINS = [o.strip() for o in os.getenv("CORS_ALLOWED_ORIGINS", "*").split(",")] if os.getenv("CORS_ALLOWED_ORIGINS") else ["*"]

# -------------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------------
app = FastAPI(title="Coral Bridge Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# MCP Custom Tool definition (the tool Coral agents will call back)
# -------------------------------------------------------------------
customTools = {
    "search-result": {
        "transport": {
            "type": "http",
            "url": f"{THIS_HOST}/mcp/search-result"
        },
        "toolSchema": {
            "name": "send-search-result",
            "description": "Send a single result of your search. You can call this multiple times as you find more info.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "result": {
                        "type": "string",
                        "description": "The text of the result, as markdown"
                    }
                },
                "required": ["result"]
            }
        }
    }
}

# Base agent graph. We'll deep-copy this per request.
BASE_AGENT_GRAPH = {
    "agents": [
        # Index 0 will be the "interface" agent (filled per request)
        {},
        {
            "id": {"name": "firecrawl", "version": "0.0.1"},
            "name": "firecrawl",
            "coralPlugins": [],
            "provider": {"type": "local", "runtime": "executable"},
            "blocking": True,
            "options": {
                "MODEL_API_KEY": {"type": "string", "value": OPENAI_KEY},
                "FIRECRAWL_API_KEY": {"type": "string", "value": FIRECRAWL_KEY},
            },
            "customToolAccess": [],
        },
        {
            "id": {"name": "github", "version": "0.0.1"},
            "name": "github",
            "coralPlugins": [],
            "provider": {"type": "local", "runtime": "executable"},
            "blocking": True,
            "options": {
                "MODEL_API_KEY": {"type": "string", "value": OPENAI_KEY},
                "GITHUB_PERSONAL_ACCESS_TOKEN": {"type": "string", "value": GH_PAT},
            },
            "customToolAccess": [],
        },
    ],
    "groups": [["interface", "firecrawl", "github"]],
    "customTools": customTools,
}

def make_agent_graph_request(user_query: str) -> Dict[str, Any]:
    """
    Build a fresh agentGraphRequest for Coral using a deep copy of BASE_AGENT_GRAPH.
    """
    graph = copy.deepcopy(BASE_AGENT_GRAPH)
    interface_agent = {
        "id": {"name": "interface", "version": "0.0.1"},
        "name": "interface",
        "coralPlugins": [],
        "provider": {"type": "local", "runtime": "executable"},
        "blocking": True,
        "options": {
            "MODEL_API_KEY": {"type": "string", "value": OPENAI_KEY},
            "USER_REQUEST": {"type": "string", "value": user_query},
        },
        "customToolAccess": ["search-result"],
    }
    graph["agents"][0] = interface_agent
    return graph

# -------------------------------------------------------------------
# In-memory session state
# -------------------------------------------------------------------
class ClientMessage(BaseModel):
    type: str  # e.g., "search-result", "status", "error"
    coral_session_id: Optional[str] = None
    payload: Any = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class CoralSessionState(BaseModel):
    coral_session_id: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_status: str = "created"
    last_error: Optional[str] = None

class ClientSessionState(BaseModel):
    client_session_id: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    coral_sessions: Dict[str, CoralSessionState] = Field(default_factory=dict)
    messages: List[ClientMessage] = Field(default_factory=list)

    # Async delivery queue for streaming outbound messages to frontend
    # Not part of pydantic model serialization:
    def __init__(self, **data):
        super().__init__(**data)
        self.queue: asyncio.Queue[ClientMessage] = asyncio.Queue()

# Global registries
sessions_by_client_id: Dict[str, ClientSessionState] = {}
client_id_by_coral_id: Dict[str, str] = {}  # reverse mapping to route MCP callbacks

def get_or_create_client_session(client_session_id: str) -> ClientSessionState:
    state = sessions_by_client_id.get(client_session_id)
    if state is None:
        state = ClientSessionState(client_session_id=client_session_id)
        sessions_by_client_id[client_session_id] = state
    return state

async def push_message_to_client(client_session_id: str, message: ClientMessage):
    state = sessions_by_client_id.get(client_session_id)
    if not state:
        # Create on the fly to avoid losing messages (optional behavior)
        state = get_or_create_client_session(client_session_id)
    state.messages.append(message)
    await state.queue.put(message)

# -------------------------------------------------------------------
# Request/Response models
# -------------------------------------------------------------------
class QueryRequest(BaseModel):
    client_session_id: str
    query: str
    application_id: Optional[str] = "app"
    privacy_key: Optional[str] = "priv"
    # Optional passthrough to Coral payload "metadata" if needed
    metadata: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    client_session_id: str
    coral_session_id: str
    status: str = "queued"

class SessionsListResponse(BaseModel):
    sessions: List[Dict[str, Any]]

# -------------------------------------------------------------------
# Frontend -> Backend: Start a query for a client session
# This will create a Coral session and immediately return ids.
# Results will arrive via MCP callback and be streamed to frontend.
# -------------------------------------------------------------------
@app.post("/bridge/query", response_model=QueryResponse)
async def start_query(req: QueryRequest):
    """
    Frontend calls this to start a new task for a given client_session_id.
    """
    if not OPENAI_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_KEY is not configured")

    client_state = get_or_create_client_session(req.client_session_id)

    payload = {
        "privacyKey": req.privacy_key or "priv",
        "applicationId": req.application_id or "app",
        # Leave empty to let Coral allocate a sessionId; we will map it after creation:
        "sessionId": "",
        "agentGraphRequest": make_agent_graph_request(req.query),
    }
    # Attach optional metadata (if your Coral server supports it)
    if req.metadata:
        payload["metadata"] = req.metadata

    # Create the Coral session
    coral_session_id: Optional[str] = None
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(f"{CORAL_SERVER_HOST}/api/v1/sessions", json=payload)
            resp.raise_for_status()
            data = resp.json()
            coral_session_id = data.get("sessionId")
        except httpx.HTTPError as e:
            # Record error for the client session
            msg = ClientMessage(type="error", payload={"error": f"Failed to create Coral session: {str(e)}"})
            await push_message_to_client(req.client_session_id, msg)
            raise HTTPException(status_code=502, detail="Coral server error creating session")

    if not coral_session_id:
        msg = ClientMessage(type="error", payload={"error": "Coral server did not return a sessionId"})
        await push_message_to_client(req.client_session_id, msg)
        raise HTTPException(status_code=502, detail="Coral sessionId missing")

    # Map coral -> client
    client_id_by_coral_id[coral_session_id] = req.client_session_id
    coral_state = CoralSessionState(coral_session_id=coral_session_id, last_status="created")
    client_state.coral_sessions[coral_session_id] = coral_state

    # Inform frontend we queued the request
    await push_message_to_client(
        req.client_session_id,
        ClientMessage(
            type="status",
            coral_session_id=coral_session_id,
            payload={"message": "Coral session created"}
        )
    )

    return QueryResponse(client_session_id=req.client_session_id, coral_session_id=coral_session_id, status="created")

# -------------------------------------------------------------------
# Frontend: list active client sessions
# -------------------------------------------------------------------
@app.get("/bridge/sessions", response_model=SessionsListResponse)
async def list_sessions():
    result = []
    for cs in sessions_by_client_id.values():
        result.append({
            "client_session_id": cs.client_session_id,
            "created_at": cs.created_at,
            "coral_session_ids": list(cs.coral_sessions.keys()),
            "message_count": len(cs.messages),
        })
    return SessionsListResponse(sessions=result)

# -------------------------------------------------------------------
# Frontend: inspect a single client session (messages and coral sessions)
# -------------------------------------------------------------------
@app.get("/bridge/sessions/{client_session_id}")
async def get_session(client_session_id: str):
    cs = sessions_by_client_id.get(client_session_id)
    if not cs:
        raise HTTPException(status_code=404, detail="Client session not found")
    return {
        "client_session_id": cs.client_session_id,
        "created_at": cs.created_at,
        "coral_sessions": {k: v.dict() for k, v in cs.coral_sessions.items()},
        "messages": [m.dict() for m in cs.messages],
    }

# -------------------------------------------------------------------
# Streaming back to the frontend (SSE)
# The frontend can open this to receive streamed messages for the session.
# -------------------------------------------------------------------
@app.get("/bridge/events-sse/{client_session_id}")
async def sse_events(client_session_id: str):
    cs = get_or_create_client_session(client_session_id)

    async def event_generator():
        # Send a hello/status immediately so the client knows the stream is alive
        hello = ClientMessage(type="status", payload={"message": "SSE stream opened"})
        yield f"data: {hello.json()}\n\n"

        try:
            while True:
                msg: ClientMessage = await cs.queue.get()
                yield f"data: {msg.json()}\n\n"
        except asyncio.CancelledError:
            # client disconnected
            return

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# -------------------------------------------------------------------
# Streaming back to the frontend (WebSocket)
# -------------------------------------------------------------------
@app.websocket("/bridge/ws/{client_session_id}")
async def websocket_events(websocket: WebSocket, client_session_id: str):
    await websocket.accept()
    cs = get_or_create_client_session(client_session_id)

    async def producer():
        # Push a greeting/status
        await websocket.send_text(ClientMessage(type="status", payload={"message": "WebSocket connected"}).json())
        try:
            while True:
                msg: ClientMessage = await cs.queue.get()
                await websocket.send_text(msg.json())
        except (WebSocketDisconnect, RuntimeError):
            return

    async def consumer():
        # Optional: receive messages from frontend (e.g., pings, cancel commands)
        try:
            while True:
                _ = await websocket.receive_text()
                # No-op; you can add command handling here
        except WebSocketDisconnect:
            return

    # Run both tasks; if either ends, close the socket
    producer_task = asyncio.create_task(producer())
    consumer_task = asyncio.create_task(consumer())
    done, pending = await asyncio.wait({producer_task, consumer_task}, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    await websocket.close()

# -------------------------------------------------------------------
# MCP Callback from Coral:
# Coral calls this when the interface agent invokes the "send-search-result" tool.
# The URL includes {sessionId} and {agentId}; we map sessionId back to client_session_id.
# -------------------------------------------------------------------
class SearchResult(BaseModel):
    result: str

@app.post("/mcp/search-result/{sessionId}/{agentId}")
async def mcp_search_result(sessionId: str, agentId: str, body: SearchResult):
    client_session_id = client_id_by_coral_id.get(sessionId)
    if not client_session_id:
        # We don't have a mapping; accept but log. (You might want to 404 in stricter setups.)
        print(f"[WARN] Unknown Coral sessionId {sessionId} for MCP callback")
        raise HTTPException(status_code=404, detail="No mapped client session for this Coral session")

    # Update coral session status
    cs = sessions_by_client_id.get(client_session_id)
    if cs and sessionId in cs.coral_sessions:
        cs.coral_sessions[sessionId].last_status = "message"

    # Push result downstream
    await push_message_to_client(
        client_session_id,
        ClientMessage(
            type="search-result",
            coral_session_id=sessionId,
            payload={"agentId": agentId, "markdown": body.result},
        )
    )
    return {"status": "ok"}

# -------------------------------------------------------------------
# Optional: Endpoint to clean up a client session (free memory)
# -------------------------------------------------------------------
@app.delete("/bridge/sessions/{client_session_id}")
async def delete_session(client_session_id: str):
    cs = sessions_by_client_id.pop(client_session_id, None)
    if cs:
        # Remove reverse mappings
        for coral_id in cs.coral_sessions.keys():
            client_id_by_coral_id.pop(coral_id, None)
        return {"deleted": True}
    return {"deleted": False}

# -------------------------------------------------------------------
# Simple health check
# -------------------------------------------------------------------
@app.get("/healthz")
async def healthz():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}