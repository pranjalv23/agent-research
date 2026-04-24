import asyncio
import json
import logging
import os
from collections import defaultdict
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from agent_sdk.logging import configure_logging
from agent_sdk.middleware.infra import RequestIDMiddleware, SecurityHeadersMiddleware, VerifyInternalKeyMiddleware
from agent_sdk.utils.env import validate_required_env_vars
from agent_sdk.utils.validation import SAFE_SESSION_RE
from agent_sdk.server.error_handlers import register_error_handlers
from agent_sdk.metrics import metrics_response
from agent_sdk.server.streaming import StreamingMathFixer, _fix_math_delimiters
from agents.agent import create_agent, run_query, stream_query, create_stream, save_memory
from database.mongo import MongoDB
from a2a_service.server import create_a2a_app


configure_logging("agent_research")
logger = logging.getLogger("agent_research.api")
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_required_env_vars(
        ["MONGO_URI", "AZURE_AI_FOUNDRY_ENDPOINT", "AZURE_AI_FOUNDRY_API_KEY",
         "PINECONE_API_KEY"],
        "agent-research",
    )
    if not os.getenv("INTERNAL_API_KEY"):
        logger.warning("INTERNAL_API_KEY is not set — internal API is unprotected. Set this in production.")
    # Connect MCP servers on startup
    agent = create_agent()
    try:
        await agent._ensure_initialized()
        if getattr(agent, '_degraded', False):
            logger.warning("Agent started in DEGRADED mode — MCP tools unavailable")
        else:
            logger.info("MCP servers connected, agent ready")
    except Exception as e:
        logger.error("Agent initialization failed (continuing without MCP): %s", e)
    await MongoDB.ensure_indexes()
    yield
    await agent._disconnect_mcp()
    await MongoDB.close()
    logger.info("Shutdown complete")


app = FastAPI(
    title="Research Agent API",
    description="Ask research questions and get AI-powered summaries from arXiv papers.",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
register_error_handlers(app)

_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173")
_allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Internal-API-Key", "X-User-Id", "X-Request-ID"],
)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(VerifyInternalKeyMiddleware)

# Mount the A2A server as a sub-application
a2a_app = create_a2a_app()
app.mount("/a2a", a2a_app.build())


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=8000)
    session_id: str | None = None
    response_format: str | None = None
    model_id: str | None = None

    model_config = {"json_schema_extra": {"examples": [{"query": "", "session_id": None, "response_format": "detailed", "model_id": None}]}}


class AskResponse(BaseModel):
    session_id: str
    query: str
    response: str


class HistoryResponse(BaseModel):
    session_id: str
    history: list[dict]


_session_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

def get_session_lock(session_id: str) -> asyncio.Lock:
    return _session_locks[session_id]


# ── Agent endpoints ──

@app.post("/ask", response_model=AskResponse)
@limiter.limit("30/minute")
async def ask(body: AskRequest, request: Request):
    user_id = request.headers.get("X-User-Id") or None
    is_new = body.session_id is None
    session_id = body.session_id or MongoDB.generate_session_id()

    if not is_new and user_id:
        owned_history = await MongoDB.get_history(session_id, user_id=user_id)
        if not owned_history:
            any_history = await MongoDB.get_history(session_id)
            if any_history:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    logger.info("POST /ask — session='%s' (%s), user='%s', query='%s'",
                session_id, "new" if is_new else "existing", user_id or "anonymous", body.query[:100])

    result = await run_query(body.query, session_id=session_id,
                             response_format=body.response_format, model_id=body.model_id,
                             user_id=user_id)
    response = _fix_math_delimiters(result["response"])
    steps = result["steps"]

    await MongoDB.save_conversation(
        session_id=session_id,
        query=body.query,
        response=response,
        steps=steps,
        user_id=user_id,
        plan=result.get("plan"),
    )

    logger.info("POST /ask complete — session='%s', response length: %d chars, tool_calls: %d",
                session_id, len(response),
                sum(1 for s in steps if s.get("action") == "tool_call"))

    return AskResponse(
        session_id=session_id,
        query=body.query,
        response=response,
    )


@app.post("/ask/stream")
@limiter.limit("30/minute")
async def ask_stream(body: AskRequest, request: Request):
    """Stream the agent's response as Server-Sent Events (SSE).

    Each event is a JSON object with a `text` field containing a chunk.
    The stream ends with a `[DONE]` sentinel.
    """
    user_id = request.headers.get("X-User-Id") or None
    is_new = body.session_id is None
    session_id = body.session_id or MongoDB.generate_session_id()

    if not is_new and user_id:
        owned_history = await MongoDB.get_history(session_id, user_id=user_id)
        if not owned_history:
            any_history = await MongoDB.get_history(session_id)
            if any_history:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    logger.info("POST /ask/stream — session='%s', user='%s', query='%s'",
                session_id, user_id or "anonymous", body.query[:100])

    stream = await create_stream(body.query, session_id=session_id,
                           response_format=body.response_format, model_id=body.model_id,
                           user_id=user_id)

    _STREAM_TIMEOUT = float(os.getenv("STREAM_TIMEOUT_SECONDS", "300"))

    async def event_stream():
        full_response = []
        queue = asyncio.Queue(maxsize=100)
        _HEARTBEAT_INTERVAL = 15.0

        async def heartbeat_worker():
            try:
                while True:
                    await asyncio.sleep(_HEARTBEAT_INTERVAL)
                    await queue.put(f": heartbeat {int(asyncio.get_running_loop().time())}\n\n")
            except asyncio.CancelledError:
                pass

        async def agent_worker():
            try:
                async with get_session_lock(session_id):
                    async with asyncio.timeout(_STREAM_TIMEOUT):
                        async for chunk in StreamingMathFixer(stream):
                            try:
                                await asyncio.wait_for(queue.put(chunk), timeout=30.0)
                            except asyncio.TimeoutError:
                                logger.warning("Stream queue full for session='%s' — client likely disconnected", session_id)
                                return
            except TimeoutError:
                logger.error("Stream timed out after %.0fs", _STREAM_TIMEOUT)
                await queue.put(f"__ERROR__:Response timed out after {_STREAM_TIMEOUT:.0f} seconds.")
            except Exception as e:
                logger.error("Stream failed: %s", e)
                await queue.put("__ERROR__:An internal error occurred while generating the response.")
            finally:
                try:
                    queue.put_nowait(None)
                except asyncio.QueueFull:
                    pass

        heartbeat_task = asyncio.create_task(heartbeat_worker())
        agent_task = asyncio.create_task(agent_worker())

        try:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break

                if chunk.startswith(": heartbeat"):
                    yield chunk
                elif chunk.startswith("__ERROR__:"):
                    error_msg = chunk[len("__ERROR__:"):]
                    yield f"event: error\ndata: {json.dumps({'message': error_msg})}\n\n"
                    fallback = f"\n\n[{error_msg}]"
                    yield f"data: {json.dumps({'text': fallback})}\n\n"
                    full_response.append(fallback)
                else:
                    full_response.append(chunk)
                    yield f"data: {json.dumps({'text': chunk})}\n\n"

            response_text = "".join(full_response)
            if not response_text.strip():
                fallback = "Sorry, the model returned an empty response. Please try again."
                yield f"data: {json.dumps({'text': fallback})}\n\n"
                response_text = fallback

            try:
                save_memory(user_id=user_id or session_id, query=body.query, response=response_text)
                await MongoDB.save_conversation(
                    session_id=session_id, query=body.query, response=response_text,
                    steps=stream.steps, user_id=user_id, plan=stream.plan,
                )
            except Exception as e:
                logger.error("Failed to save conversation: %s", e)

            yield f"data: {json.dumps({'session_id': session_id})}\n\n"
        finally:
            heartbeat_task.cancel()
            agent_task.cancel()
            try:
                await agent_task
            except asyncio.CancelledError:
                pass
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/history/user/me", response_model=HistoryResponse)
@limiter.limit("60/minute")
async def get_history_by_user(request: Request):
    user_id = request.headers.get("X-User-Id") or None
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    logger.info("GET /history/user/me — user='%s'", user_id)
    history = await MongoDB.get_history_by_user(user_id)
    return HistoryResponse(session_id=user_id, history=history)


@app.get("/history/{session_id}", response_model=HistoryResponse)
@limiter.limit("60/minute")
async def get_history(request: Request, session_id: str):
    user_id = request.headers.get("X-User-Id") or None
    logger.info("GET /history — session='%s'", session_id)
    history = await MongoDB.get_history(session_id, user_id=user_id)
    logger.info("Returning %d history entries for session='%s'", len(history), session_id)
    return HistoryResponse(session_id=session_id, history=history)


class SessionsHistoryRequest(BaseModel):
    session_ids: list[str]

@app.post("/history/sessions")
@limiter.limit("30/minute")
async def get_history_by_sessions(request: Request, body: SessionsHistoryRequest):
    user_id = request.headers.get("X-User-Id") or None
    safe_ids = [s for s in body.session_ids[:20] if isinstance(s, str) and SAFE_SESSION_RE.match(s)]
    logger.info("POST /history/sessions — %d session(s)", len(safe_ids))
    history = await MongoDB.get_history_by_sessions(safe_ids, user_id=user_id)
    return {"history": history}


@app.get("/metrics")
async def metrics():
    content, content_type = metrics_response()
    return Response(content=content, media_type=content_type)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "agent-research"}


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8081))
    ssl_certfile = os.getenv("SSL_CERTFILE") or None
    ssl_keyfile = os.getenv("SSL_KEYFILE") or None
    uvicorn.run(app, host="0.0.0.0", port=port, ssl_certfile=ssl_certfile, ssl_keyfile=ssl_keyfile)
