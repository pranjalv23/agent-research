from agent_sdk.secrets.akv import load_akv_secrets
load_akv_secrets()

import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import Response, StreamingResponse

from agent_sdk.logging import configure_logging
from agent_sdk.utils.env import validate_required_env_vars
from agent_sdk.utils.validation import SAFE_SESSION_RE
from agent_sdk.metrics import metrics_response
from agent_sdk.server.app_factory import create_agent_app
from agent_sdk.server.models import AskRequest, AskResponse, HistoryResponse, SessionsHistoryRequest
from agent_sdk.server.sse import create_sse_stream
from agent_sdk.server.session import verify_session_ownership
from agent_sdk.server.streaming import StreamingMathFixer, _fix_math_delimiters
from agents.agent import create_agent, run_query, create_stream, save_memory
from database.mongo import MongoDB
from a2a_service.server import create_a2a_app


configure_logging("agent_research")
logger = logging.getLogger("agent_research.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    from agent_sdk.observability import init_sentry
    init_sentry("agent-research")
    validate_required_env_vars(
        ["MONGO_URI", "AZURE_AI_FOUNDRY_ENDPOINT", "AZURE_AI_FOUNDRY_API_KEY",
         "PINECONE_API_KEY"],
        "agent-research",
    )
    if not os.getenv("INTERNAL_API_KEY"):
        logger.warning("INTERNAL_API_KEY is not set — internal API is unprotected. Set this in production.")
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


app, limiter = create_agent_app("Research Agent API", lifespan)

a2a_app = create_a2a_app()
app.mount("/a2a", a2a_app.build())


# ── Agent endpoints ──

@app.post("/ask", response_model=AskResponse)
@limiter.limit("30/minute")
async def ask(body: AskRequest, request: Request):
    user_id = request.headers.get("X-User-Id") or None
    is_new = body.session_id is None
    session_id = body.session_id or MongoDB.generate_session_id()

    if not is_new:
        await verify_session_ownership(session_id, user_id, MongoDB)

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

    return AskResponse(session_id=session_id, query=body.query, response=response)


@app.post("/ask/stream")
@limiter.limit("30/minute")
async def ask_stream(body: AskRequest, request: Request):
    """Stream the agent's response as Server-Sent Events (SSE)."""
    user_id = request.headers.get("X-User-Id") or None
    is_new = body.session_id is None
    session_id = body.session_id or MongoDB.generate_session_id()

    if not is_new:
        await verify_session_ownership(session_id, user_id, MongoDB)

    logger.info("POST /ask/stream — session='%s', user='%s', query='%s'",
                session_id, user_id or "anonymous", body.query[:100])

    raw_stream = await create_stream(body.query, session_id=session_id,
                                     response_format=body.response_format, model_id=body.model_id,
                                     user_id=user_id)
    stream = StreamingMathFixer(raw_stream)

    async def _on_complete(response_text: str, steps: list, plan: str | None) -> None:
        save_memory(user_id=user_id or session_id, query=body.query, response=response_text)
        await MongoDB.save_conversation(
            session_id=session_id, query=body.query, response=response_text,
            steps=steps, user_id=user_id, plan=plan,
        )

    return StreamingResponse(
        create_sse_stream(stream, session_id=session_id, query=body.query, on_complete=_on_complete),
        media_type="text/event-stream",
    )


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
    return HistoryResponse(session_id=session_id, history=history)


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
