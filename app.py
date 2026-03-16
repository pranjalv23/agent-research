import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from agents.agent import run_query
from database.mongo import MongoDB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("agent_research.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await MongoDB.close()


app = FastAPI(
    title="Research Agent API",
    description="Ask research questions and get AI-powered summaries from arXiv papers.",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    query: str
    session_id: str | None = None

    model_config = {"json_schema_extra": {"examples": [{"query": "", "session_id": None}]}}


class AskResponse(BaseModel):
    session_id: str
    query: str
    response: str


class HistoryResponse(BaseModel):
    session_id: str
    history: list[dict]


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    is_new = request.session_id is None
    session_id = request.session_id or MongoDB.generate_session_id()

    logger.info("POST /ask — session='%s' (%s), query='%s'",
                session_id, "new" if is_new else "existing", request.query[:100])

    result = await run_query(request.query, session_id=session_id)
    response = result["response"]
    steps = result["steps"]

    await MongoDB.save_conversation(
        session_id=session_id,
        query=request.query,
        response=response,
        steps=steps,
    )

    logger.info("POST /ask complete — session='%s', response length: %d chars, tool_calls: %d",
                session_id, len(response),
                sum(1 for s in steps if s.get("action") == "tool_call"))

    return AskResponse(
        session_id=session_id,
        query=request.query,
        response=response,
    )


@app.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str):
    logger.info("GET /history — session='%s'", session_id)
    history = await MongoDB.get_history(session_id)
    logger.info("Returning %d history entries for session='%s'", len(history), session_id)
    return HistoryResponse(session_id=session_id, history=history)


@app.get("/health")
async def health():
    return {"status": "ok"}
