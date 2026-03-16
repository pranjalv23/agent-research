from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from agents.agent import run_query
from database.mongo import MongoDB


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


class AskResponse(BaseModel):
    session_id: str
    query: str
    response: str


class HistoryResponse(BaseModel):
    session_id: str
    history: list[dict]


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    session_id = request.session_id or MongoDB.generate_session_id()

    response = await run_query(request.query, session_id=session_id)

    await MongoDB.save_conversation(
        session_id=session_id,
        query=request.query,
        response=response,
    )

    return AskResponse(
        session_id=session_id,
        query=request.query,
        response=response,
    )


@app.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str):
    history = await MongoDB.get_history(session_id)
    return HistoryResponse(session_id=session_id, history=history)


@app.get("/health")
async def health():
    return {"status": "ok"}
