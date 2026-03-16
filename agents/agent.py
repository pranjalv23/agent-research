import logging

from agent_sdk.agents import BaseAgent
from tools.arxiv_retriever import download_arxiv_papers
from tools.db_queries import check_papers_in_db, retrieve_from_db

logger = logging.getLogger("agent_research.agent")

SYSTEM_PROMPT = (
    "You are an autonomous research assistant. "
    "Your task is to help the user find and summarize research papers relevant to their query. "
    "You have access to the following tools:\n"
    "- `arxiv_download(query: str, max_results: int)`: Search arXiv, download the PDFs, "
    "convert them to markdown, and store them in the vector database.\n"
    "- `check_papers_in_db(links: List[str])`: Check if research papers are already stored "
    "in the vector database.\n"
    "- `retrieve_from_db(query: str, top_k: int)`: Retrieve relevant research papers from "
    "the vector database using semantic search.\n\n"
    "Workflow:\n"
    "1. First use `retrieve_from_db` to check if relevant papers already exist.\n"
    "2. If not enough results, use `arxiv_download` to fetch new papers.\n"
    "3. After downloading, use `retrieve_from_db` again to get the most relevant chunks.\n"
    "4. Synthesize the findings into a clear, structured summary for the user.\n\n"
    "Always cite paper titles and authors in your response."
)


def create_agent() -> BaseAgent:
    logger.info("Creating research agent")
    return BaseAgent(
        tools=[download_arxiv_papers, check_papers_in_db, retrieve_from_db],
        system_prompt=SYSTEM_PROMPT,
    )


async def run_query(query: str, session_id: str = "default") -> str:
    logger.info("run_query called — session='%s', query='%s'", session_id, query[:100])
    agent = create_agent()
    response = await agent.arun(query, session_id=session_id)
    logger.info("run_query finished — session='%s'", session_id)
    return response
