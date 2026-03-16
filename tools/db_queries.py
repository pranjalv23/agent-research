import asyncio
import logging
from typing import List, Dict, Any

from langchain_core.tools import tool

from database.vector_db import VectorDB

logger = logging.getLogger("agent_research.db_queries")


def _check_papers_sync(links: List[str]) -> dict:
    logger.info("Checking if papers exist in DB — %d link(s)", len(links))
    db = VectorDB()
    result = db.papers_exist(links)
    logger.info("Papers exist check result: %s", result)
    return result


@tool("check_papers_in_db", return_direct=True)
async def check_papers_in_db(links: List[str]) -> dict:
    """
    Check if research papers are already stored in the vector database.

    Args:
        links: List of arXiv paper entry_id URLs to check

    Returns:
        Dictionary mapping each link to a boolean (True if present, False if not)
    """
    return await asyncio.to_thread(_check_papers_sync, links)



def _retrieve_sync(query: str, top_k: int) -> List[Dict[str, Any]]:
    logger.info("Retrieving from vector DB — query='%s', top_k=%d", query[:80], top_k)
    db = VectorDB()
    results = db.retrieve(query, top_k)
    logger.info("Retrieved %d chunks from vector DB", len(results))
    return results


@tool("retrieve_from_db", return_direct=True)
async def retrieve_from_db(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve relevant research papers from the vector database using semantic search.

    Args:
        query: Search query to find relevant papers
        top_k: Number of top results to return (default: 5)

    Returns:
        List of papers with title, authors, summary, link, and similarity score
    """
    return await asyncio.to_thread(_retrieve_sync, query, top_k)
