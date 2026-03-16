import asyncio
import arxiv
from pathlib import Path
from langchain_core.tools import tool

from database.vector_db import VectorDB


_arxiv_client = arxiv.Client()


def _download_arxiv_papers_sync(query: str, max_results: int = 5):

    search = arxiv.Search(
        query=query,
        max_results=max_results
    )

    download_dir = Path("papers")
    download_dir.mkdir(exist_ok=True)

    results = []

    for paper in _arxiv_client.results(search):

        file_name = f"{paper.get_short_id()}.pdf"
        file_path = download_dir / file_name

        paper.download_pdf(
            dirpath=str(download_dir),
            filename=file_name
        )

        results.append(
            {
                "title": paper.title,
                "authors": [a.name for a in paper.authors],
                "summary": paper.summary,
                "pdf_path": str(file_path),
                "pdf_url": paper.pdf_url,
            }
        )

    db = VectorDB()
    db.upsert_papers(results)

    return results


@tool("arxiv_download", return_direct=True)
async def download_arxiv_papers(query: str, max_results: int = 5) -> list[dict]:
    """
    Search arXiv, download the PDFs, convert them to markdown, and store them in the vector database.
    """

    papers = await asyncio.to_thread(
        _download_arxiv_papers_sync,
        query,
        max_results
    )

    return papers