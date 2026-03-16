import os
from typing import List, Dict, Any

import pymupdf4llm
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

_CHUNK_SIZE = 1000
_CHUNK_OVERLAP = 200
_EXIST_THRESHOLD = 0.7
_UPSERT_BATCH = 100


class VectorDB:
    # Embedding dimensions per provider
    _DIMENSIONS = {"gemini": 3072, "nvidia": 4096}

    def __init__(self, provider: str = "nvidia"):
        self.provider = provider
        self.pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = "research-papers"

        if provider == "gemini":
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="gemini-embedding-2-preview",
                google_api_key=os.getenv("GEMINI_API_KEY"),
            )
        else:
            self.embeddings = NVIDIAEmbeddings(
                model="nvidia/nv-embed-v1",
                nvidia_api_key=os.getenv("NVIDIA_API_KEY"),
            )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=_CHUNK_SIZE,
            chunk_overlap=_CHUNK_OVERLAP,
        )
        self._ensure_index()

    def _ensure_index(self):
        expected_dim = self._DIMENSIONS[self.provider]
        existing = {idx.name: idx for idx in self.pinecone.list_indexes()}

        if self.index_name in existing:
            current_dim = existing[self.index_name].dimension
            if current_dim != expected_dim:
                self.pinecone.delete_index(self.index_name)
                existing.pop(self.index_name)

        if self.index_name not in existing:
            self.pinecone.create_index(
                name=self.index_name,
                dimension=expected_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        self.index = self.pinecone.Index(self.index_name)

    def _paper_id(self, pdf_path: str) -> str:
        """Derive a stable paper ID from the PDF filename (e.g. 'papers/2301.00001v1.pdf' -> '2301.00001v1')."""
        return os.path.splitext(os.path.basename(pdf_path))[0]

    def pdf_to_markdown(self, pdf_path: str) -> str:
        """Convert a PDF to markdown text using pymupdf4llm."""
        return pymupdf4llm.to_markdown(pdf_path)

    def upsert_papers(self, papers: List[Dict[str, Any]]):
        """
        Convert PDFs to markdown, chunk, embed, and upsert into Pinecone.

        Each paper dict must have: title, authors, summary, pdf_path, pdf_url
        """
        upsert_data = []

        for paper in papers:
            paper_id = self._paper_id(paper["pdf_path"])
            markdown = self.pdf_to_markdown(paper["pdf_path"])
            chunks = self.splitter.split_text(markdown)

            vectors = self.embeddings.embed_documents(chunks)

            authors = (
                ", ".join(paper["authors"])
                if isinstance(paper["authors"], list)
                else paper["authors"]
            )

            for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
                upsert_data.append({
                    "id": f"{paper_id}_chunk_{i}",
                    "values": vector,
                    "metadata": {
                        "paper_id": paper_id,
                        "title": paper["title"],
                        "authors": authors,
                        "summary": paper["summary"],
                        "pdf_path": paper["pdf_path"],
                        "pdf_url": paper["pdf_url"],
                        "chunk_index": i,
                        "text": chunk,
                    },
                })

        for i in range(0, len(upsert_data), _UPSERT_BATCH):
            self.index.upsert(vectors=upsert_data[i:i + _UPSERT_BATCH])

    def papers_exist(self, query: str) -> bool:
        """Check if relevant documents exist in the vector DB for a given query."""
        query_vector = self.embeddings.embed_query(query)
        results = self.index.query(
            vector=query_vector,
            top_k=2,
            include_metadata=False,
        )
        return bool(results.matches) and results.matches[0].score >= _EXIST_THRESHOLD

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Semantically search the vector DB and return the top_k matching chunks."""
        query_vector = self.embeddings.embed_query(query)
        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
        )

        chunks = []
        for match in results.matches:
            meta = match.metadata
            chunks.append({
                "title": meta.get("title"),
                "authors": meta.get("authors"),
                "summary": meta.get("summary"),
                "pdf_url": meta.get("pdf_url"),
                "text": meta.get("text"),
                "score": match.score,
            })
        return chunks
