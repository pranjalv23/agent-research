import logging
import os
from datetime import datetime, timezone

from agent_sdk.agents import BaseAgent
from agent_sdk.checkpoint import AsyncMongoDBSaver
from database.memory import get_memories, save_memory
from database.mongo import MongoDB

logger = logging.getLogger("agent_research.agent")

SYSTEM_PROMPT = """\
You are an autonomous research assistant with access to academic papers (arXiv), \
web search (Tavily), web scraping (Firecrawl), and a vector database of previously downloaded papers.

## Your Tools

**Paper tools (arXiv + Vector DB):**
- `retrieve_papers(query: str, top_k: int)` — Semantic search over previously downloaded papers in the vector DB.
- `download_and_store_arxiv_papers(query: str, max_results: int)` — Search arXiv, download PDFs, and store in the vector DB. \
The `query` param is passed directly to the arXiv search API.
- `check_papers_in_db(query: str)` — Check if relevant papers already exist in the vector DB.

**Web tools:**
- `tavily_quick_search(query: str, max_results: int)` — Web search. Great for finding recent information, \
blog posts, tutorials, news, or identifying specific papers/authors to then search on arXiv.
- `firecrawl_deep_scrape(url: str)` — Deep scrape a URL to get full markdown content. \
Use when you find a promising link from Tavily and need the full text (e.g., a blog post, report, or paper page).

## When to Use Which Tool

**USE arXiv tools when the user:**
- Explicitly asks for research papers, academic literature, or scholarly work
- Wants a literature review or survey of a research topic
- Asks about specific research methodologies, algorithms, or theoretical frameworks
- Wants to know the state-of-the-art in a specific research area

**DO NOT use arXiv tools when the user:**
- Asks general knowledge questions (e.g., "what is machine learning?")
- Asks about practical how-tos, tutorials, or implementation guidance
- Asks about news, current events, or industry trends
- Asks follow-up or clarifying questions about previously retrieved content
- Asks for opinions, comparisons, or recommendations that don't require papers
- Asks something you can confidently answer from conversation context or general knowledge

For these non-paper queries, use `tavily_quick_search` if you need external info, or just answer directly.

## Smart arXiv Searching

When you DO need to fetch papers, craft effective arXiv queries:
1. **Reformulate the user's query** into precise academic search terms. \
Don't pass casual/conversational queries directly to arXiv. \
Example: User asks "how do LLMs handle long documents?" → search for "long context transformers efficient attention mechanisms"
2. **Use multiple targeted searches** instead of one broad search when the topic has distinct sub-areas. \
Example: For "multi-modal AI for healthcare", do separate searches for "vision language models medical imaging" and "clinical NLP electronic health records".
3. **Use Tavily first** to identify key papers, authors, or specific terms when the topic is unfamiliar, \
then use those specific terms to search arXiv.
4. **Keep max_results small** (3-5) per search to get only the most relevant papers. It's better to do \
2-3 targeted searches with 3 results each than one broad search with 10 results.

## Workflow Guidelines

1. **Classify the query first** — decide if it actually needs papers or can be answered otherwise.
2. If papers are needed, check `retrieve_papers` first to see if the vector DB already has relevant content.
3. If the DB doesn't have what you need, craft precise arXiv search queries and call `download_and_store_arxiv_papers`.
4. After downloading, call `retrieve_papers` to get the stored content.
5. Supplement with `tavily_quick_search` for recent developments or practical context that papers might miss.
6. Use `firecrawl_deep_scrape` when a specific URL from search results has valuable in-depth content.
7. Synthesize everything into a clear, structured response.

**Always cite paper titles and authors when referencing academic work.**

## Behavioral Rule

Never narrate ANY part of your internal process. This is absolute — no exceptions. \
Forbidden categories (not exhaustive — the spirit of the rule covers all similar phrasing):
- Planning announcements: "Let me...", "Let's...", "I'll...", "First, I will...", "Step N:"
- Tool discovery narration: "Let's check if...", "I'll start by looking...", "Let me search..."
- Tool result narration: "Since there are no papers...", "The DB contains...", "I found..."
- Failure/retry narration: "It seems there was an issue...", "Let's try a different approach...", \
  "Unfortunately...", "Due to rate limiting..."
- Transition phrases: "Now let's...", "Next, I'll...", "Moving on to..."

Call tools silently. If a tool fails, silently try the next one. \
Only write text when delivering the final synthesized answer.

## Math & Equations

When writing mathematical expressions or equations:
- Use Markdown math notation, NOT LaTeX parenthesis delimiters.
- Inline math: $expression$ (e.g., $i_t = \\sigma(W_i \\cdot [h_{t-1}, x_t] + b_i)$)
- Block/display math: $$expression$$ on its own line
- Never use bare (expression) or \\(expression\\) as math delimiters — these render as plaintext.
"""

# MCP server configuration — all tools served from a single combined MCP server
MCP_SERVERS = {
    "mcp-tool-servers": {
        "url": os.getenv("MCP_SERVER_URL", "http://localhost:8010/mcp"),
        "transport": "http",
    },
}

_agent_instance: BaseAgent | None = None
_checkpointer: AsyncMongoDBSaver | None = None

RESPONSE_FORMAT_INSTRUCTIONS = {
    "summary": (
        "\n\nRESPONSE FORMAT OVERRIDE: The user wants a QUICK SUMMARY. "
        "Keep your response concise — 5-7 bullet points maximum. "
        "Focus on key findings and takeaways. Skip lengthy explanations."
    ),
    "flash_cards": (
        "\n\nRESPONSE FORMAT OVERRIDE: The user wants INSIGHT CARDS. "
        "Format your response as a series of insight cards using this EXACT format for each card:\n\n"
        "### [Topic Label]\n"
        "**Key Insight:** [The main finding or takeaway — keep it short and prominent]\n"
        "[1-2 sentence explanation with context]\n\n"
        "STRICT FORMATTING RULES:\n"
        "- Use exactly ### (three hashes) for each card topic — NOT ## or ####\n"
        "- Do NOT wrap topic names in **bold** — just plain text after ###\n"
        "- Do NOT use bullet points (- or *) for the Key Insight line — start it directly with **Key Insight:**\n"
        "- Every card MUST have a **Key Insight:** line\n"
        "- Start directly with the first ### card — no title header, preamble, or introductory text before the cards\n\n"
        "Generate 8-12 cards covering the most important research findings and takeaways."
    ),
    "detailed": "",
}


def _get_checkpointer() -> AsyncMongoDBSaver:
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = AsyncMongoDBSaver.from_conn_string(
            conn_string=os.getenv("MONGO_URI", "mongodb://localhost:27017"),
            db_name=os.getenv("MONGO_DB_NAME", "agent_research"),
        )
    return _checkpointer


def create_agent() -> BaseAgent:
    global _agent_instance
    if _agent_instance is None:
        logger.info("Creating research agent (singleton) with MCP servers")
        _agent_instance = BaseAgent(
            tools=[],
            mcp_servers=MCP_SERVERS,
            system_prompt=SYSTEM_PROMPT,
            checkpointer=_get_checkpointer(),
        )
    return _agent_instance


def _build_dynamic_context(session_id: str, query: str, response_format: str | None = None,
                            user_id: str | None = None) -> str:
    """Build dynamic context block (date, memories, format instructions) to prepend to the user query."""
    mem_key = user_id or session_id
    memories = get_memories(user_id=mem_key, query=query)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    year = today[:4]

    parts = []
    parts.append(
        f"Today's date: {today}. Include the current year ({year}) in search queries "
        "to get the latest papers and developments."
    )

    if memories:
        memory_lines = "\n".join(f"- {m}" for m in memories)
        parts.append(f"User context (long-term memory):\n{memory_lines}")
        logger.info("Injected %d memories into context for session='%s'", len(memories), session_id)

    format_instruction = RESPONSE_FORMAT_INSTRUCTIONS.get(response_format or "detailed", "")
    if format_instruction:
        parts.append(format_instruction.strip())

    context_block = "\n\n".join(parts)
    return f"[CONTEXT]\n{context_block}\n[/CONTEXT]\n\n"


async def run_query(query: str, session_id: str = "default",
                    response_format: str | None = None, model_id: str | None = None,
                    user_id: str | None = None) -> dict:
    logger.info("run_query called — session='%s', user='%s', query='%s', model='%s'",
                session_id, user_id or "anonymous", query[:100], model_id or "default")

    dynamic_context = _build_dynamic_context(session_id, query, response_format=response_format, user_id=user_id)
    enriched_query = dynamic_context + query

    agent = create_agent()
    result = await agent.arun(enriched_query, session_id=session_id, system_prompt=SYSTEM_PROMPT, model_id=model_id)
    logger.info("run_query finished — session='%s', steps: %d", session_id, len(result["steps"]))

    save_memory(user_id=user_id or session_id, query=query, response=result["response"])

    return result


def create_stream(query: str, session_id: str = "default",
                  response_format: str | None = None, model_id: str | None = None,
                  user_id: str | None = None):
    """Create a StreamResult for the query. Returns the stream object directly."""
    logger.info("create_stream called — session='%s', user='%s', query='%s', model='%s'",
                session_id, user_id or "anonymous", query[:100], model_id or "default")

    dynamic_context = _build_dynamic_context(session_id, query, response_format=response_format, user_id=user_id)
    enriched_query = dynamic_context + query
    agent = create_agent()
    return agent.astream(enriched_query, session_id=session_id, system_prompt=SYSTEM_PROMPT, model_id=model_id)


async def stream_query(query: str, session_id: str = "default"):
    """Async generator that yields text chunks for SSE streaming."""
    logger.info("stream_query called — session='%s', query='%s'", session_id, query[:100])

    dynamic_context = _build_dynamic_context(session_id, query)
    enriched_query = dynamic_context + query

    agent = create_agent()
    full_response = []

    async for chunk in agent.astream(enriched_query, session_id=session_id, system_prompt=SYSTEM_PROMPT):
        full_response.append(chunk)
        yield chunk

    # Save the complete response to Mem0 after streaming finishes
    response_text = "".join(full_response)
    save_memory(user_id=session_id, query=query, response=response_text)
    logger.info("stream_query finished — session='%s', response length: %d chars",
                session_id, len(response_text))
