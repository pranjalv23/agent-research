import logging
from datetime import datetime, timezone

from agent_sdk.agents import BaseAgent
from database.memory import get_memories, save_memory

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
"""

# MCP server configuration — tools are served remotely via MCP protocol
MCP_SERVERS = {
    "web-search": {
        "url": "http://localhost:8010/mcp",
        "transport": "streamable_http",
    },
    "vector-db": {
        "url": "http://localhost:8012/mcp",
        "transport": "streamable_http",
    },
}

_agent_instance: BaseAgent | None = None


def create_agent() -> BaseAgent:
    global _agent_instance
    if _agent_instance is None:
        logger.info("Creating research agent (singleton) with MCP servers")
        _agent_instance = BaseAgent(
            tools=[],
            mcp_servers=MCP_SERVERS,
            system_prompt=SYSTEM_PROMPT,
        )
    return _agent_instance


async def run_query(query: str, session_id: str = "default") -> dict:
    logger.info("run_query called — session='%s', query='%s'", session_id, query[:100])

    # --- Fetch long-term memories from Mem0 ---
    memories = get_memories(user_id=session_id, query=query)

    # Inject the current date so the LLM grounds searches in the right timeframe
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    date_block = (
        f"\n\nTODAY'S DATE: {today}\n"
        "When searching for recent research, include the current year in your queries "
        "to get the latest papers and developments."
    )

    # Dynamically enrich the system prompt with any known user context
    if memories:
        memory_block = "\n".join(f"- {m}" for m in memories)
        enriched_prompt = (
            SYSTEM_PROMPT
            + date_block
            + f"\n\nCONTEXT ABOUT THIS USER (from long-term memory, use this to personalize your response):\n{memory_block}"
        )
        logger.info("Injected %d memories into system_prompt for session='%s'", len(memories), session_id)
    else:
        enriched_prompt = SYSTEM_PROMPT + date_block

    # --- Run the singleton agent (checkpointer persists across calls per session) ---
    agent = create_agent()
    result = await agent.arun(query, session_id=session_id, system_prompt=enriched_prompt)
    logger.info("run_query finished — session='%s', steps: %d", session_id, len(result["steps"]))

    # --- Save this conversation turn back to Mem0 ---
    save_memory(user_id=session_id, query=query, response=result["response"])

    return result
