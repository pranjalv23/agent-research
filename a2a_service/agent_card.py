import os

from a2a.types import AgentCard, AgentCapabilities, AgentInterface, AgentSkill


RESEARCH_AGENT_CARD = AgentCard(
    name="Research Paper Agent",
    description=(
        "Autonomous research assistant that finds, downloads, and summarizes "
        "academic papers from arXiv. Specializes in AI, ML, computer science, "
        "and scientific research literature."
    ),
    supported_interfaces=[
        AgentInterface(
            url=os.getenv("AGENT_PUBLIC_URL", "http://localhost:9002"),
            protocol_binding="JSONRPC",
        )
    ],
    version="1.0.0",
    skills=[
        AgentSkill(
            id="paper-search",
            name="Paper Search",
            description="Search and download academic papers from arXiv, store in vector DB for retrieval.",
            tags=["research", "arxiv", "papers", "academic", "science"],
        ),
        AgentSkill(
            id="literature-review",
            name="Literature Review",
            description="Synthesize findings from multiple papers into structured literature reviews.",
            tags=["research", "AI", "ML", "review", "synthesis"],
        ),
    ],
    default_input_modes=["text"],
    default_output_modes=["text"],
    capabilities=AgentCapabilities(streaming=True, push_notifications=False),
)
