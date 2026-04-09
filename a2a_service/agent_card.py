import os

from a2a.types import AgentCard, AgentSkill, AgentCapabilities


RESEARCH_AGENT_CARD = AgentCard(
    name="Research Paper Agent",
    description=(
        "Autonomous research assistant that finds, downloads, and summarizes "
        "academic papers from arXiv. Specializes in AI, ML, computer science, "
        "and scientific research literature."
    ),
    url=os.getenv("AGENT_PUBLIC_URL", "http://localhost:9002"),
    version="1.0.0",
    metadata={"mode": "researcher"},
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
    defaultInputModes=["text"],
    defaultOutputModes=["text"],
    capabilities=AgentCapabilities(streaming=False, pushNotifications=False),
)
