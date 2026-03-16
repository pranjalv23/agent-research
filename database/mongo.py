import logging
import os
import uuid
from datetime import datetime, timezone

from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

logger = logging.getLogger("agent_research.mongo")

load_dotenv()

_MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
_DB_NAME = os.getenv("MONGO_DB_NAME", "agent_research")


class MongoDB:
    _client: AsyncIOMotorClient | None = None

    @classmethod
    def get_client(cls) -> AsyncIOMotorClient:
        if cls._client is None:
            logger.info("Connecting to MongoDB at %s", _MONGO_URI)
            cls._client = AsyncIOMotorClient(_MONGO_URI)
        return cls._client

    @classmethod
    def _collection(cls):
        return cls.get_client()[_DB_NAME]["conversations"]

    @classmethod
    def generate_session_id(cls) -> str:
        return uuid.uuid4().hex

    @classmethod
    async def save_conversation(
        cls,
        session_id: str,
        query: str,
        response: str,
    ) -> str:
        doc = {
            "session_id": session_id,
            "query": query,
            "response": response,
            "created_at": datetime.now(timezone.utc),
        }
        result = await cls._collection().insert_one(doc)
        logger.info("Saved conversation — session='%s', doc_id='%s'", session_id, result.inserted_id)
        return str(result.inserted_id)

    @classmethod
    async def get_history(cls, session_id: str) -> list[dict]:
        cursor = cls._collection().find(
            {"session_id": session_id},
            {"_id": 0, "query": 1, "response": 1, "created_at": 1},
        ).sort("created_at", 1)
        return await cursor.to_list(length=100)

    @classmethod
    async def close(cls):
        if cls._client:
            cls._client.close()
            cls._client = None
