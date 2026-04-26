"""채팅용 인메모리 메모리 스토어(재시작 시 초기화)."""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

MemRow = Dict[str, Any]

_memories: List[MemRow] = []


def list_memories(*, limit: int = 50) -> List[MemRow]:
    if limit <= 0:
        return []
    return list(reversed(_memories[-limit:]))


def add_memory(
    *,
    title: str,
    content: str,
    tags: List[str],
    source: str = "manual",
    session_id: Optional[str] = None,
    user_message: Optional[str] = None,
) -> MemRow:
    row: MemRow = {
        "memory_id": str(uuid.uuid4()),
        "title": title,
        "content": content,
        "tags": tags,
        "source": source,
        "session_id": session_id,
        "user_message": user_message,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _memories.append(row)
    return row


def clear_all_memories() -> None:
    _memories.clear()
