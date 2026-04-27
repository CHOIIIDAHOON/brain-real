"""
Ollama 공통 옵션(스레드 제한)과 임베딩 /api/embeddings 호출.
"""

import json
from typing import Any, Dict, List, Optional
from urllib import error, request

from config import settings

OLLAMA_NUM_THREAD = 3


def ollama_thread_options(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    out = {"num_thread": OLLAMA_NUM_THREAD}
    if overrides:
        out.update(overrides)
    return out


def ollama_embeddings(prompt: str, model: str) -> List[float]:
    """Ollama /api/embeddings — 본문 임베딩 벡터를 반환한다."""
    url = f"{settings.ollama_base_url}/api/embeddings"
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "options": ollama_thread_options(),
    }
    http_request = request.Request(
        url=url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(http_request, timeout=settings.ollama_timeout_seconds) as response:
        data = json.loads(response.read().decode("utf-8"))
    vector = data.get("embedding")
    if not isinstance(vector, list) or not vector:
        raise ValueError("Ollama embeddings: missing or empty 'embedding'")
    return [float(x) for x in vector]
