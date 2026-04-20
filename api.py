import json
import uuid
from typing import Any, Callable, Dict, Iterator, List, Optional
from urllib import error, request

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config import settings
from hermes.service import hermes_service
from mcp.connection import get_budget_summary, get_job_summary


chat_router = APIRouter(tags=["chat"])
chroma_router = APIRouter(prefix="/chroma", tags=["chroma"])
mcp_router = APIRouter(prefix="/mcp", tags=["mcp"])


_chroma_collection = None
_memory_store: List[Dict[str, Any]] = []
_chat_sessions: Dict[str, List[Dict[str, str]]] = {}
_max_session_turns = 20
_stream_headers = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

try:
    import chromadb  # type: ignore

    _chroma_client = chromadb.PersistentClient(path=settings.chroma_path)
    _chroma_collection = _chroma_client.get_or_create_collection(name="chat_memory")
except Exception:
    _chroma_collection = None


class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = None
    think: Optional[bool] = None
    keep_alive: Optional[str] = None
    stream: bool = False
    system_prompt: Optional[str] = None
    session_id: Optional[str] = None
    new_chat: bool = False


class ChromaAddRequest(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None
    id: Optional[str] = None


class ChromaSearchRequest(BaseModel):
    query: str
    n_results: int = Field(default=3, ge=1, le=20)


def _build_session_prompt(history: List[Dict[str, str]], current_message: str) -> str:
    lines: List[str] = []
    for item in history:
        role = item.get("role", "user").upper()
        content = item.get("content", "")
        if content:
            lines.append(f"{role}: {content}")
    lines.append(f"USER: {current_message}")
    lines.append("ASSISTANT:")
    return "\n".join(lines)


def _build_memory_context() -> str:
    rows = hermes_service.list_memory(limit=settings.chat_first_scan_results)
    if not rows:
        return ""

    lines = ["[HERMES_GLOBAL_MEMORY_CONTEXT]"]
    for idx, row in enumerate(rows, start=1):
        title = row.get("title", "").strip()
        content = row.get("content", "").strip()
        if not content:
            continue
        prefix = f"{idx}. {title} - " if title else f"{idx}. "
        lines.append(f"{prefix}{content}")
    if len(lines) == 1:
        return ""
    lines.append("[END_HERMES_GLOBAL_MEMORY_CONTEXT]")
    return "\n".join(lines)


def _memory_candidate(session_id: str, user_message: str, answer: str) -> Dict[str, Any]:
    reason_parts: List[str] = []
    score = 0
    text = f"{user_message}\n{answer}".strip()
    lowered = text.lower()

    if len(text) >= settings.chat_memory_min_len:
        score += 1
        reason_parts.append("length")

    matched_keywords = [kw for kw in settings.chat_memory_keywords if kw and kw.lower() in lowered]
    if matched_keywords:
        score += 1
        reason_parts.append("keywords")

    duplicate = False
    existing = hermes_service.list_memory(limit=50)
    for row in existing:
        existing_content = str(row.get("content", "")).strip().lower()
        if existing_content and (existing_content in lowered or lowered in existing_content):
            duplicate = True
            break
    if duplicate:
        reason_parts.append("duplicate")

    should_add = score >= 2 and not duplicate
    title_base = user_message.strip().replace("\n", " ")
    if len(title_base) > 50:
        title_base = f"{title_base[:50]}..."
    candidate = {
        "should_add": should_add,
        "title": title_base or "chat_memory",
        "content": text,
        "tags": [kw for kw in matched_keywords[:5]],
        "reason": ",".join(reason_parts) or "none",
        "session_id": session_id,
        "source": "chat",
        "user_message": user_message,
    }
    return candidate


def _chat_stream(
    url: str, payload: Dict[str, Any], on_complete: Optional[Callable[[str], Dict[str, Any]]] = None
) -> Iterator[str]:
    req_obj = request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    answer_parts: List[str] = []
    try:
        with request.urlopen(req_obj, timeout=settings.ollama_timeout_seconds) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                piece = data.get("response", "")
                answer_parts.append(piece)
                for ch in piece:
                    yield f"data: {json.dumps({'text': ch}, ensure_ascii=False)}\n\n"
                if data.get("done"):
                    completion_data = None
                    if on_complete:
                        completion_data = on_complete("".join(answer_parts))
                    if completion_data is not None:
                        yield (
                            "event: memory_candidate\n"
                            f"data: {json.dumps(completion_data, ensure_ascii=False)}\n\n"
                        )
                    yield "event: done\ndata: [DONE]\n\n"
                    break
    except Exception as ex:
        yield f"event: error\ndata: {json.dumps({'error': str(ex)}, ensure_ascii=False)}\n\n"


def _client_meta(req: Request) -> Dict[str, str]:
    return {
        "client": req.headers.get("x-dabo-client", "unknown"),
        "build": req.headers.get("x-dabo-build", "unknown"),
        "api_version": req.headers.get("x-api-version", "unknown"),
    }


@chat_router.post("/chat")
def chat(req: ChatRequest, request_info: Request) -> Any:
    session_id = req.session_id or str(uuid.uuid4())
    if req.new_chat:
        _chat_sessions.pop(session_id, None)
    history = _chat_sessions.get(session_id, [])
    prompt = _build_session_prompt(history, req.message)
    is_first_turn = len(history) == 0
    memory_context = _build_memory_context() if is_first_turn else ""
    if memory_context:
        prompt = f"{memory_context}\n\n{prompt}"

    url = f"{settings.ollama_base_url}/api/generate"
    payload = {
        "model": req.model or settings.ollama_model,
        "prompt": prompt,
        "stream": req.stream,
    }
    system_prompt = req.system_prompt or settings.default_system_prompt
    if system_prompt:
        payload["system"] = system_prompt
    if req.think is not None:
        payload["think"] = req.think
    keep_alive = req.keep_alive or settings.ollama_keep_alive
    if keep_alive:
        payload["keep_alive"] = keep_alive
    client = _client_meta(request_info)

    def _save_turn(answer: str) -> Dict[str, Any]:
        turns = _chat_sessions.setdefault(session_id, [])
        turns.append({"role": "user", "content": req.message})
        turns.append({"role": "assistant", "content": answer})
        if len(turns) > _max_session_turns * 2:
            _chat_sessions[session_id] = turns[-(_max_session_turns * 2) :]
        return _memory_candidate(session_id=session_id, user_message=req.message, answer=answer)

    if req.stream:
        return StreamingResponse(
            _chat_stream(url, payload, on_complete=_save_turn),
            media_type="text/event-stream; charset=utf-8",
            headers={
                **_stream_headers,
                "X-DABO-Client": client["client"],
                "X-DABO-Build": client["build"],
                "X-API-Version": client["api_version"],
                "X-Session-Id": session_id,
            },
        )

    try:
        req_obj = request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req_obj, timeout=settings.ollama_timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
        answer = body.get("response", "")
        memory_candidate = _save_turn(answer)
        return {
            "answer": answer,
            "model": payload["model"],
            "client": client,
            "session_id": session_id,
            "memory_candidate": memory_candidate,
        }
    except error.URLError as ex:
        raise HTTPException(status_code=502, detail=f"Ollama connection failed: {ex}") from ex
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Chat error: {ex}") from ex


@chroma_router.post("/add")
def chroma_add(req: ChromaAddRequest) -> Dict[str, Any]:
    doc_id = req.id or str(uuid.uuid4())

    if _chroma_collection is not None:
        _chroma_collection.add(
            ids=[doc_id],
            documents=[req.text],
            metadatas=[req.metadata or {}],
        )
    else:
        _memory_store.append({"id": doc_id, "text": req.text, "metadata": req.metadata or {}})

    return {"status": "ok", "id": doc_id}


@chroma_router.post("/search")
def chroma_search(req: ChromaSearchRequest) -> Dict[str, Any]:
    if _chroma_collection is not None:
        data = _chroma_collection.query(query_texts=[req.query], n_results=req.n_results)
        ids = data.get("ids", [[]])[0]
        docs = data.get("documents", [[]])[0]
        metas = data.get("metadatas", [[]])[0]
        distances = data.get("distances", [[]])[0]

        rows = []
        for idx, doc in enumerate(docs):
            rows.append(
                {
                    "id": ids[idx] if idx < len(ids) else None,
                    "text": doc,
                    "metadata": metas[idx] if idx < len(metas) else {},
                    "distance": distances[idx] if idx < len(distances) else None,
                }
            )
        return {"results": rows}

    query_lc = req.query.lower()
    matches = [item for item in _memory_store if query_lc in item["text"].lower()]
    return {"results": matches[: req.n_results]}


@mcp_router.get("/job")
def mcp_job() -> Dict[str, Any]:
    return get_job_summary()


@mcp_router.get("/budget")
def mcp_budget() -> Dict[str, Any]:
    return get_budget_summary()
