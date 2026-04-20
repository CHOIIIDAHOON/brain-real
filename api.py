import json
import os
import uuid
from datetime import datetime, timezone
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


def _write_chat_log(event: str, payload: Dict[str, Any]) -> None:
    if not settings.chat_log_enabled:
        return
    try:
        path = settings.chat_log_path
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "payload": payload,
        }
        with open(path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        # Logging must never break chat flow.
        return


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


def _extract_json_object(raw_text: str) -> Optional[Dict[str, Any]]:
    text = (raw_text or "").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        return None
    return None


def _is_duplicate_memory(text: str) -> bool:
    lowered = text.strip().lower()
    if not lowered:
        return True
    existing = hermes_service.list_memory(limit=100)
    for row in existing:
        existing_content = str(row.get("content", "")).strip().lower()
        if existing_content and (existing_content in lowered or lowered in existing_content):
            return True
    return False


def _decide_memory_with_ollama(model: str, user_message: str, answer: str) -> Dict[str, Any]:
    decision_prompt = (
        "You are a memory gate for a chat assistant.\n"
        "Decide whether this turn should be saved to long-term global memory.\n"
        "Return ONLY JSON object with keys:\n"
        'action: "add" or "skip"\n'
        "title: short title string\n"
        "content: memory sentence string\n"
        "tags: array of short strings\n"
        "reason: short reason\n"
        "Rules:\n"
        "- add only if stable preference/rule/fact that may help future chats.\n"
        "- skip for small talk, identity of the model, or one-off trivial Q&A.\n"
        "- If action is skip, still return title/content/tags as empty values.\n\n"
        f"USER: {user_message}\n"
        f"ASSISTANT: {answer}\n"
    )
    url = f"{settings.ollama_base_url}/api/generate"
    payload = {
        "model": model,
        "prompt": decision_prompt,
        "stream": False,
        "keep_alive": settings.ollama_keep_alive,
    }
    _write_chat_log(
        "memory_decision_http_request",
        {
            "model": model,
            "url": url,
            "payload": payload,
            "user_message": user_message,
        },
    )
    req_obj = request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req_obj, timeout=settings.ollama_timeout_seconds) as response:
        body = json.loads(response.read().decode("utf-8"))
    text = str(body.get("response", "")).strip()
    parsed = _extract_json_object(text) or {}

    action = str(parsed.get("action", "skip")).strip().lower()
    if action not in {"add", "skip"}:
        action = "skip"
    title = str(parsed.get("title", "")).strip()
    content = str(parsed.get("content", "")).strip()
    tags_raw = parsed.get("tags", [])
    tags: List[str] = []
    if isinstance(tags_raw, list):
        tags = [str(item).strip() for item in tags_raw if str(item).strip()]
    reason = str(parsed.get("reason", "")).strip()
    result = {
        "action": action,
        "title": title,
        "content": content,
        "tags": tags[:5],
        "reason": reason,
    }
    _write_chat_log(
        "memory_decision_http_response",
        {
            "model": model,
            "raw_response": text,
            "parsed": result,
        },
    )
    return result


def _maybe_store_global_memory(session_id: str, user_message: str, answer: str, model: str) -> None:
    if settings.chat_memory_decision_mode.lower() != "ollama":
        _write_chat_log(
            "memory_store_skip",
            {"session_id": session_id, "reason": "decision_mode_off", "mode": settings.chat_memory_decision_mode},
        )
        return
    if not answer.strip():
        _write_chat_log("memory_store_skip", {"session_id": session_id, "reason": "empty_answer"})
        return

    try:
        decision = _decide_memory_with_ollama(model=model, user_message=user_message, answer=answer)
    except Exception as ex:
        _write_chat_log(
            "memory_store_skip",
            {"session_id": session_id, "reason": "decision_error", "error": str(ex)},
        )
        return

    if decision["action"] != "add":
        _write_chat_log(
            "memory_store_skip",
            {"session_id": session_id, "reason": "decision_skip", "decision": decision},
        )
        return

    content = decision["content"] or f"{user_message}\n{answer}"
    if _is_duplicate_memory(content):
        _write_chat_log(
            "memory_store_skip",
            {"session_id": session_id, "reason": "duplicate", "content_preview": content[:200]},
        )
        return

    title = decision["title"] or user_message.strip().replace("\n", " ")[:50] or "chat_memory"
    row = hermes_service.add_memory(
        title=title,
        content=content,
        tags=decision["tags"],
        source="chat_auto",
        session_id=session_id,
        user_message=user_message,
    )
    _write_chat_log(
        "memory_store_add",
        {"session_id": session_id, "decision": decision, "memory_id": row.get("memory_id"), "title": title},
    )


def _normalize_answer_text(answer: str) -> str:
    text = (answer or "").strip()
    if not text:
        return ""

    # Handles cases like "A...A..." where whole answer is duplicated once.
    half = len(text) // 2
    if len(text) % 2 == 0 and half >= 10 and text[:half] == text[half:]:
        return text[:half].strip()

    # Handles line-based duplicated blocks.
    lines = text.splitlines()
    line_half = len(lines) // 2
    if len(lines) % 2 == 0 and line_half >= 1 and lines[:line_half] == lines[line_half:]:
        return "\n".join(lines[:line_half]).strip()

    return text


def _chat_stream(
    url: str,
    payload: Dict[str, Any],
    on_complete: Optional[Callable[[str], None]] = None,
    log_context: Optional[Dict[str, Any]] = None,
) -> Iterator[str]:
    _write_chat_log(
        "chat_http_request",
        {
            **(log_context or {}),
            "url": url,
            "payload": payload,
            "stream": True,
        },
    )
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
                    final_answer = "".join(answer_parts)
                    if on_complete:
                        on_complete(final_answer)
                    _write_chat_log(
                        "chat_http_response",
                        {
                            **(log_context or {}),
                            "stream": True,
                            "answer": _normalize_answer_text(final_answer),
                        },
                    )
                    yield "event: done\ndata: [DONE]\n\n"
                    break
    except Exception as ex:
        _write_chat_log(
            "chat_http_error",
            {**(log_context or {}), "stream": True, "error": str(ex)},
        )
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
    model_name = str(payload["model"])

    def _save_turn(answer: str) -> None:
        normalized = _normalize_answer_text(answer)
        turns = _chat_sessions.setdefault(session_id, [])
        if len(turns) >= 2:
            last_user = turns[-2]
            last_assistant = turns[-1]
            if (
                last_user.get("role") == "user"
                and last_assistant.get("role") == "assistant"
                and last_user.get("content") == req.message
                and last_assistant.get("content") == normalized
            ):
                return

        turns.append({"role": "user", "content": req.message})
        turns.append({"role": "assistant", "content": normalized})
        if len(turns) > _max_session_turns * 2:
            _chat_sessions[session_id] = turns[-(_max_session_turns * 2) :]
        _maybe_store_global_memory(
            session_id=session_id,
            user_message=req.message,
            answer=normalized,
            model=model_name,
        )

    if req.stream:
        return StreamingResponse(
            _chat_stream(
                url,
                payload,
                on_complete=_save_turn,
                log_context={"session_id": session_id, "model": model_name},
            ),
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
        _write_chat_log(
            "chat_http_request",
            {"session_id": session_id, "model": model_name, "url": url, "payload": payload, "stream": False},
        )
        req_obj = request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req_obj, timeout=settings.ollama_timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
        answer = _normalize_answer_text(body.get("response", ""))
        _write_chat_log(
            "chat_http_response",
            {"session_id": session_id, "model": model_name, "stream": False, "answer": answer, "raw_body": body},
        )
        _save_turn(answer)
        return {
            "answer": answer,
            "model": payload["model"],
            "client": client,
            "session_id": session_id,
        }
    except error.URLError as ex:
        _write_chat_log(
            "chat_http_error",
            {"session_id": session_id, "model": model_name, "stream": False, "error": str(ex)},
        )
        raise HTTPException(status_code=502, detail=f"Ollama connection failed: {ex}") from ex
    except Exception as ex:
        _write_chat_log(
            "chat_http_error",
            {"session_id": session_id, "model": model_name, "stream": False, "error": str(ex)},
        )
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
