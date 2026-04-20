import json
import os
import threading
import uuid
from datetime import datetime, timezone
from time import perf_counter
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
_memory_decision_marker = "<<<MEMORY_DECISION>>>"

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


def _normalize_keep_alive_for_log(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.endswith("h") and text[:-1].isdigit():
        return text[:-1]
    return text


def _log_prompt_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "prompt": str(payload.get("prompt", "")),
        "stream": bool(payload.get("stream", False)),
        "system": str(payload.get("system", "")),
        "keep_alive": _normalize_keep_alive_for_log(payload.get("keep_alive")),
    }


def _log_chat_flow(session_id: str, flow_id: str, step: str, data: Optional[Dict[str, Any]] = None) -> None:
    _write_chat_log(
        "chat_flow",
        {
            "session_id": session_id,
            "flow_id": flow_id,
            "step": step,
            "data": data or {},
        },
    )


def _fetch_hermes_memory(
    limit: int, session_id: Optional[str] = None, flow_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    _write_chat_log(
        "hermes_memory_query_request",
        {"limit": limit, "session_id": session_id, "flow_id": flow_id},
    )
    rows = hermes_service.list_memory(limit=limit)
    _write_chat_log(
        "hermes_memory_query_response",
        {"limit": limit, "count": len(rows), "session_id": session_id, "flow_id": flow_id},
    )
    return rows


def _save_hermes_memory(
    title: str,
    content: str,
    tags: List[str],
    source: str,
    session_id: str,
    user_message: str,
    flow_id: Optional[str] = None,
) -> Dict[str, Any]:
    _write_chat_log(
        "hermes_memory_add_request",
        {
            "title": title,
            "content_preview": content[:200],
            "tags": tags,
            "source": source,
            "session_id": session_id,
            "flow_id": flow_id,
        },
    )
    row = hermes_service.add_memory(
        title=title,
        content=content,
        tags=tags,
        source=source,
        session_id=session_id,
        user_message=user_message,
    )
    _write_chat_log(
        "hermes_memory_add_response",
        {
            "memory_id": row.get("memory_id"),
            "title": row.get("title"),
            "session_id": session_id,
            "flow_id": flow_id,
        },
    )
    return row


def _build_memory_context(session_id: Optional[str] = None, flow_id: Optional[str] = None) -> str:
    rows = _fetch_hermes_memory(limit=settings.chat_first_scan_results, session_id=session_id, flow_id=flow_id)
    if not rows:
        _write_chat_log(
            "hermes_context_built",
            {"session_id": session_id, "flow_id": flow_id, "applied": False, "count": 0},
        )
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
        _write_chat_log(
            "hermes_context_built",
            {"session_id": session_id, "flow_id": flow_id, "applied": False, "count": 0},
        )
        return ""
    lines.append("[END_HERMES_GLOBAL_MEMORY_CONTEXT]")
    _write_chat_log(
        "hermes_context_built",
        {"session_id": session_id, "flow_id": flow_id, "applied": True, "count": max(len(lines) - 2, 0)},
    )
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


def _build_memory_decision_instruction() -> str:
    return (
        "OUTPUT FORMAT RULE:\n"
        "1) First, write your normal assistant answer.\n"
        f"2) Then append a new line with exactly: {_memory_decision_marker}\n"
        '3) Then append a single-line JSON object: {"action":"add|skip","title":"","content":"","tags":[],"reason":""}\n'
        "4) add only for stable user profile/preference/rule/fact useful later.\n"
        "5) skip for greeting/small-talk/one-off.\n"
    )


def _normalize_memory_decision(parsed: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    data = parsed or {}
    action = str(data.get("action", "skip")).strip().lower()
    if action not in {"add", "skip"}:
        action = "skip"
    title = str(data.get("title", "")).strip()
    content = str(data.get("content", "")).strip()
    reason = str(data.get("reason", "")).strip()
    tags_raw = data.get("tags", [])
    tags: List[str] = []
    if isinstance(tags_raw, list):
        tags = [str(item).strip() for item in tags_raw if str(item).strip()]
    return {
        "action": action,
        "title": title,
        "content": content,
        "tags": tags[:5],
        "reason": reason,
    }


def _extract_answer_and_memory_decision(raw_text: str) -> tuple[str, Dict[str, Any], str, bool]:
    text = str(raw_text or "")
    marker_idx = text.find(_memory_decision_marker)
    if marker_idx < 0:
        answer = _normalize_answer_text(text)
        return (
            answer,
            {"action": "skip", "title": "", "content": "", "tags": [], "reason": "marker_missing"},
            "",
            False,
        )

    answer = _normalize_answer_text(text[:marker_idx])
    decision_raw = text[marker_idx + len(_memory_decision_marker) :].strip()
    decision = _normalize_memory_decision(_extract_json_object(decision_raw))
    if not decision["reason"] and decision["action"] == "skip":
        decision["reason"] = "model_skip"
    return (answer, decision, decision_raw, True)


def _apply_memory_decision(
    session_id: str,
    flow_id: Optional[str],
    user_message: str,
    answer: str,
    decision: Dict[str, Any],
) -> None:
    _log_chat_flow(
        session_id,
        flow_id or "n/a",
        "memory_decision_completed",
        {
            "action": decision.get("action", "skip"),
            "reason": decision.get("reason", ""),
            "title": decision.get("title", ""),
        },
    )
    _write_chat_log(
        "memory_decision_result",
        {
            "session_id": session_id,
            "flow_id": flow_id,
            "action": decision.get("action", "skip"),
            "reason": decision.get("reason", ""),
            "title": decision.get("title", ""),
            "tags": decision.get("tags", []),
            "user_message": user_message,
            "assistant_answer": answer,
        },
    )
    if decision.get("action") != "add":
        _write_chat_log(
            "memory_store_skip",
            {"session_id": session_id, "flow_id": flow_id, "reason": "decision_skip", "decision": decision},
        )
        return

    content = str(decision.get("content", "")).strip() or f"{user_message}\n{answer}"
    if _is_duplicate_memory(content, session_id=session_id, flow_id=flow_id):
        _write_chat_log(
            "memory_store_skip",
            {"session_id": session_id, "flow_id": flow_id, "reason": "duplicate", "content_preview": content[:200]},
        )
        return

    title = str(decision.get("title", "")).strip() or user_message.strip().replace("\n", " ")[:50] or "chat_memory"
    _log_chat_flow(session_id, flow_id or "n/a", "memory_save_started", {"title": title})
    row = _save_hermes_memory(
        title=title,
        content=content,
        tags=decision.get("tags", []),
        source="chat_auto",
        session_id=session_id,
        user_message=user_message,
        flow_id=flow_id,
    )
    _write_chat_log(
        "memory_store_add",
        {
            "session_id": session_id,
            "flow_id": flow_id,
            "decision": decision,
            "memory_id": row.get("memory_id"),
            "title": title,
        },
    )
    _log_chat_flow(session_id, flow_id or "n/a", "memory_save_completed", {"memory_id": row.get("memory_id")})


def _is_duplicate_memory(text: str, session_id: Optional[str] = None, flow_id: Optional[str] = None) -> bool:
    lowered = text.strip().lower()
    if not lowered:
        return True
    existing = _fetch_hermes_memory(limit=100, session_id=session_id, flow_id=flow_id)
    for row in existing:
        existing_content = str(row.get("content", "")).strip().lower()
        if existing_content and (existing_content in lowered or lowered in existing_content):
            return True
    return False


def _trim_for_memory_decision(text: str, max_chars: int) -> str:
    normalized = (text or "").strip()
    if max_chars <= 0 or len(normalized) <= max_chars:
        return normalized
    return f"{normalized[:max_chars].rstrip()}..."


def _should_skip_memory_decision(user_message: str, answer: str) -> Optional[str]:
    user_trimmed = (user_message or "").strip()
    answer_trimmed = (answer or "").strip()
    if not user_trimmed or not answer_trimmed:
        return "empty_text"

    # Avoid costly classifier calls for short, one-off guess/small-talk questions.
    if (
        "?" in user_trimmed
        and len(user_trimmed) <= settings.chat_memory_skip_short_question_len
        and len(answer_trimmed) <= settings.chat_memory_decision_max_chars
    ):
        return "short_question"
    return None


def _decide_memory_with_ollama(
    model: str,
    user_message: str,
    answer: str,
    session_id: Optional[str] = None,
    flow_id: Optional[str] = None,
) -> Dict[str, Any]:
    max_chars = settings.chat_memory_decision_max_chars
    user_message_trimmed = _trim_for_memory_decision(user_message, max_chars)
    answer_trimmed = _trim_for_memory_decision(answer, max_chars)
    decision_prompt = (
        "Return ONLY this JSON:\n"
        '{"action":"add|skip","title":"","content":"","tags":[],"reason":""}\n'
        "Rule:\n"
        "- add: stable user profile/preference/rule/fact useful later\n"
        "- skip: greeting/small talk/one-off\n\n"
        f"USER: {user_message_trimmed}\n"
        f"ASSISTANT: {answer_trimmed}\n"
    )
    url = f"{settings.ollama_base_url}/api/generate"
    payload = {
        "model": model,
        "prompt": decision_prompt,
        "stream": False,
        "keep_alive": settings.ollama_keep_alive,
        "options": {
            "temperature": 0,
            "num_predict": settings.chat_memory_decision_num_predict,
        },
    }
    _write_chat_log(
        "memory_decision_http_request",
        {
            "model": model,
            "url": url,
            "payload": payload,
            "user_message": user_message,
            "session_id": session_id,
            "flow_id": flow_id,
        },
    )
    req_obj = request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started_at = perf_counter()
    with request.urlopen(req_obj, timeout=settings.chat_memory_decision_timeout_seconds) as response:
        body = json.loads(response.read().decode("utf-8"))
    latency_ms = int((perf_counter() - started_at) * 1000)
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
            "latency_ms": latency_ms,
            "truncated": {
                "user_message": user_message_trimmed != user_message.strip(),
                "assistant_answer": answer_trimmed != answer.strip(),
                "max_chars": max_chars,
            },
            "session_id": session_id,
            "flow_id": flow_id,
        },
    )
    return result


def _maybe_store_global_memory(
    session_id: str,
    user_message: str,
    answer: str,
    model: str,
    flow_id: Optional[str] = None,
) -> None:
    if settings.chat_memory_decision_mode.lower() != "ollama":
        _write_chat_log(
            "memory_store_skip",
            {
                "session_id": session_id,
                "flow_id": flow_id,
                "reason": "decision_mode_off",
                "mode": settings.chat_memory_decision_mode,
            },
        )
        return
    if not answer.strip():
        _write_chat_log(
            "memory_store_skip",
            {"session_id": session_id, "flow_id": flow_id, "reason": "empty_answer"},
        )
        return
    skip_reason = _should_skip_memory_decision(user_message=user_message, answer=answer)
    if skip_reason:
        _write_chat_log(
            "memory_store_skip",
            {"session_id": session_id, "flow_id": flow_id, "reason": skip_reason},
        )
        return

    decision_model = settings.chat_memory_decision_model or model
    try:
        _log_chat_flow(
            session_id,
            flow_id or "n/a",
            "memory_decision_started",
            {"model": decision_model},
        )
        decision = _decide_memory_with_ollama(
            model=decision_model,
            user_message=user_message,
            answer=answer,
            session_id=session_id,
            flow_id=flow_id,
        )
    except Exception as ex:
        _write_chat_log(
            "memory_store_skip",
            {"session_id": session_id, "flow_id": flow_id, "reason": "decision_error", "error": str(ex)},
        )
        return
    _log_chat_flow(
        session_id,
        flow_id or "n/a",
        "memory_decision_completed",
        {
            "action": decision.get("action", "skip"),
            "reason": decision.get("reason", ""),
            "title": decision.get("title", ""),
        },
    )
    _write_chat_log(
        "memory_decision_result",
        {
            "session_id": session_id,
            "flow_id": flow_id,
            "action": decision.get("action", "skip"),
            "reason": decision.get("reason", ""),
            "title": decision.get("title", ""),
            "tags": decision.get("tags", []),
            "user_message": user_message,
            "assistant_answer": answer,
        },
    )

    if decision["action"] != "add":
        _write_chat_log(
            "memory_store_skip",
            {"session_id": session_id, "flow_id": flow_id, "reason": "decision_skip", "decision": decision},
        )
        return

    content = decision["content"] or f"{user_message}\n{answer}"
    if _is_duplicate_memory(content, session_id=session_id, flow_id=flow_id):
        _write_chat_log(
            "memory_store_skip",
            {"session_id": session_id, "flow_id": flow_id, "reason": "duplicate", "content_preview": content[:200]},
        )
        return

    title = decision["title"] or user_message.strip().replace("\n", " ")[:50] or "chat_memory"
    _log_chat_flow(
        session_id,
        flow_id or "n/a",
        "memory_save_started",
        {"title": title},
    )
    row = _save_hermes_memory(
        title=title,
        content=content,
        tags=decision["tags"],
        source="chat_auto",
        session_id=session_id,
        user_message=user_message,
        flow_id=flow_id,
    )
    _write_chat_log(
        "memory_store_add",
        {
            "session_id": session_id,
            "flow_id": flow_id,
            "decision": decision,
            "memory_id": row.get("memory_id"),
            "title": title,
        },
    )
    _log_chat_flow(
        session_id,
        flow_id or "n/a",
        "memory_save_completed",
        {"memory_id": row.get("memory_id")},
    )


def _schedule_memory_store(
    session_id: str,
    user_message: str,
    answer: str,
    model: str,
    flow_id: Optional[str] = None,
) -> None:
    _write_chat_log(
        "memory_store_scheduled",
        {"session_id": session_id, "flow_id": flow_id, "model": model, "user_message": user_message},
    )
    worker = threading.Thread(
        target=_maybe_store_global_memory,
        kwargs={
            "session_id": session_id,
            "user_message": user_message,
            "answer": answer,
            "model": model,
            "flow_id": flow_id,
        },
        daemon=True,
    )
    worker.start()


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
    on_complete: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    log_context: Optional[Dict[str, Any]] = None,
) -> Iterator[str]:
    flow_id = str((log_context or {}).get("flow_id", "n/a"))
    session_id = str((log_context or {}).get("session_id", "n/a"))
    _log_chat_flow(session_id, flow_id, "llm_request_started", {"stream": True})
    _write_chat_log(
        "chat_http_request",
        {
            **(log_context or {}),
            "url": url,
            **_log_prompt_fields(payload),
        },
    )
    req_obj = request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    response_text = ""
    emitted_upto = 0
    done_sent = False
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
                response_text += piece
                marker_idx = response_text.find(_memory_decision_marker)
                if marker_idx >= 0:
                    flush_upto = marker_idx
                else:
                    flush_upto = max(0, len(response_text) - (len(_memory_decision_marker) - 1))
                if flush_upto > emitted_upto:
                    safe_chunk = response_text[emitted_upto:flush_upto]
                    for ch in safe_chunk:
                        yield f"data: {json.dumps({'text': ch}, ensure_ascii=False)}\n\n"
                    emitted_upto = flush_upto
                if data.get("done"):
                    final_answer, decision, _, marker_found = _extract_answer_and_memory_decision(response_text)
                    if len(final_answer) > emitted_upto:
                        tail_chunk = final_answer[emitted_upto:]
                        for ch in tail_chunk:
                            yield f"data: {json.dumps({'text': ch}, ensure_ascii=False)}\n\n"
                        emitted_upto = len(final_answer)
                    if on_complete:
                        on_complete(final_answer, decision)
                    _write_chat_log(
                        "chat_http_response",
                        {
                            **(log_context or {}),
                            "stream": True,
                            "answer": final_answer,
                            "decision": decision,
                            "decision_marker_found": marker_found,
                        },
                    )
                    _log_chat_flow(session_id, flow_id, "llm_response_completed", {"stream": True})
                    yield "event: done\ndata: [DONE]\n\n"
                    done_sent = True
                    break
            if not done_sent:
                final_answer, decision, _, marker_found = _extract_answer_and_memory_decision(response_text)
                if len(final_answer) > emitted_upto:
                    tail_chunk = final_answer[emitted_upto:]
                    for ch in tail_chunk:
                        yield f"data: {json.dumps({'text': ch}, ensure_ascii=False)}\n\n"
                    emitted_upto = len(final_answer)
                if on_complete:
                    on_complete(final_answer, decision)
                _write_chat_log(
                    "chat_stream_done_fallback",
                    {
                        **(log_context or {}),
                        "reason": "upstream_closed_without_done",
                        "answer": final_answer,
                        "decision": decision,
                        "decision_marker_found": marker_found,
                    },
                )
                _log_chat_flow(
                    session_id,
                    flow_id,
                    "llm_response_completed",
                    {"stream": True, "fallback": "upstream_closed_without_done"},
                )
                yield "event: done\ndata: [DONE]\n\n"
                done_sent = True
    except Exception as ex:
        _write_chat_log(
            "chat_http_error",
            {**(log_context or {}), "stream": True, "error": str(ex)},
        )
        _log_chat_flow(session_id, flow_id, "llm_response_error", {"stream": True, "error": str(ex)})
        yield f"event: error\ndata: {json.dumps({'error': str(ex)}, ensure_ascii=False)}\n\n"
    finally:
        if not done_sent:
            _write_chat_log(
                "chat_stream_done_fallback",
                {**(log_context or {}), "reason": "finally_guard"},
            )
            _log_chat_flow(
                session_id,
                flow_id,
                "llm_response_completed",
                {"stream": True, "fallback": "finally_guard"},
            )
            yield "event: done\ndata: [DONE]\n\n"


def _client_meta(req: Request) -> Dict[str, str]:
    return {
        "client": req.headers.get("x-dabo-client", "unknown"),
        "build": req.headers.get("x-dabo-build", "unknown"),
        "api_version": req.headers.get("x-api-version", "unknown"),
    }


@chat_router.post("/chat")
def chat(req: ChatRequest, request_info: Request) -> Any:
    session_id = req.session_id or str(uuid.uuid4())
    flow_id = uuid.uuid4().hex[:12]
    _log_chat_flow(
        session_id,
        flow_id,
        "request_received",
        {"new_chat": req.new_chat, "stream": req.stream},
    )
    if req.new_chat:
        _chat_sessions.pop(session_id, None)
    history = _chat_sessions.get(session_id, [])
    _log_chat_flow(
        session_id,
        flow_id,
        "session_context_loaded",
        {"history_turns": len(history) // 2},
    )
    prompt = _build_session_prompt(history, req.message)
    _log_chat_flow(session_id, flow_id, "hermes_memory_lookup_started")
    memory_context = _build_memory_context(session_id=session_id, flow_id=flow_id)
    if memory_context:
        prompt = f"{memory_context}\n\n{prompt}"
    prompt = f"{prompt}\n\n{_build_memory_decision_instruction()}"
    _log_chat_flow(
        session_id,
        flow_id,
        "hermes_memory_lookup_completed",
        {"memory_context_applied": bool(memory_context)},
    )
    _write_chat_log(
        "chat_prompt_ready",
        {
            "session_id": session_id,
            "flow_id": flow_id,
            "history_turns": len(history) // 2,
            "memory_context_applied": bool(memory_context),
        },
    )

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

    def _save_turn(answer: str, decision: Dict[str, Any]) -> None:
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
        _apply_memory_decision(
            session_id=session_id,
            user_message=req.message,
            answer=normalized,
            decision=decision,
            flow_id=flow_id,
        )

    if req.stream:
        return StreamingResponse(
            _chat_stream(
                url,
                payload,
                on_complete=_save_turn,
                log_context={"session_id": session_id, "model": model_name, "flow_id": flow_id},
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
        _log_chat_flow(session_id, flow_id, "llm_request_started", {"stream": False})
        _write_chat_log(
            "chat_http_request",
            {
                "session_id": session_id,
                "flow_id": flow_id,
                "model": model_name,
                "url": url,
                **_log_prompt_fields(payload),
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
        raw_response = str(body.get("response", ""))
        answer, decision, _, marker_found = _extract_answer_and_memory_decision(raw_response)
        _write_chat_log(
            "chat_http_response",
            {
                "session_id": session_id,
                "flow_id": flow_id,
                "model": model_name,
                "stream": False,
                "answer": answer,
                "decision": decision,
                "decision_marker_found": marker_found,
                "raw_body": body,
            },
        )
        _log_chat_flow(session_id, flow_id, "llm_response_completed", {"stream": False})
        _save_turn(answer, decision)
        _log_chat_flow(session_id, flow_id, "request_completed", {"status": "ok"})
        return {
            "answer": answer,
            "model": payload["model"],
            "client": client,
            "session_id": session_id,
        }
    except error.URLError as ex:
        _write_chat_log(
            "chat_http_error",
            {
                "session_id": session_id,
                "flow_id": flow_id,
                "model": model_name,
                "stream": False,
                "error": str(ex),
            },
        )
        _log_chat_flow(session_id, flow_id, "request_completed", {"status": "error", "error": str(ex)})
        raise HTTPException(status_code=502, detail=f"Ollama connection failed: {ex}") from ex
    except Exception as ex:
        _write_chat_log(
            "chat_http_error",
            {
                "session_id": session_id,
                "flow_id": flow_id,
                "model": model_name,
                "stream": False,
                "error": str(ex),
            },
        )
        _log_chat_flow(session_id, flow_id, "request_completed", {"status": "error", "error": str(ex)})
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
