"""
채팅 처리 핵심 서비스.
CHAT_BACKEND=ollama(기본): Ollama /api/generate 직접.
CHAT_BACKEND=hermes(선택): Nous Hermes Agent(AIAgent) — hermes-agent 패키지·Python 3.11+ 필요.
"""

import json
import os
import re
import threading
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from queue import Queue
from time import perf_counter
from urllib import error, request
from urllib.parse import urlparse

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from config import settings
from hermes.service import hermes_service

chat_sessions = {}
# Hermes multi-turn: AIAgent run_conversation이 기대하는 messages 누적
hermes_session_messages: dict = {}
hermes_locks: defaultdict = defaultdict(threading.Lock)
max_session_turns = 20
stream_headers = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}
korean_timezone = timezone(timedelta(hours=9))
memory_task_queue = Queue()
memory_worker_started = False
memory_worker_lock = threading.Lock()
memory_schedule_lock = threading.Lock()
session_last_memory_schedule_time = {}


# 채팅 세션 히스토리로 최종 프롬프트를 만든다.
def build_session_prompt(history, current_message):
    prompt_max_turns = settings.chat_prompt_max_turns
    if prompt_max_turns > 0 and len(history) > prompt_max_turns * 2:
        history = history[-(prompt_max_turns * 2) :]

    prompt_lines = []
    for history_item in history:
        role = history_item.get("role", "user").upper()
        content = history_item.get("content", "")
        if content:
            prompt_lines.append(f"{role}: {content}")

    prompt_lines.append(f"USER: {current_message}")
    prompt_lines.append("ASSISTANT:")
    return "\n".join(prompt_lines)


# 채팅 진단 로그를 파일에 남긴다. event는 사람이 읽기 쉬운 한글 문자열을 쓴다.
def write_chat_log(event, payload):
    if not settings.chat_log_enabled:
        return
    try:
        log_file_path = settings.chat_log_path
        log_directory = os.path.dirname(log_file_path)
        if log_directory:
            os.makedirs(log_directory, exist_ok=True)
        log_time = datetime.now(korean_timezone).strftime("%Y.%m.%d %H:%M:%S")
        row = {"event": event, "payload": payload}
        with open(log_file_path, "a", encoding="utf-8") as file_pointer:
            file_pointer.write(f"[{log_time}] {json.dumps(row, ensure_ascii=False)}\n\n")
    except Exception:
        # 로그 실패가 채팅 기능을 막지 않도록 무시한다.
        return


# keep_alive 로그 표기를 사람이 읽기 쉽게 맞춘다.
def normalize_keep_alive_for_log(value):
    text_value = str(value or "").strip()
    if not text_value:
        return ""
    if text_value.endswith("h") and text_value[:-1].isdigit():
        return text_value[:-1]
    return text_value


# Ollama 요청 로그에 필요한 필드만 추려서 만든다.
def build_log_prompt_fields(payload):
    return {
        "prompt": str(payload.get("prompt", "")),
        "stream": bool(payload.get("stream", False)),
        "system": str(payload.get("system", "")),
        "keep_alive": normalize_keep_alive_for_log(payload.get("keep_alive")),
    }


def ollama_request_error_fields(exception):
    """Ollama/urllib 요청 실패를 로그에 남길 필드로 정리한다. HTTP 4xx/5xx일 때 응답 본문을 읽는다."""
    fields = {"error": str(exception)}
    if isinstance(exception, error.HTTPError):
        fields["http_status"] = exception.code
        try:
            raw = exception.read().decode("utf-8", errors="replace")
            if raw:
                fields["error_body"] = raw[:8000]
        except Exception:
            pass
    return fields


def build_ollama_request_diagnostics(payload, url):
    """Ollama /api/generate 요청 부하·용량 이슈 추적용 (CPU 스파이크·메모리와 함께 상관 분석)."""
    prompt = str(payload.get("prompt", "") or "")
    system = str(payload.get("system", "") or "")
    parsed = urlparse(url)
    try:
        body_bytes = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    except Exception:
        body_bytes = None
    return {
        "ollama_url_host": parsed.netloc or "",
        "ollama_url_path": parsed.path or "",
        "request_json_bytes": body_bytes,
        "prompt_chars": len(prompt),
        "system_chars": len(system),
        "model": str(payload.get("model", "") or ""),
        "stream": bool(payload.get("stream", False)),
        "keep_alive": normalize_keep_alive_for_log(payload.get("keep_alive")),
        "think": payload.get("think"),
        "options": payload.get("options"),
    }


def ollama_packet_stats(packet):
    """Ollama /api/generate JSON 한 줄(완료 패킷 등)의 eval/load 타이밍·토큰 수를 로그용으로 정리한다."""
    if not isinstance(packet, dict):
        return {}
    out = {}
    for key in ("model", "done_reason", "context", "prompt_eval_count", "eval_count"):
        if key in packet and packet[key] is not None:
            out[key] = packet[key]
    for key in ("total_duration", "load_duration", "eval_duration", "prompt_eval_duration"):
        value = packet.get(key)
        if value is not None:
            out[key + "_ms"] = int(value / 1_000_000)
    return out


def summarize_inter_chunk_gaps_ms(gaps):
    if not gaps:
        return {}
    ordered = sorted(gaps)
    amount = len(ordered)
    p95_i = min(max(0, int(amount * 0.95) - 1), amount - 1)
    return {
        "inter_chunk_count": amount,
        "inter_chunk_ms_min": round(min(gaps), 2),
        "inter_chunk_ms_max": round(max(gaps), 2),
        "inter_chunk_ms_avg": round(sum(gaps) / amount, 2),
        "inter_chunk_ms_p95": round(ordered[p95_i], 2),
    }


def write_ollama_stream_diagnostics(
    log_context,
    chunk_gaps_ms,
    chunk_count,
    first_token_latency_ms,
    done_received_latency_ms,
    total_latency_ms,
    ollama_metrics,
    extra=None,
):
    payload = {**(log_context or {}), "stream": True}
    payload.update(summarize_inter_chunk_gaps_ms(chunk_gaps_ms))
    if first_token_latency_ms is not None and done_received_latency_ms is not None:
        decode_ms = done_received_latency_ms - first_token_latency_ms
        payload["decode_window_ms"] = decode_ms
        if decode_ms > 0 and chunk_count:
            payload["approx_chunk_events_per_sec"] = round(chunk_count / (decode_ms / 1000.0), 2)
    if total_latency_ms is not None:
        payload["client_total_ms"] = total_latency_ms
    if ollama_metrics:
        payload["ollama_metrics"] = ollama_metrics
    if extra:
        payload["extra"] = extra
    write_chat_log("chat_ollama_stream_diagnostics", payload)


# 채팅 플로우 단계 로그를 통일된 형식으로 기록한다. step_name은 event로 그대로 쓰이므로 한글로 넘긴다.
def log_chat_flow(session_id, flow_id, step_name, data=None):
    write_chat_log(
        step_name,
        {
            "session_id": session_id,
            "flow_id": flow_id,
            "data": data or {},
        },
    )


# 백그라운드 메모리 저장 워커를 실행한다.
def memory_worker_loop():
    while True:
        task = memory_task_queue.get()
        try:
            maybe_store_global_memory(
                session_id=str(task.get("session_id", "")),
                user_message=str(task.get("user_message", "")),
                answer=str(task.get("answer", "")),
                model=str(task.get("model", "")),
                flow_id=str(task.get("flow_id", "")),
            )
        except Exception as exception:
            write_chat_log("메모리 워커 오류", {"error": str(exception), "task": task})
        finally:
            memory_task_queue.task_done()


# 메모리 저장 워커를 1회만 시작한다.
def ensure_memory_worker():
    global memory_worker_started
    if memory_worker_started:
        return
    with memory_worker_lock:
        if memory_worker_started:
            return
        worker_thread = threading.Thread(target=memory_worker_loop, daemon=True)
        worker_thread.start()
        memory_worker_started = True
        write_chat_log("메모리 워커 기동", {})


# Hermes 글로벌 메모리를 조회한다.
def fetch_hermes_memory(limit, session_id=None, flow_id=None):
    write_chat_log(
        "hermes 조회 요청",
        {"limit": limit, "session_id": session_id, "flow_id": flow_id},
    )
    memory_rows = hermes_service.list_memory(limit=limit)
    write_chat_log(
        "hermes 조회 응답",
        {"limit": limit, "count": len(memory_rows), "session_id": session_id, "flow_id": flow_id},
    )
    return memory_rows


# Hermes 글로벌 메모리에 새 항목을 저장한다.
def save_hermes_memory(title, content, tags, source, session_id, user_message, flow_id=None):
    write_chat_log(
        "hermes 저장 요청",
        {
            "title": title,
            "content_preview": content[:200],
            "tags": tags,
            "source": source,
            "session_id": session_id,
            "flow_id": flow_id,
        },
    )
    memory_row = hermes_service.add_memory(
        title=title,
        content=content,
        tags=tags,
        source=source,
        session_id=session_id,
        user_message=user_message,
    )
    write_chat_log(
        "hermes 저장 응답",
        {
            "memory_id": memory_row.get("memory_id"),
            "title": memory_row.get("title"),
            "session_id": session_id,
            "flow_id": flow_id,
        },
    )
    return memory_row


# 프롬프트에 붙일 Hermes 메모리 컨텍스트 문자열을 만든다.
def build_memory_context(session_id=None, flow_id=None):
    memory_rows = fetch_hermes_memory(
        limit=settings.chat_first_scan_results,
        session_id=session_id,
        flow_id=flow_id,
    )
    if not memory_rows:
        write_chat_log(
            "프롬프트 메모리 문맥",
            {"session_id": session_id, "flow_id": flow_id, "applied": False, "count": 0},
        )
        return ""

    context_lines = ["[HERMES_GLOBAL_MEMORY_CONTEXT]"]
    for row_index, memory_row in enumerate(memory_rows, start=1):
        title = memory_row.get("title", "").strip()
        content = memory_row.get("content", "").strip()
        if not content:
            continue
        prefix = f"{row_index}. {title} - " if title else f"{row_index}. "
        context_lines.append(f"{prefix}{content}")

    if len(context_lines) == 1:
        write_chat_log(
            "프롬프트 메모리 문맥",
            {"session_id": session_id, "flow_id": flow_id, "applied": False, "count": 0},
        )
        return ""

    context_lines.append("[END_HERMES_GLOBAL_MEMORY_CONTEXT]")
    write_chat_log(
        "프롬프트 메모리 문맥",
        {"session_id": session_id, "flow_id": flow_id, "applied": True, "count": max(len(context_lines) - 2, 0)},
    )
    return "\n".join(context_lines)


# 문자열에서 JSON 객체를 안전하게 추출한다.
def extract_json_object(raw_text):
    text = (raw_text or "").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    first_brace_index = text.find("{")
    last_brace_index = text.rfind("}")
    if first_brace_index == -1 or last_brace_index == -1 or last_brace_index <= first_brace_index:
        return None
    try:
        data = json.loads(text[first_brace_index : last_brace_index + 1])
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        return None
    return None


# 메모리 저장 의사결정 결과를 안전한 값으로 정규화한다.
def normalize_memory_decision(parsed_data):
    def to_stored_field(value, fallback, max_len):
        """Strip/whitespace collapse; keep UTF-8 (Korean etc.) so facts are not collapsed to ASCII fallbacks."""
        text = str(value or "").strip()
        if not text:
            return fallback
        text = re.sub(r"\s+", " ", text)
        if max_len > 0 and len(text) > max_len:
            text = text[:max_len].rstrip()
        return text or fallback

    def to_ascii_tag(value):
        text = str(value or "").strip()
        tag_map = {
            "직업": "job",
            "회사": "company",
            "이름": "name",
            "사용자 정보": "user_profile",
            "개인 정보": "personal_info",
            "职业信息": "job_info",
        }
        if text in tag_map:
            return tag_map[text]
        ascii_text = text.encode("ascii", "ignore").decode("ascii").lower()
        ascii_text = re.sub(r"[^a-z0-9]+", "_", ascii_text).strip("_")
        return ascii_text[:24] if ascii_text else ""

    decision_data = parsed_data or {}
    action = str(decision_data.get("action", "skip")).strip().lower()
    if action not in {"add", "skip"}:
        action = "skip"
    # add만 저장용 placeholder를 쓴다. skip이면 빈 값은 그대로 두어 로그/의미가 섞이지 않게 한다.
    title = to_stored_field(
        decision_data.get("title", ""),
        "user_info" if action == "add" else "",
        40,
    )
    content = to_stored_field(
        decision_data.get("content", ""),
        "user_profile_update" if action == "add" else "",
        120,
    )
    reason = to_stored_field(
        decision_data.get("reason", ""),
        "useful" if action == "add" else "",
        40,
    )

    tags_raw = decision_data.get("tags", [])
    tags = []
    if isinstance(tags_raw, list):
        tags = [tag for item in tags_raw if (tag := to_ascii_tag(item))]

    return {
        "action": action,
        "title": title,
        "content": content,
        "tags": tags[:5],
        "reason": reason,
    }


# 저장하려는 메모리가 기존 메모리와 중복되는지 확인한다.
def is_duplicate_memory(text, session_id=None, flow_id=None):
    normalized_text = text.strip().lower()
    if not normalized_text:
        return True
    existing_rows = fetch_hermes_memory(limit=100, session_id=session_id, flow_id=flow_id)
    for existing_row in existing_rows:
        existing_content = str(existing_row.get("content", "")).strip().lower()
        if existing_content and (existing_content in normalized_text or normalized_text in existing_content):
            return True
    return False


# 메모리 의사결정 요청 텍스트를 최대 길이로 잘라낸다.
def trim_for_memory_decision(text, max_chars):
    normalized_text = (text or "").strip()
    if max_chars <= 0 or len(normalized_text) <= max_chars:
        return normalized_text
    return f"{normalized_text[:max_chars].rstrip()}..."


# 짧은 소통은 메모리 저장 의사결정을 생략한다.
def should_skip_memory_decision(user_message, answer):
    user_message_trimmed = (user_message or "").strip()
    answer_trimmed = (answer or "").strip()
    if not user_message_trimmed or not answer_trimmed:
        return "empty_text"

    if (
        "?" in user_message_trimmed
        and len(user_message_trimmed) <= settings.chat_memory_skip_short_question_len
        and len(answer_trimmed) <= settings.chat_memory_decision_max_chars
    ):
        return "short_question"
    return None


# Ollama를 사용해 메모리 저장 여부를 결정한다.
def decide_memory_with_ollama(model, user_message, answer, session_id=None, flow_id=None):
    max_chars = settings.chat_memory_decision_max_chars
    trimmed_user_message = trim_for_memory_decision(user_message, max_chars)
    trimmed_answer = trim_for_memory_decision(answer, max_chars)
    decision_prompt = (
        "One line JSON only:\n"
        '{"action":"add|skip","title":"","content":"","tags":[],"reason":""}\n'
        "add=durable facts about the user or their close circle the assistant should recall later: identity, "
        "job, place, prefs, rules. Include family/relationship facts (spouse/children/parents: names, jobs, "
        "roles) whenever the user states them as fact—e.g. Korean: 아내/남편/자녀 + 직업·이름. "
        "skip only for thanks/greetings, chit-chat with no storable fact, or one-off task detail. "
        "Non-English in title/content ok; tags: <=5 Latin snake_case. Limits: title<=40, content<=120, "
        "reason<=40. If skip: \"\" for title, content, tags; reason explains skip (e.g. no_new_fact). "
        "Valid UTF-8; no control chars in strings.\n"
        f"USER: {trimmed_user_message}\n"
        f"ASSISTANT: {trimmed_answer}\n"
    )
    ollama_url = f"{settings.ollama_base_url}/api/generate"
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
    write_chat_log(
        "저장 판단 API 요청",
        {
            "model": model,
            "payload": payload,
            "user_message": user_message,
            "session_id": session_id,
            "flow_id": flow_id,
        },
    )
    http_request = request.Request(
        url=ollama_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started_time = perf_counter()
    timeout_seconds = settings.chat_memory_decision_timeout_seconds
    if timeout_seconds <= 0:
        timeout_seconds = settings.ollama_timeout_seconds
    with request.urlopen(http_request, timeout=timeout_seconds) as response:
        response_body = json.loads(response.read().decode("utf-8"))
    latency_milliseconds = int((perf_counter() - started_time) * 1000)
    raw_response_text = str(response_body.get("response", "")).strip()
    parsed_decision = normalize_memory_decision(extract_json_object(raw_response_text))

    write_chat_log(
        "저장 판단 API 응답",
        {
            "model": model,
            "raw_response": raw_response_text,
            "parsed": parsed_decision,
            "latency_ms": latency_milliseconds,
            "truncated": {
                "user_message": trimmed_user_message != user_message.strip(),
                "assistant_answer": trimmed_answer != answer.strip(),
                "max_chars": max_chars,
            },
            "session_id": session_id,
            "flow_id": flow_id,
            "timeout_seconds": timeout_seconds,
        },
    )
    return parsed_decision


# 응답 완료 후 메모리 저장 파이프라인을 실행한다.
def maybe_store_global_memory(session_id, user_message, answer, model, flow_id=None):
    if settings.chat_memory_decision_mode.lower() != "ollama":
        write_chat_log(
            "메모리 저장 생략",
            {
                "session_id": session_id,
                "flow_id": flow_id,
                "reason": "decision_mode_off",
                "mode": settings.chat_memory_decision_mode,
            },
        )
        return
    if not answer.strip():
        write_chat_log(
            "메모리 저장 생략",
            {"session_id": session_id, "flow_id": flow_id, "reason": "empty_answer"},
        )
        return
    skip_reason = should_skip_memory_decision(user_message=user_message, answer=answer)
    if skip_reason:
        write_chat_log(
            "메모리 저장 생략",
            {"session_id": session_id, "flow_id": flow_id, "reason": skip_reason},
        )
        return

    decision_model = settings.chat_memory_decision_model or model
    try:
        log_chat_flow(
            session_id,
            flow_id or "n/a",
            "메모리 판단 시작",
            {"model": decision_model},
        )
        decision = decide_memory_with_ollama(
            model=decision_model,
            user_message=user_message,
            answer=answer,
            session_id=session_id,
            flow_id=flow_id,
        )
    except Exception as exception:
        write_chat_log(
            "메모리 저장 생략",
            {"session_id": session_id, "flow_id": flow_id, "reason": "decision_error", "error": str(exception)},
        )
        return

    log_chat_flow(
        session_id,
        flow_id or "n/a",
        "메모리 판단 끝",
        {
            "action": decision.get("action", "skip"),
            "reason": decision.get("reason", ""),
            "title": decision.get("title", ""),
        },
    )
    write_chat_log(
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
        write_chat_log(
            "메모리 저장 생략",
            {"session_id": session_id, "flow_id": flow_id, "reason": "decision_skip", "decision": decision},
        )
        return

    content = decision["content"] or f"{user_message}\n{answer}"
    if is_duplicate_memory(content, session_id=session_id, flow_id=flow_id):
        write_chat_log(
            "메모리 저장 생략",
            {"session_id": session_id, "flow_id": flow_id, "reason": "duplicate", "content_preview": content[:200]},
        )
        return

    title = decision["title"] or user_message.strip().replace("\n", " ")[:50] or "chat_memory"
    log_chat_flow(session_id, flow_id or "n/a", "메모리 저장 시작", {"title": title})
    memory_row = save_hermes_memory(
        title=title,
        content=content,
        tags=decision["tags"],
        source="chat_auto",
        session_id=session_id,
        user_message=user_message,
        flow_id=flow_id,
    )
    write_chat_log(
        "memory_store_add",
        {
            "session_id": session_id,
            "flow_id": flow_id,
            "decision": decision,
            "memory_id": memory_row.get("memory_id"),
            "title": title,
        },
    )
    log_chat_flow(
        session_id,
        flow_id or "n/a",
        "메모리 저장 끝",
        {"memory_id": memory_row.get("memory_id")},
    )


# 메모리 저장 작업을 큐에 안전하게 등록한다.
def schedule_memory_store(session_id, user_message, answer, model, flow_id=None):
    if settings.chat_backend == "hermes":
        return
    ensure_memory_worker()
    cooldown_seconds = settings.chat_memory_session_cooldown_seconds
    now = perf_counter()
    if cooldown_seconds > 0:
        with memory_schedule_lock:
            last_scheduled_time = session_last_memory_schedule_time.get(session_id)
            if last_scheduled_time is not None and (now - last_scheduled_time) < cooldown_seconds:
                remaining_milliseconds = int((cooldown_seconds - (now - last_scheduled_time)) * 1000)
                write_chat_log(
                    "메모리 저장 생략",
                    {
                        "session_id": session_id,
                        "flow_id": flow_id,
                        "reason": "session_cooldown",
                        "cooldown_seconds": cooldown_seconds,
                        "remaining_ms": max(remaining_milliseconds, 0),
                    },
                )
                return
            session_last_memory_schedule_time[session_id] = now

    task = {
        "session_id": session_id,
        "flow_id": flow_id or "",
        "model": model,
        "user_message": user_message,
        "answer": answer,
    }
    write_chat_log(
        "memory_store_scheduled",
        {
            "session_id": session_id,
            "flow_id": flow_id,
            "model": model,
            "user_message": user_message,
            "queue_size_before": memory_task_queue.qsize(),
        },
    )
    memory_task_queue.put(task)
    write_chat_log(
        "memory_store_enqueued",
        {"session_id": session_id, "flow_id": flow_id, "queue_size_after": memory_task_queue.qsize()},
    )


# 중복 생성된 답변 텍스트를 정규화한다.
def normalize_answer_text(answer):
    text = (answer or "").strip()
    if not text:
        return ""

    half_length = len(text) // 2
    if len(text) % 2 == 0 and half_length >= 10 and text[:half_length] == text[half_length:]:
        return text[:half_length].strip()

    lines = text.splitlines()
    line_half_length = len(lines) // 2
    if len(lines) % 2 == 0 and line_half_length >= 1 and lines[:line_half_length] == lines[line_half_length:]:
        return "\n".join(lines[:line_half_length]).strip()

    return text


# Ollama 스트리밍 응답을 SSE 이벤트로 변환한다.
def chat_stream(url, payload, on_complete=None, log_context=None):
    flow_id = str((log_context or {}).get("flow_id", "n/a"))
    session_id = str((log_context or {}).get("session_id", "n/a"))
    started_time = perf_counter()
    first_token_latency_ms = None
    done_received_latency_ms = None
    on_complete_latency_ms = None
    chunk_count = 0
    chunk_gaps_ms = []
    last_piece_at = None
    log_chat_flow(session_id, flow_id, "LLM 호출 시작", {"stream": True})
    write_chat_log(
        "chat_http_request",
        {
            **(log_context or {}),
            "url": url,
            **build_log_prompt_fields(payload),
        },
    )
    write_chat_log(
        "chat_ollama_diagnostics",
        {
            **(log_context or {}),
            "wall_started_at": datetime.now(korean_timezone).isoformat(),
            **build_ollama_request_diagnostics(payload, url),
        },
    )
    http_request = request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    answer_parts = []
    done_sent = False
    try:
        with request.urlopen(http_request, timeout=settings.ollama_timeout_seconds) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                piece = data.get("response", "")
                if piece:
                    now = perf_counter()
                    chunk_count += 1
                    if first_token_latency_ms is None:
                        first_token_latency_ms = int((now - started_time) * 1000)
                        write_chat_log(
                            "chat_stream_first_token",
                            {
                                **(log_context or {}),
                                "latency_ms": first_token_latency_ms,
                            },
                        )
                    elif last_piece_at is not None:
                        chunk_gaps_ms.append((now - last_piece_at) * 1000)
                    last_piece_at = now
                    answer_parts.append(piece)
                    # Ollama NDJSON 한 덩어리씩 보낸다. 글자 단위 SSE는 이벤트가 과도해
                    # 클라이언트/프록시가 끊거나 GeneratorExit → finally_guard로 이어질 수 있다.
                    yield f"data: {json.dumps({'text': piece}, ensure_ascii=False)}\n\n"

                if data.get("done"):
                    done_received_latency_ms = int((perf_counter() - started_time) * 1000)
                    final_answer = normalize_answer_text("".join(answer_parts))
                    if on_complete:
                        on_complete_started_time = perf_counter()
                        on_complete(final_answer)
                        on_complete_latency_ms = int((perf_counter() - on_complete_started_time) * 1000)
                    total_latency_ms = int((perf_counter() - started_time) * 1000)
                    write_chat_log(
                        "chat_http_response",
                        {
                            **(log_context or {}),
                            "stream": True,
                            "answer": final_answer,
                        },
                    )
                    write_chat_log(
                        "chat_stream_timing",
                        {
                            **(log_context or {}),
                            "total_latency_ms": total_latency_ms,
                            "first_token_latency_ms": first_token_latency_ms,
                            "done_received_latency_ms": done_received_latency_ms,
                            "on_complete_latency_ms": on_complete_latency_ms,
                            "chunk_count": chunk_count,
                            "answer_chars": len(final_answer),
                        },
                    )
                    write_chat_log(
                        "chat_stream_phase_breakdown",
                        {
                            **(log_context or {}),
                            "prefill_ms": first_token_latency_ms,
                            "answer_generation_ms": (
                                done_received_latency_ms - first_token_latency_ms
                                if first_token_latency_ms is not None and done_received_latency_ms is not None
                                else None
                            ),
                            "postprocess_ms": (
                                total_latency_ms - done_received_latency_ms
                                if done_received_latency_ms is not None
                                else None
                            ),
                        },
                    )
                    write_ollama_stream_diagnostics(
                        log_context,
                        chunk_gaps_ms,
                        chunk_count,
                        first_token_latency_ms,
                        done_received_latency_ms,
                        total_latency_ms,
                        ollama_packet_stats(data),
                    )
                    log_chat_flow(session_id, flow_id, "LLM 응답 완료", {"stream": True})
                    yield "event: done\ndata: [DONE]\n\n"
                    done_sent = True
                    break

            if not done_sent:
                done_received_latency_ms = int((perf_counter() - started_time) * 1000)
                final_answer = normalize_answer_text("".join(answer_parts))
                if on_complete:
                    on_complete_started_time = perf_counter()
                    on_complete(final_answer)
                    on_complete_latency_ms = int((perf_counter() - on_complete_started_time) * 1000)
                total_latency_ms = int((perf_counter() - started_time) * 1000)
                write_chat_log(
                    "chat_stream_done_fallback",
                    {
                        **(log_context or {}),
                        "reason": "upstream_closed_without_done",
                        "answer": final_answer,
                    },
                )
                write_chat_log(
                    "chat_stream_timing",
                    {
                        **(log_context or {}),
                        "total_latency_ms": total_latency_ms,
                        "first_token_latency_ms": first_token_latency_ms,
                        "done_received_latency_ms": done_received_latency_ms,
                        "on_complete_latency_ms": on_complete_latency_ms,
                        "chunk_count": chunk_count,
                        "answer_chars": len(final_answer),
                        "fallback": "upstream_closed_without_done",
                    },
                )
                write_chat_log(
                    "chat_stream_phase_breakdown",
                    {
                        **(log_context or {}),
                        "prefill_ms": first_token_latency_ms,
                        "answer_generation_ms": (
                            done_received_latency_ms - first_token_latency_ms
                            if first_token_latency_ms is not None and done_received_latency_ms is not None
                            else None
                        ),
                        "postprocess_ms": (
                            total_latency_ms - done_received_latency_ms
                            if done_received_latency_ms is not None
                            else None
                        ),
                        "fallback": "upstream_closed_without_done",
                    },
                )
                write_ollama_stream_diagnostics(
                    log_context,
                    chunk_gaps_ms,
                    chunk_count,
                    first_token_latency_ms,
                    done_received_latency_ms,
                    total_latency_ms,
                    {},
                    extra={"note": "upstream_closed_without_done"},
                )
                log_chat_flow(
                    session_id,
                    flow_id,
                    "LLM 응답 완료",
                    {"stream": True, "fallback": "upstream_closed_without_done"},
                )
                yield "event: done\ndata: [DONE]\n\n"
                done_sent = True
    except Exception as exception:
        err_fields = ollama_request_error_fields(exception)
        write_chat_log("chat_http_error", {**(log_context or {}), "stream": True, **err_fields})
        flow_err = {"stream": True, "error": err_fields["error"]}
        if "http_status" in err_fields:
            flow_err["http_status"] = err_fields["http_status"]
        log_chat_flow(session_id, flow_id, "LLM 응답 오류", flow_err)
        yield f"event: error\ndata: {json.dumps({'error': err_fields['error']}, ensure_ascii=False)}\n\n"
    finally:
        if not done_sent:
            write_chat_log(
                "chat_stream_done_fallback",
                {**(log_context or {}), "reason": "finally_guard"},
            )
            log_chat_flow(
                session_id,
                flow_id,
                "LLM 응답 완료",
                {"stream": True, "fallback": "finally_guard"},
            )
            write_chat_log(
                "chat_stream_timing",
                {
                    **(log_context or {}),
                    "total_latency_ms": int((perf_counter() - started_time) * 1000),
                    "first_token_latency_ms": first_token_latency_ms,
                    "done_received_latency_ms": done_received_latency_ms,
                    "on_complete_latency_ms": on_complete_latency_ms,
                    "chunk_count": chunk_count,
                    "fallback": "finally_guard",
                },
            )
            _fg_total = int((perf_counter() - started_time) * 1000)
            write_ollama_stream_diagnostics(
                log_context,
                chunk_gaps_ms,
                chunk_count,
                first_token_latency_ms,
                done_received_latency_ms,
                _fg_total,
                {},
                extra={"note": "finally_guard"},
            )
            yield "event: done\ndata: [DONE]\n\n"


# 요청 헤더에서 클라이언트 메타 정보를 추출한다.
def client_meta(request_info):
    return {
        "client": request_info.headers.get("x-dabo-client", "unknown"),
        "build": request_info.headers.get("x-dabo-build", "unknown"),
        "api_version": request_info.headers.get("x-api-version", "unknown"),
    }


# 단일 채팅 턴을 세션 히스토리에 저장한다.
def save_chat_turn(session_id, user_message, answer, model_name, flow_id):
    normalized_answer = normalize_answer_text(answer)
    turns = chat_sessions.setdefault(session_id, [])
    if len(turns) >= 2:
        last_user_turn = turns[-2]
        last_assistant_turn = turns[-1]
        if (
            last_user_turn.get("role") == "user"
            and last_assistant_turn.get("role") == "assistant"
            and last_user_turn.get("content") == user_message
            and last_assistant_turn.get("content") == normalized_answer
        ):
            return

    turns.append({"role": "user", "content": user_message})
    turns.append({"role": "assistant", "content": normalized_answer})
    if len(turns) > max_session_turns * 2:
        chat_sessions[session_id] = turns[-(max_session_turns * 2) :]

    schedule_memory_store(
        session_id=session_id,
        user_message=user_message,
        answer=normalized_answer,
        model=model_name,
        flow_id=flow_id,
    )


# CHAT_BACKEND=hermes: AIAgent + Ollama(OpenAI /v1) — 스텁 hermes_service 메모리 주입/직접 Ollama와 병행하지 않는다.
def _process_hermes_chat_request(chat_request, request_info, session_id, flow_id):
    from service import hermes_chat

    try:
        hermes_chat.ensure_hermes_import()
    except Exception as exc:
        write_chat_log("hermes_import_failed", {"error": str(exc), "flow_id": flow_id, "session_id": session_id})
        raise HTTPException(
            status_code=503,
            detail=(
                "Hermes Agent(run_agent)를 불러올 수 없습니다. "
                "requirements의 hermes-agent 설치를 확인하세요. "
                f"({exc})"
            ),
        ) from exc

    write_chat_log(
        "hermes_http_request",
        {
            "session_id": session_id,
            "flow_id": flow_id,
            "new_chat": chat_request.new_chat,
            "stream": chat_request.stream,
            **hermes_chat.hermes_config_snapshot(),
        },
    )
    log_chat_flow(
        session_id,
        flow_id,
        "요청 수신(hermes)",
        {"new_chat": chat_request.new_chat, "stream": chat_request.stream, "path": "hermes_agent"},
    )
    with hermes_locks[session_id]:
        conversation_history = hermes_session_messages.get(session_id)
        if conversation_history is not None:
            conversation_history = list(conversation_history)
    log_chat_flow(
        session_id,
        flow_id,
        "세션(hermes) 로드",
        {
            "path": "hermes_agent",
            "message_count": len(conversation_history or []),
        },
    )
    raw_system = chat_request.system_prompt or settings.default_system_prompt or ""
    system_message = raw_system.strip() or None
    model_name = str(chat_request.model or settings.ollama_model)
    user_message = chat_request.message
    client = client_meta(request_info)

    def persist_hermes_turn(_final: str, result: dict) -> None:
        with hermes_locks[session_id]:
            hermes_session_messages[session_id] = list(result.get("messages") or [])

    if chat_request.stream:

        def on_complete(final_text, result):
            persist_hermes_turn(final_text, result)
            log_chat_flow(session_id, flow_id, "요청 처리 종료", {"status": "ok", "path": "hermes_agent", "stream": True})

        # StreamingResponse: 클라이언트가 SSE body를 읽기 시작한 뒤에만 제너레이터가 돌고 hermes_turn_start가 찍힌다.
        log_chat_flow(
            session_id,
            flow_id,
            "Hermes 스트리밍 응답 반환(이후 클라이언트 수신 시 턴·툴 로딩이 시작됨)",
            {
                "stream": True,
                "path": "hermes_agent",
                "client_build": client.get("build", ""),
            },
        )
        return StreamingResponse(
            hermes_chat.hermes_stream_generator(
                user_message=user_message,
                system_message=system_message,
                conversation_history=conversation_history,
                model=model_name,
                session_id=session_id,
                on_complete=on_complete,
                log_context={
                    "session_id": session_id,
                    "model": model_name,
                    "flow_id": flow_id,
                    **hermes_chat.hermes_config_snapshot(),
                },
            ),
            media_type="text/event-stream; charset=utf-8",
            headers={
                **stream_headers,
                "X-DABO-Client": client["client"],
                "X-DABO-Build": client["build"],
                "X-API-Version": client["api_version"],
                "X-Session-Id": session_id,
                "X-Chat-Backend": "hermes",
            },
        )

    try:
        log_chat_flow(session_id, flow_id, "Hermes LLM 호출 시작", {"stream": False, "path": "hermes_agent"})
        result = hermes_chat.run_hermes_conversation(
            user_message=user_message,
            system_message=system_message,
            conversation_history=conversation_history,
            model=model_name,
            session_id=session_id,
            flow_id=flow_id,
        )
        answer = normalize_answer_text(result.get("final_response", ""))
        write_chat_log(
            "hermes_http_response",
            {
                "session_id": session_id,
                "flow_id": flow_id,
                "model": model_name,
                "stream": False,
                "answer_preview": hermes_chat.hermes_answer_preview(answer, 500),
                "result_messages_persisted": len(result.get("messages") or []),
            },
        )
        persist_hermes_turn(answer, result)
        log_chat_flow(session_id, flow_id, "LLM 응답 완료", {"stream": False, "path": "hermes_agent"})
        log_chat_flow(session_id, flow_id, "요청 처리 종료", {"status": "ok", "path": "hermes_agent"})
        return {
            "answer": answer,
            "model": model_name,
            "client": client,
            "session_id": session_id,
        }
    except Exception as exc:
        import traceback

        write_chat_log(
            "hermes_conversation_error",
            {
                "session_id": session_id,
                "flow_id": flow_id,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                **hermes_chat.hermes_config_snapshot(),
            },
        )
        log_chat_flow(session_id, flow_id, "요청 처리 종료", {"status": "error", "path": "hermes_agent", "error": str(exc)})
        raise HTTPException(status_code=500, detail=f"Hermes conversation error: {exc}") from exc


# 채팅 요청 1건을 처리하고 일반/스트리밍 응답을 반환한다.
def process_chat_request(chat_request, request_info):
    session_id = chat_request.session_id or str(uuid.uuid4())
    flow_id = uuid.uuid4().hex[:12]
    log_chat_flow(
        session_id,
        flow_id,
        "요청 수신",
        {"new_chat": chat_request.new_chat, "stream": chat_request.stream},
    )
    if chat_request.new_chat:
        chat_sessions.pop(session_id, None)
        hermes_session_messages.pop(session_id, None)

    if settings.chat_backend == "hermes":
        return _process_hermes_chat_request(chat_request, request_info, session_id, flow_id)

    history = chat_sessions.get(session_id, [])
    log_chat_flow(
        session_id,
        flow_id,
        "세션 히스토리 로드",
        {"history_turns": len(history) // 2, "prompt_max_turns": settings.chat_prompt_max_turns},
    )
    prompt = build_session_prompt(history, chat_request.message)

    log_chat_flow(session_id, flow_id, "Hermes 메모리 조회 시작")
    use_memory_context = True
    if settings.chat_memory_context_first_turn_only and len(history) > 0:
        use_memory_context = False
    memory_context = build_memory_context(session_id=session_id, flow_id=flow_id) if use_memory_context else ""
    if memory_context:
        prompt = f"{memory_context}\n\n{prompt}"
    log_chat_flow(
        session_id,
        flow_id,
        "Hermes 메모리 조회 끝",
        {
            "memory_context_applied": bool(memory_context),
            "memory_context_first_turn_only": settings.chat_memory_context_first_turn_only,
        },
    )
    write_chat_log(
        "chat_prompt_ready",
        {
            "session_id": session_id,
            "flow_id": flow_id,
            "history_turns": len(history) // 2,
            "memory_context_applied": bool(memory_context),
            "prompt_chars": len(prompt),
        },
    )

    ollama_url = f"{settings.ollama_base_url}/api/generate"
    payload = {
        "model": chat_request.model or settings.ollama_model,
        "prompt": prompt,
        "stream": chat_request.stream,
    }
    system_prompt = chat_request.system_prompt or settings.default_system_prompt
    if system_prompt:
        payload["system"] = system_prompt
    if chat_request.think is not None:
        payload["think"] = chat_request.think
    keep_alive = chat_request.keep_alive or settings.ollama_keep_alive
    if keep_alive:
        payload["keep_alive"] = keep_alive

    client = client_meta(request_info)
    model_name = str(payload["model"])

    if chat_request.stream:
        def on_complete(answer):
            save_chat_turn(
                session_id=session_id,
                user_message=chat_request.message,
                answer=answer,
                model_name=model_name,
                flow_id=flow_id,
            )

        return StreamingResponse(
            chat_stream(
                ollama_url,
                payload,
                on_complete=on_complete,
                log_context={"session_id": session_id, "model": model_name, "flow_id": flow_id},
            ),
            media_type="text/event-stream; charset=utf-8",
            headers={
                **stream_headers,
                "X-DABO-Client": client["client"],
                "X-DABO-Build": client["build"],
                "X-API-Version": client["api_version"],
                "X-Session-Id": session_id,
            },
        )

    try:
        log_chat_flow(session_id, flow_id, "LLM 호출 시작", {"stream": False})
        write_chat_log(
            "chat_http_request",
            {
                "session_id": session_id,
                "flow_id": flow_id,
                "model": model_name,
                "url": ollama_url,
                **build_log_prompt_fields(payload),
            },
        )
        write_chat_log(
            "chat_ollama_diagnostics",
            {
                "session_id": session_id,
                "flow_id": flow_id,
                "model": model_name,
                "stream": False,
                "wall_started_at": datetime.now(korean_timezone).isoformat(),
                **build_ollama_request_diagnostics(payload, ollama_url),
            },
        )
        http_request = request.Request(
            url=ollama_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        generate_start = perf_counter()
        with request.urlopen(http_request, timeout=settings.ollama_timeout_seconds) as response:
            response_body = json.loads(response.read().decode("utf-8"))
        client_http_roundtrip_ms = int((perf_counter() - generate_start) * 1000)
        answer = normalize_answer_text(response_body.get("response", ""))
        write_chat_log(
            "chat_ollama_generate_diagnostics",
            {
                "session_id": session_id,
                "flow_id": flow_id,
                "model": model_name,
                "stream": False,
                "client_http_roundtrip_ms": client_http_roundtrip_ms,
                "ollama_metrics": ollama_packet_stats(response_body),
            },
        )
        write_chat_log(
            "chat_http_response",
            {
                "session_id": session_id,
                "flow_id": flow_id,
                "model": model_name,
                "stream": False,
                "answer": answer,
                "raw_body": response_body,
            },
        )
        log_chat_flow(session_id, flow_id, "LLM 응답 완료", {"stream": False})
        save_chat_turn(
            session_id=session_id,
            user_message=chat_request.message,
            answer=answer,
            model_name=model_name,
            flow_id=flow_id,
        )
        log_chat_flow(session_id, flow_id, "요청 처리 종료", {"status": "ok"})
        return {
            "answer": answer,
            "model": payload["model"],
            "client": client,
            "session_id": session_id,
        }
    except error.URLError as exception:
        err_fields = ollama_request_error_fields(exception)
        write_chat_log(
            "chat_http_error",
            {
                "session_id": session_id,
                "flow_id": flow_id,
                "model": model_name,
                "stream": False,
                **err_fields,
            },
        )
        log_chat_flow(session_id, flow_id, "요청 처리 종료", {"status": "error", "error": err_fields["error"]})
        detail = err_fields["error"]
        if isinstance(exception, error.HTTPError):
            raise HTTPException(status_code=502, detail=f"Ollama HTTP error: {detail}") from exception
        raise HTTPException(status_code=502, detail=f"Ollama connection failed: {detail}") from exception
    except Exception as exception:
        write_chat_log(
            "chat_http_error",
            {
                "session_id": session_id,
                "flow_id": flow_id,
                "model": model_name,
                "stream": False,
                "error": str(exception),
            },
        )
        log_chat_flow(session_id, flow_id, "요청 처리 종료", {"status": "error", "error": str(exception)})
        raise HTTPException(status_code=500, detail=f"Chat error: {exception}") from exception
