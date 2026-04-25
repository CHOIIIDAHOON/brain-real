"""
Hermes Agent(AIAgent) + 로컬 Ollama(OpenAI 호환) 연동.
run_agent.AIAgent가 세션/메모리/툴 루프를 담당하므로 별도 Ollama + 스텁 Hermes 메모리 주입은 쓰지 않는다.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from queue import Queue
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional

from config import settings


def ollama_openai_base_url() -> str:
    return f"{str(settings.ollama_base_url).rstrip('/')}/v1"


def _parse_disabled_toolsets() -> Optional[List[str]]:
    raw = (getattr(settings, "hermes_disabled_toolsets", None) or "").strip()
    if not raw:
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def ensure_hermes_import() -> None:
    """앱 기동 시 한 번 호출해 의존성·임포트 오류를 조기에 낸다."""
    from run_agent import AIAgent  # noqa: F401


def _build_agent(
    *,
    model: str,
    session_id: str,
    ephemeral_system_prompt: Optional[str],
) -> Any:
    from run_agent import AIAgent

    return AIAgent(
        model=model,
        base_url=ollama_openai_base_url(),
        api_key=settings.hermes_ollama_api_key,
        quiet_mode=True,
        max_iterations=settings.hermes_max_iterations,
        session_id=session_id,
        skip_context_files=settings.hermes_skip_context_files,
        skip_memory=settings.hermes_skip_memory,
        ephemeral_system_prompt=ephemeral_system_prompt,
        platform=settings.hermes_platform,
        disabled_toolsets=_parse_disabled_toolsets(),
    )


def run_hermes_conversation(
    *,
    user_message: str,
    system_message: Optional[str],
    conversation_history: Optional[List[Dict[str, Any]]],
    model: str,
    session_id: str,
    stream_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    agent = _build_agent(
        model=model,
        session_id=session_id,
        ephemeral_system_prompt=None,
    )
    return agent.run_conversation(
        user_message,
        system_message=system_message,
        conversation_history=conversation_history,
        stream_callback=stream_callback,
    )


def hermes_stream_generator(
    *,
    user_message: str,
    system_message: Optional[str],
    conversation_history: Optional[List[Dict[str, Any]]],
    model: str,
    session_id: str,
    on_complete: Optional[Callable[[str, Dict[str, Any]], None]],
    log_context: Optional[Dict[str, Any]] = None,
):
    """
    Ollama chat_stream과 동일한 SSE(text JSON + event: done) 형식을 낸다.
    AIAgent는 동기식이므로 전용 스레드에서 run_conversation을 돌리고, 스트림 콜백을 큐로 넘긴다.
    """
    from service.chat_service import (  # 지연 import — chat_service 쪽 임포트 순서 회피
        korean_timezone,
        log_chat_flow,
        write_chat_log,
    )

    q: Queue = Queue()
    result_holder: list = []
    flow_id = str((log_context or {}).get("flow_id", "n/a"))

    def work():
        try:
            result = run_hermes_conversation(
                user_message=user_message,
                system_message=system_message,
                conversation_history=conversation_history,
                model=model,
                session_id=session_id,
                stream_callback=lambda t: q.put(("delta", t)),
            )
            result_holder.append(result)
            q.put(("finished", None))
        except Exception as e:
            q.put(("error", str(e)))

    started_time = perf_counter()
    log_chat_flow(session_id, flow_id, "Hermes LLM 호출 시작", {"stream": True, "path": "hermes_agent"})

    thread = threading.Thread(target=work, daemon=True)
    thread.start()

    first_token_latency_ms = None
    answer_parts: List[str] = []
    while True:
        # 툴 루프·긴 추론 시 첫 토큰까지 수분 걸릴 수 있으므로 큐 get에 짧은 타임아웃을 두지 않는다.
        item = q.get()
        kind = item[0]
        if kind == "delta":
            text = item[1] or ""
            if text:
                now = perf_counter()
                if first_token_latency_ms is None:
                    first_token_latency_ms = int((now - started_time) * 1000)
                    write_chat_log(
                        "chat_stream_first_token",
                        {**(log_context or {}), "latency_ms": first_token_latency_ms, "backend": "hermes"},
                    )
                answer_parts.append(text)
                yield f"data: {json.dumps({'text': text}, ensure_ascii=False)}\n\n"
        elif kind == "finished":
            break
        elif kind == "error":
            write_chat_log("hermes_stream_error", {**(log_context or {}), "stream": True, "error": item[1]})
            yield f"event: error\ndata: {json.dumps({'error': item[1]}, ensure_ascii=False)}\n\n"
            return
        else:
            continue

    if not result_holder:
        write_chat_log("hermes_stream_missing_result", {**(log_context or {}), "stream": True})
        yield f"event: error\ndata: {json.dumps({'error': 'hermes: empty result'}, ensure_ascii=False)}\n\n"
        return

    result = result_holder[0]
    from service.chat_service import normalize_answer_text

    final_text = normalize_answer_text(result.get("final_response", ""))
    if on_complete:
        on_complete(final_text, result)
    done_ms = int((perf_counter() - started_time) * 1000)
    write_chat_log(
        "chat_stream_timing",
        {
            **(log_context or {}),
            "stream": True,
            "total_latency_ms": done_ms,
            "first_token_latency_ms": first_token_latency_ms,
            "chunk_count": len(answer_parts),
            "answer_chars": len(final_text),
            "backend": "hermes",
        },
    )
    log_chat_flow(session_id, flow_id, "LLM 응답 완료", {"stream": True, "path": "hermes_agent"})
    write_chat_log(
        "chat_http_response",
        {**(log_context or {}), "stream": True, "answer": final_text, "wall_ended": datetime.now(korean_timezone).isoformat()},
    )
    yield "event: done\ndata: [DONE]\n\n"
