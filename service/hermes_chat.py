"""
Hermes Agent(AIAgent) + 로컬 Ollama(OpenAI 호환) 연동.
run_agent.AIAgent가 세션/메모리/툴 루프를 담당하므로 별도 Ollama + 스텁 Hermes 메모리 주입은 쓰지 않는다.
"""

from __future__ import annotations

import json
import threading
import traceback
from datetime import datetime
from queue import Queue
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional

from config import settings


def ollama_openai_base_url() -> str:
    return f"{str(settings.ollama_base_url).rstrip('/')}/v1"


def _user_text_preview(text: str, max_chars: int = 160) -> str:
    t = (text or "").replace("\n", " ").strip()
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 3] + "..."


def hermes_answer_preview(text: str, max_chars: int = 500) -> str:
    """로그용 답변 미리보기(전체 답은 CHAT_LOG에 과하게 남기지 않음)."""
    return _user_text_preview(text, max_chars)


def _parse_disabled_toolsets() -> Optional[List[str]]:
    raw = (getattr(settings, "hermes_disabled_toolsets", None) or "").strip()
    if not raw:
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_enabled_toolsets() -> Optional[List[str]]:
    raw = (getattr(settings, "hermes_enabled_toolsets", None) or "").strip()
    if not raw:
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def hermes_config_snapshot() -> Dict[str, Any]:
    return {
        "ollama_v1_url": ollama_openai_base_url(),
        "hermes_max_iterations": settings.hermes_max_iterations,
        "hermes_skip_memory": settings.hermes_skip_memory,
        "hermes_skip_context_files": settings.hermes_skip_context_files,
        "hermes_platform": settings.hermes_platform,
        "hermes_enabled_toolsets": _parse_enabled_toolsets(),
        "hermes_disabled_toolsets": _parse_disabled_toolsets(),
        "hermes_trace_log": getattr(settings, "hermes_trace_log", True),
        "hermes_heartbeat_sec": int(getattr(settings, "hermes_heartbeat_interval_seconds", 0) or 0),
    }


def ensure_hermes_import() -> None:
    """앱 기동 시 한 번 호출해 의존성·임포트 오류를 조기에 낸다."""
    from run_agent import AIAgent  # noqa: F401


def _clip_repr(value: Any, max_len: int = 900) -> str:
    try:
        text = repr(value)
    except Exception:
        text = f"<{type(value).__name__}>"
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _hermes_trace_callback_kwargs(
    session_id: str, flow_id: Optional[str]
) -> Dict[str, Any]:
    """HERMES_TRACE_LOG=1 일 때 AIAgent 툴/상태/스텝 콜백 — 어디서 오래 걸리는지 추적용."""
    if not getattr(settings, "hermes_trace_log", True):
        return {}
    from service.chat_service import log_chat_flow, write_chat_log

    b = {"session_id": session_id, "flow_id": flow_id or "n/a"}

    def on_tool_progress(*args: Any, **kwargs: Any) -> None:
        write_chat_log("hermes_cb_tool_progress", {**b, "args": _clip_repr(args), "kwargs": _clip_repr(kwargs) if kwargs else {}})

    def on_tool_start(*args: Any, **kwargs: Any) -> None:
        try:
            name = str(args[0]) if args else ""
        except Exception:
            name = "?"
        write_chat_log("hermes_cb_tool_start", {**b, "args": _clip_repr(args, 1200)})
        log_chat_flow(
            session_id,
            b["flow_id"],
            "Hermes 툴 시작",
            {
                "tool": name[:120] if name else "?",
                "arg_preview": _clip_repr(args[1:3] if len(args) > 1 else args, 400),
            },
        )

    def on_tool_complete(*args: Any, **kwargs: Any) -> None:
        write_chat_log("hermes_cb_tool_complete", {**b, "args": _clip_repr(args, 1200)})

    def on_status(*args: Any, **kwargs: Any) -> None:
        write_chat_log("hermes_cb_status", {**b, "payload": _clip_repr(args, 1500)})

    def on_step(*args: Any, **kwargs: Any) -> None:
        write_chat_log("hermes_cb_step", {**b, "payload": _clip_repr(args, 1500)})

    def on_thinking(*args: Any, **kwargs: Any) -> None:
        write_chat_log("hermes_cb_thinking", {**b, "payload": _clip_repr(args, 800)})

    def on_reasoning(*args: Any, **kwargs: Any) -> None:
        write_chat_log("hermes_cb_reasoning", {**b, "payload": _clip_repr(args, 800)})

    def on_interim_assistant(*args: Any, **kwargs: Any) -> None:
        write_chat_log("hermes_cb_interim_assistant", {**b, "payload": _clip_repr(args, 600)})

    def on_tool_gen(*args: Any, **kwargs: Any) -> None:
        write_chat_log("hermes_cb_tool_gen", {**b, "args": _clip_repr(args, 600)})

    return {
        "tool_progress_callback": on_tool_progress,
        "tool_start_callback": on_tool_start,
        "tool_complete_callback": on_tool_complete,
        "status_callback": on_status,
        "step_callback": on_step,
        "thinking_callback": on_thinking,
        "reasoning_callback": on_reasoning,
        "interim_assistant_callback": on_interim_assistant,
        "tool_gen_callback": on_tool_gen,
    }


def _build_agent(
    *,
    model: str,
    session_id: str,
    ephemeral_system_prompt: Optional[str],
    flow_id: Optional[str] = None,
) -> Any:
    from run_agent import AIAgent

    params: Dict[str, Any] = {
        "model": model,
        "base_url": ollama_openai_base_url(),
        "api_key": settings.hermes_ollama_api_key,
        "quiet_mode": True,
        "max_iterations": settings.hermes_max_iterations,
        "session_id": session_id,
        "skip_context_files": settings.hermes_skip_context_files,
        "skip_memory": settings.hermes_skip_memory,
        "ephemeral_system_prompt": ephemeral_system_prompt,
        "platform": settings.hermes_platform,
        "enabled_toolsets": _parse_enabled_toolsets(),
        "disabled_toolsets": _parse_disabled_toolsets(),
    }
    params.update(_hermes_trace_callback_kwargs(session_id, flow_id))
    return AIAgent(**params)


def run_hermes_conversation(
    *,
    user_message: str,
    system_message: Optional[str],
    conversation_history: Optional[List[Dict[str, Any]]],
    model: str,
    session_id: str,
    stream_callback: Optional[Callable[[str], None]] = None,
    flow_id: Optional[str] = None,
) -> Dict[str, Any]:
    from service.chat_service import log_chat_flow, write_chat_log

    fid = flow_id or "n/a"
    payload_start = {
        **hermes_config_snapshot(),
        "session_id": session_id,
        "flow_id": fid,
        "model": model,
        "user_message_chars": len(user_message or ""),
        "user_message_preview": _user_text_preview(user_message),
        "has_system_message": bool((system_message or "").strip()),
        "history_message_count": len(conversation_history or []),
        "streaming": stream_callback is not None,
    }
    write_chat_log("hermes_turn_start", payload_start)
    log_chat_flow(
        session_id,
        fid,
        "Hermes turn 시작",
        {"model": model, "streaming": stream_callback is not None, "history_messages": len(conversation_history or [])},
    )
    started = perf_counter()
    try:
        t_build0 = perf_counter()
        agent = _build_agent(
            model=model,
            session_id=session_id,
            ephemeral_system_prompt=None,
            flow_id=fid,
        )
        build_ms = int((perf_counter() - t_build0) * 1000)
        write_chat_log("hermes_agent_built", {**payload_start, "build_ms": build_ms})
        log_chat_flow(
            session_id,
            fid,
            "Hermes AIAgent 초기화 끝",
            {"build_ms": build_ms, "hint": "툴/클라이언트 로딩. 오래 걸리면 HERMES_ENABLED_TOOLSETS로 툴을 줄이세요."},
        )
        t_run0 = perf_counter()
        write_chat_log(
            "hermes_invoking_run_conversation",
            {
                **payload_start,
                "build_ms": build_ms,
                "note": "이후 Ollama/툴 콜백: hermes_cb_*",
            },
        )
        log_chat_flow(
            session_id,
            fid,
            "Hermes run_conversation 호출(모델·툴 루프)",
            {
                "model": model,
                "ollama_v1": ollama_openai_base_url(),
                "streaming": stream_callback is not None,
            },
        )
        hb_sec = int(getattr(settings, "hermes_heartbeat_interval_seconds", 30) or 0)
        stop_heartbeat = threading.Event()

        def _heartbeat_thread_fn() -> None:
            n = 0
            while not stop_heartbeat.wait(float(hb_sec)):
                n += 1
                w = int((perf_counter() - t_run0) * 1000)
                write_chat_log(
                    "hermes_heartbeat",
                    {
                        **payload_start,
                        "build_ms": build_ms,
                        "wait_ms": w,
                        "heartbeat_n": n,
                        "ollama_v1": ollama_openai_base_url(),
                        "note": "Ollama/툴 응답 대기( thinking 직후 오래 잠잠 = 첫 API 지연·GPU·툴 중 하나)",
                    },
                )
                log_chat_flow(
                    session_id,
                    fid,
                    "Hermes 응답 대기(heartbeat)",
                    {"wait_ms": w, "n": n},
                )

        hb_thread: Optional[threading.Thread] = None
        if hb_sec > 0:
            hb_thread = threading.Thread(
                target=_heartbeat_thread_fn,
                name=f"hermes-hb-{fid[:8]}",
                daemon=True,
            )
            hb_thread.start()
        try:
            result = agent.run_conversation(
                user_message,
                system_message=system_message,
                conversation_history=conversation_history,
                stream_callback=stream_callback,
            )
        finally:
            stop_heartbeat.set()
        run_ms = int((perf_counter() - t_run0) * 1000)
        write_chat_log(
            "hermes_run_conversation_returned",
            {**payload_start, "run_conversation_ms": run_ms, "build_ms": build_ms},
        )
    except Exception as exc:
        elapsed_ms = int((perf_counter() - started) * 1000)
        write_chat_log(
            "hermes_turn_error",
            {
                **payload_start,
                "elapsed_ms": elapsed_ms,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            },
        )
        log_chat_flow(
            session_id,
            fid,
            "Hermes turn 실패",
            {"error": str(exc), "elapsed_ms": elapsed_ms},
        )
        raise

    elapsed_ms = int((perf_counter() - started) * 1000)
    messages = result.get("messages") if isinstance(result, dict) else None
    final_text = (result.get("final_response") or "") if isinstance(result, dict) else ""
    write_chat_log(
        "hermes_turn_complete",
        {
            **payload_start,
            "elapsed_ms": elapsed_ms,
            "final_response_chars": len(final_text),
            "result_messages_count": len(messages) if isinstance(messages, list) else None,
        },
    )
    log_chat_flow(
        session_id,
        fid,
        "Hermes turn 완료",
        {
            "elapsed_ms": elapsed_ms,
            "final_response_chars": len(final_text),
            "result_messages": len(messages) if isinstance(messages, list) else 0,
        },
    )
    return result


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
    base_ctx = {**(log_context or {}), **hermes_config_snapshot()}

    def work():
        write_chat_log(
            "hermes_stream_worker_begin",
            {
                **base_ctx,
                "session_id": session_id,
                "flow_id": flow_id,
                "thread": threading.current_thread().name,
                "thread_id": threading.get_ident(),
            },
        )
        try:
            result = run_hermes_conversation(
                user_message=user_message,
                system_message=system_message,
                conversation_history=conversation_history,
                model=model,
                session_id=session_id,
                stream_callback=lambda t: q.put(("delta", t)),
                flow_id=flow_id,
            )
            result_holder.append(result)
            q.put(("finished", None))
        except Exception as e:
            write_chat_log(
                "hermes_stream_worker_error",
                {
                    **base_ctx,
                    "session_id": session_id,
                    "flow_id": flow_id,
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                },
            )
            q.put(("error", str(e)))

    started_time = perf_counter()
    write_chat_log(
        "hermes_stream_sse_start",
        {
            **base_ctx,
            "session_id": session_id,
            "flow_id": flow_id,
            "model": model,
        },
    )

    thread = threading.Thread(target=work, name=f"hermes-sse-{flow_id[:8]}", daemon=True)
    thread.start()
    write_chat_log(
        "hermes_stream_thread_started",
        {
            **base_ctx,
            "session_id": session_id,
            "flow_id": flow_id,
            "thread_name": thread.name,
        },
    )

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
                        "hermes_stream_first_token",
                        {
                            **base_ctx,
                            "stream": True,
                            "latency_ms": first_token_latency_ms,
                            "model": model,
                        },
                    )
                    warn_after = int(getattr(settings, "hermes_ttfb_warn_ms", 120_000) or 0)
                    if warn_after > 0 and first_token_latency_ms >= warn_after:
                        log_chat_flow(
                            session_id,
                            flow_id,
                            "Hermes 첫 토큰 지연(매우 느림)",
                            {
                                "latency_ms": first_token_latency_ms,
                                "hint": "툴·반복: HERMES_MAX_ITERATIONS↓ HERMES_ENABLED_TOOLSETS(작은 집합) Ollama cold/GPU·Docker 127.0.0.1",
                            },
                        )
                        write_chat_log(
                            "hermes_ttfb_warn",
                            {
                                **base_ctx,
                                "stream": True,
                                "latency_ms": first_token_latency_ms,
                                "warn_threshold_ms": warn_after,
                                "check_docker_ollama_host": "API 컨테이너는 127.0.0.1≠호스트",
                            },
                        )
                answer_parts.append(text)
                yield f"data: {json.dumps({'text': text}, ensure_ascii=False)}\n\n"
        elif kind == "finished":
            break
        elif kind == "error":
            err_msg = item[1] if len(item) > 1 else ""
            write_chat_log(
                "hermes_stream_sse_error",
                {**base_ctx, "stream": True, "error": err_msg},
            )
            log_chat_flow(session_id, flow_id, "Hermes SSE 중단", {"error": err_msg, "stream": True})
            yield f"event: error\ndata: {json.dumps({'error': err_msg}, ensure_ascii=False)}\n\n"
            return
        else:
            continue

    if not result_holder:
        write_chat_log("hermes_stream_missing_result", {**base_ctx, "stream": True})
        yield f"event: error\ndata: {json.dumps({'error': 'hermes: empty result'}, ensure_ascii=False)}\n\n"
        return

    result = result_holder[0]
    from service.chat_service import normalize_answer_text

    final_text = normalize_answer_text(result.get("final_response", ""))
    if on_complete:
        on_complete(final_text, result)
    done_ms = int((perf_counter() - started_time) * 1000)
    write_chat_log(
        "hermes_stream_timing",
        {
            **base_ctx,
            "stream": True,
            "total_latency_ms": done_ms,
            "first_token_latency_ms": first_token_latency_ms,
            "stream_delta_events": len(answer_parts),
            "final_response_chars": len(final_text),
            "model": model,
        },
    )
    log_chat_flow(
        session_id,
        flow_id,
        "Hermes 응답 전송 완료(SSE)",
        {"stream": True, "total_latency_ms": done_ms, "final_response_chars": len(final_text)},
    )
    write_chat_log(
        "hermes_stream_response",
        {
            **base_ctx,
            "stream": True,
            "answer_preview": hermes_answer_preview(final_text, 500),
            "wall_ended": datetime.now(korean_timezone).isoformat(),
        },
    )
    yield "event: done\ndata: [DONE]\n\n"
