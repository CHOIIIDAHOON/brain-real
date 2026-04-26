import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# 프로젝트 루트의 .env를 os.environ에 반영한다 (import 시점에 한 번).
load_dotenv(Path(__file__).resolve().parent / ".env")


class Settings:
    def __init__(self) -> None:
        self.env = os.getenv("APP_ENV", "local")

        if self.env == "server":
            self.host = os.getenv("APP_HOST", "0.0.0.0")
            self.port = int(os.getenv("APP_PORT", "8000"))
            self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
            self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3")
        else:
            self.host = os.getenv("APP_HOST", "127.0.0.1")
            self.port = int(os.getenv("APP_PORT", "8000"))
            self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
            self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3")

        self.chroma_path = os.getenv("CHROMA_PATH", "./chromaDB")
        self.mcp_base_url = os.getenv("MCP_BASE_URL", "http://api.strangeway.life")
        self.ollama_keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "24h")
        self.ollama_timeout_seconds = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "300"))
        cors_origins = os.getenv(
            "CORS_ALLOW_ORIGINS",
            "http://localhost:3000,http://localhost:5000,http://localhost:8000,http://127.0.0.1:3000,http://127.0.0.1:5000,http://127.0.0.1:8000",
        )
        self.cors_allow_origins = [origin.strip() for origin in cors_origins.split(",") if origin.strip()]
        self.cors_allow_origin_regex = os.getenv(
            "CORS_ALLOW_ORIGIN_REGEX",
            r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        )
        self.default_system_prompt = os.getenv(
            "DEFAULT_SYSTEM_PROMPT",
            (
                "Daboa AI: executive-assistant style, concise and useful.\n"
                "No emojis/decorative symbols unless asked. Korean: consistent 존댓말; English: polite "
                "professional tone (no slang).\n"
                "Address the real question; infer intent; do not dodge. If the user is torn: one-line "
                "tradeoff, a default with brief reason, one concrete next step.\n"
                "Default ~3 lines: conclusion or exact action first, no filler. Longer only if they ask "
                'for detail (explain, why, in detail, step by step, etc.). Errors: likely cause + fix first; '
                "more only on request. No extra background after the answer unless asked."
            ),
        )
        self.chat_prompt_max_turns = int(os.getenv("CHAT_PROMPT_MAX_TURNS", "6"))
        self.chat_memory_context_first_turn_only = os.getenv(
            "CHAT_MEMORY_CONTEXT_FIRST_TURN_ONLY",
            "true",
        ).lower() in {"1", "true", "yes", "on"}
        self.chat_first_scan_results = int(os.getenv("CHAT_FIRST_SCAN_RESULTS", "3"))
        self.chat_memory_min_len = int(os.getenv("CHAT_MEMORY_MIN_LEN", "20"))
        keywords = os.getenv("CHAT_MEMORY_KEYWORDS", "기억,중요,규칙,선호,일정,해야함")
        self.chat_memory_keywords = [item.strip() for item in keywords.split(",") if item.strip()]
        self.chat_memory_decision_mode = os.getenv("CHAT_MEMORY_DECISION_MODE", "ollama")
        self.chat_memory_decision_model = os.getenv("CHAT_MEMORY_DECISION_MODEL", "").strip()
        self.chat_memory_decision_max_chars = int(os.getenv("CHAT_MEMORY_DECISION_MAX_CHARS", "600"))
        # <= 0 means "use OLLAMA_TIMEOUT_SECONDS" for decision calls.
        self.chat_memory_decision_timeout_seconds = int(os.getenv("CHAT_MEMORY_DECISION_TIMEOUT_SECONDS", "0"))
        decision_num_predict = os.getenv("CHAT_MEMORY_DECISION_NUM_PREDICT", "").strip()
        if not decision_num_predict:
            # Backward compatibility for earlier env name.
            decision_num_predict = os.getenv("CHAT_MEMORY_DECISION_MAX_TOKENS", "80").strip()
        self.chat_memory_decision_num_predict = int(decision_num_predict)
        self.chat_memory_skip_short_question_len = int(os.getenv("CHAT_MEMORY_SKIP_SHORT_QUESTION_LEN", "18"))
        self.chat_memory_session_cooldown_seconds = int(os.getenv("CHAT_MEMORY_SESSION_COOLDOWN_SECONDS", "10"))
        self.chat_log_enabled = os.getenv("CHAT_LOG_ENABLED", "true").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.chat_log_path = os.getenv("CHAT_LOG_PATH", "./logs/chat_events.jsonl")

        # ollama = 이전 직접 /api/generate, hermes = Nous Hermes Agent(AIAgent) 경유 (hermes 는 Python 3.11+)
        _chat_backend_env = os.getenv("CHAT_BACKEND", "").strip().lower()
        if _chat_backend_env:
            self.chat_backend = _chat_backend_env
        else:
            self.chat_backend = "hermes" if sys.version_info >= (3, 11) else "ollama"
        # OpenAI 클라이언트에 넘기는 키. 로컬 Ollama는 임의 비공백 값이면 된다( run_agent: api_key and base_url ).
        self.hermes_ollama_api_key = (os.getenv("HERMES_OLLAMA_API_KEY", "not-needed") or "not-needed").strip()
        self.hermes_max_iterations = int(os.getenv("HERMES_MAX_ITERATIONS", "30"))
        self.hermes_skip_context_files = os.getenv("HERMES_SKIP_CONTEXT_FILES", "true").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.hermes_skip_memory = os.getenv("HERMES_SKIP_MEMORY", "false").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        _hp = os.getenv("HERMES_PLATFORM", "").strip()
        self.hermes_platform = _hp or None
        self.hermes_disabled_toolsets = os.getenv("HERMES_DISABLED_TOOLSETS", "").strip()
        # 쉼표. 비어 있으면 툴 전부(필터는 disabled와 조합). 지정 시 초기화·툴 수 감소로 첫 응답이 빨라질 수 있음(예: reasoning, web)
        self.hermes_enabled_toolsets = os.getenv("HERMES_ENABLED_TOOLSETS", "").strip()
        # Hermes 콜백(툴/상태 등) — False면 hermes_cb_* / log_chat_flow 상세 끔
        self.hermes_trace_log = os.getenv("HERMES_TRACE_LOG", "true").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }


settings = Settings()
