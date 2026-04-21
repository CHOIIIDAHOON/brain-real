import os


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
                "응답 길이 규칙:\n"
                "- 기본은 3줄 이내, 명령어/결론 우선.\n"
                '- 사용자가 "자세히", "왜", "설명", "가이드", "단계별"을 요청한 경우에만 상세 설명 허용.\n'
                "- 에러/장애 해결 상황에서는 원인 1줄 + 조치 명령어만 먼저 제시.\n"
                "- 추가 설명은 사용자가 요청할 때만 제공."
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


settings = Settings()
