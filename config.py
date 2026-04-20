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
        self.chat_first_scan_results = int(os.getenv("CHAT_FIRST_SCAN_RESULTS", "3"))
        self.chat_memory_min_len = int(os.getenv("CHAT_MEMORY_MIN_LEN", "20"))
        keywords = os.getenv("CHAT_MEMORY_KEYWORDS", "기억,중요,규칙,선호,일정,해야함")
        self.chat_memory_keywords = [item.strip() for item in keywords.split(",") if item.strip()]


settings = Settings()
