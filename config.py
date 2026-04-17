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


settings = Settings()
