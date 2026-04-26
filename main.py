import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import chat_router, chroma_router, mcp_router
from config import settings
from hermes import hermes_router


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    if settings.chat_backend == "hermes":
        if sys.version_info < (3, 11):
            raise RuntimeError(
                "CHAT_BACKEND=hermes needs Python >=3.11 (hermes-agent). "
                "Recreate the venv with `python3.12 -m venv .venv` or set CHAT_BACKEND=ollama in .env."
            )
        from service import hermes_chat
        from service.chat_service import write_chat_log

        hermes_chat.ensure_hermes_import()
        write_chat_log(
            "hermes_app_startup_ok",
            {
                "python": sys.version,
                "ollama_model": settings.ollama_model,
                "hermes_trace_log": settings.hermes_trace_log,
                **hermes_chat.hermes_config_snapshot(),
            },
        )
    yield


app = FastAPI(title="Personal Assistant AI API", lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_origin_regex=settings.cors_allow_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(chat_router)
app.include_router(chroma_router)
app.include_router(mcp_router)
app.include_router(hermes_router)


@app.get("/")
def health() -> dict:   
    return {"status": "ok", "env": settings.env}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=settings.env == "local")
