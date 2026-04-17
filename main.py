from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import chat_router, chroma_router, mcp_router
from config import settings


app = FastAPI(title="Personal Assistant AI API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(chat_router)
app.include_router(chroma_router)
app.include_router(mcp_router)


@app.get("/")
def health() -> dict:
    return {"status": "ok", "env": settings.env}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=settings.env == "local")
