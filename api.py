import json
import uuid
from typing import Any, Dict, Iterator, List, Optional
from urllib import error, request

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config import settings
from mcp.connection import get_budget_summary, get_job_summary


chat_router = APIRouter(tags=["chat"])
chroma_router = APIRouter(prefix="/chroma", tags=["chroma"])
mcp_router = APIRouter(prefix="/mcp", tags=["mcp"])


_chroma_collection = None
_memory_store: List[Dict[str, Any]] = []
_stream_headers = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

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


class ChromaAddRequest(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None
    id: Optional[str] = None


class ChromaSearchRequest(BaseModel):
    query: str
    n_results: int = Field(default=3, ge=1, le=20)


def _chat_stream(url: str, payload: Dict[str, Any]) -> Iterator[str]:
    req_obj = request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
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
                for ch in piece:
                    yield f"data: {json.dumps({'text': ch}, ensure_ascii=False)}\n\n"
                if data.get("done"):
                    yield "event: done\ndata: [DONE]\n\n"
                    break
    except Exception as ex:
        yield f"event: error\ndata: {json.dumps({'error': str(ex)}, ensure_ascii=False)}\n\n"


def _client_meta(req: Request) -> Dict[str, str]:
    return {
        "client": req.headers.get("x-dabo-client", "unknown"),
        "build": req.headers.get("x-dabo-build", "unknown"),
        "api_version": req.headers.get("x-api-version", "unknown"),
    }


@chat_router.post("/chat")
def chat(req: ChatRequest, request_info: Request) -> Any:
    url = f"{settings.ollama_base_url}/api/generate"
    payload = {
        "model": req.model or settings.ollama_model,
        "prompt": req.message,
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
    if req.stream:
        return StreamingResponse(
            _chat_stream(url, payload),
            media_type="text/event-stream; charset=utf-8",
            headers={
                **_stream_headers,
                "X-DABO-Client": client["client"],
                "X-DABO-Build": client["build"],
                "X-API-Version": client["api_version"],
            },
        )

    try:
        req_obj = request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req_obj, timeout=settings.ollama_timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
        return {
            "answer": body.get("response", ""),
            "model": payload["model"],
            "client": client,
        }
    except error.URLError as ex:
        raise HTTPException(status_code=502, detail=f"Ollama connection failed: {ex}") from ex
    except Exception as ex:
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
