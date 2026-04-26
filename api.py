"""
API 라우터 전용 파일.
엔드포인트 정의만 두고 상세 로직은 service 폴더로 분리한다.
"""

from fastapi import APIRouter, Request

from config import settings
from service import chat_service, chroma_service, mcp_service
from service import hermes_chat
from service.schemas import ChatRequest, ChromaAddRequest, ChromaSearchRequest

chat_router = APIRouter(tags=["chat"])
chroma_router = APIRouter(prefix="/chroma", tags=["chroma"])
mcp_router = APIRouter(prefix="/mcp", tags=["mcp"])


# 채팅 요청을 처리하고 일반/스트리밍 응답을 반환한다.
@chat_router.post("/chat")
def chat(chat_request: ChatRequest, request_info: Request):
    return chat_service.process_chat_request(chat_request=chat_request, request_info=request_info)


# Hermes run_conversation()이 아직 끝나지 않은 요청(동일 API 프로세스). stuck 디버깅용.
@chat_router.get("/hermes-in-flight")
def hermes_in_flight() -> dict:
    items = hermes_chat.get_hermes_in_flight_items()
    return {
        "ok": True,
        "chat_backend": settings.chat_backend,
        "count": len(items),
        "items": items,
    }


# Chroma 저장소에 문서를 추가한다.
@chroma_router.post("/add")
def chroma_add(chroma_add_request: ChromaAddRequest):
    return chroma_service.add_document(
        text=chroma_add_request.text,
        metadata=chroma_add_request.metadata,
        document_id=chroma_add_request.id,
    )


# Chroma 저장소에서 문서를 검색한다.
@chroma_router.post("/search")
def chroma_search(chroma_search_request: ChromaSearchRequest):
    return chroma_service.search_documents(
        query=chroma_search_request.query,
        number_of_results=chroma_search_request.n_results,
    )


# MCP 작업 요약 정보를 조회한다.
@mcp_router.get("/job")
def mcp_job():
    return mcp_service.get_job_summary_data()


# MCP 예산 요약 정보를 조회한다.
@mcp_router.get("/budget")
def mcp_budget():
    return mcp_service.get_budget_summary_data()
