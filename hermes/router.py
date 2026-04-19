from typing import Any, Dict, List

from fastapi import APIRouter

from .schemas import (
    ConfigSetRequest,
    CronJobCreateRequest,
    GenericAckResponse,
    GatewayStatusResponse,
    HermesFeatureResponse,
    MCPServerConnectRequest,
    MemoryAddRequest,
    MemorySearchRequest,
    ModelSelectRequest,
    SessionCreateRequest,
    SkillCreateRequest,
    ToolToggleRequest,
)
from .service import hermes_service

hermes_router = APIRouter(prefix="/hermes", tags=["hermes"])


# Hermes 기능 목록(카탈로그)을 조회한다.
@hermes_router.get("/features", response_model=HermesFeatureResponse)
def list_hermes_features() -> HermesFeatureResponse:
    return HermesFeatureResponse(features=hermes_service.list_features())


# 현재 세션 목록을 조회한다.
@hermes_router.get("/sessions")
def list_sessions() -> Dict[str, Any]:
    return {"items": hermes_service.list_sessions()}


# 새 세션을 생성한다.
@hermes_router.post("/sessions")
def create_session(req: SessionCreateRequest) -> Dict[str, Any]:
    row = hermes_service.create_session(title=req.title, personality=req.personality)
    return {"status": "ok", "session": row}


# 모든 세션을 초기화한다.
@hermes_router.post("/sessions/reset", response_model=GenericAckResponse)
def reset_sessions() -> GenericAckResponse:
    hermes_service.reset_sessions()
    return GenericAckResponse(status="ok", message="All sessions were reset.")


# 현재 선택된 모델 정보를 조회한다.
@hermes_router.get("/models/current")
def get_current_model() -> Dict[str, str]:
    return hermes_service.current_model()


# 사용할 모델(provider/model)을 변경한다.
@hermes_router.post("/models/select")
def select_model(req: ModelSelectRequest) -> Dict[str, Any]:
    row = hermes_service.select_model(provider=req.provider, model=req.model)
    return {"status": "ok", "model": row}


# 등록된 툴 활성화 상태를 조회한다.
@hermes_router.get("/tools")
def list_tools() -> Dict[str, Dict[str, bool]]:
    return {"tools": hermes_service.list_tools()}


# 특정 툴의 활성화 상태를 변경한다.
@hermes_router.post("/tools/toggle")
def toggle_tool(req: ToolToggleRequest) -> Dict[str, Any]:
    row = hermes_service.toggle_tool(tool_name=req.tool_name, enabled=req.enabled)
    return {"status": "ok", "tools": row}


# 게이트웨이 실행 상태와 플랫폼 목록을 조회한다.
@hermes_router.get("/gateway/status", response_model=GatewayStatusResponse)
def gateway_status() -> GatewayStatusResponse:
    row = hermes_service.gateway_status()
    return GatewayStatusResponse(running=row["running"], platforms=row["platforms"])


# 게이트웨이를 시작하고 대상 플랫폼을 설정한다.
@hermes_router.post("/gateway/start")
def gateway_start(platforms: List[str]) -> Dict[str, Any]:
    row = hermes_service.start_gateway(platforms=platforms)
    return {"status": "ok", "gateway": row}


# 게이트웨이를 중지한다.
@hermes_router.post("/gateway/stop")
def gateway_stop() -> Dict[str, Any]:
    row = hermes_service.stop_gateway()
    return {"status": "ok", "gateway": row}


# 등록된 스킬 목록을 조회한다.
@hermes_router.get("/skills")
def list_skills() -> Dict[str, List[Dict[str, str]]]:
    return {"items": hermes_service.list_skills()}


# 새 스킬 메타데이터를 등록한다.
@hermes_router.post("/skills")
def create_skill(req: SkillCreateRequest) -> Dict[str, Any]:
    row = hermes_service.create_skill(name=req.name, description=req.description)
    return {"status": "ok", "skill": row}


# 저장된 메모리 목록을 조회한다.
@hermes_router.get("/memory")
def list_memory() -> Dict[str, Any]:
    return {"items": hermes_service.list_memory()}


# 새 메모리 항목을 저장한다.
@hermes_router.post("/memory")
def add_memory(req: MemoryAddRequest) -> Dict[str, Any]:
    row = hermes_service.add_memory(title=req.title, content=req.content, tags=req.tags)
    return {"status": "ok", "memory": row}


# 메모리를 키워드로 검색한다.
@hermes_router.post("/memory/search")
def search_memory(req: MemorySearchRequest) -> Dict[str, Any]:
    rows = hermes_service.search_memory(query=req.query, limit=req.limit)
    return {"results": rows}


# 등록된 크론 작업 목록을 조회한다.
@hermes_router.get("/cron/jobs")
def list_cron_jobs() -> Dict[str, Any]:
    return {"items": hermes_service.list_cron_jobs()}


# 새 크론 작업을 생성한다.
@hermes_router.post("/cron/jobs")
def create_cron_job(req: CronJobCreateRequest) -> Dict[str, Any]:
    row = hermes_service.create_cron_job(name=req.name, schedule=req.schedule, action=req.action)
    return {"status": "ok", "job": row}


# 연결된 MCP 서버 목록을 조회한다.
@hermes_router.get("/mcp/servers")
def list_mcp_servers() -> Dict[str, Any]:
    return {"items": hermes_service.list_mcp_servers()}


# MCP 서버 연결 정보를 등록한다.
@hermes_router.post("/mcp/servers/connect")
def connect_mcp_server(req: MCPServerConnectRequest) -> Dict[str, Any]:
    row = hermes_service.connect_mcp_server(name=req.name, url=req.url)
    return {"status": "ok", "mcp_server": row}


# 현재 설정값 전체를 조회한다.
@hermes_router.get("/config")
def get_config() -> Dict[str, Any]:
    return {"config": hermes_service.get_config()}


# 단일 설정값(key/value)을 저장한다.
@hermes_router.post("/config/set")
def set_config(req: ConfigSetRequest) -> Dict[str, Any]:
    row = hermes_service.set_config(key=req.key, value=req.value)
    return {"status": "ok", "config": row}
