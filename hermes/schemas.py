from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class FeatureItem(BaseModel):
    key: str
    name: str
    description: str
    endpoints: List[str]
    docs: str


class HermesFeatureResponse(BaseModel):
    features: List[FeatureItem]


class SessionCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=100)
    personality: Optional[str] = None


class SessionInfo(BaseModel):
    session_id: str
    title: str
    personality: Optional[str] = None
    status: str


class ModelSelectRequest(BaseModel):
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)


class ConfigSetRequest(BaseModel):
    key: str = Field(min_length=1)
    value: Any


class GatewayStatusResponse(BaseModel):
    running: bool
    platforms: List[str]


class ToolToggleRequest(BaseModel):
    tool_name: str = Field(min_length=1)
    enabled: bool


class SkillCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=500)


class MemoryAddRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1)
    tags: List[str] = Field(default_factory=list)
    source: str = Field(default="manual", min_length=1, max_length=50)
    session_id: Optional[str] = None
    user_message: Optional[str] = None


class MemorySearchRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=20)


class CronJobCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    schedule: str = Field(min_length=1, max_length=50)
    action: str = Field(min_length=1, max_length=200)


class MCPServerConnectRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    url: str = Field(min_length=1, max_length=300)


class GenericAckResponse(BaseModel):
    status: str
    message: str
    data: Dict[str, Any] = Field(default_factory=dict)
