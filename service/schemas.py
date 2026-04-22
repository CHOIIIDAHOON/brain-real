"""
API 요청 본문 모델 모음.
라우터 파일에서는 이 모델만 가져와서 사용한다.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


# 채팅 요청 데이터를 검증한다.
class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = None
    think: Optional[bool] = None
    keep_alive: Optional[str] = None
    stream: bool = False
    system_prompt: Optional[str] = None
    session_id: Optional[str] = None
    new_chat: bool = False


# Chroma 문서 추가 요청 데이터를 검증한다.
class ChromaAddRequest(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None
    id: Optional[str] = None


# Chroma 검색 요청 데이터를 검증한다.
class ChromaSearchRequest(BaseModel):
    query: str
    n_results: int = Field(default=3, ge=1, le=20)
