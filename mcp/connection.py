import json
from urllib import error, request

from fastapi import HTTPException

from config import settings


def _fetch_mcp(path: str) -> dict:
    url = f"{settings.mcp_base_url.rstrip('/')}{path}"
    try:
        req_obj = request.Request(url=url, method="GET")
        with request.urlopen(req_obj, timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.URLError as ex:
        raise HTTPException(status_code=502, detail=f"MCP connection failed: {ex}") from ex
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"MCP parse error: {ex}") from ex


def get_job_summary() -> dict:
    return _fetch_mcp("/job")


def get_budget_summary() -> dict:
    return _fetch_mcp("/budget")


__all__ = ["get_job_summary", "get_budget_summary"]
