"""
MCP 관련 요약 조회 기능을 담당한다.
"""

from mcp.connection import get_budget_summary, get_job_summary


# 작업 현황 MCP 요약을 반환한다.
def get_job_summary_data():
    return get_job_summary()


# 예산 MCP 요약을 반환한다.
def get_budget_summary_data():
    return get_budget_summary()
