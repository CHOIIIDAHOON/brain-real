import uuid
from typing import Any, Dict, List

from .schemas import FeatureItem


class HermesService:
    def __init__(self) -> None:
        self._current_model: Dict[str, str] = {"provider": "openai", "model": "gpt-4o-mini"}
        self._gateway: Dict[str, Any] = {"running": False, "platforms": []}
        self._tools: Dict[str, bool] = {}
        self._configs: Dict[str, Any] = {}
        self._skills: List[Dict[str, str]] = []
        self._memories: List[Dict[str, Any]] = []
        self._cron_jobs: List[Dict[str, str]] = []
        self._mcp_servers: List[Dict[str, str]] = []
        self._sessions: List[Dict[str, Any]] = []

    def list_features(self) -> List[FeatureItem]:
        return [
            FeatureItem(
                key="sessions",
                name="CLI Session Control",
                description="Start/reset conversations and track active sessions.",
                endpoints=["GET /hermes/sessions", "POST /hermes/sessions", "POST /hermes/sessions/reset"],
                docs="https://hermes-agent.nousresearch.com/docs/user-guide/cli",
            ),
            FeatureItem(
                key="models",
                name="Model Management",
                description="Select provider and model dynamically without code changes.",
                endpoints=["GET /hermes/models/current", "POST /hermes/models/select"],
                docs="https://hermes-agent.nousresearch.com/docs/user-guide/configuration",
            ),
            FeatureItem(
                key="tools",
                name="Tools and Toolsets",
                description="Enable or disable tool usage by tool name.",
                endpoints=["GET /hermes/tools", "POST /hermes/tools/toggle"],
                docs="https://hermes-agent.nousresearch.com/docs/user-guide/features/tools",
            ),
            FeatureItem(
                key="gateway",
                name="Messaging Gateway",
                description="Control gateway-like status for Telegram/Discord/Slack style integrations.",
                endpoints=["GET /hermes/gateway/status", "POST /hermes/gateway/start", "POST /hermes/gateway/stop"],
                docs="https://hermes-agent.nousresearch.com/docs/user-guide/messaging",
            ),
            FeatureItem(
                key="skills",
                name="Skills System",
                description="Create and list agent skills metadata.",
                endpoints=["GET /hermes/skills", "POST /hermes/skills"],
                docs="https://hermes-agent.nousresearch.com/docs/user-guide/features/skills",
            ),
            FeatureItem(
                key="memory",
                name="Memory",
                description="Add and search persistent memory records.",
                endpoints=["GET /hermes/memory", "POST /hermes/memory", "POST /hermes/memory/search"],
                docs="https://hermes-agent.nousresearch.com/docs/user-guide/features/memory",
            ),
            FeatureItem(
                key="cron",
                name="Cron Scheduling",
                description="Register and inspect scheduled automation jobs.",
                endpoints=["GET /hermes/cron/jobs", "POST /hermes/cron/jobs"],
                docs="https://hermes-agent.nousresearch.com/docs/user-guide/features/cron",
            ),
            FeatureItem(
                key="mcp",
                name="MCP Integration",
                description="Register and inspect connected MCP servers.",
                endpoints=["GET /hermes/mcp/servers", "POST /hermes/mcp/servers/connect"],
                docs="https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp",
            ),
            FeatureItem(
                key="config",
                name="Configuration",
                description="Set and fetch key-value based runtime config.",
                endpoints=["GET /hermes/config", "POST /hermes/config/set"],
                docs="https://hermes-agent.nousresearch.com/docs/user-guide/configuration",
            ),
        ]

    def create_session(self, title: str, personality: str | None) -> Dict[str, Any]:
        row = {
            "session_id": str(uuid.uuid4()),
            "title": title,
            "personality": personality,
            "status": "active",
        }
        self._sessions.append(row)
        return row

    def list_sessions(self) -> List[Dict[str, Any]]:
        return self._sessions

    def reset_sessions(self) -> None:
        self._sessions = []

    def select_model(self, provider: str, model: str) -> Dict[str, str]:
        self._current_model = {"provider": provider, "model": model}
        return self._current_model

    def current_model(self) -> Dict[str, str]:
        return self._current_model

    def list_tools(self) -> Dict[str, bool]:
        return self._tools

    def toggle_tool(self, tool_name: str, enabled: bool) -> Dict[str, bool]:
        self._tools[tool_name] = enabled
        return self._tools

    def gateway_status(self) -> Dict[str, Any]:
        return self._gateway

    def start_gateway(self, platforms: List[str]) -> Dict[str, Any]:
        self._gateway = {"running": True, "platforms": platforms}
        return self._gateway

    def stop_gateway(self) -> Dict[str, Any]:
        self._gateway["running"] = False
        return self._gateway

    def list_skills(self) -> List[Dict[str, str]]:
        return self._skills

    def create_skill(self, name: str, description: str) -> Dict[str, str]:
        row = {"name": name, "description": description}
        self._skills.append(row)
        return row

    def list_memory(self) -> List[Dict[str, Any]]:
        return self._memories

    def add_memory(self, title: str, content: str, tags: List[str]) -> Dict[str, Any]:
        row = {
            "memory_id": str(uuid.uuid4()),
            "title": title,
            "content": content,
            "tags": tags,
        }
        self._memories.append(row)
        return row

    def search_memory(self, query: str, limit: int) -> List[Dict[str, Any]]:
        query_lc = query.lower()
        matches = [
            item
            for item in self._memories
            if query_lc in item["title"].lower() or query_lc in item["content"].lower()
        ]
        return matches[:limit]

    def list_cron_jobs(self) -> List[Dict[str, str]]:
        return self._cron_jobs

    def create_cron_job(self, name: str, schedule: str, action: str) -> Dict[str, str]:
        row = {"job_id": str(uuid.uuid4()), "name": name, "schedule": schedule, "action": action}
        self._cron_jobs.append(row)
        return row

    def list_mcp_servers(self) -> List[Dict[str, str]]:
        return self._mcp_servers

    def connect_mcp_server(self, name: str, url: str) -> Dict[str, str]:
        row = {"name": name, "url": url}
        self._mcp_servers.append(row)
        return row

    def set_config(self, key: str, value: Any) -> Dict[str, Any]:
        self._configs[key] = value
        return self._configs

    def get_config(self) -> Dict[str, Any]:
        return self._configs


hermes_service = HermesService()
