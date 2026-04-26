#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

APP_HOST="${APP_HOST:-127.0.0.1}"
APP_PORT="${APP_PORT:-8000}"
LOG_FILE="${LOG_FILE:-app.log}"

DECISION_MODEL_MSG="${CHAT_MEMORY_DECISION_MODEL:-chat model fallback}"

# --- 기존 uvicorn 종료 (TERM → 남으면 -9) ------------------------------------
echo "[restart] stopping old uvicorn (main:app)…"
pkill -f "uvicorn main:app" 2>/dev/null || true
pkill -f "uvicorn.*main:app" 2>/dev/null || true
sleep 2
# 늦게 죽는·좀비에 가까운 잔류: SIGKILL
pkill -9 -f "uvicorn main:app" 2>/dev/null || true
pkill -9 -f "uvicorn.*main:app" 2>/dev/null || true
sleep 1
# 이 포트에 LISTEN 중인 프로세스(패턴이 달라도) 마무리
if command -v fuser >/dev/null 2>&1; then
  fuser -k "${APP_PORT}/tcp" 2>/dev/null || true
  fuser -k -9 "${APP_PORT}/tcp" 2>/dev/null || true
fi
if command -v lsof >/dev/null 2>&1; then
  pids_lsof=$(lsof -t -iTCP:"${APP_PORT}" -sTCP:LISTEN 2>/dev/null || true)
  for p in ${pids_lsof}; do
    if [ -n "${p}" ]; then
      kill "${p}" 2>/dev/null || true
      sleep 1
      kill -9 "${p}" 2>/dev/null || true
    fi
  done
fi
sleep 1
# 여전히 포트가 쓰이면: systemd/다른 유저/수동 nohup 등 — 로그로 보이게만 함
if command -v ss >/dev/null 2>&1; then
  if ss -tlnp 2>/dev/null | grep -qE ":${APP_PORT}\s"; then
    echo "[restart] warn: :${APP_PORT} 아직 점유 중. 아래에 LISTEN이 보이면 PID를 수동 kill 하거나( kill -9 PID ) 같은 포트 쓰는 서비스를 끄세요."
    ss -tlnp 2>/dev/null | grep -E ":${APP_PORT}\s" || true
  fi
fi

echo "[restart] starting uvicorn on ${APP_HOST}:${APP_PORT}..."
echo "[restart] memory decision model=${DECISION_MODEL_MSG}"
nohup uvicorn main:app --host "$APP_HOST" --port "$APP_PORT" > "$LOG_FILE" 2>&1 &

sleep 1
PID="$(pgrep -f "uvicorn main:app" | head -n 1 || true)"

if [ -n "$PID" ]; then
  echo "[restart] started pid=$PID"
  echo "[restart] log: $PROJECT_DIR/$LOG_FILE"
else
  echo "[restart] failed to start. check log: $PROJECT_DIR/$LOG_FILE"
  exit 1
fi
