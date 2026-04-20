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
CHAT_MEMORY_DECISION_MODEL="${CHAT_MEMORY_DECISION_MODEL:-exaone3.5:2.4b}"
CHAT_MEMORY_DECISION_NUM_PREDICT="${CHAT_MEMORY_DECISION_NUM_PREDICT:-64}"
CHAT_MEMORY_DECISION_TIMEOUT_SECONDS="${CHAT_MEMORY_DECISION_TIMEOUT_SECONDS:-10}"

export CHAT_MEMORY_DECISION_MODEL
export CHAT_MEMORY_DECISION_NUM_PREDICT
export CHAT_MEMORY_DECISION_TIMEOUT_SECONDS

echo "[restart] stopping old uvicorn..."
pkill -f "uvicorn main:app" || true
sleep 1

echo "[restart] starting uvicorn on ${APP_HOST}:${APP_PORT}..."
echo "[restart] memory decision model=${CHAT_MEMORY_DECISION_MODEL}, num_predict=${CHAT_MEMORY_DECISION_NUM_PREDICT}, timeout=${CHAT_MEMORY_DECISION_TIMEOUT_SECONDS}s"
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
