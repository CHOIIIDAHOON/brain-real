#!/usr/bin/env python3
"""
새벽 배치: pending 로그 전부를 본문 임베딩으로 Chroma에 반영하고 archive로 이동한다.

예: 0 3 * * * cd /path/to/real-brain && /path/to/venv/bin/python sync_hybrid_memory.py
"""

import json
import sys
from pathlib import Path

# 프로젝트 루트를 path에 넣는다
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from service.hybrid_memory_service import sync_pending_to_archive  # noqa: E402


def main() -> int:
    result = sync_pending_to_archive()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result.get("errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())
