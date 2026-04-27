"""
하이브리드 AI 메모리: pending .md, Ollama nomic 임베딩, Chroma 벡터 검색.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from config import settings
from service import ollama_client

# chat_service와 순환 import 방지 — 로그·파일명 날짜는 동일 KST
_kst = timezone(timedelta(hours=9))


def _wlog(event: str, payload: Any) -> None:
    from service.chat_service import write_chat_log

    write_chat_log(event, payload)


_hybrid_chroma_collection = None
_chroma_hybrid_unavailable = False

try:
    import chromadb  # type: ignore
except Exception:
    chromadb = None  # type: ignore


def _get_collection():
    global _hybrid_chroma_collection, _chroma_hybrid_unavailable
    if _chroma_hybrid_unavailable:
        return None
    if _hybrid_chroma_collection is not None:
        return _hybrid_chroma_collection
    if chromadb is None:
        _chroma_hybrid_unavailable = True
        return None
    try:
        client = chromadb.PersistentClient(path=settings.chroma_path)
        _hybrid_chroma_collection = client.get_or_create_collection(
            name=settings.hybrid_memory_chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
    except Exception:
        _chroma_hybrid_unavailable = True
        return None
    return _hybrid_chroma_collection


def ensure_dirs() -> None:
    for path_str in (settings.hybrid_memory_pending_path, settings.hybrid_memory_archive_path):
        if path_str:
            os.makedirs(path_str, exist_ok=True)


# --- front matter (simple YAML subset) ---


def _parse_front_matter(raw: str) -> Tuple[Dict[str, Any], str]:
    if not raw.lstrip().startswith("---"):
        return {}, raw
    after_first = raw.split("---", 2)
    if len(after_first) < 3:
        return {}, raw
    fm_block = after_first[1]
    body = after_first[2].lstrip("\n")
    meta: Dict[str, Any] = {}
    for line in fm_block.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        k = key.strip().lower()
        v = value.strip()
        if k == "tags":
            if v.startswith("[") and v.endswith("]"):
                # JSON list
                try:
                    loaded = json.loads(v.replace("'", '"'))
                    meta["tags"] = loaded if isinstance(loaded, list) else []
                except Exception:
                    meta["tags"] = [p.strip() for p in v.strip("[]").split(",") if p.strip()]
            else:
                meta["tags"] = [p.strip() for p in v.split(",") if p.strip()]
        else:
            meta[k] = v.strip('"').strip("'")
    return meta, body


def _build_file_content(
    title: str,
    tags: List[str],
    chroma_doc_id: str,
    embed_kind: str,
    body: str,
) -> str:
    tags_json = json.dumps(tags, ensure_ascii=False)
    return (
        "---\n"
        f'title: {json.dumps(title, ensure_ascii=False)}\n'
        f"tags: {tags_json}\n"
        f"chroma_doc_id: {chroma_doc_id}\n"
        f"embed_kind: {embed_kind}\n"
        "---\n\n"
        f"{body}"
    )


# --- title / file naming ---

_SAFE_TITLE_RE = re.compile(r"[^\w\uac00-\ud7a3\s-]+", re.UNICODE)


def _filename_slug(title: str, max_len: int = 60) -> str:
    text = (title or "").strip().replace("\n", " ")
    text = _SAFE_TITLE_RE.sub("", text)
    text = re.sub(r"[\s_]+", "_", text).strip("_")
    if not text:
        text = "turn"
    if len(text) > max_len:
        text = text[:max_len].rstrip("_")
    return text or "turn"


# --- Chroma (manual embeddings) ---


def _chroma_upsert_light(
    chroma_doc_id: str,
    embed_text: str,
    document_body: str,
    metadata: Dict[str, Any],
) -> bool:
    col = _get_collection()
    if col is None:
        return False
    vector = ollama_client.ollama_embeddings(
        embed_text,
        model=settings.ollama_embed_model,
    )
    # Chroma metadata: only str, int, float, bool; flatten tags
    meta: Dict[str, Any] = {}
    for k, v in (metadata or {}).items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            meta[k] = v
        elif isinstance(v, list):
            meta[k] = ",".join(str(x) for x in v)[:2000]
        else:
            meta[k] = str(v)[:2000]
    col.upsert(
        ids=[chroma_doc_id],
        embeddings=[vector],
        documents=[document_body[:8000]],
        metadatas=[meta],
    )
    return True


def chroma_upsert_full_embedding(
    chroma_doc_id: str,
    full_text: str,
    metadata: Dict[str, Any],
) -> bool:
    """동기화 시 본문 전체를 nomic으로 임베딩해 동일 id로 Chroma를 갱신한다."""
    col = _get_collection()
    if col is None:
        return False
    vector = ollama_client.ollama_embeddings(
        full_text,
        model=settings.ollama_embed_model,
    )
    meta: Dict[str, Any] = {}
    for k, v in (metadata or {}).items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            meta[k] = v
        elif isinstance(v, list):
            meta[k] = ",".join(str(x) for x in v)[:2000]
        else:
            meta[k] = str(v)[:2000]
    col.upsert(
        ids=[chroma_doc_id],
        embeddings=[vector],
        documents=[full_text[:80000]],
        metadatas=[meta],
    )
    return True


def _chroma_query(user_question: str, n_results: int) -> List[Dict[str, Any]]:
    col = _get_collection()
    if col is None:
        return []
    try:
        qvec = ollama_client.ollama_embeddings(
            user_question,
            model=settings.ollama_embed_model,
        )
    except Exception as exc:
        _wlog("chroma_검색_오류", {"phase": "embed_query", "error": str(exc)})
        return []
    try:
        res = col.query(
            query_embeddings=[qvec],
            n_results=min(max(1, n_results), 50),
        )
    except Exception as exc:
        _wlog("chroma_검색_오류", {"phase": "chroma_query", "error": str(exc)})
        return []
    rows: List[Dict[str, Any]] = []
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    for i, doc_id in enumerate(ids or []):
        rows.append(
            {
                "id": doc_id,
                "text": docs[i] if i < len(docs) else "",
                "metadata": metas[i] if i < len(metas) else {},
                "distance": dists[i] if i < len(dists) else None,
            }
        )
    return rows


# --- recent pending files ---


def _pending_dir() -> Path:
    return Path(settings.hybrid_memory_pending_path)


def _read_recent_pending_md_text(max_age_seconds: int) -> str:
    ensure_dirs()
    directory = _pending_dir()
    if not directory.is_dir():
        return ""
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=max_age_seconds)
    candidates: List[Tuple[float, Path]] = []
    for p in directory.glob("*.md"):
        try:
            st = p.stat()
            mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)
        except OSError:
            continue
        if mtime < cutoff:
            continue
        candidates.append((st.st_mtime, p))
    candidates.sort(key=lambda x: x[0])
    chunks: List[str] = []
    for _, path in candidates:
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            continue
        chunks.append(f"### {path.name}\n{raw}\n")
    return "\n".join(chunks).strip()


# --- public: prompt context ---


def build_hybrid_memory_context(user_message: str, session_id: str, flow_id: str) -> str:
    if not settings.hybrid_memory_enabled and not settings.chroma_memory_persist_enabled:
        return ""
    n = settings.hybrid_memory_search_n
    db_rows = _chroma_query(user_message, n)
    recent_text = _read_recent_pending_md_text(settings.hybrid_memory_pending_max_age_seconds)

    lines: List[str] = ["[HYBRID_MEMORY_CONTEXT]"]

    if db_rows:
        lines.append("## 유사한 과거 기록 (ChromaDB)")
        for i, row in enumerate(db_rows, start=1):
            snippet = (row.get("text") or "").strip()
            if len(snippet) > 1200:
                snippet = snippet[:1200] + "…"
            meta = row.get("metadata") or {}
            extra = f" (id={row.get('id')}, d={row.get('distance')})" if row.get("distance") is not None else ""
            lines.append(f"{i}. {meta.get('title', '')}{extra}\n{snippet}\n")
    else:
        lines.append("## 유사한 과거 기록 (ChromaDB)\n(검색 결과 없음)")

    if recent_text:
        lines.append("## 최근 1시간 이내 pending 기록(원문)")
        lines.append(recent_text)
    else:
        lines.append("## 최근 1시간 이내 pending 기록(원문)\n(해당 없음)")

    lines.append("[/HYBRID_MEMORY_CONTEXT]")
    return "\n".join(lines)


# --- Ollama 메모리 판단 add → Chroma + pending(동기화 스크립트 호환) ---


def persist_memory_decision_add(
    *,
    user_message: str,
    answer: str,
    decision: Dict[str, Any],
    session_id: str = "",
    flow_id: str = "",
) -> None:
    """
    메모리_판단 action=add 시에만: 팩트 요약·태그·제목을 nomic으로 임베딩해 Chroma에 upsert,
    pending .md(답변 본문)는 동기화·1시간 맥락용.
    """
    if not settings.chroma_memory_persist_enabled:
        return
    if not (answer or "").strip():
        return
    title = (decision.get("title") or "").strip() or "memory"
    tags = decision.get("tags") if isinstance(decision.get("tags"), list) else []
    tags = [str(t) for t in tags][:10]
    content = (decision.get("content") or "").strip()
    if not content:
        content = f"{user_message}\n{answer}".strip()[:2000]

    ensure_dirs()
    date_str = datetime.now(_kst).strftime("%Y-%m-%d")
    base_slug = _filename_slug(title)
    chroma_doc_id = str(uuid.uuid4())
    # 검색/임베딩: 제목·태그·요약 본문
    embed_str = f"{title}\n{', '.join(tags) if tags else ''}\n{content}"
    # 컬렉션 문서 표시용
    document_for_chroma = f"{title}\n{content}\n\n[assistant]\n{answer.strip()[:6000]}"
    file_body = _build_file_content(
        title=title,
        tags=tags,
        chroma_doc_id=chroma_doc_id,
        embed_kind="light",
        body=answer.strip(),
    )
    out_dir = _pending_dir()
    out_path = out_dir / f"{date_str}_{base_slug}.md"
    suffix = 0
    while out_path.exists():
        suffix += 1
        out_path = out_dir / f"{date_str}_{base_slug}_{suffix}.md"
    try:
        out_path.write_text(file_body, encoding="utf-8")
    except OSError as exc:
        _wlog("chroma_pending_쓰기_오류", {"error": str(exc), "path": str(out_path)})
        return

    doc_meta: Dict[str, Any] = {
        "title": title[:2000],
        "source": "memory_decision_add",
        "session_id": session_id or "",
        "path": out_path.name,
        "flow_id": flow_id or "",
        "embed_kind": "light",
    }
    if tags:
        doc_meta["tags"] = tags

    ok = _chroma_upsert_light(
        chroma_doc_id=chroma_doc_id,
        embed_text=embed_str,
        document_body=document_for_chroma[:8000],
        metadata=doc_meta,
    )
    _wlog(
        "chroma_메모리저장",
        {
            "path": out_path.name,
            "chroma_ok": ok,
            "chroma_doc_id": chroma_doc_id,
            "title": title,
            "tags": tags,
            "session_id": session_id,
            "flow_id": flow_id,
        },
    )


def sync_pending_to_archive() -> Dict[str, Any]:
    """
    pending의 모든 .md를 읽어 본문 전체를 임베딩·Chroma 반영 후 archive로 옮긴다.
    새벽 cron 등에서 단독 실행한다.
    """
    ensure_dirs()
    pending = _pending_dir()
    archive = Path(settings.hybrid_memory_archive_path)
    archive.mkdir(parents=True, exist_ok=True)
    processed: List[str] = []
    errors: List[Dict[str, str]] = []
    if not pending.is_dir():
        return {"ok": True, "processed": [], "errors": [], "message": "pending missing"}
    for path in sorted(pending.glob("*.md")):
        try:
            raw = path.read_text(encoding="utf-8")
            meta, body = _parse_front_matter(raw)
            if not (body or "").strip():
                errors.append({"file": str(path), "error": "empty_body"})
                continue
            doc_id = str(meta.get("chroma_doc_id") or "").strip()
            if not doc_id:
                errors.append({"file": str(path), "error": "missing_chroma_doc_id"})
                continue
            title = str(meta.get("title") or path.stem)
            tags = meta.get("tags")
            if not isinstance(tags, list):
                tags = []
            full_text = body.strip()
            meta_out: Dict[str, Any] = {
                "title": title[:2000],
                "source": "hybrid_archive_full",
                "path": path.name,
                "embed_kind": "full",
            }
            if tags:
                meta_out["tags"] = tags
            ok = chroma_upsert_full_embedding(doc_id, full_text, meta_out)
            if not ok:
                errors.append({"file": str(path), "error": "chroma_upsert_failed"})
                continue
            new_content = _build_file_content(
                title=title,
                tags=[str(t) for t in tags],
                chroma_doc_id=doc_id,
                embed_kind="full",
                body=full_text,
            )
            dest = archive / path.name
            if dest.exists():
                stem, suf = dest.stem, dest.suffix
                dest = archive / f"{stem}_moved_{uuid.uuid4().hex[:8]}{suf}"
            dest.write_text(new_content, encoding="utf-8")
            path.unlink(missing_ok=True)
            processed.append(str(dest))
            _wlog("chroma_동기화_아카이브", {"from": path.name, "to": dest.name, "chroma_doc_id": doc_id})
        except Exception as exc:
            errors.append({"file": str(path), "error": str(exc)})
    return {"ok": not errors, "processed": processed, "errors": errors}
