"""
Chroma 저장/검색 기능을 담당한다.
Chroma 연결이 실패하면 메모리 리스트로 동작한다.
"""

import uuid

from config import settings

in_memory_documents = []
chroma_collection = None

try:
    import chromadb  # type: ignore

    chroma_client = chromadb.PersistentClient(path=settings.chroma_path)
    chroma_collection = chroma_client.get_or_create_collection(name="chat_memory")
except Exception:
    chroma_collection = None


# 문서를 Chroma 또는 메모리 저장소에 추가한다.
def add_document(text, metadata=None, document_id=None):
    saved_document_id = document_id or str(uuid.uuid4())

    if chroma_collection is not None:
        chroma_collection.add(
            ids=[saved_document_id],
            documents=[text],
            metadatas=[metadata or {}],
        )
    else:
        in_memory_documents.append(
            {
                "id": saved_document_id,
                "text": text,
                "metadata": metadata or {},
            }
        )

    return {"status": "ok", "id": saved_document_id}


# 질의어로 문서를 검색한다.
def search_documents(query, number_of_results):
    if chroma_collection is not None:
        query_result = chroma_collection.query(query_texts=[query], n_results=number_of_results)
        document_ids = query_result.get("ids", [[]])[0]
        documents = query_result.get("documents", [[]])[0]
        metadata_list = query_result.get("metadatas", [[]])[0]
        distance_list = query_result.get("distances", [[]])[0]

        search_rows = []
        for document_index, document_text in enumerate(documents):
            search_rows.append(
                {
                    "id": document_ids[document_index] if document_index < len(document_ids) else None,
                    "text": document_text,
                    "metadata": metadata_list[document_index] if document_index < len(metadata_list) else {},
                    "distance": distance_list[document_index] if document_index < len(distance_list) else None,
                }
            )
        return {"results": search_rows}

    lower_query = query.lower()
    matched_rows = [row for row in in_memory_documents if lower_query in row["text"].lower()]
    return {"results": matched_rows[:number_of_results]}
