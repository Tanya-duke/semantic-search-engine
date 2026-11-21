"""Dataset preparation and Chroma-backed semantic retrieval."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


DATASET_ID = "fancyzhx/ag_news"
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "ag_news_articles"
CATEGORY_NAMES = ("World", "Sports", "Business", "Sci/Tech")


def chunk_text(text: str, chunk_size: int = 180, overlap: int = 30) -> list[str]:
    """Split text into overlapping word windows, preserving short documents."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between zero and chunk_size - 1")

    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    return [
        " ".join(words[start : start + chunk_size])
        for start in range(0, len(words), step)
    ]


class SemanticSearchEngine:
    """Load AG News, create a vector index, and retrieve relevant chunks."""

    def __init__(
        self,
        dataset_limit: int = 500,
        chunk_size: int = 180,
        overlap: int = 30,
        db_path: str | Path = "chroma_db",
        model_id: str = MODEL_ID,
    ) -> None:
        if dataset_limit < 1:
            raise ValueError("dataset_limit must be greater than zero")

        import chromadb
        from datasets import load_dataset
        from fastembed import TextEmbedding

        self.dataset_limit = dataset_limit
        self.model = TextEmbedding(model_name=model_id)
        dataset = load_dataset(DATASET_ID, split=f"train[:{dataset_limit}]")
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        ids: list[str] = []

        for article_id, row in enumerate(dataset):
            for chunk_id, chunk in enumerate(
                chunk_text(row["text"], chunk_size=chunk_size, overlap=overlap)
            ):
                documents.append(chunk)
                category_id = int(row["label"])
                category = (
                    CATEGORY_NAMES[category_id]
                    if 0 <= category_id < len(CATEGORY_NAMES)
                    else "Unknown"
                )
                metadatas.append(
                    {
                        "article_id": article_id,
                        "chunk_id": chunk_id,
                        "category": category,
                    }
                )
                ids.append(f"article-{article_id}-chunk-{chunk_id}")

        self.article_count = len(dataset)
        self.indexed_chunks = len(documents)
        self.client = chromadb.PersistentClient(path=str(Path(db_path)))
        fingerprint = f"{DATASET_ID}:{dataset_limit}:{chunk_size}:{overlap}:{model_id}"

        collection = self.client.get_or_create_collection(
            COLLECTION_NAME,
            metadata={
                "hnsw:space": "cosine",
                "index_fingerprint": fingerprint,
            },
        )
        if (
            (collection.metadata or {}).get("index_fingerprint") != fingerprint
            or collection.count() != self.indexed_chunks
        ):
            self.client.delete_collection(COLLECTION_NAME)
            collection = self.client.create_collection(
                COLLECTION_NAME,
                metadata={
                    "hnsw:space": "cosine",
                    "index_fingerprint": fingerprint,
                },
            )
            for offset in range(0, len(documents), 64):
                batch_documents = documents[offset : offset + 64]
                embeddings = [
                    embedding.tolist()
                    for embedding in self.model.embed(batch_documents, batch_size=64)
                ]
                collection.add(
                    ids=ids[offset : offset + 64],
                    documents=batch_documents,
                    metadatas=metadatas[offset : offset + 64],
                    embeddings=embeddings,
                )

        self.collection = collection

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Return the closest matching text chunks and their metadata."""
        if not query.strip():
            return []

        limit = max(1, min(int(limit), 20))
        query_embedding = next(self.model.embed([query])).tolist()
        matches = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(limit, self.indexed_chunks),
            include=["documents", "metadatas", "distances"],
        )

        results = []
        for document, metadata, distance in zip(
            matches["documents"][0],
            matches["metadatas"][0],
            matches["distances"][0],
        ):
            results.append(
                {
                    "text": document,
                    "category": metadata["category"],
                    "article_id": metadata["article_id"],
                    "similarity": max(0.0, min(1.0, 1.0 - float(distance))),
                }
            )
        return results


def configured_dataset_limit() -> int:
    """Read the number of dataset articles from the environment."""
    return max(1, int(os.getenv("DATASET_LIMIT", "500")))