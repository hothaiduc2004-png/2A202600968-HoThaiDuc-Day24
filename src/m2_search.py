from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    try:
        from underthesea import word_tokenize
        segmented = word_tokenize(text, format="text")
        return segmented.replace("_", " ")
    except Exception:
        return text.replace("_", " ")


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = chunks
        self.corpus_tokens = [segment_vietnamese(chunk["text"]).split() for chunk in chunks]
        try:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi(self.corpus_tokens)
        except Exception:
            self.bm25 = None

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None:
            return []
        tokenized_query = segment_vietnamese(query).split()
        if not tokenized_query:
            return []
        scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = []
        for idx in ranked[:top_k]:
            score = float(scores[idx])
            if score <= 0:
                continue
            chunk = self.documents[idx]
            results.append(SearchResult(text=chunk["text"], score=score,
                                        metadata=chunk.get("metadata", {}), method="bm25"))
        return results


class DenseSearch:
    def __init__(self):
        self.client = None
        self._encoder = None
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            sock.connect((QDRANT_HOST, QDRANT_PORT))
            sock.close()
            from qdrant_client import QdrantClient
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, check_compatibility=False)
        except Exception:
            self.client = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        if self.client is None:
            return

        try:
            from qdrant_client.models import Distance, VectorParams, PointStruct
            self.client.recreate_collection(collection_name=collection,
                                            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE))
        except Exception:
            try:
                self.client.delete_collection(collection_name=collection)
            except Exception:
                pass
            try:
                from qdrant_client.models import Distance, VectorParams, PointStruct
                self.client.create_collection(collection_name=collection,
                                              vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE))
            except Exception:
                self.client = None
                return

        if self.client is None:
            return

        texts = [c["text"] for c in chunks]
        encoder = self._get_encoder()
        try:
            vectors = encoder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        except TypeError:
            vectors = encoder.encode(texts, show_progress_bar=True)
        points = []
        for i, (text, chunk) in enumerate(zip(texts, chunks)):
            vector = vectors[i].tolist() if hasattr(vectors[i], 'tolist') else list(vectors[i])
            payload = {**chunk.get("metadata", {}), "text": text}
            points.append(PointStruct(id=i, vector=vector, payload=payload))
        try:
            self.client.upsert(collection_name=collection, points=points)
        except TypeError:
            self.client.upsert(collection=collection, points=points)
        except Exception:
            self.client = None

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        if self.client is None:
            return []

        try:
            query_vector = self._get_encoder().encode(query, convert_to_numpy=True).tolist()
        except TypeError:
            query_vector = self._get_encoder().encode(query).tolist()

        response = None
        try:
            response = self.client.search(collection_name=collection, query_vector=query_vector, limit=top_k)
        except Exception:
            try:
                response = self.client.query_points(collection_name=collection, query=query_vector, limit=top_k)
            except Exception:
                return []

        points = getattr(response, "points", response)
        results = []
        for pt in points or []:
            score = float(getattr(pt, "score", 0.0))
            payload = getattr(pt, "payload", {}) or pt
            text = payload.get("text", "")
            if not text:
                continue
            results.append(SearchResult(text=text, score=score, metadata=payload, method="dense"))
        return results


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    rrf_scores: dict[str, dict] = {}
    for result_list in results_list:
        for rank, result in enumerate(result_list):
            key = result.text
            if key not in rrf_scores:
                rrf_scores[key] = {"score": 0.0, "result": result}
            rrf_scores[key]["score"] += 1.0 / (k + rank + 1)

    merged = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    results = []
    for item in merged:
        result = item["result"]
        results.append(SearchResult(
            text=result.text,
            score=float(item["score"]),
            metadata=result.metadata,
            method="hybrid",
        ))
    return results


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        if not dense_results:
            return bm25_results[:top_k]
        if not bm25_results:
            return dense_results[:top_k]
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
