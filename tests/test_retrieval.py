import unittest
from unittest.mock import MagicMock, patch

from app.retrieval.schema import RetrievedChunk
from app.retrieval.rrf import rrf_fuse
from app.retrieval.vector import vector_search
from app.retrieval.service import hybrid_search


class TestRetrieval(unittest.TestCase):
    def test_rrf_fuse_combines_vector_and_bm25(self):
        bm25_chunks = [
            RetrievedChunk(
                chunk_id="chunk-1",
                text="Vector databases are great for semantic search.",
                retriever="bm25",
                rank=1,
                score=3.5,
                source_id="src-1",
                title="Doc 1",
                url=None,
                source_type="internal",
                chunk_index=0,
            )
        ]

        vector_chunks = [
            RetrievedChunk(
                chunk_id="chunk-1",
                text="Vector databases are great for semantic search.",
                retriever="vector",
                rank=1,
                score=0.25,
                source_id="src-1",
                title="Doc 1",
                url=None,
                source_type="internal",
                chunk_index=0,
            )
        ]

        fused = rrf_fuse([bm25_chunks, vector_chunks])
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].chunk_id, "chunk-1")
        self.assertIn("bm25", fused[0].retrievers)
        self.assertIn("vector", fused[0].retrievers)
        self.assertEqual(fused[0].bm25_rank, 1)
        self.assertEqual(fused[0].vector_rank, 1)
        self.assertEqual(fused[0].vector_score, 0.25)

    @patch("app.retrieval.vector.get_vectorstore")
    def test_vector_search_with_score(self, mock_get_vectorstore):
        mock_doc = MagicMock()
        mock_doc.page_content = "Chroma is a vector DB."
        mock_doc.metadata = {
            "chunk_id": "c-123",
            "source_id": "s-456",
            "title": "Chroma Info",
            "url": "",
            "source_type": "internal",
            "chunk_index": 0,
        }

        mock_vs = MagicMock()
        mock_vs.similarity_search_with_score.return_value = [(mock_doc, 0.4496)]
        mock_get_vectorstore.return_value = mock_vs

        results = vector_search(query="vector db", k=5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk_id, "c-123")
        self.assertEqual(results[0].retriever, "vector")
        self.assertEqual(results[0].score, 0.4496)

    @patch("app.retrieval.service.vector_search")
    @patch("app.retrieval.service.bm25_search")
    def test_hybrid_search_logs_vector_failure(self, mock_bm25, mock_vector):
        mock_bm25.return_value = [
            RetrievedChunk(
                chunk_id="chunk-1",
                text="BM25 text",
                retriever="bm25",
                rank=1,
                score=1.0,
                source_id="src-1",
                title="Title",
                url=None,
                source_type="internal",
                chunk_index=0,
            )
        ]
        mock_vector.side_effect = Exception("Chroma connection refused")

        mock_db = MagicMock()
        results = hybrid_search(db=mock_db, query="test", top_k=5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk_id, "chunk-1")
        self.assertEqual(results[0].retrievers, ["bm25"])
        self.assertIsNone(results[0].vector_rank)


if __name__ == "__main__":
    unittest.main()
