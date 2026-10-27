"""Checks that do not require a running Milvus or Ollama service."""
import unittest
from unittest.mock import patch

from app import rag
from app import main


class RagTests(unittest.TestCase):
    def test_blank_question_does_not_call_services(self):
        with patch.object(rag, "search") as search:
            result = rag.ask("  ")
        self.assertEqual(result["sources"], [])
        self.assertTrue(result["answer"])
        search.assert_not_called()

    def test_ask_passes_sources_to_generation(self):
        sources = [{"id": 1, "question": "Q", "answer": "A", "score": 0.9}]
        with patch.object(rag, "search", return_value=sources) as search, \
             patch.object(rag, "generate", return_value="answer") as generate:
            result = rag.ask(" question ", top_k=3)
        search.assert_called_once_with("question", top_k=3)
        generate.assert_called_once_with("question", sources)
        self.assertEqual(result, {"answer": "answer", "sources": sources})

    def test_generation_failure_falls_back_to_first_source(self):
        sources = [{"question": "Q", "answer": "fallback"}]
        with patch.object(rag.requests, "post", side_effect=TimeoutError("timeout")):
            self.assertEqual(rag.generate("Q", sources), "fallback")

    def test_health_reports_unavailable_collection(self):
        with patch.object(main, "is_ready", return_value=False):
            self.assertFalse(main.health()["collection_ready"])


if __name__ == "__main__":
    unittest.main()
