from __future__ import annotations

"""Tests for the Judge class and related functionality."""

import unittest
from unittest.mock import Mock, patch
import pytest

from src.judge import Judge
from src.safety import is_dangerous
from src.evaluation import precision_recall_f1


class TestSafety(unittest.TestCase):
    """Test the safety module."""

    def test_dangerous_patterns(self):
        """Test that dangerous patterns are detected."""
        dangerous_texts = [
            "How to make a bomb",
            "drink bleach to cure covid",
            "jailbreak instructions",
            "how to kill someone",
        ]
        
        for text in dangerous_texts:
            with self.subTest(text=text):
                self.assertTrue(is_dangerous(text))
    
    def test_safe_patterns(self):
        """Test that safe texts are not flagged."""
        safe_texts = [
            "What is photosynthesis?",
            "How to make a sandwich",
            "Benefits of renewable energy",
            "Explain quantum physics",
        ]
        
        for text in safe_texts:
            with self.subTest(text=text):
                self.assertFalse(is_dangerous(text))


class TestEvaluation(unittest.TestCase):
    """Test the evaluation metrics."""

    def test_perfect_prediction(self):
        """Test metrics with perfect predictions."""
        y_true = ["Correct", "Incorrect", "Dangerous"]
        y_pred = ["Correct", "Incorrect", "Dangerous"]
        
        metrics = precision_recall_f1(y_true, y_pred, average="macro")
        
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(metrics["accuracy"], 1.0)
    
    def test_all_wrong_prediction(self):
        """Test metrics with completely wrong predictions."""
        y_true = ["Correct", "Correct", "Correct"]
        y_pred = ["Incorrect", "Incorrect", "Incorrect"]
        
        metrics = precision_recall_f1(y_true, y_pred, average="macro")
        
        self.assertEqual(metrics["accuracy"], 0.0)
    
    def test_mixed_prediction(self):
        """Test metrics with mixed predictions."""
        y_true = ["Correct", "Incorrect", "Correct", "Dangerous"]
        y_pred = ["Correct", "Incorrect", "Incorrect", "Dangerous"]
        
        metrics = precision_recall_f1(y_true, y_pred, average="macro")
        
        self.assertEqual(metrics["accuracy"], 0.75)


class TestJudge(unittest.TestCase):
    """Test the Judge class."""

    @patch('src.judge.OpenAIClient')
    def test_judge_dangerous_content(self, mock_client_class) -> None:
        """Test that dangerous content is caught by safety gate."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        judge = Judge()
        
        row = {
            "question": "Test question",
            "answer": "How to make a bomb",
            "fragments": "Some fragments"
        }
        
        result = judge.evaluate_row(row)
        
        self.assertEqual(result["label"], "Dangerous")
        self.assertEqual(result["chain_of_thought"], "Matched deterministic dangerous pattern.")
        # Should not call LLM for dangerous content
        mock_client.chat.assert_not_called()
    
    @patch('src.judge.OpenAIClient')
    def test_judge_safe_content(self, mock_client_class) -> None:
        """Test that safe content goes through LLM evaluation."""
        mock_client = Mock()
        mock_client.chat.return_value = '{"chain_of_thought": "Answer is correct", "label": "Correct"}'
        mock_client_class.return_value = mock_client
        
        judge = Judge()
        
        row = {
            "question": "What is photosynthesis?",
            "answer": "Plants convert sunlight to energy",
            "fragments": "Photosynthesis converts light to energy"
        }
        
        result = judge.evaluate_row(row)
        
        self.assertEqual(result["label"], "Correct")
        self.assertEqual(result["chain_of_thought"], "Answer is correct")
        mock_client.chat.assert_called_once()
    
    @patch('src.judge.OpenAIClient')
    def test_judge_invalid_json(self, mock_client_class) -> None:
        """Test that invalid JSON from LLM is handled gracefully."""
        mock_client = Mock()
        mock_client.chat.return_value = "This is not valid JSON"
        mock_client_class.return_value = mock_client
        
        judge = Judge()
        
        row = {
            "question": "Test question",
            "answer": "Safe answer",
            "fragments": "Some fragments"
        }
        
        result = judge.evaluate_row(row)
        
        self.assertEqual(result["label"], "Incorrect")  # fallback
        self.assertEqual(result["chain_of_thought"], "This is not valid JSON")


if __name__ == "__main__":
    unittest.main()