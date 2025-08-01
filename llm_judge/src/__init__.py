"""LLM-as-a-Judge: Automated evaluation system for RAG Q&A pairs."""

__version__ = "1.0.0"
__author__ = "MoveO AI"

from .judge import Judge
from .evaluation import precision_recall_f1, metrics_report
from .safety import is_dangerous

__all__ = [
    "Judge", 
    "precision_recall_f1", 
    "metrics_report", 
    "is_dangerous",
    "judge",
    "evaluation",
    "safety",
    "openai_client",
    "io",
]
