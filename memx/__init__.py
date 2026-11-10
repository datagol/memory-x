"""memx: scoped, auditable long-term memory for agents."""

from memx.config import MemxConfig
from memx.documents import DocumentError, load_text, read_document
from memx.memory import Conflict, InputTooLarge, Memory, NotFound
from memx.scopes import InvalidScope, ScopePath
from memx.types import Category, Fact, InsertReceipt, Messages, Recall, Text, Turn

__all__ = [
    "Category", "Conflict", "DocumentError", "Fact", "InputTooLarge", "InsertReceipt", "InvalidScope", "Memory", "MemxConfig", "Messages",
    "NotFound", "Recall", "ScopePath", "Text", "Turn", "load_text", "read_document",
]
