"""memx: scoped, auditable long-term memory for agents."""

from importlib.metadata import PackageNotFoundError, version

from memx.config import MemxConfig
from memx.documents import DocumentError, load_text, read_document
from memx.memory import InputTooLarge, Memory, NotFound
from memx.scopes import InvalidScope, ScopePath
from memx.types import Category, Fact, InsertReceipt, Messages, Recall, Text, Turn

try:
    __version__ = version("memx")
except PackageNotFoundError:  # running from a checkout that was never installed
    __version__ = "0.0.0"

__all__ = [
    "Category", "DocumentError", "Fact", "InputTooLarge", "InsertReceipt", "InvalidScope", "Memory", "MemxConfig", "Messages",
    "NotFound", "Recall", "ScopePath", "Text", "Turn", "__version__", "load_text", "read_document",
]
