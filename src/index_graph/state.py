"""State management for the index graph."""

from dataclasses import dataclass
from typing import Annotated

from langchain_core.documents import Document

from shared.state import reduce_docs


# The index state defines the simple IO for the single-node index graph
@dataclass(kw_only=True)
class IndexState:
    """Represents the state for document indexing and retrieval.

    This class defines the structure of the index state, which includes
    the documents to be indexed and the retriever used for searching
    these documents.
    """

    docs: Annotated[list[Document], reduce_docs]
    """A list of documents that the agent can index."""
