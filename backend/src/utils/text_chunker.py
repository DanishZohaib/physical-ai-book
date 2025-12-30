from typing import List, Dict, Any
import re

class TextChunker:
    """
    Utility class for chunking text into smaller pieces suitable for embedding
    while preserving semantic boundaries.
    """

    def __init__(self, max_chunk_size: int = 500, overlap: int = 50):
        """
        Initialize the TextChunker

        Args:
            max_chunk_size: Maximum number of characters per chunk
            overlap: Number of characters to overlap between chunks
        """
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Chunk the input text into smaller pieces

        Args:
            text: The text to be chunked
            metadata: Additional metadata to include with each chunk

        Returns:
            List of dictionaries containing chunk content and metadata
        """
        if not text:
            return []

        # Split text into sentences to preserve semantic boundaries
        sentences = self._split_into_sentences(text)

        chunks = []
        current_chunk = ""
        chunk_index = 0

        for sentence in sentences:
            # If adding this sentence would exceed the max chunk size
            if len(current_chunk) + len(sentence) > self.max_chunk_size:
                # If current chunk is not empty, save it
                if current_chunk.strip():
                    chunks.append({
                        "content": current_chunk.strip(),
                        "chunk_index": chunk_index,
                        "metadata": metadata or {}
                    })
                    chunk_index += 1

                # Start a new chunk with potential overlap
                if len(sentence) > self.max_chunk_size:
                    # If the sentence itself is too long, split it by length
                    sub_chunks = self._split_long_sentence(sentence)
                    for sub_chunk in sub_chunks[:-1]:  # Add all but the last sub-chunk
                        chunks.append({
                            "content": sub_chunk.strip(),
                            "chunk_index": chunk_index,
                            "metadata": metadata or {}
                        })
                        chunk_index += 1

                    # Start the next iteration with the last sub-chunk
                    current_chunk = sub_chunks[-1]
                else:
                    current_chunk = sentence
            else:
                current_chunk += " " + sentence

        # Add the final chunk if it has content
        if current_chunk.strip():
            chunks.append({
                "content": current_chunk.strip(),
                "chunk_index": chunk_index,
                "metadata": metadata or {}
            })

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences while preserving the sentence structure
        """
        # Split on sentence-ending punctuation followed by whitespace
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Remove empty strings and strip whitespace
        return [s.strip() for s in sentences if s.strip()]

    def _split_long_sentence(self, sentence: str) -> List[str]:
        """
        Split a sentence that's longer than max_chunk_size into smaller pieces
        """
        if len(sentence) <= self.max_chunk_size:
            return [sentence]

        chunks = []
        start = 0

        while start < len(sentence):
            end = start + self.max_chunk_size
            if end >= len(sentence):
                chunks.append(sentence[start:])
                break

            # Try to break at a space to avoid cutting words
            if sentence[end] != ' ':
                # Find the last space before the max length
                space_index = sentence.rfind(' ', start, end)
                if space_index != -1 and space_index > start:
                    chunk = sentence[start:space_index]
                    start = space_index
                else:
                    # If no space found, just split at the max length
                    chunk = sentence[start:end]
                    start = end
            else:
                chunk = sentence[start:end]
                start = end

            chunks.append(chunk.strip())

        return chunks

    def chunk_document(self, document_content: str, doc_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Chunk an entire document with document-specific metadata

        Args:
            document_content: The full content of the document
            doc_metadata: Metadata specific to the document (like source path, title, etc.)

        Returns:
            List of chunk dictionaries with document metadata included
        """
        chunks = self.chunk_text(document_content, doc_metadata)

        # Add document-specific metadata to each chunk
        for i, chunk in enumerate(chunks):
            chunk_metadata = (doc_metadata or {}).copy()
            chunk_metadata['chunk_number'] = i + 1
            chunk['metadata'] = chunk_metadata

        return chunks