import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'src'))

import asyncio
from backend.src.utils.indexer import DocumentIndexer

async def test_indexing():
    print("Initializing indexer...")
    indexer = DocumentIndexer()
    await indexer.initialize_services()
    print("Services initialized successfully!")

    # Index just a single document to test
    print("Indexing a single document...")
    doc_path = os.path.join("docs", "intro.md")
    doc_id = await indexer.index_single_document(doc_path, "Introduction")
    print(f"Successfully indexed document with ID: {doc_id}")

    print("Indexing completed!")

if __name__ == "__main__":
    asyncio.run(test_indexing())