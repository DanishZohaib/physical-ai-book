import asyncio
import os
import sys

# Add the backend/src directory to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'src'))

from backend.src.utils.indexer import DocumentIndexer

async def test_indexing():
    print("Initializing indexer...")
    indexer = DocumentIndexer()
    await indexer.initialize_services()
    print("Services initialized successfully!")

    # Index just a single document to test - using full path
    print("Indexing a single document...")
    doc_path = os.path.join("..", "docs", "intro.md")
    doc_id = await indexer.index_single_document(doc_path, "Introduction")
    print(f"Successfully indexed document with ID: {doc_id}")

    print("Indexing completed!")

if __name__ == "__main__":
    # Change to backend directory to run the test
    os.chdir("backend")
    asyncio.run(test_indexing())