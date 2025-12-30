import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'src'))

import asyncio
from backend.src.utils.indexer import DocumentIndexer

async def main():
    """
    Main function to run the indexer
    """
    indexer = DocumentIndexer()
    await indexer.initialize_services()

    # Index all documents in the docs directory
    print("Starting document indexing...")
    indexed_docs = await indexer.index_all_documents()
    print(f"Completed indexing {len(indexed_docs)} documents")

if __name__ == "__main__":
    asyncio.run(main())