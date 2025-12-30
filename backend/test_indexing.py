import asyncio
from src.utils.indexer import DocumentIndexer

async def test_indexing():
    print("Initializing indexer...")
    indexer = DocumentIndexer()
    await indexer.initialize_services()
    print("Services initialized successfully!")

    # Index just a single document to test
    print("Indexing a single document...")
    doc_id = await indexer.index_single_document("docs/intro.md", "Introduction")
    print(f"Successfully indexed document with ID: {doc_id}")

    print("Indexing completed!")

if __name__ == "__main__":
    asyncio.run(test_indexing())