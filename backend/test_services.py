import asyncio
from src.utils.indexer import DocumentIndexer

async def test_services():
    print("Initializing indexer...")
    indexer = DocumentIndexer()
    print("Indexer created, initializing services...")
    await indexer.initialize_services()
    print("Services initialized successfully!")

if __name__ == "__main__":
    asyncio.run(test_services())