# Quickstart Guide: RAG Chatbot for Physical AI Book

## Development Setup

### Prerequisites
- Python 3.11+
- Node.js 18+ (for Docusaurus)
- Docker (optional, for local Qdrant)
- OpenAI API key
- Qdrant Cloud account
- Neon Postgres account

### Backend Setup
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and connection strings
   ```

4. Run the backend server:
   ```bash
   uvicorn src.api.main:app --reload
   ```

### Frontend Integration
1. The frontend component integrates directly with Docusaurus
2. Add the ChatWidget component to your Docusaurus layout
3. Configure the API endpoint to point to your backend

## Running the Application

### Local Development
1. Start the backend:
   ```bash
   cd backend
   uvicorn src.api.main:app --reload --port 8000
   ```

2. In a separate terminal, start the Docusaurus frontend:
   ```bash
   cd frontend  # or your Docusaurus root
   npm start
   ```

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key
- `QDRANT_URL`: Qdrant Cloud endpoint
- `QDRANT_API_KEY`: Qdrant Cloud API key
- `NEON_DATABASE_URL`: Neon Postgres connection string

## Indexing Book Content
1. Run the indexing script to process all book content:
   ```bash
   python -m src.utils.indexer
   ```

2. The script will:
   - Parse all markdown files in the docs directory
   - Chunk the content appropriately
   - Generate embeddings
   - Store in Qdrant and metadata in Postgres

## API Endpoints
- `POST /chat` - Submit a question and receive a response
- `POST /documents/index` - Index new document content
- `GET /health` - Check service health

## Testing
- Run backend tests: `pytest`
- Run frontend tests: `npm test`

## Deployment
- Backend: Deploy to Railway, Fly.io, or Render
- Frontend: Remains on GitHub Pages as part of Docusaurus site