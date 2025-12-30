# Deployment Guide: RAG Chatbot for Physical AI Book

## Prerequisites

- Python 3.11+
- Docker (optional, for containerized deployment)
- OpenAI API key
- Qdrant Cloud account (or local Qdrant instance)
- Neon Postgres account (or local PostgreSQL instance)

## Environment Configuration

Create a `.env` file in the backend directory with the following variables:

```env
OPENAI_API_KEY=your_openai_api_key_here
QDRANT_URL=your_qdrant_url_here
QDRANT_API_KEY=your_qdrant_api_key_here
NEON_DATABASE_URL=your_neon_database_url_here
```

## Backend Deployment

### Option 1: Direct Python Deployment

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   uvicorn src.api.main:app --host 0.0.0.0 --port 8000
   ```

### Option 2: Docker Deployment

1. Build the Docker image:
   ```bash
   docker build -t rag-chatbot-backend .
   ```

2. Run the container:
   ```bash
   docker run -d -p 8000:8000 --env-file .env rag-chatbot-backend
   ```

## Frontend Integration

The frontend component is designed to integrate with Docusaurus-based documentation sites:

1. Install the frontend dependencies:
   ```bash
   cd frontend
   npm install
   ```

2. Configure the API endpoint in your Docusaurus environment:
   ```env
   REACT_APP_API_BASE_URL=https://your-backend-url.com
   ```

3. Build the frontend:
   ```bash
   npm run build
   ```

## Indexing Book Content

Before the chatbot can answer questions, you need to index the book content:

```bash
cd backend
python -m src.utils.indexer
```

This will index all markdown files in the `docs/` directory.

## API Endpoints

### Health Check
- `GET /health` - Check service availability

### Chat
- `POST /chat` - Submit a question and receive a response
- `POST /chat/start-session` - Start a new chat session

### Documents
- `POST /documents/index` - Index new document content
- `GET /documents/{document_id}` - Get information about a specific document

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `QDRANT_URL`: Qdrant Cloud endpoint or local instance
- `QDRANT_API_KEY`: Qdrant Cloud API key
- `NEON_DATABASE_URL`: Neon Postgres connection string

## Scaling Considerations

- For production use, consider implementing rate limiting
- Use a load balancer for multiple backend instances
- Consider caching frequently accessed embeddings
- Monitor API usage for cost optimization

## Troubleshooting

### Common Issues

1. **Qdrant Connection Errors**: Ensure your Qdrant instance is accessible and credentials are correct
2. **Database Connection Errors**: Verify your PostgreSQL connection string
3. **OpenAI API Errors**: Check that your API key is valid and you have sufficient quota

### Logging

The application logs to stdout. Check your deployment platform's logs for debugging information.