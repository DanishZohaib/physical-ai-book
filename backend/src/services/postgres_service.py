from typing import Dict, Any, List, Optional
import aiosqlite
from dotenv import load_dotenv
import os
from datetime import datetime
import json
import asyncio

load_dotenv()

class PostgresService:
    def __init__(self):
        self.db_path = "local_chatbot.db"  # Local SQLite database for testing
        self.connection = None

    async def connect(self):
        """Establish connection to the local SQLite database"""
        self.connection = await aiosqlite.connect(self.db_path)

        # Create necessary tables if they don't exist
        await self._create_tables()

    async def _create_tables(self):
        """Create necessary tables if they don't exist"""
        # Create documents table
        await self.connection.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                source_path TEXT NOT NULL,
                chunk_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create document_chunks table
        await self.connection.execute('''
            CREATE TABLE IF NOT EXISTS document_chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT,
                content TEXT NOT NULL,
                chunk_index INTEGER,
                vector_id TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents(id)
            )
        ''')

        # Create chat_sessions table
        await self.connection.execute('''
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id TEXT PRIMARY KEY,
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_context TEXT,
                expires_at TIMESTAMP
            )
        ''')

        # Create questions table
        await self.connection.execute('''
            CREATE TABLE IF NOT EXISTS questions (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                page_context TEXT,
                selected_text TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                session_id TEXT,
                FOREIGN KEY (session_id) REFERENCES chat_sessions(id)
            )
        ''')

        # Create responses table
        await self.connection.execute('''
            CREATE TABLE IF NOT EXISTS responses (
                id TEXT PRIMARY KEY,
                question_id TEXT,
                content TEXT NOT NULL,
                sources TEXT,
                confidence_score REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                session_id TEXT,
                FOREIGN KEY (question_id) REFERENCES questions(id),
                FOREIGN KEY (session_id) REFERENCES chat_sessions(id)
            )
        ''')

        await self.connection.commit()

    async def store_document(self, document_data: Dict[str, Any]) -> str:
        """Store a document in the database"""
        query = '''
            INSERT OR REPLACE INTO documents (id, title, content, source_path, chunk_count, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        '''
        await self.connection.execute(
            query,
            (
                document_data['id'],
                document_data['title'],
                document_data['content'],
                document_data['source_path'],
                document_data.get('chunk_count'),
                document_data.get('created_at', datetime.now().isoformat()),
                document_data.get('updated_at', datetime.now().isoformat())
            )
        )
        await self.connection.commit()
        return document_data['id']

    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document by ID"""
        query = 'SELECT * FROM documents WHERE id = ?'
        cursor = await self.connection.execute(query, (document_id,))
        row = await cursor.fetchone()
        if row:
            # Get column names
            columns = [description[0] for description in cursor.description]
            return dict(zip(columns, row))
        return None

    async def store_document_chunk(self, chunk_data: Dict[str, Any]) -> str:
        """Store a document chunk in the database"""
        # Convert metadata to JSON string for SQLite
        metadata_str = json.dumps(chunk_data.get('metadata', {}))

        query = '''
            INSERT INTO document_chunks (id, document_id, content, chunk_index, vector_id, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        '''
        await self.connection.execute(
            query,
            (
                chunk_data['id'],
                chunk_data['document_id'],
                chunk_data['content'],
                chunk_data['chunk_index'],
                chunk_data['vector_id'],
                metadata_str
            )
        )
        await self.connection.commit()
        return chunk_data['id']

    async def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Retrieve all chunks for a document"""
        query = 'SELECT * FROM document_chunks WHERE document_id = ? ORDER BY chunk_index'
        cursor = await self.connection.execute(query, (document_id,))
        rows = await cursor.fetchall()
        if rows:
            # Get column names
            columns = [description[0] for description in cursor.description]
            results = []
            for row in rows:
                row_dict = dict(zip(columns, row))
                # Parse metadata JSON string back to dict
                if 'metadata' in row_dict and row_dict['metadata']:
                    row_dict['metadata'] = json.loads(row_dict['metadata'])
                results.append(row_dict)
            return results
        return []

    async def store_chat_session(self, session_data: Dict[str, Any]) -> str:
        """Store a chat session in the database"""
        # Convert user_context to JSON string for SQLite
        user_context_str = json.dumps(session_data.get('user_context', {}))

        query = '''
            INSERT INTO chat_sessions (id, start_time, last_activity, user_context, expires_at)
            VALUES (?, ?, ?, ?, ?)
        '''
        await self.connection.execute(
            query,
            (
                session_data['id'],
                session_data['start_time'],
                session_data['last_activity'],
                user_context_str,
                session_data['expires_at']
            )
        )
        await self.connection.commit()
        return session_data['id']

    async def store_question(self, question_data: Dict[str, Any]) -> str:
        """Store a question in the database"""
        query = '''
            INSERT INTO questions (id, content, page_context, selected_text, timestamp, session_id)
            VALUES (?, ?, ?, ?, ?, ?)
        '''
        await self.connection.execute(
            query,
            (
                question_data['id'],
                question_data['content'],
                question_data['page_context'],
                question_data.get('selected_text'),
                question_data['timestamp'],
                question_data['session_id']
            )
        )
        await self.connection.commit()
        return question_data['id']

    async def store_response(self, response_data: Dict[str, Any]) -> str:
        """Store a response in the database"""
        # Convert sources to JSON string for SQLite
        sources_str = json.dumps(response_data['sources'])

        query = '''
            INSERT INTO responses (id, question_id, content, sources, confidence_score, timestamp, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        '''
        await self.connection.execute(
            query,
            (
                response_data['id'],
                response_data['question_id'],
                response_data['content'],
                sources_str,
                response_data['confidence_score'],
                response_data['timestamp'],
                response_data['session_id']
            )
        )
        await self.connection.commit()
        return response_data['id']