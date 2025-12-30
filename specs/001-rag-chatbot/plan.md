# Implementation Plan: RAG Chatbot for Physical AI Book

**Branch**: `001-rag-chatbot` | **Date**: 2025-01-08 | **Spec**: [link](spec.md)
**Input**: Feature specification from `/specs/001-rag-chatbot/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Implementation of an embedded RAG (Retrieval Augmented Generation) chatbot that allows students to ask questions about Physical AI book content. The system will use FastAPI backend with Qdrant vector database and Neon Postgres for metadata, integrated with a Docusaurus React frontend component.

## Technical Context

**Language/Version**: Python 3.11 (backend), JavaScript/TypeScript (frontend)
**Primary Dependencies**: FastAPI, OpenAI SDK, Qdrant client, Neon Postgres client, React for Docusaurus
**Storage**: Qdrant vector database (for embeddings), Neon Postgres (for document metadata)
**Testing**: pytest (backend), Jest/React Testing Library (frontend)
**Target Platform**: Web application (backend: Linux server, frontend: GitHub Pages)
**Project Type**: Web (frontend + backend)
**Performance Goals**: <10 seconds response time for typical queries, support 50+ concurrent users
**Constraints**: <10 seconds p95 response time, must work with static site hosting (GitHub Pages)
**Scale/Scope**: Up to 1000+ book pages, 50+ concurrent users, 1M+ tokens of content

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Alignment with Physical AI Book Constitution

- **Content-First**: ✅ The RAG approach prioritizes educational clarity by directly connecting student questions to book content
- **Beginner-Friendly Accuracy**: ✅ Using authoritative book content ensures technical accuracy while OpenAI's GPT-4 provides accessible explanations
- **Progressive Learning**: ✅ The chatbot preserves current page context, supporting the logical progression of learning concepts
- **Real-World Robotics Examples**: ✅ The system can reference specific examples from the book when answering questions
- **Structured Presentation**: ✅ The floating chat widget integrates cleanly without disrupting the existing documentation structure
- **University-Level Standards**: ✅ Responses are based solely on book content, maintaining academic rigor and preventing hallucination
- **Simplicity in Tooling**: ✅ The separation of backend and frontend services keeps complexity manageable while enabling functionality
- **GitHub Pages Deployment**: ✅ The frontend component works within static site constraints while backend is deployed separately

## Project Structure

### Documentation (this feature)

```text
specs/001-rag-chatbot/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
backend/
├── src/
│   ├── models/
│   │   ├── document.py
│   │   ├── chat_session.py
│   │   └── response.py
│   ├── services/
│   │   ├── embedding_service.py
│   │   ├── rag_service.py
│   │   ├── qdrant_service.py
│   │   └── postgres_service.py
│   ├── api/
│   │   ├── chat.py
│   │   ├── documents.py
│   │   └── main.py
│   └── utils/
│       ├── text_chunker.py
│       └── indexer.py
└── tests/
    ├── unit/
    ├── integration/
    └── contract/

frontend/
├── src/
│   ├── components/
│   │   ├── ChatWidget.jsx
│   │   ├── ChatModal.jsx
│   │   └── Message.jsx
│   ├── services/
│   │   ├── api.js
│   │   └── chat-context.js
│   └── hooks/
│       └── useChat.js
└── tests/
    ├── unit/
    └── integration/
```

**Structure Decision**: Web application with separate backend and frontend. The backend will be deployed separately (Railway/Fly.io/Render) while the frontend remains integrated with the Docusaurus documentation site for GitHub Pages deployment.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |
