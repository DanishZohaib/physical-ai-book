# Physical AI Book with RAG Chatbot

This website is built using [Docusaurus](https://docusaurus.io/), a modern static website generator. It includes an embedded RAG (Retrieval Augmented Generation) chatbot that allows students to ask questions about the Physical AI book content.

## Features

- Interactive textbook about Physical AI and Humanoid Robotics
- Embedded RAG chatbot for answering questions based on book content
- Access to all chapters covering ROS 2, Gazebo simulation, NVIDIA Isaac, and more

## Installation

```bash
yarn
```

## Local Development

```bash
yarn start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

## Build

```bash
yarn build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.

## Backend Setup (RAG Chatbot)

The RAG chatbot requires a separate backend service:

1. Navigate to the backend directory: `cd backend`
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Copy `.env.example` to `.env` and add your API keys
6. Start the server: `uvicorn src.api.main:app --reload`

## Deployment

Using SSH:

```bash
USE_SSH=true yarn deploy
```

Not using SSH:

```bash
GIT_USER=<Your GitHub username> yarn deploy
```

If you are using GitHub pages for hosting, this command is a convenient way to build the website and push to the `gh-pages` branch.
