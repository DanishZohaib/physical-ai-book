# Quickstart Guide: Physical AI & Humanoid Robotics Textbook

## Prerequisites

- Node.js v18 or higher
- npm or yarn package manager
- Git
- A GitHub account for deployment

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd physical-ai-book
   ```

2. **Install dependencies**
   ```bash
   npm install
   # or
   yarn install
   ```

3. **Start the development server**
   ```bash
   npm run start
   # or
   yarn start
   ```
   This will start a local development server at http://localhost:3000

4. **Build for production**
   ```bash
   npm run build
   # or
   yarn build
   ```

## Adding New Content

1. **Create a new chapter** in the `docs/` directory:
   ```bash
   # Create a new markdown file in the appropriate chapter directory
   # Example: docs/chapter-1-introduction-physical-ai/new-topic.md
   ```

2. **Update navigation** in `sidebars.ts`:
   - Add your new page to the appropriate section in the sidebars configuration

3. **Use Docusaurus markdown features**:
   - Frontmatter for metadata
   - Admonitions for notes and warnings
   - Code blocks with syntax highlighting
   - Images and diagrams

## Deployment

The site is configured for GitHub Pages deployment:

1. **GitHub Actions** will automatically build and deploy when you push to the main branch
2. **Manual deployment** (if needed):
   ```bash
   GIT_USER=<Your GitHub username> \
   CURRENT_BRANCH=main \
   USE_SSH=true \
   npm run deploy
   ```

## Content Guidelines

- Write in clear, accessible language (beginner-friendly)
- Include practical examples using ROS 2, Gazebo, Unity, NVIDIA Isaac Sim, Isaac ROS
- Add learning outcomes at the beginning of each chapter
- Structure content to support progressive learning
- Use diagrams and visual aids where helpful

## Development Workflow

1. Create a new branch for your changes
2. Add or modify content in the `docs/` directory
3. Test locally with `npm run start`
4. Commit your changes
5. Push and create a pull request
6. Review and merge after approval