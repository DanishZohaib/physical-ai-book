# Data Model: Physical AI & Humanoid Robotics Textbook

## Entities

### Textbook Chapter
- **Fields**:
  - id: string (unique identifier for the chapter)
  - title: string (chapter title)
  - content: string (markdown content)
  - learningOutcomes: array of strings (learning outcomes for the chapter)
  - examples: array of strings (practical examples in the chapter)
  - prerequisites: array of strings (prerequisite knowledge/chapters)
  - nextChapter: string (reference to the next chapter in sequence)
  - order: integer (position in the sequence)

- **Validation Rules**:
  - title must be 3-100 characters
  - content must be valid markdown
  - learningOutcomes must have at least 1 outcome
  - order must be unique across all chapters

### Learning Outcome
- **Fields**:
  - id: string (unique identifier)
  - description: string (what the student should learn)
  - chapterId: string (reference to the chapter it belongs to)
  - measurable: boolean (can the outcome be measured)

- **Validation Rules**:
  - description must be 10-200 characters
  - chapterId must reference an existing chapter

### Practical Example
- **Fields**:
  - id: string (unique identifier)
  - title: string (example title)
  - description: string (description of the example)
  - chapterId: string (reference to the chapter it belongs to)
  - tools: array of strings (tools used in the example: ROS 2, Gazebo, Unity, etc.)
  - difficulty: string (beginner, intermediate, advanced)

- **Validation Rules**:
  - title must be 5-50 characters
  - chapterId must reference an existing chapter
  - difficulty must be one of: 'beginner', 'intermediate', 'advanced'

### Capstone Project
- **Fields**:
  - id: string (unique identifier)
  - title: string (project title)
  - description: string (project description)
  - requirements: array of strings (requirements for the project)
  - learningOutcomes: array of strings (learning outcomes for the project)
  - chaptersIntegrated: array of strings (references to chapters whose concepts are used)

- **Validation Rules**:
  - title must be 5-100 characters
  - must reference at least 3 different chapters

## Relationships
- Textbook Chapter contains multiple Learning Outcomes
- Textbook Chapter contains multiple Practical Examples
- Capstone Project integrates concepts from multiple Textbook Chapters
- Learning Outcomes are specific to a Textbook Chapter
- Practical Examples are specific to a Textbook Chapter

## State Transitions
- Chapter draft → review → published (content workflow)
- Example draft → review → published (example workflow)
- Capstone project draft → review → published (project workflow)