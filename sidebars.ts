import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  // Manual sidebar for the Physical AI & Humanoid Robotics textbook
  textbookSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Chapter 1: Introduction to Physical AI and Embodied Intelligence',
      items: [
        'chapter-1-introduction-physical-ai/embodied-intelligence',
        'chapter-1-introduction-physical-ai/ai-in-physical-world'
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Chapter 2: ROS 2 fundamentals for humanoid robots',
      items: [
        'chapter-2-ros2-fundamentals/ros2-basics',
        'chapter-2-ros2-fundamentals/humanoid-robot-control'
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Chapter 3: Robot simulation using Gazebo and Unity',
      items: [
        'chapter-3-robot-simulation/gazebo-simulation',
        'chapter-3-robot-simulation/unity-integration'
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Chapter 4: NVIDIA Isaac Sim and Isaac ROS',
      items: [
        'chapter-4-nvidia-isaac/isaac-sim-overview',
        'chapter-4-nvidia-isaac/isaac-ros-integration'
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Chapter 5: Vision-Language-Action systems',
      items: [
        'chapter-5-vision-language-action/computer-vision',
        'chapter-5-vision-language-action/action-systems'
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Chapter 6: Conversational humanoid robots',
      items: [
        'chapter-6-conversational-robots/dialogue-systems',
        'chapter-6-conversational-robots/human-robot-interaction'
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Chapter 7: Capstone - Autonomous simulated humanoid robot',
      items: [
        'chapter-7-capstone-project/autonomous-humanoid-robot'
      ],
      collapsed: false,
    },
    {
      type: 'doc',
      id: 'assessment-framework',
      label: 'Assessment Framework',
    },
  ],
};

export default sidebars;
