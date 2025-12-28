---
sidebar_position: 1
description: Dialogue systems for conversational robotics
---

# Dialogue Systems for Conversational Robotics

## Learning Outcomes

By the end of this chapter, you should be able to:

- Design and implement dialogue systems for robots
- Integrate natural language processing with robot behavior
- Create context-aware conversational agents
- Implement multi-modal dialogue systems combining speech, vision, and action
- Evaluate the effectiveness of conversational robot interfaces

## Introduction to Conversational Robotics

Conversational robotics combines natural language processing, dialogue management, and robot behavior to create robots that can communicate naturally with humans. Unlike simple voice assistants, conversational robots must understand context, maintain dialogue state, and coordinate verbal communication with physical actions.

### Key Challenges in Conversational Robotics

- **Multi-modality**: Integrating speech, vision, gesture, and action
- **Context Awareness**: Understanding and maintaining context across dialogue turns
- **Embodied Interaction**: Coordinating verbal and non-verbal communication
- **Real-time Processing**: Handling dialogue in real-time with limited computational resources
- **Social Cues**: Recognizing and responding to social signals

## Architecture of Dialogue Systems

### Components of a Dialogue System

```python
class DialogueSystem:
    def __init__(self):
        self.speech_recognizer = None
        self.nlp_processor = None
        self.dialogue_manager = None
        self.response_generator = None
        self.speech_synthesizer = None
        self.context_manager = None

    def process_input(self, audio_input):
        # 1. Speech recognition
        text = self.speech_recognizer.recognize(audio_input)

        # 2. Natural language understanding
        intent, entities = self.nlp_processor.parse(text)

        # 3. Dialogue management
        response = self.dialogue_manager.generate_response(intent, entities)

        # 4. Response generation
        response_text = self.response_generator.format(response)

        # 5. Speech synthesis
        audio_response = self.speech_synthesizer.synthesize(response_text)

        return audio_response
```

### Speech Recognition

```python
import speech_recognition as sr
import rospy
from std_msgs.msg import String

class SpeechRecognizer:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Adjust for ambient noise
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)

        # Publishers for recognized text
        self.text_pub = rospy.Publisher('/speech/text', String, queue_size=10)

    def listen_once(self):
        """Listen for a single utterance"""
        try:
            with self.microphone as source:
                print("Listening...")
                audio = self.recognizer.listen(source, timeout=5.0)

            # Use Google's speech recognition
            text = self.recognizer.recognize_google(audio)
            print(f"Recognized: {text}")

            # Publish recognized text
            msg = String()
            msg.data = text
            self.text_pub.publish(msg)

            return text

        except sr.WaitTimeoutError:
            print("Timeout: No speech detected")
            return None
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Error with speech recognition service: {e}")
            return None

    def listen_continuously(self):
        """Listen continuously for speech"""
        def callback(recognizer, audio):
            try:
                text = recognizer.recognize_google(audio)
                msg = String()
                msg.data = text
                self.text_pub.publish(msg)
            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                print(f"Error: {e}")

        # Start continuous listening
        stop_listening = self.recognizer.listen_in_background(self.microphone, callback)
        return stop_listening
```

### Natural Language Understanding (NLU)

```python
import spacy
import re
from typing import Dict, List, Tuple

class NaturalLanguageUnderstanding:
    def __init__(self):
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Please install en_core_web_sm: python -m spacy download en_core_web_sm")
            self.nlp = None

        # Define intent patterns
        self.intent_patterns = {
            'navigation': [
                r'go to (.+)',
                r'move to (.+)',
                r'go (.+)',
                r'find (.+)',
                r'locate (.+)'
            ],
            'grasp': [
                r'pick up (.+)',
                r'grasp (.+)',
                r'get (.+)',
                r'take (.+)'
            ],
            'greeting': [
                r'hello',
                r'hi',
                r'hey',
                r'good morning',
                r'good afternoon'
            ],
            'question': [
                r'what (.+)',
                r'where (.+)',
                r'how (.+)',
                r'can you (.+)'
            ]
        }

    def parse(self, text: str) -> Tuple[str, Dict]:
        """Parse text to extract intent and entities"""
        if not self.nlp:
            return 'unknown', {}

        doc = self.nlp(text.lower())

        # Extract intent
        intent = self.extract_intent(text)

        # Extract entities
        entities = self.extract_entities(doc)

        return intent, entities

    def extract_intent(self, text: str) -> str:
        """Extract intent from text using pattern matching"""
        text_lower = text.lower()

        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return intent

        # If no pattern matches, use spaCy for POS tagging and dependency parsing
        doc = self.nlp(text_lower)

        # Simple heuristic-based intent detection
        for token in doc:
            if token.lemma_ in ['go', 'move', 'navigate']:
                return 'navigation'
            elif token.lemma_ in ['pick', 'grasp', 'take', 'get']:
                return 'grasp'
            elif token.lemma_ in ['hello', 'hi', 'hey']:
                return 'greeting'

        return 'unknown'

    def extract_entities(self, doc) -> Dict:
        """Extract named entities and noun chunks"""
        entities = {
            'locations': [],
            'objects': [],
            'people': [],
            'quantities': []
        }

        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in ['GPE', 'LOC', 'FAC']:  # Geopolitical entities, locations, facilities
                entities['locations'].append(ent.text)
            elif ent.label_ == 'PERSON':
                entities['people'].append(ent.text)

        # Extract noun chunks
        for chunk in doc.noun_chunks:
            # Filter out determiners and pronouns
            if chunk.root.pos_ in ['NOUN', 'PROPN']:
                # Check if it's likely an object
                if self.is_object(chunk.text):
                    entities['objects'].append(chunk.text)

        return entities

    def is_object(self, text: str) -> bool:
        """Simple heuristic to determine if text refers to an object"""
        # This is a simplified check - in practice, you'd use more sophisticated methods
        object_keywords = ['cup', 'bottle', 'box', 'book', 'ball', 'toy', 'object', 'item']
        return any(keyword in text.lower() for keyword in object_keywords)
```

### Dialogue Manager

```python
from enum import Enum
from typing import Dict, Any, Optional
import json

class DialogueState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    RESPONDING = "responding"
    WAITING = "waiting"

class DialogueManager:
    def __init__(self):
        self.state = DialogueState.IDLE
        self.context = {}
        self.conversation_history = []
        self.active_intent = None
        self.pending_action = None

    def process_input(self, intent: str, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Process user input and generate response"""
        # Update conversation history
        self.conversation_history.append({
            'turn': 'user',
            'intent': intent,
            'entities': entities,
            'timestamp': time.time()
        })

        # Update context
        self.update_context(entities)

        # Generate response based on intent
        response = self.generate_response(intent, entities)

        # Update conversation history with system response
        self.conversation_history.append({
            'turn': 'system',
            'response': response,
            'timestamp': time.time()
        })

        return response

    def generate_response(self, intent: str, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response based on intent and context"""
        if intent == 'greeting':
            return self.handle_greeting(entities)
        elif intent == 'navigation':
            return self.handle_navigation(entities)
        elif intent == 'grasp':
            return self.handle_grasp(entities)
        elif intent == 'question':
            return self.handle_question(entities)
        else:
            return self.handle_unknown(entities)

    def handle_greeting(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Handle greeting intent"""
        responses = [
            "Hello! How can I help you today?",
            "Hi there! What would you like me to do?",
            "Greetings! I'm ready to assist you."
        ]

        import random
        return {
            'text': random.choice(responses),
            'action': 'none'
        }

    def handle_navigation(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Handle navigation intent"""
        if 'locations' in entities and entities['locations']:
            target_location = entities['locations'][0]

            # Check if location is known
            if self.is_known_location(target_location):
                return {
                    'text': f"Okay, I'll navigate to {target_location}.",
                    'action': 'navigate',
                    'location': target_location
                }
            else:
                return {
                    'text': f"I don't know where {target_location} is. Can you guide me there?",
                    'action': 'request_guidance',
                    'location': target_location
                }
        else:
            return {
                'text': "Where would you like me to go?",
                'action': 'request_location'
            }

    def handle_grasp(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Handle grasp intent"""
        if 'objects' in entities and entities['objects']:
            target_object = entities['objects'][0]

            return {
                'text': f"Okay, I'll try to pick up the {target_object}.",
                'action': 'grasp',
                'object': target_object
            }
        else:
            return {
                'text': "What would you like me to pick up?",
                'action': 'request_object'
            }

    def handle_question(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Handle question intent"""
        # This would typically connect to a knowledge base or other systems
        return {
            'text': "I'm still learning how to answer questions. Can you ask me to do something instead?",
            'action': 'none'
        }

    def handle_unknown(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Handle unknown intent"""
        return {
            'text': "I'm not sure what you mean. Can you rephrase that?",
            'action': 'none'
        }

    def update_context(self, entities: Dict[str, Any]):
        """Update dialogue context with new information"""
        for entity_type, entity_list in entities.items():
            if entity_list:
                self.context[entity_type] = entity_list

    def is_known_location(self, location: str) -> bool:
        """Check if location is in known locations map"""
        # This would typically check against a map of known locations
        known_locations = ['kitchen', 'living room', 'bedroom', 'office', 'dining room']
        return location.lower() in known_locations
```

## Multi-modal Dialogue Systems

### Vision Integration

```python
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class VisionEnhancedDialogue:
    def __init__(self):
        self.bridge = CvBridge()
        self.object_detector = None  # Object detection system
        self.gaze_controller = None  # Gaze control system
        self.current_scene_objects = []

    def analyze_scene(self, image_msg):
        """Analyze scene to identify objects and context"""
        cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

        # Detect objects in scene
        detections = self.object_detector.detect(cv_image)

        # Update current scene objects
        self.current_scene_objects = detections

        return detections

    def resolve_reference(self, noun_phrase: str) -> Optional[Dict]:
        """Resolve noun phrase to actual object in scene"""
        for obj in self.current_scene_objects:
            if noun_phrase.lower() in obj['name'].lower():
                return obj

        # If exact match not found, try partial matching
        for obj in self.current_scene_objects:
            if self.is_similar(noun_phrase, obj['name']):
                return obj

        return None

    def is_similar(self, phrase1: str, phrase2: str) -> bool:
        """Check if two phrases are similar"""
        # Simple similarity check
        return phrase1.lower() in phrase2.lower() or phrase2.lower() in phrase1.lower()

    def look_at_object(self, object_name: str):
        """Direct robot's gaze toward specified object"""
        for obj in self.current_scene_objects:
            if object_name.lower() in obj['name'].lower():
                # Calculate position for gaze
                center_x = (obj['bbox'][0] + obj['bbox'][2]) / 2
                center_y = (obj['bbox'][1] + obj['bbox'][3]) / 2

                # Direct gaze to object
                self.gaze_controller.look_at_pixel(center_x, center_y)
                return True

        return False
```

### Gesture Integration

```python
class GestureEnhancedDialogue:
    def __init__(self):
        self.gesture_recognizer = None
        self.gesture_executor = None

    def interpret_gesture(self, gesture_data):
        """Interpret gesture and update dialogue context"""
        gesture_type = self.gesture_recognizer.recognize(gesture_data)

        if gesture_type == 'pointing':
            # Pointing often indicates reference to an object
            return {
                'type': 'deixis',
                'target': gesture_data['pointed_location']
            }
        elif gesture_type == 'beckoning':
            # Beckoning might indicate request for approach
            return {
                'type': 'request',
                'action': 'approach'
            }
        elif gesture_type == 'waving':
            # Waving might indicate greeting
            return {
                'type': 'greeting',
                'intensity': gesture_data['intensity']
            }

        return None

    def generate_gesture(self, dialogue_act: str):
        """Generate appropriate gesture for dialogue act"""
        if dialogue_act == 'greeting':
            return 'wave'
        elif dialogue_act == 'acknowledgment':
            return 'nod'
        elif dialogue_act == 'uncertainty':
            return 'shrug'
        elif dialogue_act == 'direction':
            return 'point'

        return 'neutral'
```

## Context Management

### Dialogue Context

```python
import time
from typing import Any, Dict, List

class ContextManager:
    def __init__(self):
        self.context = {
            'current_topic': None,
            'recent_entities': [],
            'user_preferences': {},
            'task_state': {},
            'spatial_context': {},
            'temporal_context': {}
        }
        self.max_history = 10

    def update_context(self, new_info: Dict[str, Any]):
        """Update dialogue context with new information"""
        for key, value in new_info.items():
            if key == 'entity':
                # Add to recent entities with timestamp
                self.context['recent_entities'].append({
                    'value': value,
                    'timestamp': time.time(),
                    'type': new_info.get('entity_type', 'unknown')
                })

                # Keep only recent entities
                if len(self.context['recent_entities']) > self.max_history:
                    self.context['recent_entities'] = self.context['recent_entities'][-self.max_history:]
            else:
                self.context[key] = value

    def resolve_coreference(self, pronoun: str) -> Optional[Any]:
        """Resolve pronouns to actual entities"""
        if pronoun.lower() in ['it', 'this', 'that']:
            # Get most recent entity
            if self.context['recent_entities']:
                return self.context['recent_entities'][-1]['value']
        elif pronoun.lower() in ['they', 'these', 'those']:
            # Get recent entities of the same type
            if len(self.context['recent_entities']) >= 2:
                return [ent['value'] for ent in self.context['recent_entities'][-2:]]

        return None

    def get_context_for_nlu(self) -> Dict[str, Any]:
        """Get relevant context for NLU processing"""
        return {
            'current_topic': self.context['current_topic'],
            'recent_entities': self.context['recent_entities'][-5:],  # Last 5 entities
            'spatial_context': self.context['spatial_context'],
            'user_preferences': self.context['user_preferences']
        }
```

## Response Generation

### Template-based Responses

```python
import random
from string import Template

class ResponseGenerator:
    def __init__(self):
        self.response_templates = {
            'greeting': [
                "Hello! How can I assist you today?",
                "Hi there! What would you like me to do?",
                "Greetings! I'm ready to help you."
            ],
            'acknowledgment': [
                "I understand.",
                "Got it.",
                "Okay, I'll do that."
            ],
            'navigation_success': [
                "I've reached the $location.",
                "I'm now at the $location.",
                "Arrived at the $location."
            ],
            'navigation_failure': [
                "I couldn't reach the $location.",
                "I'm having trouble navigating to the $location.",
                "I can't find a path to the $location."
            ],
            'grasp_success': [
                "I've picked up the $object.",
                "Successfully grasped the $object.",
                "Now holding the $object."
            ],
            'grasp_failure': [
                "I couldn't pick up the $object.",
                "Failed to grasp the $object.",
                "The $object is too difficult to pick up."
            ],
            'request_confirmation': [
                "Should I $action?",
                "Do you want me to $action?",
                "Is it okay if I $action?"
            ]
        }

    def generate_response(self, response_type: str, **kwargs) -> str:
        """Generate response based on type and parameters"""
        if response_type in self.response_templates:
            template = random.choice(self.response_templates[response_type])

            # If template contains placeholders, substitute them
            if '$' in template:
                try:
                    t = Template(template)
                    return t.substitute(**kwargs)
                except KeyError:
                    # If substitution fails, return original template
                    return template
            else:
                return template
        else:
            return "I'm not sure how to respond to that."

    def generate_contextual_response(self, intent: str, entities: Dict, context: Dict) -> str:
        """Generate response based on intent, entities, and context"""
        # This would implement more sophisticated response generation
        # considering the current context and conversation history
        pass
```

## Speech Synthesis

```python
import pyttsx3
import rospy
from std_msgs.msg import String

class SpeechSynthesizer:
    def __init__(self):
        self.engine = pyttsx3.init()

        # Configure speech parameters
        self.engine.setProperty('rate', 150)  # Speed of speech
        self.engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)

        # Get available voices
        voices = self.engine.getProperty('voices')
        if voices:
            self.engine.setProperty('voice', voices[0].id)  # Set to first available voice

    def synthesize(self, text: str, blocking: bool = True):
        """Synthesize speech from text"""
        if blocking:
            # Speak and wait for completion
            self.engine.say(text)
            self.engine.runAndWait()
        else:
            # Speak in background
            self.engine.say(text)
            # Note: For non-blocking, you'd need to handle the event loop separately

    def set_voice_parameters(self, rate: int = None, volume: float = None, voice_id: str = None):
        """Set voice parameters"""
        if rate is not None:
            self.engine.setProperty('rate', rate)
        if volume is not None:
            self.engine.setProperty('volume', volume)
        if voice_id is not None:
            self.engine.setProperty('voice', voice_id)

    def get_available_voices(self):
        """Get list of available voices"""
        voices = self.engine.getProperty('voices')
        return [{'id': v.id, 'name': v.name, 'languages': v.languages} for v in voices]
```

## Integration with Robot Actions

### Action Execution

```python
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseStamped
import tf2_ros

class RobotActionExecutor:
    def __init__(self):
        # Navigation client
        self.nav_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.nav_client.wait_for_server()

        # TF buffer for coordinate transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Gripper client (example)
        self.gripper_client = actionlib.SimpleActionClient('gripper_controller', GripperCommandAction)
        self.gripper_client.wait_for_server()

    def execute_navigation(self, location_name: str) -> bool:
        """Execute navigation to specified location"""
        # Get coordinates for location name
        pose = self.get_location_pose(location_name)
        if not pose:
            return False

        # Create navigation goal
        goal = MoveBaseGoal()
        goal.target_pose = pose

        # Send goal
        self.nav_client.send_goal(goal)

        # Wait for result with timeout
        finished_within_time = self.nav_client.wait_for_result(rospy.Duration(60.0))

        if not finished_within_time:
            self.nav_client.cancel_goal()
            return False

        # Check result
        state = self.nav_client.get_state()
        return state == actionlib.GoalStatus.SUCCEEDED

    def execute_grasp(self, object_name: str) -> bool:
        """Execute grasping action"""
        # Find object in environment
        object_pose = self.locate_object(object_name)
        if not object_pose:
            return False

        # Plan grasp
        grasp_pose = self.calculate_grasp_pose(object_pose)

        # Execute grasp
        success = self.move_to_pose(grasp_pose)
        if success:
            return self.close_gripper()

        return False

    def get_location_pose(self, location_name: str) -> Optional[PoseStamped]:
        """Get pre-defined pose for location name"""
        # This would typically look up poses from a map
        location_map = {
            'kitchen': PoseStamped(),
            'living room': PoseStamped(),
            'bedroom': PoseStamped(),
            # Add more locations
        }

        return location_map.get(location_name.lower())

    def locate_object(self, object_name: str) -> Optional[PoseStamped]:
        """Locate object in environment using vision system"""
        # This would integrate with computer vision system
        # to find the object in the current scene
        pass

    def calculate_grasp_pose(self, object_pose: PoseStamped) -> PoseStamped:
        """Calculate appropriate grasp pose for object"""
        # Calculate approach vector and orientation
        grasp_pose = PoseStamped()
        grasp_pose.header = object_pose.header

        # Approach from the front of the object
        grasp_pose.pose.position.x = object_pose.pose.position.x
        grasp_pose.pose.position.y = object_pose.pose.position.y
        grasp_pose.pose.position.z = object_pose.pose.position.z + 0.1  # 10cm above object

        # Set orientation for grasping
        grasp_pose.pose.orientation.w = 1.0  # Simple orientation

        return grasp_pose
```

## Complete Dialogue System Integration

```python
class ConversationalRobot:
    def __init__(self):
        # Initialize components
        self.speech_recognizer = SpeechRecognizer()
        self.nlu = NaturalLanguageUnderstanding()
        self.dialogue_manager = DialogueManager()
        self.response_generator = ResponseGenerator()
        self.speech_synthesizer = SpeechSynthesizer()
        self.context_manager = ContextManager()
        self.action_executor = RobotActionExecutor()
        self.vision_system = VisionEnhancedDialogue()

        # ROS subscribers and publishers
        self.text_sub = rospy.Subscriber('/speech/text', String, self.text_callback)
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback)

    def text_callback(self, msg):
        """Handle recognized text input"""
        # Process with NLU
        intent, entities = self.nlu.parse(msg.data)

        # Update context
        self.context_manager.update_context({'input_text': msg.data})

        # Generate response
        response = self.dialogue_manager.process_input(intent, entities)

        # Execute any required actions
        if 'action' in response and response['action'] != 'none':
            success = self.execute_action(response)
            if not success:
                response['text'] = f"I couldn't complete the {response['action']} task."

        # Synthesize speech
        self.speech_synthesizer.synthesize(response['text'])

    def image_callback(self, msg):
        """Handle camera input for vision enhancement"""
        # Analyze current scene
        objects = self.vision_system.analyze_scene(msg)

        # Update context with scene information
        self.context_manager.update_context({
            'current_objects': [obj['name'] for obj in objects]
        })

    def execute_action(self, response: Dict) -> bool:
        """Execute robot action based on response"""
        action_type = response.get('action')

        if action_type == 'navigate':
            location = response.get('location')
            return self.action_executor.execute_navigation(location)
        elif action_type == 'grasp':
            object_name = response.get('object')
            return self.action_executor.execute_grasp(object_name)
        elif action_type == 'look_at':
            object_name = response.get('object')
            return self.vision_system.look_at_object(object_name)

        return False

    def start_listening(self):
        """Start the conversational system"""
        print("Starting conversational robot...")

        # Start speech recognition
        self.speech_recognizer.listen_continuously()

        # Keep the system running
        rospy.spin()
```

## Evaluation and Improvement

### Dialogue Quality Metrics

```python
class DialogueEvaluator:
    def __init__(self):
        self.metrics = {
            'understanding_rate': 0.0,
            'task_completion_rate': 0.0,
            'user_satisfaction': 0.0,
            'dialogue_coherence': 0.0
        }

    def evaluate_conversation(self, conversation_history: List[Dict]) -> Dict:
        """Evaluate conversation quality"""
        metrics = {}

        # Calculate understanding rate
        total_inputs = len([turn for turn in conversation_history if turn['turn'] == 'user'])
        understood_inputs = len([turn for turn in conversation_history
                                if turn['turn'] == 'user' and turn.get('intent') != 'unknown'])

        metrics['understanding_rate'] = understood_inputs / total_inputs if total_inputs > 0 else 0.0

        # Calculate task completion
        task_successes = len([turn for turn in conversation_history
                             if turn['turn'] == 'system' and turn.get('action_success', False)])
        total_tasks = len([turn for turn in conversation_history
                          if turn['turn'] == 'system' and turn.get('action')])

        metrics['task_completion_rate'] = task_successes / total_tasks if total_tasks > 0 else 0.0

        return metrics

    def collect_user_feedback(self) -> Dict:
        """Collect user feedback on conversation quality"""
        # This would typically involve asking users to rate the interaction
        # For now, we'll simulate feedback collection
        return {
            'satisfaction_rating': 4,  # 1-5 scale
            'ease_of_use': 4,
            'naturalness': 3,
            'helpfulness': 4
        }
```

## Performance Optimization

### Real-time Processing

```python
import threading
import queue
from collections import deque

class RealTimeDialogueProcessor:
    def __init__(self):
        self.input_queue = queue.Queue(maxsize=10)
        self.output_queue = queue.Queue(maxsize=10)

        # Processing pipelines
        self.nlu_pipeline = deque()
        self.dialogue_pipeline = deque()

        # Threading for parallel processing
        self.processing_thread = threading.Thread(target=self.process_pipeline)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def process_pipeline(self):
        """Process dialogue pipeline in real-time"""
        while True:
            try:
                # Get input from queue
                input_data = self.input_queue.get(timeout=1.0)

                # Process through pipeline stages
                processed_data = self.run_nlu(input_data)
                response = self.run_dialogue_manager(processed_data)

                # Put response in output queue
                self.output_queue.put(response)

                self.input_queue.task_done()

            except queue.Empty:
                continue

    def run_nlu(self, text):
        """Run natural language understanding"""
        # This would call the NLU system
        return {'text': text, 'intent': 'unknown', 'entities': {}}

    def run_dialogue_manager(self, nlu_output):
        """Run dialogue manager"""
        # This would call the dialogue manager
        return {'text': 'Hello', 'action': 'none'}
```

## Summary

Conversational robotics represents a significant advancement in human-robot interaction, enabling more natural and intuitive communication. By integrating speech, vision, and action systems, conversational robots can engage in meaningful interactions that go beyond simple command execution.

## Next Steps

In the next section, we'll explore human-robot interaction principles that complement conversational capabilities.