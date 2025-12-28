---
sidebar_position: 2
description: Action systems for robotics that combine perception and control
---

# Action Systems for Robotics

## Learning Outcomes

By the end of this chapter, you should be able to:

- Design and implement action systems that combine perception and control
- Create behavior trees and state machines for robot behavior
- Implement planning and execution systems for complex robot tasks
- Integrate vision systems with action execution
- Evaluate the performance of action systems in robotics applications

## Introduction to Action Systems

Action systems in robotics are responsible for translating high-level goals into low-level motor commands that enable robots to interact with their environment. These systems bridge the gap between perception (understanding the world) and action (changing the world), requiring sophisticated planning, control, and execution capabilities.

### Key Components of Action Systems

- **Task Planning**: Breaking down high-level goals into executable actions
- **Motion Planning**: Computing trajectories for robot movement
- **Control Systems**: Executing planned motions with precision
- **Behavior Management**: Coordinating multiple behaviors and handling conflicts
- **Execution Monitoring**: Tracking action progress and handling failures

## Behavior Trees

Behavior trees provide a structured approach to robot behavior:

### Basic Behavior Tree Structure

```python
class BehaviorNode:
    def __init__(self, name):
        self.name = name
        self.status = None

    def tick(self):
        pass

class SequenceNode(BehaviorNode):
    def __init__(self, name, children):
        super().__init__(name)
        self.children = children
        self.current_child_idx = 0

    def tick(self):
        for i in range(self.current_child_idx, len(self.children)):
            child = self.children[i]
            child_status = child.tick()

            if child_status == 'RUNNING':
                self.current_child_idx = i
                return 'RUNNING'
            elif child_status == 'FAILURE':
                self.current_child_idx = 0
                return 'FAILURE'

        # All children succeeded
        self.current_child_idx = 0
        return 'SUCCESS'

class SelectorNode(BehaviorNode):
    def __init__(self, name, children):
        super().__init__(name)
        self.children = children
        self.current_child_idx = 0

    def tick(self):
        for i in range(self.current_child_idx, len(self.children)):
            child = self.children[i]
            child_status = child.tick()

            if child_status == 'RUNNING':
                self.current_child_idx = i
                return 'RUNNING'
            elif child_status == 'SUCCESS':
                self.current_child_idx = 0
                return 'SUCCESS'

        # All children failed
        self.current_child_idx = 0
        return 'FAILURE'
```

### Action Nodes

```python
import time
from enum import Enum

class ActionStatus(Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RUNNING = "RUNNING"

class ActionNode(BehaviorNode):
    def __init__(self, name):
        super().__init__(name)
        self.start_time = None

    def tick(self):
        pass

class MoveToNode(ActionNode):
    def __init__(self, name, target_pose):
        super().__init__(name)
        self.target_pose = target_pose
        self.nav_client = None  # Navigation client
        self.sent_goal = False

    def tick(self):
        if not self.sent_goal:
            # Send navigation goal
            self.nav_client.send_goal(self.target_pose)
            self.sent_goal = True
            self.start_time = time.time()

        # Check navigation status
        status = self.nav_client.get_result()
        if status == 'SUCCESS':
            self.sent_goal = False
            return ActionStatus.SUCCESS
        elif status == 'FAILURE':
            self.sent_goal = False
            return ActionStatus.FAILURE
        else:
            return ActionStatus.RUNNING

class DetectObjectNode(ActionNode):
    def __init__(self, name, object_type):
        super().__init__(name)
        self.object_type = object_type
        self.vision_client = None  # Vision client

    def tick(self):
        # Request object detection
        detection_result = self.vision_client.detect_object(self.object_type)

        if detection_result.success:
            self.last_detection = detection_result
            return ActionStatus.SUCCESS
        else:
            return ActionStatus.FAILURE
```

### Complete Behavior Tree Example

```python
class RobotBehaviorTree:
    def __init__(self):
        # Create action nodes
        self.detect_cup = DetectObjectNode("detect_cup", "cup")
        self.move_to_cup = MoveToNode("move_to_cup", [0.5, 0.5, 0.0])
        self.grasp_cup = ActionNode("grasp_cup")

        # Create behavior tree structure
        # Sequence: detect cup -> move to cup -> grasp cup
        self.root = SequenceNode("pick_cup_sequence", [
            self.detect_cup,
            self.move_to_cup,
            self.grasp_cup
        ])

    def execute(self):
        status = self.root.tick()
        return status
```

## State Machines

State machines provide an alternative approach to robot behavior:

### Basic State Machine

```python
from enum import Enum

class RobotState(Enum):
    IDLE = "idle"
    NAVIGATING = "navigating"
    DETECTING = "detecting"
    MANIPULATING = "manipulating"
    ERROR = "error"

class RobotStateMachine:
    def __init__(self):
        self.state = RobotState.IDLE
        self.last_state = None
        self.state_start_time = time.time()

    def update(self):
        current_time = time.time()

        if self.state == RobotState.IDLE:
            self.handle_idle()
        elif self.state == RobotState.NAVIGATING:
            self.handle_navigating()
        elif self.state == RobotState.DETECTING:
            self.handle_detecting()
        elif self.state == RobotState.MANIPULATING:
            self.handle_manipulating()
        elif self.state == RobotState.ERROR:
            self.handle_error()

    def handle_idle(self):
        # Wait for command
        if self.new_goal_available():
            self.transition_to(RobotState.NAVIGATING)

    def handle_navigating(self):
        # Check navigation status
        nav_status = self.get_navigation_status()
        if nav_status == 'SUCCESS':
            self.transition_to(RobotState.DETECTING)
        elif nav_status == 'FAILURE':
            self.transition_to(RobotState.ERROR)

    def handle_detecting(self):
        # Detect object
        detection_result = self.detect_object()
        if detection_result.success:
            self.transition_to(RobotState.MANIPULATING)
        elif current_time - self.state_start_time > 10:  # Timeout after 10 seconds
            self.transition_to(RobotState.ERROR)

    def handle_manipulating(self):
        # Perform manipulation
        manipulation_status = self.perform_manipulation()
        if manipulation_status == 'SUCCESS':
            self.transition_to(RobotState.IDLE)
        elif manipulation_status == 'FAILURE':
            self.transition_to(RobotState.ERROR)

    def handle_error(self):
        # Handle error state
        if self.error_resolved():
            self.transition_to(RobotState.IDLE)

    def transition_to(self, new_state):
        self.last_state = self.state
        self.state = new_state
        self.state_start_time = time.time()
        self.on_state_enter(new_state)

    def on_state_enter(self, state):
        # Actions to perform when entering a state
        pass
```

## Motion Planning

Motion planning computes feasible paths for robot movement:

### Path Planning with A*

```python
import heapq
import numpy as np

class PathPlanner:
    def __init__(self, occupancy_grid):
        self.grid = occupancy_grid  # 2D grid with 0 for free, 1 for occupied

    def a_star(self, start, goal):
        # Convert to grid coordinates
        start_grid = self.world_to_grid(start)
        goal_grid = self.world_to_grid(goal)

        # Priority queue: (cost, x, y)
        open_set = [(0, start_grid[0], start_grid[1])]
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self.heuristic(start_grid, goal_grid)}

        while open_set:
            current_cost, x, y = heapq.heappop(open_set)
            current = (x, y)

            if current == goal_grid:
                return self.reconstruct_path(came_from, current)

            for neighbor in self.get_neighbors(current):
                tentative_g_score = g_score[current] + self.distance(current, neighbor)

                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal_grid)

                    if neighbor not in [item[1:] for item in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor[0], neighbor[1]))

        return None  # No path found

    def heuristic(self, a, b):
        # Manhattan distance
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_neighbors(self, pos):
        x, y = pos
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]:
            nx, ny = x + dx, y + dy
            if (0 <= nx < self.grid.shape[0] and 0 <= ny < self.grid.shape[1] and
                self.grid[nx, ny] == 0):  # Free space
                neighbors.append((nx, ny))
        return neighbors

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def world_to_grid(self, world_pos):
        # Convert world coordinates to grid coordinates
        # This is a simplified conversion
        grid_x = int(world_pos[0] / self.resolution)
        grid_y = int(world_pos[1] / self.resolution)
        return (grid_x, grid_y)
```

### Trajectory Planning

```python
from scipy.interpolate import CubicSpline
import numpy as np

class TrajectoryPlanner:
    def __init__(self):
        self.max_velocity = 1.0  # m/s
        self.max_acceleration = 0.5  # m/s^2

    def plan_trajectory(self, waypoints, dt=0.1):
        # Plan smooth trajectory through waypoints
        n_points = len(waypoints)
        if n_points < 2:
            return []

        # Create time vector
        distances = [0]
        for i in range(1, n_points):
            dist = np.linalg.norm(np.array(waypoints[i]) - np.array(waypoints[i-1]))
            distances.append(dist + distances[-1])

        total_distance = distances[-1]
        estimated_time = total_distance / self.max_velocity
        time_points = np.linspace(0, estimated_time, n_points)

        # Create splines for each dimension
        x_coords = [wp[0] for wp in waypoints]
        y_coords = [wp[1] for wp in waypoints]

        x_spline = CubicSpline(time_points, x_coords)
        y_spline = CubicSpline(time_points, y_coords)

        # Generate trajectory points
        trajectory = []
        t = 0
        while t <= estimated_time:
            pos_x = x_spline(t)
            pos_y = y_spline(t)

            # Calculate velocity (derivative)
            vel_x = x_spline(t, 1)
            vel_y = y_spline(t, 1)

            # Calculate acceleration (second derivative)
            acc_x = x_spline(t, 2)
            acc_y = y_spline(t, 2)

            trajectory.append({
                'time': t,
                'position': (pos_x, pos_y),
                'velocity': (vel_x, vel_y),
                'acceleration': (acc_x, acc_y)
            })

            t += dt

        return trajectory
```

## Control Systems

Control systems execute planned motions with precision:

### PID Controller

```python
class PIDController:
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, dt=0.01):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt

        self.prev_error = 0
        self.integral = 0

    def update(self, error):
        # Proportional term
        p_term = self.kp * error

        # Integral term
        self.integral += error * self.dt
        i_term = self.ki * self.integral

        # Derivative term
        derivative = (error - self.prev_error) / self.dt
        d_term = self.kd * derivative

        # Store error for next iteration
        self.prev_error = error

        # Calculate output
        output = p_term + i_term + d_term
        return output

class RobotController:
    def __init__(self):
        # PID controllers for different joints/axes
        self.position_controller = PIDController(kp=2.0, ki=0.1, kd=0.05)
        self.velocity_controller = PIDController(kp=1.5, ki=0.05, kd=0.02)

    def control_position(self, current_pos, target_pos):
        error = target_pos - current_pos
        control_signal = self.position_controller.update(error)
        return control_signal

    def control_velocity(self, current_vel, target_vel):
        error = target_vel - current_vel
        control_signal = self.velocity_controller.update(error)
        return control_signal
```

### Model Predictive Control (MPC)

```python
import numpy as np
from scipy.optimize import minimize

class MPCController:
    def __init__(self, horizon=10, dt=0.1):
        self.horizon = horizon
        self.dt = dt
        self.Q = np.eye(3)  # State cost matrix
        self.R = np.eye(2)  # Control cost matrix

    def predict_states(self, initial_state, control_sequence):
        # Simple linear model prediction
        states = [initial_state]
        current_state = initial_state.copy()

        for control in control_sequence:
            # Update state based on control input
            # This is a simplified model
            new_state = current_state + control * self.dt
            states.append(new_state)
            current_state = new_state

        return states

    def cost_function(self, control_sequence, initial_state, reference_trajectory):
        states = self.predict_states(initial_state, control_sequence.reshape(-1, 2))
        cost = 0

        for i, state in enumerate(states):
            if i < len(reference_trajectory):
                # State tracking cost
                state_error = state - reference_trajectory[i]
                cost += state_error.T @ self.Q @ state_error

        # Control effort cost
        for control in control_sequence.reshape(-1, 2):
            cost += control.T @ self.R @ control

        return cost

    def compute_control(self, initial_state, reference_trajectory):
        # Optimize control sequence
        n_controls = self.horizon * 2  # 2 control inputs per step
        initial_guess = np.zeros(n_controls)

        result = minimize(
            self.cost_function,
            initial_guess,
            args=(initial_state, reference_trajectory),
            method='SLSQP'
        )

        if result.success:
            optimal_controls = result.x.reshape(-1, 2)
            return optimal_controls[0]  # Return first control
        else:
            return np.zeros(2)  # Return zero control if optimization fails
```

## Vision-Action Integration

Combining vision and action systems:

### Visual Servoing

```python
class VisualServoingController:
    def __init__(self):
        self.position_controller = PIDController(kp=1.0, ki=0.1, kd=0.05)
        self.orientation_controller = PIDController(kp=0.5, ki=0.05, kd=0.02)

    def compute_visual_servo_command(self, current_features, desired_features):
        # Calculate feature error
        position_error = desired_features[:2] - current_features[:2]
        orientation_error = desired_features[2:] - current_features[2:]

        # Compute control commands
        position_cmd = self.position_controller.update(np.linalg.norm(position_error))
        orientation_cmd = self.orientation_controller.update(orientation_error[0])

        # Convert to robot velocity
        cmd_vel = Twist()
        cmd_vel.linear.x = position_cmd * position_error[0] / (np.linalg.norm(position_error) + 1e-6)
        cmd_vel.linear.y = position_cmd * position_error[1] / (np.linalg.norm(position_error) + 1e-6)
        cmd_vel.angular.z = orientation_cmd

        return cmd_vel
```

### Grasping with Vision Feedback

```python
class VisionGuidedGrasping:
    def __init__(self):
        self.vision_system = None  # Object detection system
        self.motion_planner = None  # Motion planning system
        self.gripper_controller = None  # Gripper control system

    def grasp_object(self, object_type):
        # Detect object
        object_info = self.vision_system.detect_object(object_type)
        if not object_info.success:
            return False

        # Calculate grasp pose
        grasp_pose = self.calculate_grasp_pose(object_info)

        # Plan approach trajectory
        approach_trajectory = self.motion_planner.plan_trajectory_to_grasp(grasp_pose)

        # Execute approach
        success = self.execute_trajectory(approach_trajectory)
        if not success:
            return False

        # Close gripper
        self.gripper_controller.close()

        # Lift object
        lift_trajectory = self.calculate_lift_trajectory(grasp_pose)
        success = self.execute_trajectory(lift_trajectory)

        return success

    def calculate_grasp_pose(self, object_info):
        # Calculate optimal grasp pose based on object properties
        # This would involve object shape, orientation, and grasp point selection
        object_center = object_info.center
        object_orientation = object_info.orientation

        # Calculate grasp pose (simplified)
        grasp_pose = {
            'position': object_center + np.array([0, 0, 0.1]),  # 10cm above object
            'orientation': self.calculate_approach_orientation(object_orientation)
        }

        return grasp_pose
```

## Task Planning

Task planning decomposes high-level goals into sequences of actions:

### STRIPS-style Planning

```python
class STRIPSPlanner:
    def __init__(self):
        self.actions = []
        self.initial_state = set()
        self.goal_state = set()

    def add_action(self, name, preconditions, effects):
        self.actions.append({
            'name': name,
            'preconditions': set(preconditions),
            'effects': set(effects)
        })

    def plan(self, initial_state, goal_state):
        # Simple backward chaining planner
        plan = []
        current_state = set(goal_state)

        while not current_state.issubset(set(initial_state)):
            applicable_actions = []
            for action in self.actions:
                # Check if action effects can satisfy any goal condition
                if action['effects'] & current_state:
                    # Check if preconditions can be satisfied
                    if self.can_satisfy_preconditions(action['preconditions'], initial_state):
                        applicable_actions.append(action)

            if not applicable_actions:
                return None  # No solution found

            # Choose first applicable action
            chosen_action = applicable_actions[0]
            plan.insert(0, chosen_action['name'])

            # Update current state with action preconditions
            current_state = current_state - chosen_action['effects']
            current_state = current_state | chosen_action['preconditions']

        return plan

    def can_satisfy_preconditions(self, preconditions, initial_state):
        # Check if preconditions can be satisfied from initial state
        # This is a simplified check
        return True
```

### Hierarchical Task Networks (HTN)

```python
class HTNPlanner:
    def __init__(self):
        self.tasks = {}
        self.methods = {}

    def define_task(self, task_name, subtasks):
        self.tasks[task_name] = subtasks

    def define_method(self, method_name, task_name, decomposition):
        if task_name not in self.methods:
            self.methods[task_name] = []
        self.methods[task_name].append({
            'name': method_name,
            'decomposition': decomposition
        })

    def plan(self, task, state):
        if task not in self.tasks:
            # Primitive task - return as is
            return [task]

        if task in self.methods:
            # Try each method for this task
            for method in self.methods[task]:
                decomposition = method['decomposition']
                plan = []
                success = True

                for subtask in decomposition:
                    subplan = self.plan(subtask, state)
                    if subplan is None:
                        success = False
                        break
                    plan.extend(subplan)

                if success:
                    return plan

        return None  # No valid method found
```

## Execution Monitoring

Monitoring and handling execution failures:

### Action Monitoring

```python
class ActionMonitor:
    def __init__(self):
        self.active_actions = {}
        self.timeout_threshold = 30.0  # seconds

    def start_action(self, action_id, expected_duration):
        self.active_actions[action_id] = {
            'start_time': time.time(),
            'expected_duration': expected_duration,
            'status': 'RUNNING'
        }

    def check_action_status(self, action_id):
        if action_id not in self.active_actions:
            return 'UNKNOWN'

        action_info = self.active_actions[action_id]
        elapsed_time = time.time() - action_info['start_time']

        # Check for timeout
        if elapsed_time > action_info['expected_duration'] * 2:
            action_info['status'] = 'TIMEOUT'
            return 'TIMEOUT'

        # Check for external status updates
        # This would come from action servers or other systems
        return action_info['status']

    def complete_action(self, action_id, success):
        if action_id in self.active_actions:
            del self.active_actions[action_id]

    def handle_failure(self, action_id):
        # Handle action failure
        # This might involve recovery strategies
        print(f"Action {action_id} failed, initiating recovery...")
```

### Recovery Strategies

```python
class RecoveryManager:
    def __init__(self):
        self.recovery_strategies = {
            'navigation_failure': self.recovery_navigation,
            'manipulation_failure': self.recovery_manipulation,
            'detection_failure': self.recovery_detection
        }

    def handle_failure(self, failure_type, context):
        if failure_type in self.recovery_strategies:
            strategy = self.recovery_strategies[failure_type]
            return strategy(context)
        else:
            return False  # No recovery strategy available

    def recovery_navigation(self, context):
        # Try alternative navigation approach
        current_pos = context['current_position']
        goal_pos = context['goal_position']

        # Try different path planning algorithm
        alternative_path = self.plan_alternative_path(current_pos, goal_pos)
        if alternative_path:
            return self.execute_path(alternative_path)

        # Try going around obstacle
        detour_path = self.plan_detour_path(current_pos, goal_pos)
        if detour_path:
            return self.execute_path(detour_path)

        return False

    def recovery_manipulation(self, context):
        # Try different grasp approach
        object_info = context['object_info']

        # Try different grasp points
        alternative_grasps = self.generate_alternative_grasps(object_info)
        for grasp in alternative_grasps:
            success = self.attempt_grasp(grasp)
            if success:
                return True

        return False
```

## Integration Example: Fetch and Carry Task

```python
class FetchAndCarryTask:
    def __init__(self):
        self.state_machine = RobotStateMachine()
        self.behavior_tree = RobotBehaviorTree()
        self.planner = PathPlanner()
        self.controller = RobotController()
        self.vision_system = None
        self.manipulator = None

    def execute_fetch_and_carry(self, object_type, destination):
        # High-level task: go to object, pick it up, bring to destination

        # 1. Navigate to object location
        object_location = self.find_object_location(object_type)
        if not object_location:
            return False

        navigation_success = self.navigate_to(object_location)
        if not navigation_success:
            return False

        # 2. Detect and grasp object
        grasp_success = self.grasp_object(object_type)
        if not grasp_success:
            return False

        # 3. Navigate to destination
        navigation_success = self.navigate_to(destination)
        if not navigation_success:
            return False

        # 4. Release object
        self.release_object()

        return True

    def find_object_location(self, object_type):
        # Use vision system to find object
        detection = self.vision_system.detect_object(object_type)
        if detection.success:
            return detection.location
        return None

    def navigate_to(self, target_location):
        # Plan and execute navigation
        path = self.planner.plan_path(self.get_robot_position(), target_location)
        if path:
            return self.execute_navigation_path(path)
        return False

    def grasp_object(self, object_type):
        # Detect object and perform grasping
        object_info = self.vision_system.detect_object(object_type)
        if not object_info.success:
            return False

        grasp_pose = self.calculate_grasp_pose(object_info)
        return self.manipulator.grasp_at_pose(grasp_pose)
```

## Performance Optimization

Action systems require careful optimization for real-time performance:

### Multi-threading for Parallel Processing

```python
import threading
import queue
import time

class ParallelActionSystem:
    def __init__(self):
        self.action_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.threads = []

        # Start worker threads
        for i in range(4):  # 4 worker threads
            thread = threading.Thread(target=self.worker_loop)
            thread.daemon = True
            thread.start()
            self.threads.append(thread)

    def worker_loop(self):
        while True:
            try:
                action = self.action_queue.get(timeout=1.0)
                result = self.execute_action(action)
                self.result_queue.put(result)
                self.action_queue.task_done()
            except queue.Empty:
                continue

    def execute_action(self, action):
        # Execute the action and return result
        # This is where the actual action execution happens
        return {'action': action, 'success': True, 'timestamp': time.time()}
```

### Resource Management

```python
class ResourceManager:
    def __init__(self):
        self.resources = {
            'arm': {'busy': False, 'priority': 0},
            'navigation': {'busy': False, 'priority': 0},
            'vision': {'busy': False, 'priority': 0}
        }

    def acquire_resource(self, resource_type, priority=0, timeout=5.0):
        start_time = time.time()

        while time.time() - start_time < timeout:
            if not self.resources[resource_type]['busy']:
                self.resources[resource_type]['busy'] = True
                self.resources[resource_type]['priority'] = priority
                return True
            time.sleep(0.1)

        return False  # Timeout

    def release_resource(self, resource_type):
        self.resources[resource_type]['busy'] = False
        self.resources[resource_type]['priority'] = 0
```

## Summary

Action systems are crucial for robotics, enabling robots to translate perception into meaningful actions. By combining planning, control, and execution monitoring, robots can perform complex tasks in dynamic environments. The integration of vision and action systems allows for adaptive behavior based on environmental feedback.

## Next Steps

In the next chapter, we'll explore conversational systems for robots that enable natural human-robot interaction.