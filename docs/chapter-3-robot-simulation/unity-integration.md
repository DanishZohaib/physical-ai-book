---
sidebar_position: 2
description: Unity integration for robot simulation and development
---

# Unity Integration for Robot Simulation

## Learning Outcomes

By the end of this chapter, you should be able to:

- Set up Unity for robotics simulation
- Create robot models and environments in Unity
- Implement robot control systems using Unity
- Integrate Unity with ROS/ROS 2 for robot development
- Evaluate the benefits and limitations of Unity for robotics

## Introduction to Unity for Robotics

Unity is a powerful game engine that has found significant applications in robotics research and development. While traditionally used for game development, Unity's physics engine, rendering capabilities, and scripting flexibility make it an excellent platform for robotics simulation and development.

### Why Unity for Robotics?

- **High-Fidelity Graphics**: Realistic rendering for computer vision applications
- **Flexible Physics**: PhysX physics engine for realistic simulation
- **Cross-Platform**: Deploy to various platforms and hardware
- **Asset Store**: Extensive library of models and tools
- **Scripting**: C# scripting for complex robot behaviors
- **VR/AR Support**: Natural integration with virtual and augmented reality

## Unity Robotics Hub

Unity provides the Robotics Hub package that facilitates robotics development:

### Installation

1. Install Unity Hub and Unity 2021.3 LTS or later
2. Install the Unity Robotics Hub package
3. Import the required packages for robotics simulation

### Key Components

- **ROS TCP Connector**: Enables communication with ROS/ROS 2
- **ML-Agents**: Machine learning framework for training robot behaviors
- **Synthetic Data Tools**: Generate training data for perception systems
- **Simulation Framework**: Tools for creating robot simulation environments

## Creating Robot Models in Unity

Unlike URDF/SDF in Gazebo, Unity uses its own format for 3D models:

### Basic Robot Structure

```csharp
using UnityEngine;

public class RobotController : MonoBehaviour
{
    public float moveSpeed = 5.0f;
    public float turnSpeed = 100.0f;

    // Components
    public Transform chassis;
    public Transform[] wheels;

    // Physics
    private Rigidbody rb;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
    }

    void Update()
    {
        // Basic movement controls
        float moveInput = Input.GetAxis("Vertical");
        float turnInput = Input.GetAxis("Horizontal");

        Vector3 movement = transform.forward * moveInput * moveSpeed * Time.deltaTime;
        transform.Translate(movement, Space.World);

        transform.Rotate(Vector3.up, turnInput * turnSpeed * Time.deltaTime);
    }
}
```

### Physics Setup

In Unity, physics properties are set using Rigidbody components:

- **Mass**: Set in the Rigidbody component
- **Drag**: Controls motion resistance
- **Angular Drag**: Controls rotational resistance
- **Material**: Set physics material for friction properties

## ROS/Unity Integration

The ROS TCP Connector enables communication between Unity and ROS:

### Setup

1. Add the ROS TCP Connector component to your scene
2. Configure IP address and port for ROS communication
3. Create custom message publishers/subscribers

### Example: Publishing Robot State

```csharp
using System.Collections;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Std;
using UnityEngine;

public class RobotStatePublisher : MonoBehaviour
{
    private ROSConnection ros;
    private float publishFrequency = 10f; // 10 Hz

    void Start()
    {
        ros = ROSConnection.instance;
    }

    void Update()
    {
        if (Time.time % (1.0f / publishFrequency) < Time.deltaTime)
        {
            // Publish robot position
            var positionMsg = new Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry.PoseMsg
            {
                position = new Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry.PointMsg
                {
                    x = transform.position.x,
                    y = transform.position.y,
                    z = transform.position.z
                }
            };

            ros.Send("robot_pose", positionMsg);
        }
    }
}
```

## Practical Example: Unity Robot Simulation

Let's create a simple robot simulation:

### Scene Setup

1. Create a new 3D scene
2. Add a ground plane
3. Create robot model using primitives or imported models
4. Add physics materials and colliders
5. Attach control scripts

### Robot Control Script

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry;

public class UnityRobotController : MonoBehaviour
{
    [Header("Robot Configuration")]
    public float maxSpeed = 2.0f;
    public float maxAngularSpeed = 1.0f;

    [Header("Components")]
    public Transform leftWheel;
    public Transform rightWheel;
    public float wheelRadius = 0.1f;

    private ROSConnection ros;
    private float leftWheelVelocity = 0f;
    private float rightWheelVelocity = 0f;

    void Start()
    {
        ros = ROSConnection.instance;
        // Subscribe to velocity commands
        ros.Subscribe<TwistMsg>("cmd_vel", OnVelocityCommand);
    }

    void OnVelocityCommand(TwistMsg cmd)
    {
        // Convert linear and angular velocity to wheel velocities
        float linearVel = (float)cmd.linear.x;
        float angularVel = (float)cmd.angular.z;

        // Differential drive kinematics
        leftWheelVelocity = linearVel - (angularVel * 0.5f); // Assuming 1m track width
        rightWheelVelocity = linearVel + (angularVel * 0.5f);
    }

    void Update()
    {
        // Apply wheel rotation based on velocities
        if (leftWheel != null)
        {
            leftWheel.Rotate(Vector3.right, leftWheelVelocity * Time.deltaTime / wheelRadius * Mathf.Rad2Deg);
        }

        if (rightWheel != null)
        {
            rightWheel.Rotate(Vector3.right, rightWheelVelocity * Time.deltaTime / wheelRadius * Mathf.Rad2Deg);
        }

        // Update robot position based on wheel velocities
        float avgVelocity = (leftWheelVelocity + rightWheelVelocity) / 2.0f;
        transform.Translate(Vector3.forward * avgVelocity * Time.deltaTime);

        float angularVelocity = (rightWheelVelocity - leftWheelVelocity) / 1.0f; // track width
        transform.Rotate(Vector3.up, angularVelocity * Time.deltaTime * Mathf.Rad2Deg);
    }
}
```

## Computer Vision in Unity

Unity excels at generating synthetic data for computer vision:

### Camera Setup

```csharp
using UnityEngine;

public class RgbdCamera : MonoBehaviour
{
    public Camera rgbCamera;
    public Camera depthCamera;

    void Start()
    {
        SetupCameras();
    }

    void SetupCameras()
    {
        // Configure RGB camera
        rgbCamera = GetComponent<Camera>();
        rgbCamera.depth = 0;

        // Create depth camera
        GameObject depthCamObj = new GameObject("DepthCamera");
        depthCamObj.transform.SetParent(transform);
        depthCamObj.transform.localPosition = Vector3.zero;
        depthCamObj.transform.localRotation = Quaternion.identity;

        depthCamera = depthCamObj.AddComponent<Camera>();
        depthCamera.depth = -1; // Render after RGB camera
        depthCamera.backgroundColor = Color.black;
        depthCamera.clearFlags = CameraClearFlags.SolidColor;
    }

    void Update()
    {
        // Capture RGB and depth images
        CaptureImages();
    }

    void CaptureImages()
    {
        // Implementation for capturing synthetic images
        // This would typically involve custom shaders and image processing
    }
}
```

## Unity ML-Agents Integration

Unity's ML-Agents framework allows for training AI agents in simulation:

### Basic Agent Setup

```csharp
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class RobotAgent : Agent
{
    public override void OnEpisodeBegin()
    {
        // Reset robot position and state
        transform.position = new Vector3(0, 0.5f, 0);
        transform.rotation = Quaternion.identity;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Add observations about the environment
        sensor.AddObservation(transform.position);
        sensor.AddObservation(transform.rotation);

        // Add sensor data if available
        // sensor.AddObservation(GetLidarData());
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        // Process actions from the neural network
        float forward = actions.ContinuousActions[0];
        float turn = actions.ContinuousActions[1];

        // Apply actions to robot
        transform.Translate(Vector3.forward * forward * Time.deltaTime);
        transform.Rotate(Vector3.up, turn * Time.deltaTime);

        // Provide rewards
        SetReward(CalculateReward());
    }

    float CalculateReward()
    {
        // Calculate reward based on task
        return 0f; // Implement reward logic
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // Manual control for testing
        var continuousActionsOut = actionsOut.ContinuousActions;
        continuousActionsOut[0] = Input.GetAxis("Vertical");
        continuousActionsOut[1] = Input.GetAxis("Horizontal");
    }
}
```

## Best Practices for Unity Robotics

1. **Performance Optimization**: Optimize scenes for real-time simulation
2. **Physics Accuracy**: Tune physics parameters for realistic behavior
3. **ROS Integration**: Use proper message types and communication patterns
4. **Testing**: Validate simulation results with real-world data when possible
5. **Modular Design**: Create reusable components for different robot types

## Summary

Unity provides a powerful platform for robotics simulation with high-fidelity graphics and flexible physics. The integration with ROS/ROS 2 and ML-Agents makes it particularly valuable for computer vision and machine learning applications in robotics.

## Next Steps

In the next chapter, we'll explore NVIDIA Isaac Sim and Isaac ROS for advanced robotics simulation and development.