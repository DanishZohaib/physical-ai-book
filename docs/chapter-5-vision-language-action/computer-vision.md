---
sidebar_position: 1
description: Computer vision systems for robotics applications
---

# Computer Vision for Robotics

## Learning Outcomes

By the end of this chapter, you should be able to:

- Understand fundamental computer vision techniques for robotics
- Implement vision-based perception systems for robots
- Integrate computer vision with robot control systems
- Evaluate the performance of vision systems in robotics applications
- Apply deep learning techniques for robot perception

## Introduction to Computer Vision in Robotics

Computer vision is a critical component of robotics that enables robots to perceive and understand their environment. Unlike traditional computer vision applications that process images for analysis, robot vision systems must process visual information in real-time to enable navigation, manipulation, and interaction with the environment.

### Challenges in Robot Vision

- **Real-time Processing**: Vision systems must operate within strict timing constraints
- **Dynamic Environments**: Robots operate in constantly changing environments
- **Motion Blur**: Robot movement can cause image blur
- **Lighting Variations**: Indoor and outdoor lighting conditions vary significantly
- **Occlusions**: Objects may be partially or fully occluded
- **Scale Variations**: Objects appear at different scales based on distance

## Camera Systems for Robotics

Robots typically use multiple camera types for different tasks:

### Monocular Cameras

Monocular cameras provide 2D image data:

```python
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class MonocularVisionNode(Node):
    def __init__(self):
        super().__init__('monocular_vision_node')
        self.bridge = CvBridge()

        # Subscribe to camera topic
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # Publishers for processed data
        self.object_pub = self.create_publisher(
            # Object detection results
        )

    def image_callback(self, msg):
        # Convert ROS image to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Process image
        processed_image = self.process_image(cv_image)

        # Publish results
        # self.publish_results(processed_image)

    def process_image(self, image):
        # Example: Simple edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return edges
```

### Stereo Vision

Stereo vision provides depth information:

```python
import cv2
import numpy as np

class StereoVisionNode(Node):
    def __init__(self):
        super().__init__('stereo_vision_node')

        # Load stereo calibration parameters
        self.left_cam_matrix = None
        self.right_cam_matrix = None
        self.distortion_coeffs = None

    def compute_disparity(self, left_image, right_image):
        # Create stereo matcher
        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

        # Compute disparity map
        gray_left = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

        disparity = stereo.compute(gray_left, gray_right)
        return disparity

    def disparity_to_depth(self, disparity):
        # Convert disparity to depth using calibration parameters
        # This is a simplified example
        baseline = 0.1  # Camera baseline in meters
        focal_length = 640  # Focal length in pixels

        # Depth = (baseline * focal_length) / disparity
        depth = (baseline * focal_length) / (disparity + 1e-6)
        return depth
```

### RGB-D Cameras

RGB-D cameras provide both color and depth information:

```python
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point

class RGBDNode(Node):
    def __init__(self):
        super().__init__('rgbd_node')

        # Subscribers for RGB and depth images
        self.rgb_sub = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.rgb_callback,
            10
        )

        self.depth_sub = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_callback,
            10
        )

        self.camera_info = None
        self.intrinsic_matrix = None

    def rgb_callback(self, msg):
        # Process RGB image
        pass

    def depth_callback(self, msg):
        # Process depth image
        depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')

        # Convert pixel coordinates to 3D points
        points_3d = self.depth_to_3d(depth_image)

    def depth_to_3d(self, depth_image):
        # Convert depth image to 3D point cloud
        height, width = depth_image.shape

        # Create coordinate grids
        u_coords, v_coords = np.meshgrid(
            np.arange(width), np.arange(height)
        )

        # Convert to 3D coordinates
        z_coords = depth_image
        x_coords = (u_coords - self.intrinsic_matrix[0, 2]) * z_coords / self.intrinsic_matrix[0, 0]
        y_coords = (v_coords - self.intrinsic_matrix[1, 2]) * z_coords / self.intrinsic_matrix[1, 1]

        # Stack coordinates
        points_3d = np.stack([x_coords, y_coords, z_coords], axis=-1)
        return points_3d
```

## Object Detection and Recognition

Object detection is fundamental for robot perception:

### Traditional Methods

```python
import cv2
import numpy as np

class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')

        # Pre-trained Haar cascade classifiers
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # Feature-based detectors
        self.sift = cv2.SIFT_create()

    def detect_faces(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return faces

    def detect_features(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        return keypoints, descriptors
```

### Deep Learning Methods

```python
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn

class DeepDetectionNode(Node):
    def __init__(self):
        super().__init__('deep_detection_node')

        # Load pre-trained model
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()

        # Image preprocessing
        self.transform = T.Compose([
            T.ToTensor(),
        ])

    def detect_objects(self, image):
        # Preprocess image
        image_tensor = self.transform(image).unsqueeze(0)

        # Run inference
        with torch.no_grad():
            predictions = self.model(image_tensor)

        # Process results
        boxes = predictions[0]['boxes'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()

        return boxes, labels, scores
```

## Visual SLAM (Simultaneous Localization and Mapping)

Visual SLAM enables robots to build maps while localizing themselves:

### Feature-Based SLAM

```python
import cv2
import numpy as np
from collections import deque

class VisualSLAMNode(Node):
    def __init__(self):
        super().__init__('visual_slam_node')

        # ORB feature detector
        self.orb = cv2.ORB_create(nfeatures=1000)

        # FLANN matcher
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6,
                           key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Pose estimation
        self.current_pose = np.eye(4)
        self.keyframes = deque(maxlen=100)
        self.map_points = []

    def process_frame(self, image):
        # Detect features
        keypoints, descriptors = self.orb.detectAndCompute(image, None)

        if len(keypoints) < 10:
            return  # Not enough features

        # Store keyframe if needed
        if len(self.keyframes) == 0 or self.is_keyframe(keypoints):
            self.keyframes.append({
                'image': image,
                'keypoints': keypoints,
                'descriptors': descriptors,
                'pose': self.current_pose.copy()
            })

        # Estimate motion if we have previous frames
        if len(self.keyframes) > 1:
            self.estimate_motion(keypoints, descriptors)

    def estimate_motion(self, curr_kp, curr_desc):
        # Match features with previous keyframe
        prev_frame = self.keyframes[-2]
        matches = self.flann.knnMatch(prev_frame['descriptors'], curr_desc, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        if len(good_matches) >= 10:
            # Extract matched points
            src_pts = np.float32([prev_frame['keypoints'][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([curr_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Estimate fundamental matrix
            F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC, 4, 0.999)

            # Estimate pose (simplified)
            # In practice, you'd use camera calibration to get essential matrix and decompose it
```

## Semantic Segmentation

Semantic segmentation provides pixel-level understanding:

```python
import torch
import torch.nn.functional as F
import torchvision.transforms as T

class SemanticSegmentationNode(Node):
    def __init__(self):
        super().__init__('semantic_segmentation_node')

        # Load pre-trained segmentation model
        self.model = torch.hub.load(
            'pytorch/vision:v0.10.0',
            'deeplabv3_resnet101',
            pretrained=True
        )
        self.model.eval()

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
        ])

    def segment_image(self, image):
        # Preprocess
        input_tensor = self.transform(image).unsqueeze(0)

        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]
            output_predictions = output.argmax(0)

        # Convert to color image for visualization
        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
        colors = (colors % 255).numpy().astype("uint8")

        # Map predictions to colors
        segmentation = colors[output_predictions.cpu().numpy()]
        return segmentation
```

## 3D Object Detection and Pose Estimation

For manipulation tasks, robots need to understand 3D object poses:

```python
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

class ObjectPoseEstimator:
    def __init__(self):
        # Predefined 3D models of objects
        self.object_models = {}

    def estimate_pose(self, point_cloud, object_type):
        # Load object model
        model_cloud = self.object_models[object_type]

        # Align point cloud to model using ICP
        threshold = 0.02  # 2cm threshold
        trans_init = np.eye(4)  # Initial alignment

        reg_p2p = o3d.pipelines.registration.registration_icp(
            point_cloud, model_cloud, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

        # Extract transformation
        transformation = reg_p2p.transformation
        rotation_matrix = transformation[:3, :3]
        translation = transformation[:3, 3]

        # Convert to rotation vector
        rotation = R.from_matrix(rotation_matrix)
        rotation_vector = rotation.as_rotvec()

        return {
            'translation': translation,
            'rotation': rotation_vector,
            'confidence': reg_p2p.fitness
        }
```

## Visual Servoing

Visual servoing uses visual feedback to control robot motion:

```python
class VisualServoingNode(Node):
    def __init__(self):
        super().__init__('visual_servoing_node')

        # PID controllers for different axes
        self.pid_x = PIDController(kp=1.0, ki=0.1, kd=0.05)
        self.pid_y = PIDController(kp=1.0, ki=0.1, kd=0.05)
        self.pid_z = PIDController(kp=1.0, ki=0.1, kd=0.05)

        self.target_pixel = None
        self.current_pixel = None

    def servo_to_target(self):
        if self.target_pixel is None or self.current_pixel is None:
            return

        # Calculate error in pixel space
        pixel_error = self.target_pixel - self.current_pixel

        # Convert pixel error to camera frame error
        camera_error = self.pixel_to_camera_error(pixel_error)

        # Generate velocity command using PID
        vel_x = self.pid_x.update(camera_error[0])
        vel_y = self.pid_y.update(camera_error[1])
        vel_z = self.pid_z.update(camera_error[2])

        # Create twist message
        twist = Twist()
        twist.linear.x = vel_x
        twist.linear.y = vel_y
        twist.linear.z = vel_z

        # Publish command
        self.cmd_pub.publish(twist)

    def pixel_to_camera_error(self, pixel_error):
        # Convert pixel coordinates to camera frame coordinates
        # This requires camera calibration parameters
        fx, fy = self.camera_info.K[0, 0], self.camera_info.K[1, 1]
        cx, cy = self.camera_info.K[0, 2], self.camera_info.K[1, 2]

        # Simplified conversion (assuming known depth)
        z_depth = 1.0  # Known depth to target
        camera_error_x = (pixel_error[0] * z_depth) / fx
        camera_error_y = (pixel_error[1] * z_depth) / fy
        camera_error_z = pixel_error[2]  # Depth error

        return np.array([camera_error_x, camera_error_y, camera_error_z])
```

## Performance Optimization

Computer vision in robotics requires careful optimization:

### Multi-threading

```python
import threading
import queue

class OptimizedVisionNode(Node):
    def __init__(self):
        super().__init__('optimized_vision_node')

        # Queues for image processing pipeline
        self.input_queue = queue.Queue(maxsize=2)
        self.output_queue = queue.Queue(maxsize=2)

        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_images)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def image_callback(self, msg):
        try:
            self.input_queue.put_nowait(msg)
        except queue.Full:
            # Drop frame if queue is full
            pass

    def process_images(self):
        while rclpy.ok():
            try:
                msg = self.input_queue.get(timeout=1.0)
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

                # Process image
                result = self.heavy_processing(cv_image)

                # Put result in output queue
                self.output_queue.put_nowait(result)

            except queue.Empty:
                continue
```

### GPU Acceleration

```python
import cupy as cp  # CUDA-accelerated NumPy
import cv2

class GPUVisionNode(Node):
    def __init__(self):
        super().__init__('gpu_vision_node')

        # Check GPU availability
        if cp.cuda.is_available():
            self.use_gpu = True
            self.gpu_device = cp.cuda.Device(0)
        else:
            self.use_gpu = False

    def gpu_image_processing(self, image):
        if self.use_gpu:
            # Transfer image to GPU
            gpu_image = cp.asarray(image)

            # Perform operations on GPU
            # Example: Gaussian blur
            # Note: This is simplified; actual GPU acceleration would use CuPy operations
            processed_image = cp.asnumpy(gpu_image)  # Convert back to CPU for ROS
        else:
            # Fallback to CPU processing
            processed_image = cv2.GaussianBlur(image, (15, 15), 0)

        return processed_image
```

## Integration with Robot Control

Computer vision systems must be tightly integrated with robot control:

```python
class VisionControlNode(Node):
    def __init__(self):
        super().__init__('vision_control_node')

        # Subscriptions
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.gripper_pub = self.create_publisher(GripperCommand, '/gripper/command', 10)

        # Vision processing
        self.vision_system = ObjectDetectionNode()

        # Control state
        self.target_object = 'cup'
        self.object_detected = False
        self.object_pose = None

    def image_callback(self, msg):
        # Process image to detect objects
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Detect objects
        objects = self.vision_system.detect_objects(cv_image)

        # Find target object
        for obj in objects:
            if obj['label'] == self.target_object:
                self.object_detected = True
                self.object_pose = self.calculate_object_pose(obj, msg.header)
                self.navigate_to_object()
                break
        else:
            self.object_detected = False

    def navigate_to_object(self):
        if not self.object_detected:
            return

        # Calculate approach vector
        approach_vector = self.calculate_approach_vector(self.object_pose)

        # Generate control commands
        twist = Twist()
        twist.linear.x = approach_vector[0] * 0.1  # Scale factor
        twist.angular.z = approach_vector[1] * 0.1  # Scale factor

        self.cmd_pub.publish(twist)

    def calculate_approach_vector(self, object_pose):
        # Calculate vector from robot to object
        # This would use robot's current pose and object pose
        return np.array([0.5, 0.1])  # Simplified example
```

## Summary

Computer vision is fundamental to robotics, enabling robots to perceive and understand their environment. By combining traditional computer vision techniques with deep learning and optimization strategies, robots can perform complex perception tasks in real-time.

## Next Steps

In the next section, we'll explore action systems that combine perception with robot control to perform complex tasks.