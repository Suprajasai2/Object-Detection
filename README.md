# Object-Detection

Object detection from a video captured by a remote controlled (RC) aircraft is an important task in various applications such as search and rescue, surveillance, and agriculture. In this project, I present a method for object detection in a video captured by an RC aircraft using the Python programming language, the YOLO (You Only Look Once) algorithm, and the OpenCV library.

• The first step in the proposed method is pre-processing the video to remove noise and stabilize the frame. This is done by applying image processing techniques such as median filtering, which are provided by the OpenCV library. The stabilization of the frame is done by using Optical Flow algorithm from OpenCV, which is a technique used to estimate the motion of objects in a video.

• The next step is to apply an object detection algorithm to identify and locate objects of interest in the video. We use the YOLO algorithm, a real-time object detection algorithm that is known for its speed and accuracy. YOLO can detect multiple objects in a single frame and provides the bounding boxes coordinates of the objects. Once the objects are detected, they are tracked throughout the video using a tracking algorithm. In this project, we used the Kalman filter, a popular algorithm for tracking objects in a video.

In conclusion, the proposed method is able to detect and track objects in with high accuracy and efficiency. The proposed method can be applied to various applications such as search and rescue, surveillance, and agriculture. Additionally, the use of OpenCV library provides a lot of pre-built functions that can be used in image processing and computer vision tasks.
