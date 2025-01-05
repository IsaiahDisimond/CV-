# Object Detection with YOLOv8

## Overview
This project implements an object detection system using the YOLOv8 model. It allows users to detect and count specific object classes in real-time from a webcam feed or a video stream. The detected objects are displayed with bounding boxes and labeled with their confidence scores.

## Features
- **Real-time object detection** using the YOLOv8 model.
- **Supports multiple input sources**: webcam or video stream.
- **Tracks and counts unique instances** of specified object classes.
- **Non-Maximum Suppression (NMS)** applied to remove duplicate detections.
- **Configurable confidence thresholds** for different object classes.
- **Pause and resume functionality**.

## Installation
### Prerequisites
Ensure you have Python 3.8+ installed along with the following dependencies:

```sh
pip install opencv-python numpy ultralytics vidgear
```

## Usage
1. **Download the YOLO model weights** and update the `model` path in the script:
   ```python
   model = YOLO(r"C:\\Users\\User\\Downloads\\cv_weights_21122024.pt")
   ```
2. **Run the script**:
   ```sh
   python object_detection.py
   ```
3. **Select the input source**:
   - Enter `1` for webcam.
   - Enter `2` for a video stream (provide URL or file path).
4. **Control the execution**:
   - Press `p` to pause/resume detection.
   - Press `q` to quit the program.

## Configuration
Modify the following parameters in the script:
- `classes_of_interest`: Specify the object classes to detect (e.g., `'person', 'handbag', 'paperbag'`).
- `confidence_thresholds`: Set confidence thresholds for different classes.
- `colors`: Define bounding box colors for each class.
- `count_interval`: Set the time interval (in seconds) for displaying counts.

## How It Works
1. The script loads the YOLOv8 model and processes frames from the selected input source.
2. Objects are detected, and bounding boxes are drawn with confidence scores.
3. A Non-Maximum Suppression (NMS) algorithm filters overlapping detections.
4. Unique object instances are tracked, and their counts are updated at specified intervals.
5. The user can pause/resume or quit the application using keyboard inputs.

## Example Output
When a detected object appears on screen, the output includes:
- A bounding box around the object.
- The object's class label and confidence score.
- Real-time count of detected objects displayed on the frame.

## License
This project is open-source and available for personal and academic use. Modify as needed!

