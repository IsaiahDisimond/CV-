import cv2 as cv
import numpy as np
from vidgear.gears import CamGear
from ultralytics import YOLO
import time

# Define the model and classes of interest
model = YOLO(r"C:\\Users\\User\\Downloads\\cv_weights_21122024.pt")
classes_of_interest = ['person', 'handbag', 'paperbag']
colors = {'person': (0, 255, 0), 'handbag': (0, 0, 255), 'paperbag': (255, 0, 0)}  # BGR format
confidence_thresholds = {'person': 0.5, 'handbag': 0.3, 'paperbag': 0.3}

# Variables for counting and timing
class_counts = {cls: set() for cls in classes_of_interest}  # Using sets to track unique instances
last_count_time = time.time()
count_interval = 60  # 1 minute interval in seconds
last_positions = {cls: [] for cls in classes_of_interest}  # Store last known positions of objects

# Function to handle key press events
pause = False
exit_program = False

def on_key(event):
    global pause, exit_program
    if event.key == 'q':
        exit_program = True
    elif event.key == 'p':
        pause = not pause

# Non-Maximum Suppression (NMS) function
def non_max_suppression(boxes, scores, iou_threshold):
    indices = cv.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=iou_threshold)
    if len(indices) > 0:
        indices = indices.flatten()
    return [boxes[i] for i in indices]

def display_counts():
    """Display the current counts for each class"""
    print("\n=== Object Counts (Last 1 minute) ===")
    for cls in classes_of_interest:
        print(f"{cls}: {len(class_counts[cls])}")

def is_new_instance(box, cls_name):
    """Check if the detected box is a new instance of the class"""
    x1, y1, x2, y2 = box
    for (lx1, ly1, lx2, ly2) in last_positions[cls_name]:
        if abs(x1 - lx1) < 50 and abs(y1 - ly1) < 50 and abs(x2 - lx2) < 50 and abs(y2 - ly2) < 50:
            return False
    return True

# Main loop for processing frames
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret or exit_program:
        break

    if pause:
        cv.waitKey(100)
        continue

    # Resize frame for faster processing
    frame_resized = cv.resize(frame, (640, 480))

    # Run YOLOv8 inference on the frame
    results = model(frame_resized)

    # Filter results for classes of interest and assign unique identifiers
    filtered_results = []
    instance_counters = {cls: 1 for cls in classes_of_interest}
    boxes = []
    scores = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls)
            cls_name = model.names[cls_id]
            conf = box.conf[0]
            if cls_name in classes_of_interest and conf >= confidence_thresholds[cls_name]:
                instance_id = instance_counters[cls_name]
                instance_counters[cls_name] += 1
                filtered_results.append((box, cls_name, instance_id))
                boxes.append([int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])])
                scores.append(float(conf))

    # Apply Non-Maximum Suppression (NMS)
    nms_boxes = non_max_suppression(boxes, scores, iou_threshold=0.5)

    # Draw bounding boxes and labels
    for box, cls_name, instance_id in filtered_results:
        x1, y1, x2, y2 = box.xyxy[0]
        conf = box.conf[0]
        if [int(x1), int(y1), int(x2), int(y2)] in nms_boxes:
            cv.rectangle(frame_resized, (int(x1), int(y1)), (int(x2), int(y2)), colors[cls_name], 2)
            cv.putText(frame_resized, f'{cls_name} {instance_id} {conf:.2f}', (int(x1), int(y1) - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, colors[cls_name], 2)
            if is_new_instance([int(x1), int(y1), int(x2), int(y2)], cls_name):
                class_counts[cls_name].add(instance_id)
                last_positions[cls_name].append([int(x1), int(y1), int(x2), int(y2)])

    # Display the frame
    cv.imshow('Frame', frame_resized)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    # Display counts every minute
    if time.time() - last_count_time > count_interval:
        display_counts()
        last_count_time = time.time()
        class_counts = {cls: set() for cls in classes_of_interest}  # Reset counts
        last_positions = {cls: [] for cls in classes_of_interest}  # Reset positions

cap.release()
cv.destroyAllWindows()
