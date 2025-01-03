import cv2 as cv
import numpy as np
from vidgear.gears import CamGear
from ultralytics import YOLO

# Define the model and classes of interest
model = YOLO(r"C:\\Users\\User\\Downloads\\cv_weights_21122024.pt")  # replace with your model path
classes_of_interest = ['person', 'handbag', 'paperbag']
colors = {'person': (0, 255, 0), 'handbag': (0, 0, 255), 'paperbag': (255, 0, 0)}  # BGR format for OpenCV
confidence_thresholds = {'person': 0.5, 'handbag': 0.3, 'paperbag': 0.3}  # Set confidence thresholds for each class

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

# Ask the user to select the input source
input_source = input("Select input source (1 for webcam, 2 for video stream): ")

if input_source == '1':
    # Open the webcam using OpenCV
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

        # Display the frame
        cv.imshow('Frame', frame_resized)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

elif input_source == '2':
    # Ask the user for the video stream source
    video_source = input("Enter video stream URL or local file path: ")
    stream = CamGear(source=video_source, stream_mode=True, logging=True).start()

    while True:
        frame = stream.read()
        if frame is None or exit_program:
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

        # Display the frame
        cv.imshow('Frame', frame_resized)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    stream.stop()
    cv.destroyAllWindows()

else:
    print("Invalid input source selected. Please restart the program and select either 1 or 2.")