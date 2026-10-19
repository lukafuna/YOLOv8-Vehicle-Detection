## YOLOv8 Vehicle Detection

Real-time vehicle detection in video using Ultralytics YOLOv8. 
The project runs a pretrained YOLOv8 nano model on video files, draws bounding boxes with class labels and confidence scores,
and counts the number and type of vehicles that pass the digital line. Both a Centroid and a DeepSORT tracker are implemented and
compared. 

## Short demo

<img width="800" height="451" alt="gif-yolo" src="https://github.com/user-attachments/assets/34311d76-b2fe-466b-aafd-fb2cbe6a0155"/>


## Features
- Detects vehicles in video using the pretrained yolov8n.pt model (COCO dataset)
- Detected classes: car, motorcycle, bus, truck
- Draws bounding boxes, class names and confidence scores on every frame
- Counts vehicles crossing a line / tracks objects across frames

## How it works
1. Each frame of the input video is read with OpenCV.
2. The frame is passed to the YOLOv8 model, which returns bounding boxes, class IDs and confidence scores.
3. Detections that belong to vehicle classes are kept, and the rest are ignored.
4. The boxes and labels are drawn onto the frame and displayed (or saved).

### Comparison

| | Centroid | DeepSORT |
|---|---|---|
| Uses motion prediction | No | Yes (Kalman filter) |
| Uses appearance | No | Yes (re-ID embedding) |
| Handles occlusion | Poorly | Well |
| ID switches | More | Fewer |
| Speed | Fastest | Slower |
| Complexity | Low | Higher |


