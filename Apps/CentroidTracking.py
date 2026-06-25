import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from ultralytics import YOLO


class CentroidTracker:
    def __init__(self, max_disappeared=8, max_distance=50):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, centroids):
        if len(centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        if len(self.objects) == 0:
            for c in centroids:
                self.register(c)
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        D = cdist(np.array(object_centroids), np.array(centroids))
        rows, cols = linear_sum_assignment(D)

        used_rows, used_cols = set(), set()
        for row, col in zip(rows, cols):
            if D[row, col] > self.max_distance:
                continue
            object_id = object_ids[row]
            self.objects[object_id] = centroids[col]
            self.disappeared[object_id] = 0
            used_rows.add(row)
            used_cols.add(col)

        for row in set(range(len(object_centroids))) - used_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)

        for col in set(range(len(centroids))) - used_cols:
            self.register(centroids[col])

        return self.objects


def draw_tracks(frame, tracked_objects):
    for object_id, (cx, cy) in tracked_objects.items():
        cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1)
        cv2.putText(frame, f"ID {object_id}", (cx - 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)


def draw_boxes(frame, boxes):
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)


model = YOLO("../yolov8n.pt")
video_path = "../Media/Videos/HighwayVideo.mp4"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: could not open video")
    exit()

VEHICLE_CLASSES = {2, 3, 5, 7}
CONF_THRESHOLD = 0.5
tracker = CentroidTracker(max_disappeared=8, max_distance=50)

WINDOW_NAME = "Vehicle Tracking"
frame_count = 0
PROCESS_EVERY_N = 2
last_tracked_objects = {}
last_boxes = []

LINE_Y = 180
LEFT_LINE_X_RANGE = (5, 300)
RIGHT_LINE_X_RANGE = (300, 635)

LINE_COLOR = (0, 200, 255)

left_count = 0
right_count = 0
counted_ids_left = set()
counted_ids_right = set()
prev_positions = {}

id_to_class = {}
type_counts = {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    frame = cv2.resize(frame, (640, 360))

    if frame_count % PROCESS_EVERY_N != 0:
        draw_boxes(frame, last_boxes)
        draw_tracks(frame, last_tracked_objects)
    else:
        results = model(frame, verbose=False, imgsz=480)

        centroids = []
        current_boxes = []
        detected_classes = []

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id in VEHICLE_CLASSES and conf > CONF_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                centroids.append((cx, cy))
                current_boxes.append((x1, y1, x2, y2))
                detected_classes.append(model.names[cls_id])

        last_boxes = current_boxes
        draw_boxes(frame, current_boxes)

        tracked_objects = tracker.update(centroids)
        last_tracked_objects = tracked_objects
        draw_tracks(frame, tracked_objects)

        for i, centroid in enumerate(centroids):
            for object_id, tracked_centroid in tracked_objects.items():
                if tracked_centroid == centroid:
                    id_to_class[object_id] = detected_classes[i]
                    break

        for object_id, (cx, cy) in tracked_objects.items():
            if object_id in prev_positions:
                prev_cx, prev_cy = prev_positions[object_id]
                vehicle_type = id_to_class.get(object_id, "car")

                if LEFT_LINE_X_RANGE[0] <= cx <= LEFT_LINE_X_RANGE[1]:
                    if prev_cy < LINE_Y <= cy or prev_cy > LINE_Y >= cy:
                        if object_id not in counted_ids_left:
                            left_count += 1
                            counted_ids_left.add(object_id)
                            type_counts[vehicle_type] += 1

                if RIGHT_LINE_X_RANGE[0] <= cx <= RIGHT_LINE_X_RANGE[1]:
                    if prev_cy < LINE_Y <= cy or prev_cy > LINE_Y >= cy:
                        if object_id not in counted_ids_right:
                            right_count += 1
                            counted_ids_right.add(object_id)
                            type_counts[vehicle_type] += 1

            prev_positions[object_id] = (cx, cy)

    cv2.line(frame, (LEFT_LINE_X_RANGE[0], LINE_Y),
              (LEFT_LINE_X_RANGE[1], LINE_Y), LINE_COLOR, 2)
    cv2.line(frame, (RIGHT_LINE_X_RANGE[0], LINE_Y),
              (RIGHT_LINE_X_RANGE[1], LINE_Y), LINE_COLOR, 2)

    cv2.rectangle(frame, (20, 5), (160, 30), (255, 255, 255), -1)
    cv2.putText(frame, f"Left: {left_count}", (25, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.rectangle(frame, (460, 5), (600, 30), (255, 255, 255), -1)
    cv2.putText(frame, f"Right: {right_count}", (465, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.imshow(WINDOW_NAME, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n--- FINAL RESULTS ---")
print(f"Left lane count: {left_count}")
print(f"Right lane count: {right_count}")
print(f"Total vehicles counted: {left_count + right_count}")
print("\n--- BY VEHICLE TYPE ---")
for vehicle_type, count in type_counts.items():
    print(f"{vehicle_type}: {count}")