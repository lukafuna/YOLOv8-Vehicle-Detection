import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


def draw_tracks_and_boxes(frame, tracks):
    for track in tracks:
        if not track.is_confirmed():
            continue
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1)
        cv2.putText(frame, f"ID {track.track_id}", (cx - 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)


model = YOLO("../yolov8n.pt")
video_path = "../Media/Videos/HighwayVideo.mp4"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: could not open video")
    exit()

VEHICLE_CLASSES = {2, 3, 5, 7}
CONF_THRESHOLD = 0.5

tracker = DeepSort(max_age=15, n_init=1, embedder="mobilenet")

WINDOW_NAME = "Vehicle Tracking - DeepSORT"
frame_count = 0

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

    results = model(frame, verbose=False, imgsz=416)

    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())

        if cls_id in VEHICLE_CLASSES and conf > CONF_THRESHOLD:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            w, h = int(x2 - x1), int(y2 - y1)
            x1, y1 = int(x1), int(y1)

            class_name = model.names[cls_id]
            detections.append(([x1, y1, w, h], conf, class_name))


    tracks = tracker.update_tracks(detections, frame=frame)
    draw_tracks_and_boxes(frame, tracks)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if track.det_class is not None:
            id_to_class[track_id] = track.det_class

        if track_id in prev_positions:
            prev_cx, prev_cy = prev_positions[track_id]
            vehicle_type = id_to_class.get(track_id, "car")

            if LEFT_LINE_X_RANGE[0] <= cx <= LEFT_LINE_X_RANGE[1]:
                if (prev_cy < LINE_Y <= cy) or (prev_cy > LINE_Y >= cy):
                    if track_id not in counted_ids_left:
                        left_count += 1
                        counted_ids_left.add(track_id)
                        type_counts[vehicle_type] = type_counts.get(vehicle_type, 0) + 1

            if RIGHT_LINE_X_RANGE[0] <= cx <= RIGHT_LINE_X_RANGE[1]:
                if (prev_cy < LINE_Y <= cy) or (prev_cy > LINE_Y >= cy):
                    if track_id not in counted_ids_right:
                        right_count += 1
                        counted_ids_right.add(track_id)
                        type_counts[vehicle_type] = type_counts.get(vehicle_type, 0) + 1

        prev_positions[track_id] = (cx, cy)

    cv2.line(frame, (LEFT_LINE_X_RANGE[0], LINE_Y), (LEFT_LINE_X_RANGE[1], LINE_Y), LINE_COLOR, 2)
    cv2.line(frame, (RIGHT_LINE_X_RANGE[0], LINE_Y), (RIGHT_LINE_X_RANGE[1], LINE_Y), LINE_COLOR, 2)

    cv2.rectangle(frame, (20, 5), (160, 30), (255, 255, 255), -1)
    cv2.putText(frame, f"Left: {left_count}", (25, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.rectangle(frame, (460, 5), (600, 30), (255, 255, 255), -1)
    cv2.putText(frame, f"Right: {right_count}", (465, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n--- FINAL RESULTS (DeepSORT) ---")
print(f"Left lane count: {left_count}")
print(f"Right lane count: {right_count}")
print(f"Total vehicles counted: {left_count + right_count}")
print("\n--- BY VEHICLE TYPE ---")
for vehicle_type, count in type_counts.items():
    print(f"{vehicle_type}: {count}")