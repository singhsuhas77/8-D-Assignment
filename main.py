import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv9e model
model = YOLO("yolov9e.pt")

# Input/output paths
input_path = "Port.mp4"
output_path = "Port_processed.mp4"

cap = cv2.VideoCapture(input_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Allowed port-related objects + train + airplane
allowed_classes = {
    "boat", "ship", "person", "truck", "crane", "container",
    "train", "airplane"
}

# Fixed colors per category (BGR format)
fixed_colors = {
    "boat": (255, 0, 0),         # Blue
    "ship": (255, 100, 0),       # Orange-blue
    "person": (0, 255, 0),       # Green
    "truck": (0, 0, 255),        # Red
    "crane": (0, 255, 255),      # Yellow
    "container": (200, 200, 200),# Grey
    "train": (255, 0, 255),      # Purple
    "airplane": (0, 165, 255)    # Orange
}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # --- Adaptive brightness check ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    brightness = hsv[:, :, 2].mean()

    gamma = None
    if brightness < 40:
        gamma = 1.8
    elif brightness < 70:
        gamma = 1.5
    elif brightness < 100:
        gamma = 1.3

    if gamma:
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in range(256)]).astype("uint8")
        frame = cv2.LUT(frame, table)

    # Run YOLOv9e inference
    results = model.predict(frame, conf=0.35)

    for res in results:
        for box in res.boxes:
            cls = int(box.cls[0])
            class_name = model.names[cls].lower()

            # Only allow selected objects
            if class_name not in allowed_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = f"{class_name} {conf:.2f}"

            # Use fixed color (fallback = white if not defined)
            color = fixed_colors.get(class_name, (255, 255, 255))

            # Draw bounding box + label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Processed video saved as {output_path}")
