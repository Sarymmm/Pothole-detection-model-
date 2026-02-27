import random

import cv2
import torch

detection_count = 0


def reset_count():
    global detection_count
    detection_count = 0


def detect_pothole(frame, model, device):
    global detection_count

    img = cv2.resize(frame, (128, 128))
    img = img / 255.0
    img = (
        torch.tensor(img, dtype=torch.float32)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        output = model(img)
        probs=torch.softmax(output, dim=1)                      
        probability = probs[0,1].item()  
    latitude = None
    longitude = None

    if probability > 0.9:
        detection_count += 1

        h, w, _ = frame.shape
        x1, y1 = w // 4, h // 4
        x2, y2 = w * 3 // 4, h * 3 // 4

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(
            frame,
            f"Pothole {probability:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

        # Simulated GPS near fixed base point
        latitude = 37.7749 + random.uniform(-0.01, 0.01)
        longitude = -122.4194 + random.uniform(-0.01, 0.01)

    return frame, probability, detection_count, latitude, longitude