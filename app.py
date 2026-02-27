import time
import threading

import cv2
from flask import Flask, Response, jsonify, render_template

from .detection import detect_pothole, reset_count
from .model_loader import load_model

app = Flask(__name__)

model, device = load_model()

camera = None
is_running = False
is_paused = False
camera_lock = threading.Lock()

latest_probability = 0
latest_count = 0
latest_latitude = None
latest_longitude = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start")
def start():
    global camera, is_running, is_paused

    with camera_lock:
        if camera is None:
            cap_flag = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0
            camera = cv2.VideoCapture(0, cap_flag)

    reset_count()
    is_running = True
    is_paused = False

    return jsonify({"status": "started"})


@app.route("/pause")
def pause():
    global is_paused
    is_paused = not is_paused
    status = "paused" if is_paused else "resumed"
    return jsonify({"status": status})


@app.route("/stop")
def stop():
    global camera, is_running, is_paused

    is_running = False
    is_paused = False

    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None

    return jsonify({"status": "stopped"})


@app.route("/stats")
def stats():
    return jsonify({
        "probability": round(latest_probability, 3),
        "count": latest_count,
        "latitude": latest_latitude,
        "longitude": latest_longitude,
    })


def generate_frames():
    global latest_probability, latest_count, latest_latitude, latest_longitude

    while True:
        if not is_running or camera is None:
            time.sleep(0.1)
            blank = cv2.imencode(".jpg",cv2.UMat(480,640,cv2.CV_8UC3).get())[1].tobytes()
            yield(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + blank + b"\r\n")
            continue

        if is_paused:
            time.sleep(0.1)
            continue

        with camera_lock:
            if camera is None:
                time.sleep(0.1)
                continue

            success, frame = camera.read()

        if not success:
            time.sleep(0.05)
            continue

        try:
            frame, prob, count, lat, lon = detect_pothole(frame, model, device)

            latest_probability = prob
            latest_count = count
            latest_latitude = lat
            latest_longitude = lon

        except Exception as e:
            print("Detection error:", e)

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    app.run(debug=True)

# if __name__=="__main__":
#     app.run(host="0.0.0.0", port=5000, debug=False)