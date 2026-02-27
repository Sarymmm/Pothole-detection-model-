# Pothole-detection-model-
Real time pothole detection system built using Computer Vision and Deep Learning build on flask
The system detects potholes from a live camera feed, displays detection probability, counts detections, and simulates GPS coordinates via a web dashboard.

Project Overview

This project consists of:

A custom-trained deep learning model

A real-time video streaming system

A Flask-based interactive dashboard

Detection statistics and GPS simulation

The system captures frames from a camera, runs them through a trained model, and displays:

✅ Live video stream

✅ Detection count

✅ Confidence probability

✅ Simulated GPS coordinates

pothole-detection-model/
│
├── app/
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css
│   │   └── js/
│   │       └── main.js
│   │
│   ├── templates/
│   │   └── index.html
│   │
│   ├── app.py
│   ├── detection.py
│   └── model_loader.py
│
├── dataset/
│   ├── potholes/          (2500 pothole images)
│   └── normal_road/       (2500 normal road images)
│
├── training/
│   ├── model.py
│   └── train_model.py
│
└── README.md




How the System Works
1️⃣ Model Training

Dataset contains:

2500 pothole images

2500 normal road images

The model is defined in:

training/model.py

Training script:

training/train_model.py

After training, the model weights are saved and loaded by the web application.





Run Application

From project root:

python -m app.app
