# Traffic-Light-Detection

A Streamlit application that detects traffic lights in images or videos and classifies their state as **red**, **yellow**, or **green** using a YOLOv8 model.

## Features

- Detects traffic lights in uploaded images
- Detects traffic lights in uploaded videos
- Visualizes bounding boxes and labels on detections
- Shows detection confidence scores
- Allows video download after processing

## Requirements

- Python 3.9+
- `streamlit`
- `ultralytics`
- `opencv-python`
- `Pillow`
- `numpy`

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Files

- `app.py` — Streamlit application entry point
- `requirements.txt` — Python dependencies
- `model_export/best.pt` — exported YOLO model weights used for detection
- `notebook/traffic_light_detection.ipynb` — notebook used for training or evaluation
- `notebook/yolov8n.pt` — a YOLOv8 model checkpoint used in the notebook

## Usage

Run the app from the project root:

```bash
streamlit run app.py
```

Then open the URL shown in the terminal, choose either **Image** or **Video**, and upload a file.

## Notes

- Make sure `model_export/best.pt` exists in the project directory.
- For video uploads, processing may take longer depending on file length and system performance.
- The confidence threshold slider can be used to filter low-confidence detections.

