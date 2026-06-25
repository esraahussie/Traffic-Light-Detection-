import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os


@st.cache_resource
def load_model():
    return YOLO(r"model_export\best.pt")

model = load_model()


st.title("Traffic Light Detection")

thresh = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

input_type = st.radio("Select Input Type", ("Image", "Video"))


if input_type == "Image":
    uploaded_file = st.file_uploader("Choose an image..", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        original_col, result_col = st.columns(2)

        with original_col:
            st.header("Original")
            st.image(image, use_container_width=True)

        with result_col:
            st.header("Detected")
            results = model.predict(np.array(image), conf=thresh)
            result_image = np.squeeze(results[0].plot())
            st.image(result_image, channels="BGR", use_container_width=True)

        st.divider()
        st.header("Results")
        boxes = results[0].boxes
        if len(boxes) == 0:
            st.info("No traffic lights detected.")
        else:
            for box in boxes:
                cls_id   = int(box.cls[0])
                conf_val = float(box.conf[0])
                label    = model.names[cls_id]
                st.write(f"**{label.upper()}** — {conf_val:.1%} confidence")
            st.write(f"**Total detections:** {len(boxes)}")


else:
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.flush()

        cap = cv2.VideoCapture(tfile.name)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = max(int(cap.get(cv2.CAP_PROP_FPS)), 1)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        st.info("Processing video — please wait...")
        progress = st.progress(0)
        stframe = st.empty()
        frame_n = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, conf=thresh, verbose=False)
            result_frame = np.squeeze(results[0].plot())

            out.write(result_frame)
            stframe.image(result_frame, channels="BGR", use_container_width=True)

            frame_n += 1
            if total > 0:
                progress.progress(min(frame_n / total, 1.0))

        cap.release()
        out.release()
        progress.progress(1.0)
        stframe.empty()

        with open(output_path, "rb") as f:
            st.download_button(
                label="Download Processed Video",
                data=f,
                file_name="traffic_detected.mp4",
                mime="video/mp4"
            )

        os.unlink(tfile.name)