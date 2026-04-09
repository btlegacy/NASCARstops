import streamlit as st
import cv2
import numpy as np
import tempfile
import pandas as pd
from ultralytics import YOLO

st.set_page_config(page_title="AI Pit Stop Timer", layout="wide")

# Load YOLO11 Nano (Fastest version for Streamlit Cloud)
@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()

def process_pit_stop(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Data tracking
    frame_indices = []
    y_coords = [] # Tracking vertical shift of the hood/background
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Focus on the top half of the frame where the 'hood drop' against the horizon is visible
        roi = frame[0:height//2, 0:width]
        
        # Inference: Track person (class 0) to confirm crew activity 
        # and use the bounding box of the car/objects to see vertical displacement
        results = model.track(frame, persist=True, classes=[0], verbose=False)
        
        # We use a simple point-feature track or the average Y of detections 
        # to find when the "world" moves down (meaning the car went up)
        if results[0].boxes.id is not None:
            avg_y = np.mean(results[0].boxes.xywh[:, 1].cpu().numpy())
            y_coords.append(avg_y)
        else:
            # Fallback to last known or mid if no one is in frame
            y_coords.append(y_coords[-1] if y_coords else height//2)
        
        frame_indices.append(count)
        count += 1
        
        if count % 15 == 0:
            progress_bar.progress(count / total_frames)
            status_text.text(f"Analyzing Frame {count}/{total_frames}...")

    cap.release()
    return np.array(frame_indices), np.array(y_coords), fps

# --- UI ---
st.title("🏁 NASCAR AI Pit Stop Analyzer")
st.info("The AI tracks crew movement and vertical chassis displacement to calculate split times.")

uploaded_file = st.file_uploader("Upload Pit Stop Video", type=["mp4", "mov"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    with st.spinner("Processing with YOLO11..."):
        indices, y_coords, fps = process_pit_stop(tfile.name)
        
        # Calculate Delta Y (Velocity of the lift)
        # We look for the peaks in movement
        dy = np.gradient(y_coords)
        
        # Heuristic to find the 4 main events based on the logic:
        # 1. Right Up (Sudden negative dy)
        # 2. Right Down (Sudden positive dy)
        # 3. Left Up (Sudden negative dy)
        # 4. Left Down (Sudden positive dy)
        
        # Simplified thresholding for the "Jumps"
        threshold = np.std(dy) * 2
        events = []
        for i in range(1, len(dy)):
            if abs(dy[i]) > threshold and abs(dy[i-1]) < threshold:
                events.append(indices[i] / fps)
        
        # Filter events to get the most distinct 4 (Start, Transition, Left, End)
        # We filter out events that are too close together (< 0.5s)
        filtered_events = []
        if len(events) > 0:
            filtered_events.append(events[0])
            for e in events:
                if e - filtered_events[-1] > 0.8:
                    filtered_events.append(e)

    if len(filtered_events) >= 4:
        t_r_start = filtered_events[0]
        t_r_end = filtered_events[1]
        t_l_start = filtered_events[2]
        t_l_end = filtered_events[3]

        # Calculations
        right_time = t_r_end - t_r_start
        transition = t_l_start - t_r_end
        left_time = t_l_end - t_l_start
        total_time = t_l_end - t_r_start

        # Display Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Right Side", f"{right_time:.2f}s")
        c2.metric("Transition", f"{transition:.2f}s")
        c3.metric("Left Side", f"{left_time:.2f}s")
        c4.metric("Total Stop", f"{total_time:.2f}s")
        
        # Data Table
        st.subheader("Event Log")
        log_data = {
            "Action": ["Right Side Jack Up", "Right Side Jack Down", "Left Side Jack Up", "Left Side Jack Down"],
            "Timestamp": [f"{t_r_start:.2f}s", f"{t_r_end:.2f}s", f"{t_l_start:.2f}s", f"{t_l_end:.2f}s"]
        }
        st.table(pd.DataFrame(log_data))
    else:
        st.warning(f"Detected {len(filtered_events)} events. We need 4 clear jacking motions. Try a cleaner video or adjust lighting.")
        if st.checkbox("Show Raw Motion Data"):
            st.line_chart(dy)
