import streamlit as st
import cv2
import numpy as np
import tempfile
import pandas as pd
from scipy.signal import find_peaks

st.set_page_config(page_title="Pit Stop Timing Analyzer", layout="wide")

def process_video(video_path, threshold, sensitivity):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        return None, None
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    left_y_motion = []
    right_y_motion = []
    timestamps = []
    
    width = prev_gray.shape[1]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate dense optical flow (Farneback method)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Split screen into left and right halves to track respective jacking motions
        left_flow_y = flow[:, :width//2, 1]
        right_flow_y = flow[:, width//2:, 1]
        
        # Average vertical motion (ignoring minor vibrations)
        left_y_motion.append(np.mean(left_flow_y))
        right_y_motion.append(np.mean(right_flow_y))
        timestamps.append(frame_count / fps)
        
        prev_gray = gray
        frame_count += 1
        
        if frame_count % 10 == 0:
            progress_bar.progress(frame_count / total_frames)
            status_text.text(f"Processing frame {frame_count} of {total_frames}...")

    cap.release()
    progress_bar.empty()
    status_text.empty()
    
    # Smooth the data to find distinct jacking events
    left_y_smooth = np.convolve(left_y_motion, np.ones(sensitivity)/sensitivity, mode='same')
    right_y_smooth = np.convolve(right_y_motion, np.ones(sensitivity)/sensitivity, mode='same')
    
    return timestamps, left_y_smooth, right_y_smooth

def extract_timings(timestamps, left_motion, right_motion, threshold):
    # Find peaks in vertical motion
    # Jack UP creates a spike in one direction, Jack DOWN creates a spike in the opposite
    
    # Right side analysis
    right_up_peaks, _ = find_peaks(right_motion, height=threshold, distance=30)
    right_down_peaks, _ = find_peaks(-right_motion, height=threshold, distance=30)
    
    # Left side analysis
    left_up_peaks, _ = find_peaks(left_motion, height=threshold, distance=30)
    left_down_peaks, _ = find_peaks(-left_motion, height=threshold, distance=30)
    
    # Logic to find the first sequence of: Right Up -> Right Down -> Left Up -> Left Down
    timings = {}
    
    try:
        t_right_up = timestamps[right_up_peaks[0]]
        t_right_down = timestamps[right_down_peaks[right_down_peaks > right_up_peaks[0]][0]]
        t_left_up = timestamps[left_up_peaks[left_up_peaks > right_down_peaks[0]][0]]
        t_left_down = timestamps[left_down_peaks[left_down_peaks > left_up_peaks[0]][0]]
        
        timings['Right Side Start (Hood Up)'] = t_right_up
        timings['Right Side Stop / Transition Start (Hood Down)'] = t_right_down
        timings['Transition Stop / Left Side Start (Hood Up)'] = t_left_up
        timings['Left Side Stop / Overall Stop (Hood Down)'] = t_left_down
        
        # Calculate Durations
        timings['Right Side Duration'] = t_right_down - t_right_up
        timings['Transition Duration'] = t_left_up - t_right_down
        timings['Left Side Duration'] = t_left_down - t_left_up
        timings['Overall Pit Stop Time'] = t_left_down - t_right_up
        
    except IndexError:
        st.warning("Could not automatically detect the full 4-part sequence. Try adjusting the motion threshold or sensitivity.")
        
    return timings

# --- UI Layout ---
st.title("🏁 NASCAR Pit Stop Timing Analyzer")
st.markdown("Upload an in-car pit stop video. The app uses Optical Flow to track the vertical motion of the car being jacked up and down on the left and right sides.")

uploaded_file = st.file_uploader("Upload Pit Stop Video (MP4)", type=["mp4", "mov", "avi"])

with st.sidebar:
    st.header("Analysis Settings")
    st.write("Adjust these if the automatic detection misses the jack-up/jack-down events.")
    threshold = st.slider("Motion Spike Threshold", min_value=0.1, max_value=5.0, value=1.5, step=0.1)
    sensitivity = st.slider("Smoothing Sensitivity (Frames)", min_value=1, max_value=20, value=5, step=1)

if uploaded_file is not None:
    # Save uploaded file to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.video(uploaded_file)
        
    with col2:
        with st.spinner("Analyzing video motion..."):
            timestamps, left_motion, right_motion = process_video(video_path, threshold, sensitivity)
            
        if timestamps is not None:
            timings = extract_timings(timestamps, left_motion, right_motion, threshold)
            
            if timings:
                st.subheader("⏱️ Pit Stop Split Times")
                
                # Display metrics
                m1, m2 = st.columns(2)
                m1.metric("Right Side Time", f"{timings['Right Side Duration']:.2f} s")
                m2.metric("Transition Time", f"{timings['Transition Duration']:.2f} s")
                
                m3, m4 = st.columns(2)
                m3.metric("Left Side Time", f"{timings['Left Side Duration']:.2f} s")
                m4.metric("Overall Time", f"{timings['Overall Pit Stop Time']:.2f} s")
                
                st.divider()
                
                # Display exact timestamps
                st.subheader("Event Timestamps")
                df = pd.DataFrame({
                    "Event": [
                        "Front Right Raised (Timer Start)", 
                        "Front Right Dropped", 
                        "Front Left Raised", 
                        "Front Left Dropped (Timer Stop)"
                    ],
                    "Video Time (seconds)": [
                        f"{timings['Right Side Start (Hood Up)']:.2f}",
                        f"{timings['Right Side Stop / Transition Start (Hood Down)']:.2f}",
                        f"{timings['Transition Stop / Left Side Start (Hood Up)']:.2f}",
                        f"{timings['Left Side Stop / Overall Stop (Hood Down)']:.2f}"
                    ]
                })
                st.table(df)
            
            # Optional: Plot the motion graph for debugging/visualization
            with st.expander("View Motion Analysis Graph"):
                st.markdown("This chart shows the vertical motion detected on the left and right halves of the video. Peaks represent the car being jacked up or dropped.")
                chart_data = pd.DataFrame({
                    "Time (s)": timestamps,
                    "Left Side Vertical Motion": left_motion,
                    "Right Side Vertical Motion": right_motion
                }).set_index("Time (s)")
                st.line_chart(chart_data)
