import streamlit as st
import torch
import cv2
import numpy as np
import time
import json
from collections import defaultdict

# Load YOLOv5 model
@st.cache_resource()
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # load YOLOv5s model (small)
    return model

# Function to detect objects in each frame
def detect_objects(frame, model, confidence_threshold):
    results = model(frame)  # Perform inference
    labels, cord = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
    return labels, cord

def draw_boxes(labels, cord, frame, model, confidence_threshold, object_counts):
    # Define a color map for different object classes
    color_map = {
        'person': (0, 255, 0),  # Green for persons
        'car': (0, 0, 255),     # Red for vehicles like cars
        'bus': (255, 0, 0),     # Blue for buses
        'truck': (255, 0, 0),   # blue for trucks
        'laptop': (188, 97, 237),
        # Add more object categories and colors as needed
    }
    
    n = len(labels)
    for i in range(n):
        row = cord[i]
        if row[4] >= confidence_threshold:  # Confidence threshold
            x1, y1, x2, y2 = int(row[0] * frame.shape[1]), int(row[1] * frame.shape[0]), int(row[2] * frame.shape[1]), int(row[3] * frame.shape[0])

            # Get the label for the current object
            label = model.names[int(labels[i])]
            
            # Get the color for the current label, default to white if not in color_map
            bgr = color_map.get(label, (255, 255, 255))  # Default is white if label not found in color_map

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)

            # Label and confidence score
            confidence = row[4] * 100  # Convert confidence to percentage
            text = f"{label} {confidence:.1f}%"

            # Font settings
            font_scale = 2  # Adjust font size
            font_thickness = 3  # Adjust font thickness

            # Draw label and confidence score on the bounding box
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, bgr, font_thickness)

            # Track object counts
            object_counts[label] += 1

    return frame

# Streamlit app
def main():
    st.title("Real-time Object Detection with YOLOv5")
    
    # Sidebar options
    st.sidebar.title("Options")
    btn = st.button("Export Object Counts as JSON")
    confidence_threshold = st.sidebar.slider("Confidence threshold", 0.1, 1.0, 0.5, 0.1)
    
    # Load YOLOv5 model
    model = load_model()

    # Start video stream
    run_video = st.checkbox("Start Video Stream")

    # Flag to track if the export button was pressed
    export_pressed = False


    if run_video:
        cap = cv2.VideoCapture(0)  # Use webcam (index 0)
        stframe = st.empty()
        object_count_placeholder = st.sidebar.empty()  # Create a placeholder for object counts
        object_counts = defaultdict(int)  # Dictionary to store counts of detected objects

        # Placeholder for success message
        success_message_placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image from camera.")
                break

            # Reset object counts for each frame
            object_counts.clear()

            # Perform object detection
            labels, cord = detect_objects(frame, model, confidence_threshold)
            frame = draw_boxes(labels, cord, frame, model, confidence_threshold, object_counts)

            # Display frame
            stframe.image(frame, channels="BGR")

            # Prepare a single string for object counts
            object_count_str = "Object Counts\n"
            for obj, count in object_counts.items():
                object_count_str += f"{obj}: {count}\n"
            
            # Update object counts in a single render
            object_count_placeholder.text(object_count_str)

            if btn:
                if not export_pressed:  # Check if it hasn't been pressed yet
                    json_data = json.dumps(object_counts, indent=4)
                    with open("object_counts.json", "w") as json_file:
                        json_file.write(json_data)
                    success_message_placeholder.success("Object counts exported as object_counts.json")
                    time.sleep(3) 
                    success_message_placeholder.empty()
                    export_pressed = True  # Set the flag to True

            # To allow Streamlit to handle UI events
            if not run_video:
                break
            
        cap.release()       

if __name__ == '__main__':
    main()
