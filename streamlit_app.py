import streamlit as st
import torch
import cv2
import numpy as np
import json
from collections import defaultdict
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# Load YOLOv5 model
@st.cache_resource()
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load YOLOv5s model (small)
    return model

# Function to detect objects in each frame
def detect_objects(frame, model, confidence_threshold):
    results = model(frame)  # Perform inference
    labels, cord = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
    return labels, cord

# Function to draw bounding boxes
def draw_boxes(labels, cord, frame, model, confidence_threshold):
    # Define a color map for different object classes
    color_map = {
        'person': (0, 255, 0),  # Green for persons
        'car': (0, 0, 255),     # Red for vehicles like cars
        'bus': (255, 0, 0),     # Blue for buses
        'truck': (255, 0, 0),   # Blue for trucks
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

    return frame

# Global object count tracking
global_object_counts = defaultdict(int)

# Function to update object counts
def update_object_counts(labels, model):
    # Clear counts for the current frame
    object_counts = defaultdict(int)
    
    # Count detected objects
    for label in labels:
        label_name = model.names[int(label)]
        object_counts[label_name] += 1
        global_object_counts[label_name] += 1  # Update global count
        print("Current Frame Object Counts:", object_counts)  # Debugging


    return object_counts

# Define a video transformer class for streamlit-webrtc
class YOLOv5Transformer(VideoTransformerBase):
    def __init__(self, model, confidence_threshold):
        self.model = model
        self.confidence_threshold = confidence_threshold

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")  # Convert the frame to an ndarray

        labels, cord = detect_objects(img, self.model, self.confidence_threshold)

        # Draw boxes on the frame
        img = draw_boxes(labels, cord, img, self.model, self.confidence_threshold)

        # Update object counts for the current frame
        update_object_counts(labels, self.model)

        return img  # Return the frame with bounding boxes

# Streamlit app
def main():
    st.title("Real-time Object Detection with YOLOv5 and WebRTC")
    
    # Sidebar options
    st.sidebar.title("Options")
    confidence_threshold = st.sidebar.slider("Confidence threshold", 0.1, 1.0, 0.5, 0.1)

    # Load YOLOv5 model
    model = load_model()

    # Start WebRTC video streaming
    webrtc_ctx = webrtc_streamer(
        key="yolov5-object-detection",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=lambda: YOLOv5Transformer(model, confidence_threshold),
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True,  # Asynchronous processing for better performance
    )

    object_count_str = "Object Counts (Global)\n"
    # Display the object counts in the sidebar if the video transformer is active
    if webrtc_ctx.state.playing:
        # Prepare object counts as a string
        for obj, count in global_object_counts.items():
            object_count_str += f"{obj}: {count}\n"

        # Display the object counts in the sidebar
        print(object_count_str)
        st.sidebar.text(object_count_str)
    
    # Export object counts as JSON when the button is clicked
    btn = st.button("Export Object Counts as JSON")
    if btn:
        json_data = json.dumps(global_object_counts, indent=4)
        with open("object_counts.json", "w") as json_file:
            json_file.write(json_data)
        st.success("Object counts exported as object_counts.json")

if __name__ == '__main__':
    main()
