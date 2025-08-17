YOLO + DeepSORT Vehicle Tracking 🚗📹

This project integrates YOLO (You Only Look Once) object detection with DeepSORT (Simple Online and Realtime Tracking with a Deep Association Metric) to perform real-time vehicle tracking in videos.

It detects vehicles frame by frame using YOLO and assigns consistent IDs across frames using DeepSORT, enabling multi-vehicle tracking in traffic or surveillance footage.

✨ Features

Vehicle detection using YOLO (fast and accurate).

Multi-object tracking with DeepSORT.

Tracks cars, buses, trucks, bikes, etc.

Unique IDs assigned to each vehicle.

Works with both pre-recorded videos and live camera streams.
⚙️ Installation

Clone this repository:

git clone https://github.com/ankush850/YOLO-DeepSORT-Vehicle-Tracking.git
cd YOLO-DeepSORT-Vehicle-Tracking


Create a virtual environment (recommended) and install dependencies:

pip install -r requirements.txt


Download YOLO weights (example: YOLOv8 or YOLOv5):

# Example for YOLOv8n
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -P yolov8/

▶️ Usage

Run tracking on a video:

python track.py --source input_video.mp4 --output runs/result.mp4


Run tracking on webcam (real-time):

python track.py --source 0

📊 Example Output

Tracked vehicles with bounding boxes + unique IDs:

(Add sample GIF or image here)

📌 Note on Videos

All example/demo videos used in this repository are taken from public/common sources provided in YOLO tutorials and open datasets.
They are used only for research and demonstration purposes, not for commercial use.
