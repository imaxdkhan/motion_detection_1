# Motion Crossing Counter using OpenCV

A real-time object counting application built with **Python** and **OpenCV** that detects moving objects through a webcam, tracks their centroids, and counts objects whenever they cross a predefined horizontal line.

## Features

- 🎥 Real-time webcam video processing
- 🚶 Motion detection using frame differencing
- 📦 Bounding box detection for moving objects
- 🎯 Centroid-based object tracking
- 🔢 Automatic object counting on line crossing
- 🆔 Unique ID assignment for tracked objects
- ⚡ Lightweight and runs in real time

## Technologies Used

- Python 3.x
- OpenCV
- NumPy

## Project Structure

```
.
├── project.py          # Main application
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/motion-crossing-counter.git
cd motion-crossing-counter
```

### 2. Create a virtual environment (Optional)

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

Run the application using:

```bash
python project.py
```

The webcam will open automatically.

### Controls

- Press **Q** to quit the application.

## How It Works

1. Captures live video from the webcam.
2. Converts each frame to grayscale and applies Gaussian blur.
3. Detects motion using frame differencing.
4. Extracts contours of moving objects.
5. Computes the centroid of each detected object.
6. Tracks objects using centroid matching.
7. Counts an object whenever it crosses the horizontal counting line.
8. Displays:
   - Bounding boxes
   - Object IDs
   - Counting line
   - Total object count

## Detection Pipeline

```
Webcam Input
      │
      ▼
Grayscale Conversion
      │
      ▼
Gaussian Blur
      │
      ▼
Frame Differencing
      │
      ▼
Thresholding
      │
      ▼
Contour Detection
      │
      ▼
Centroid Calculation
      │
      ▼
Object Tracking
      │
      ▼
Line Crossing Detection
      │
      ▼
Object Counter
```

## Dependencies

Install all required packages using:

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:

```
opencv-python
numpy
```

## Future Improvements

- Support multiple counting lines
- Direction-based counting (In/Out)
- Deep learning-based object detection (YOLO)
- Multi-object tracking with SORT/DeepSORT
- Save counting statistics to CSV
- Video file input support
- Object classification (Person, Vehicle, etc.)
- GUI for configuration

## Applications

- People counting
- Vehicle counting
- Smart surveillance
- Traffic monitoring
- Retail analytics
- Entrance/Exit monitoring
- Industrial automation

## Author

**MOHAMMED AMAN ULLA KHAN**

BCA Final Year Student

## License

This project is licensed under the MIT License.
