# Special CNIC Recognition System

This project demonstrates a real-time system for recognizing a specific template (e.g., a special CNIC) in a video stream using OpenCV. The system leverages the SIFT (Scale-Invariant Feature Transform) algorithm for feature detection and FLANN (Fast Library for Approximate Nearest Neighbors) for matching.

## Features
- Real-time video processing with a webcam.
- Template matching using SIFT features.
- Homography computation to detect the template's position and orientation.
- FPS (frames per second) display for performance monitoring.

## Requirements
- Python 3.x
- OpenCV

## Installation

1. Install Python 3.x from [Python.org](https://www.python.org/).
2. Install OpenCV library:
    ```bash
    pip install opencv-python opencv-contrib-python
    ```

## Usage

1. Ensure you have a template image named `template.png` in the working directory.
2. Run the script:

    ```bash
    python template_test_realtime.py
    ```
