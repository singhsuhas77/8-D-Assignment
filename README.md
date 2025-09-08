# 8-D-Assignment
# Port Object Detection with YOLOv9

This project uses the YOLOv9e object detection model to identify and track port-related objects in a video feed. The script processes an input video, applies adaptive brightness correction, detects specified objects, and saves the output as a new video with bounding boxes and labels drawn on each frame.

---

## 📸 Features

-   **High-Performance Detection**: Utilizes the pre-trained **YOLOv9e** model for accurate and efficient object detection.
-   **Selective Object Filtering**: Detects a specific set of objects relevant to a port environment, including:
    -   `boat`
    -   `ship`
    -   `person`
    -   `truck`
    -   `crane`
    -   `container`
    -   `train`
    -   `airplane`
-   **Adaptive Brightness Correction**: Automatically adjusts the gamma of dark video frames to improve visibility and detection accuracy.
-   **Customizable Bounding Boxes**: Draws colored bounding boxes and confidence labels for each detected object.
-   **Video Processing**: Reads an input video (`Port.mp4`) and writes the processed frames to an output file (`Port_processed.mp4`).

---

## ⚙️ Setup and Installation

### Prerequisites

-   Python 3.8 or higher
-   `pip` package manager

### Installation

1.  **Clone the repository or download the source code.**

2.  **Install the required Python libraries:**
    ```bash
    pip install -r requirements.txt
    ```
    If you don't have a `requirements.txt` file, you can install the packages manually:
    ```bash
    pip install opencv-python numpy ultralytics
    ```

3.  **Model Weights**: The YOLOv9e model weights (`yolov9e.pt`) will be automatically downloaded by the `ultralytics` library the first time you run the script.

---

## 🚀 How to Run

1.  **Place your input video** in the same directory as the `main.py` script and ensure it is named `Port.mp4`. You can change the input path in the script if needed.

2.  **Execute the script** from your terminal:
    ```bash
    python main.py
    ```

3.  **Check the output**: Once the script finishes processing, a new video file named `Port_processed.mp4` will be saved in the same directory. A success message will be printed to the console:
    ```
    ✅ Processed video saved as Port_processed.mp4
    ```

---

## 🔧 Configuration

You can easily customize the script's behavior by modifying these variables in `main.py`:

-   **`input_path`**: Change the path to your source video file.
-   **`output_path`**: Set the desired name for the processed output video.
-   **`allowed_classes`**: Add or remove class names from this set to change which objects are detected.
-   **`fixed_colors`**: Modify the BGR color codes for different object classes.
-   **`conf`**: Adjust the confidence threshold (e.g., `conf=0.40`) in the `model.predict()` line to filter out detections with lower confidence scores.
