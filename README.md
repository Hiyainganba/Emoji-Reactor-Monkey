# Emoji Reactor

This is a computer vision project I built using Python. It uses the webcam to track my face and hand gestures in real-time and displays a corresponding reaction image based on what I am doing.

I used the MediaPipe library to handle the face mesh and hand tracking because it is fast and accurate.

## How It Works

The script reads the video feed from the webcam. It draws a green box around the face and shows the wireframe landmarks on the hands to indicate that tracking is active.

It recognizes four specific states based on the geometry of the face and hands:

1.  **No Reaction:** The default state when I am just looking at the camera with no specific gesture.
2.  **Surprised:** Triggers when I open my mouth and hold **both hands** up.
3.  **Thinking:** Triggers when I touch my index finger to my mouth (like a "shhh" or biting finger gesture) and **look up** with my eyes.
4.  **Got It:** Triggers when I point one index finger up, open my mouth, and **look up**.

## Requirements

You need to have Python installed. The project uses these three libraries:

* opencv-python
* mediapipe
* numpy

## Setup

1.  Clone this repository or download the source code.
2.  Install the required libraries by running this command in your terminal:
    ```bash
    pip install opencv-python mediapipe numpy
    ```
3.  **Important:** You must place 4 image files in the same folder as the script. They must be named exactly as follows:
    * `noreaction.jpg`
    * `surprise.jpg`
    * `think.jpg`
    * `gotit.jpg`

## How to Run

Open your terminal in the project folder and run the script:

```bash
python monkey.py
