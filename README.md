# Face Recognition

Recognizes known people in a webcam stream with the `face_recognition` library (built on dlib) and OpenCV.

## How it works

1. Every image in `images/` is loaded, and its file name becomes the person's name (`images/alice.jpg` → `Alice`).
2. For each webcam frame, faces are found and their encodings are compared with the known ones.
3. The closest match is printed to the console, or `Unknown` if nobody matches.

Drawing boxes and names on the video is commented out in the code; uncomment the `cv2.rectangle` and `cv2.putText` lines to show them.

## Setup

```bash
pip install -r requirements.txt   # dlib needs CMake and a C++ compiler
mkdir images                      # add one photo per person
```

## Run

```bash
python face_recognation.py
```

Press `q` to quit.
