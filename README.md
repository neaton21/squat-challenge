# Squat Challenge

The prototype uses MediaPipe Pose Landmarker to draw one person's skeleton and dlib to track their face and draw 68 facial landmarks on the mirrored webcam feed. Both the desktop window and browser preview process frames locally. Squat counting is the next milestone.

## Run on Windows

From this folder with 64-bit Python 3.10 installed, run in PowerShell:

On Windows, installing dlib from source requires CMake on PATH and the Visual Studio C++ desktop development tools.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Invoke-WebRequest 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task' -OutFile pose_landmarker_lite.task
.\.venv\Scripts\python.exe pose_detection.py
```

Stand far enough back for the camera to see your whole body. Face the camera to start dlib head tracking. The head box and 68 facial landmarks appear in blue. Press **Q** or **Esc** to exit. If you have multiple webcams, pass `--camera 1` to try the second camera.

The head tracker reads `shape_predictor_68_face_landmarks.dat` from this folder or its parent. For a separate checkout, download and decompress the [dlib shape predictor](https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) into this folder. To use a model stored elsewhere, pass `--face-model "C:\models\shape_predictor_68_face_landmarks.dat"` to either script.

For the browser preview, run `.\.venv\Scripts\python.exe preview_server.py` and open [localhost:8766](http://127.0.0.1:8766/).

The GitHub Pages landing page is static. Run the Python prototype locally to use the webcam.
