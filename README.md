# Face Detection and Recognition — OpenCV, MediaPipe, face_recognition

A set of scripts comparing ways to find and identify faces in a webcam feed: Haar-cascade frontal face and eye detection, MediaPipe face detection and 468-point face mesh, and face recognition from a folder of known faces using the `face_recognition` library. Computer-vision learning project, 2022.

![Face recognition](https://user-images.githubusercontent.com/105288469/186960921-e91914d5-7bc4-4f12-9524-ceefbb8a7cb8.gif)

| Script | Method |
|---|---|
| `Frontal_Face_Detection.py`, `Frontal_Face_and_Eye_Detection.py` | Haar cascades |
| `Face Detection(Fast and Accurate) using Mediapipe.py`, `Face Mesh.py` | MediaPipe |
| `Face_recognition_os.walk_storing_face_data.py` → `..._reading_face_data.py` / `FR_..._using_cam.py` | Encode known faces, then recognise |

**Stack:** Python · OpenCV · MediaPipe · face_recognition (dlib)

---
[Kumar Utkarsh](https://github.com/Curious-Utkarsh) · [Portfolio of builds](https://www.mechamindlabs.com/projects)
