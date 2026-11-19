"""Detect and display one person's pose from a webcam."""

import argparse
from pathlib import Path
from time import monotonic_ns

import cv2
import mediapipe as mp

from head_tracking import HeadTracker


WINDOW = "Squat Challenge - Pose Detection"
DEFAULT_MODEL = Path(__file__).with_name("pose_landmarker_lite.task")
MIN_VISIBILITY = 0.5


def draw_skeleton(frame, landmarks):
    height, width = frame.shape[:2]
    points = {
        index: (int(point.x * width), int(point.y * height))
        for index, point in enumerate(landmarks)
        if point.visibility is not None
        and point.visibility >= MIN_VISIBILITY
        and 0 <= point.x < 1
        and 0 <= point.y < 1
    }

    for connection in mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS:
        if connection.start in points and connection.end in points:
            cv2.line(frame, points[connection.start], points[connection.end],
                     (90, 224, 100), 2, cv2.LINE_AA)
    for point in points.values():
        cv2.circle(frame, point, 4, (255, 255, 255), -1, cv2.LINE_AA)


def run(camera_index, model, face_model=None):
    head = HeadTracker(face_model)
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
    )

    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as detector:
        camera = cv2.VideoCapture(camera_index)
        try:
            if not camera.isOpened():
                raise RuntimeError(f"Could not open camera {camera_index}.")

            start = monotonic_ns()
            timestamp_ms = -1
            while True:
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError(f"Could not read from camera {camera_index}.")

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                elapsed_ms = (monotonic_ns() - start) // 1_000_000
                timestamp_ms = max(timestamp_ms + 1, elapsed_ms)
                result = detector.detect_for_video(image, timestamp_ms)

                status = "No pose detected"
                if result.pose_landmarks:
                    draw_skeleton(frame, result.pose_landmarks[0])
                    status = "Pose detected"
                nose = result.pose_landmarks[0][0] if result.pose_landmarks else None
                head.draw(frame, rgb, nose)

                cv2.putText(frame, status, (16, 32), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, status, (16, 32), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow(WINDOW, frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            camera.release()
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help="Path to the MediaPipe Pose Landmarker model")
    parser.add_argument("--face-model", type=Path,
                        help="Path to the dlib 68-point shape predictor")
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"Model file missing: {args.model}. See README.md for download instructions.")

    try:
        run(args.camera, args.model, args.face_model)
    except (RuntimeError, ValueError, OSError, cv2.error) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
