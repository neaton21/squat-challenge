"""Track one head and draw dlib's facial landmarks."""

from pathlib import Path

import cv2
import dlib


PREDICTOR_NAME = "shape_predictor_68_face_landmarks.dat"
DETECTION_INTERVAL = 8
MIN_TRACKING_QUALITY = 7


class HeadTracker:
    def __init__(self, predictor_path=None):
        project = Path(__file__).resolve().parent
        if predictor_path is None:
            predictor_path = project / PREDICTOR_NAME
            if not predictor_path.is_file():
                predictor_path = project.parent / PREDICTOR_NAME
        if not predictor_path.is_file():
            raise RuntimeError(f"Missing dlib model: {predictor_path}. See README.md.")

        self.detect_faces = dlib.get_frontal_face_detector()
        self.predict_landmarks = dlib.shape_predictor(str(predictor_path))
        self.tracker = dlib.correlation_tracker()
        self.tracking = False
        self.frames_until_detection = 0

    def draw(self, frame, rgb, pose_nose=None):
        """Update the track from an unannotated RGB frame and draw onto frame."""
        previous_center = None
        if self.tracking:
            quality = self.tracker.update(rgb)
            self.tracking = quality >= MIN_TRACKING_QUALITY
            if self.tracking:
                previous_center = self.tracker.get_position().center()

        height, width = frame.shape[:2]
        if not self.tracking or self.frames_until_detection <= 0:
            self.frames_until_detection = DETECTION_INTERVAL
            faces = self.detect_faces(rgb, 0)
            if faces:
                if (
                    pose_nose is not None
                    and pose_nose.visibility is not None
                    and pose_nose.visibility >= 0.5
                ):
                    x, y = pose_nose.x * width, pose_nose.y * height
                elif previous_center is not None:
                    x, y = previous_center.x, previous_center.y
                else:
                    x, y = None, None

                if x is None:
                    face = max(faces, key=lambda box: box.area())
                else:
                    face = min(faces, key=lambda box: (box.center().x - x) ** 2
                               + (box.center().y - y) ** 2)
                self.tracker.start_track(rgb, face)
                self.tracking = True

        self.frames_until_detection -= 1
        if not self.tracking:
            return

        position = self.tracker.get_position()
        face = dlib.rectangle(
            max(0, round(position.left())),
            max(0, round(position.top())),
            min(width - 1, round(position.right())),
            min(height - 1, round(position.bottom())),
        )
        if face.is_empty():
            self.tracking = False
            return

        color = (255, 207, 95)
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), color, 2)
        for point in self.predict_landmarks(rgb, face).parts():
            cv2.circle(frame, (point.x, point.y), 2, color, -1, cv2.LINE_AA)
