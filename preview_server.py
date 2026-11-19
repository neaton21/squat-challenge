"""Serve a local browser preview of the live Python pose detector."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from time import monotonic_ns

from head_tracking import HeadTracker
from pose_detection import DEFAULT_MODEL, cv2, draw_skeleton, mp

CAMERA_LOCK = Lock()

PAGE = b"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Squat Challenge | Live pose preview</title>
<style>
body { background:#111921; color:#f4f6f8; font:1rem system-ui; margin:2rem auto; width:min(94%, 58rem); }
img { display:block; width:100%; background:#1a2632; }
</style>
<h1>Squat Challenge</h1>
<p>Keep your full body in frame for the skeleton overlay. Face the camera to start head tracking.</p>
<img src="/stream" alt="Live camera feed with pose landmarks"
     onerror="this.alt='Camera unavailable. Close other previews or camera apps, then reload.'">
</html>"""


class PreviewHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(PAGE)
        elif self.path == "/stream":
            if not CAMERA_LOCK.acquire(blocking=False):
                self.send_error(503, "Camera already in use by another preview")
                return
            try:
                self.stream_pose()
            finally:
                CAMERA_LOCK.release()
        else:
            self.send_error(404)

    def stream_pose(self):
        vision = mp.tasks.vision
        options = vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(DEFAULT_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )
        camera = cv2.VideoCapture(0)
        try:
            if not camera.isOpened():
                self.send_error(503, "Could not open webcam")
                return
            try:
                head = HeadTracker()
            except RuntimeError as error:
                self.log_error("%s", error)
                self.send_error(503, "Could not load dlib head tracking model")
                return
            with vision.PoseLandmarker.create_from_options(options) as detector:
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                timestamp = -1
                start = monotonic_ns()
                while True:
                    ok, frame = camera.read()
                    if not ok:
                        break
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    timestamp = max(timestamp + 1, (monotonic_ns() - start) // 1_000_000)
                    poses = detector.detect_for_video(image, timestamp).pose_landmarks
                    if poses:
                        draw_skeleton(frame, poses[0])
                    head.draw(frame, rgb, poses[0][0] if poses else None)
                    ok, jpeg = cv2.imencode(".jpg", frame)
                    if ok:
                        data = jpeg.tobytes()
                        self.wfile.write(
                            f"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {len(data)}\r\n\r\n".encode()
                            + data + b"\r\n"
                        )
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        finally:
            camera.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    with ThreadingHTTPServer(("127.0.0.1", args.port), PreviewHandler) as server:
        print(f"Squat Challenge preview: http://127.0.0.1:{server.server_port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
