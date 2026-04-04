"""前端影像擷取模組 (Raspberry Pi)。

功能:
- 使用車載 Webcam 擷取 1920x1080 @ 30fps 影像
- 透過網路 (TCP socket) 傳輸影像至後端 GPU 伺服器
- 支援 MJPEG 壓縮減少頻寬

使用方式 (在 Raspberry Pi 上):
    python capture_stream.py --server 192.168.1.100 --port 9999
"""

import struct
import time
import argparse

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import socket
    HAS_SOCKET = True
except ImportError:
    HAS_SOCKET = False


class CaptureStream:
    """車載攝影機影像擷取與串流。"""

    def __init__(
        self,
        camera_id: int = 0,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        jpeg_quality: int = 80,
    ):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.jpeg_quality = jpeg_quality

        self.cap = None
        self.sock = None

    def open_camera(self):
        if not HAS_CV2:
            raise ImportError("需要安裝 opencv-python")

        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        if not self.cap.isOpened():
            raise RuntimeError(f"無法開啟攝影機 {self.camera_id}")

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"攝影機已開啟: {actual_w}x{actual_h}")

    def connect_server(self, server_ip: str, port: int):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((server_ip, port))
        print(f"已連線至後端伺服器: {server_ip}:{port}")

    def send_frame(self, frame):
        """壓縮並傳送單幀影像。"""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        _, encoded = cv2.imencode(".jpg", frame, encode_param)
        data = encoded.tobytes()

        # 先傳送資料大小 (4 bytes, big-endian)
        size = struct.pack(">I", len(data))
        self.sock.sendall(size + data)

    def stream(self, server_ip: str, port: int, duration: float = None):
        """開始串流影像至後端。

        Args:
            server_ip: 後端伺服器 IP
            port: 後端伺服器 port
            duration: 串流時長 (秒), None=無限
        """
        self.open_camera()
        self.connect_server(server_ip, port)

        frame_count = 0
        start_time = time.time()
        interval = 1.0 / self.fps

        print("開始串流影像...")
        try:
            while True:
                t0 = time.time()
                ret, frame = self.cap.read()
                if not ret:
                    break

                self.send_frame(frame)
                frame_count += 1

                elapsed = time.time() - start_time
                if duration and elapsed >= duration:
                    break

                # 維持目標 FPS
                process_time = time.time() - t0
                if process_time < interval:
                    time.sleep(interval - process_time)

                if frame_count % (self.fps * 5) == 0:
                    actual_fps = frame_count / elapsed
                    print(f"已傳送 {frame_count} 幀, 平均 {actual_fps:.1f} FPS")

        except (BrokenPipeError, ConnectionResetError):
            print("與伺服器的連線已中斷")
        finally:
            self.cleanup()

        total_time = time.time() - start_time
        print(f"串流結束: {frame_count} 幀, {total_time:.1f}s")

    def cleanup(self):
        if self.cap:
            self.cap.release()
        if self.sock:
            self.sock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="前端影像擷取串流")
    parser.add_argument("--server", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--duration", type=float, default=None)
    args = parser.parse_args()

    streamer = CaptureStream(camera_id=args.camera)
    streamer.stream(args.server, args.port, args.duration)
