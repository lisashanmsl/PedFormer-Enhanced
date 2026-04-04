"""後端推論伺服器 (RTX 4090 工作站)。

功能:
- 接收前端 (Raspberry Pi) 傳送的影像串流
- 即時執行 RAFT 光流計算、SAM 分割、模型推論
- 回傳預測結果 (軌跡 + 意圖)
- 目標: >= 10 FPS

使用方式 (在 GPU 伺服器上):
    python inference_server.py --port 9999
"""

import struct
import time
import threading
import argparse

import numpy as np

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

from inference import PedFormerInference


class InferenceServer:
    """GPU 後端即時推論伺服器。"""

    def __init__(
        self,
        port: int = 9999,
        weights_path: str = "weights/pedformer_best.pth",
        obs_len: int = 16,
    ):
        self.port = port
        self.obs_len = obs_len

        print("載入推論引擎...")
        self.engine = PedFormerInference(weights_path=weights_path)

        self.frame_buffer = []
        self.running = False

    def recv_exact(self, conn, size: int) -> bytes:
        """確保接收完整的 size bytes。"""
        data = b""
        while len(data) < size:
            chunk = conn.recv(size - len(data))
            if not chunk:
                raise ConnectionError("連線已中斷")
            data += chunk
        return data

    def recv_frame(self, conn) -> np.ndarray:
        """接收單幀影像。"""
        size_data = self.recv_exact(conn, 4)
        size = struct.unpack(">I", size_data)[0]
        jpg_data = self.recv_exact(conn, size)
        frame = cv2.imdecode(
            np.frombuffer(jpg_data, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        return frame

    def process_frames(self, frames: list) -> dict:
        """處理觀察序列並推論。

        Args:
            frames: List of [H, W, 3] BGR 影像 (長度 = obs_len)

        Returns:
            prediction dict
        """
        # 簡化演示: 使用假 bbox (實際需搭配行人偵測器)
        h, w = frames[0].shape[:2]
        fake_traj = np.array(
            [[w * 0.4, h * 0.5, w * 0.6, h * 0.7]] * self.obs_len,
            dtype=np.float32,
        )

        return self.engine.predict(fake_traj)

    def handle_client(self, conn, addr):
        """處理單一客戶端連線。"""
        print(f"客戶端已連線: {addr}")
        frame_count = 0
        start_time = time.time()

        try:
            while self.running:
                frame = self.recv_frame(conn)
                if frame is None:
                    continue

                self.frame_buffer.append(frame)
                frame_count += 1

                # 累積足夠觀察幀後進行推論
                if len(self.frame_buffer) >= self.obs_len:
                    t0 = time.time()
                    result = self.process_frames(
                        self.frame_buffer[-self.obs_len:]
                    )
                    latency = (time.time() - t0) * 1000

                    if frame_count % 30 == 0:
                        elapsed = time.time() - start_time
                        fps = frame_count / elapsed
                        print(
                            f"[{addr}] Frame #{frame_count} | "
                            f"FPS: {fps:.1f} | "
                            f"Latency: {latency:.0f}ms | "
                            f"Crossing: {result['crossing_prob']:.2%}"
                        )

                    # 控制 buffer 大小
                    if len(self.frame_buffer) > self.obs_len * 2:
                        self.frame_buffer = self.frame_buffer[-self.obs_len:]

        except (ConnectionError, struct.error):
            print(f"客戶端已斷線: {addr}")
        finally:
            conn.close()

    def start(self):
        """啟動伺服器。"""
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(("0.0.0.0", self.port))
        server_sock.listen(5)

        self.running = True
        print(f"推論伺服器已啟動，監聽 port {self.port}...")

        try:
            while self.running:
                conn, addr = server_sock.accept()
                client_thread = threading.Thread(
                    target=self.handle_client, args=(conn, addr), daemon=True
                )
                client_thread.start()
        except KeyboardInterrupt:
            print("\n伺服器關閉中...")
        finally:
            self.running = False
            server_sock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="後端推論伺服器")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--weights", default="weights/pedformer_best.pth")
    args = parser.parse_args()

    server = InferenceServer(port=args.port, weights_path=args.weights)
    server.start()
