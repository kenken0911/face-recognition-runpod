# client.py
import cv2
import requests
import base64
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceClient:
    def __init__(self, server_url="http://localhost:8000"):
        self.server_url = server_url
        self.cap = cv2.VideoCapture(0)
        self.font_path = "C:/Windows/Fonts/meiryo.ttc"
        self.font_size = 32
        
        # カメラの設定
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def cv2_putText_ja(self, img, text, org, color):
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        font = ImageFont.truetype(self.font_path, self.font_size)
        draw.text(org, text, fill=color, font=font)
        return np.array(pil_img)

    def encode_frame(self, frame):
        """フレームをBase64エンコード"""
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')

    def send_frame(self, frame):
        """フレームをサーバーに送信"""
        encoded_frame = self.encode_frame(frame)
        try:
            response = requests.post(
                f"{self.server_url}/analyze",
                json={"image": encoded_frame}
            )
            return response.json()
        except Exception as e:
            logger.error(f"送信エラー: {str(e)}")
            return None

    def run(self):
        logger.info("顔認識クライアントを開始します...")
        last_analysis_time = time.time()
        analysis_interval = 0.2

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                current_time = time.time()
                display_frame = frame.copy()

                if current_time - last_analysis_time >= analysis_interval:
                    # フレーム送信と結果取得
                    result = self.send_frame(frame)
                    if result and not result.get('error'):
                        # 背景描画
                        overlay = display_frame.copy()
                        cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
                        cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)

                        # 結果表示
                        display_frame = self.cv2_putText_ja(
                            display_frame,
                            f"感情: {result.get('emotion_ja', '不明')}",
                            (20, 40),
                            (255, 255, 255)
                        )
                        display_frame = self.cv2_putText_ja(
                            display_frame,
                            f"年齢: {result.get('age', '不明')}歳",
                            (20, 80),
                            (255, 255, 255)
                        )
                        display_frame = self.cv2_putText_ja(
                            display_frame,
                            f"性別: {result.get('gender_ja', '不明')}",
                            (20, 120),
                            (255, 255, 255)
                        )

                    last_analysis_time = current_time

                # FPS表示
                fps = 1.0 / (time.time() - current_time) if time.time() != current_time else 0
                display_frame = self.cv2_putText_ja(
                    display_frame,
                    f"FPS: {fps:.1f}",
                    (display_frame.shape[1] - 150, display_frame.shape[0] - 30),
                    (0, 255, 0)
                )

                # 終了案内
                display_frame = self.cv2_putText_ja(
                    display_frame,
                    "終了: Qキー",
                    (20, display_frame.shape[0] - 30),
                    (0, 255, 0)
                )

                cv2.imshow('Face Recognition Client', display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    client = FaceClient()
    client.run()