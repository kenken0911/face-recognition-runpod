# server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import cv2
import numpy as np
from deepface import DeepFace
import logging
from typing import Dict, Any
import uvicorn

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class ImageRequest(BaseModel):
    image: str  # Base64エンコードされた画像

class FaceAnalyzer:
    def __init__(self):
        self.emotion_dict = {
            'angry': '怒り',
            'disgust': '嫌悪',
            'fear': '恐れ',
            'happy': '喜び',
            'sad': '悲しみ',
            'surprise': '驚き',
            'neutral': '無表情'
        }

    def decode_image(self, base64_string: str) -> np.ndarray:
        """Base64文字列をOpenCV画像に変換"""
        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    def get_gender_label(self, gender_dict: Dict) -> str:
        """性別の判定を行い、適切なラベルを返す"""
        if isinstance(gender_dict, dict):
            gender = max(gender_dict.items(), key=lambda x: x[1])[0]
            return '女性' if gender == 'Woman' else '男性'
        return '不明'

    async def analyze(self, image_base64: str) -> Dict[str, Any]:
        """画像を分析して結果を返す"""
        try:
            # 画像のデコード
            frame = self.decode_image(image_base64)
            
            # DeepFaceによる分析
            result = DeepFace.analyze(
                frame,
                actions=['emotion', 'age', 'gender'],
                enforce_detection=False,
                silent=True
            )

            if isinstance(result, list):
                result = result[0]

            # 結果の整形
            return {
                'emotion': result.get('dominant_emotion', 'unknown'),
                'emotion_ja': self.emotion_dict.get(result.get('dominant_emotion', 'unknown'), '不明'),
                'age': result.get('age', 'unknown'),
                'gender_ja': self.get_gender_label(result.get('gender', {})),
                'error': False
            }

        except Exception as e:
            logger.error(f"分析エラー: {str(e)}")
            return {
                'error': True,
                'message': str(e)
            }

analyzer = FaceAnalyzer()

@app.post("/analyze")
async def analyze_image(request: ImageRequest):
    try:
        result = await analyzer.analyze(request.image)
        return result
    except Exception as e:
        logger.error(f"API エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)