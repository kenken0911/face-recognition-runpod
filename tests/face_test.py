import cv2
from deepface import DeepFace
import time
import os
import logging
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# TensorFlowの警告を抑制
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ログの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cv2_putText_ja(img, text, org, font_path, font_size, color):
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        logger.error(f"フォント読み込みエラー: {str(e)}")
        font = ImageFont.load_default()
    
    draw.text(org, text, fill=color, font=font)
    return np.array(pil_img)

def get_gender_label(gender_dict):
    """性別の判定を行い、適切なラベルを返す"""
    if isinstance(gender_dict, dict):
        # 最も確率の高い性別を取得
        gender = max(gender_dict.items(), key=lambda x: x[1])[0]
        return '女性' if gender == 'Woman' else '男性'
    return '不明'

def test_face_recognition():
    logger.info("カメラを初期化中...")
    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    font_path = "C:/Windows/Fonts/meiryo.ttc"
    font_size = 32
    
    if not cap.isOpened():
        logger.error("カメラを開けませんでした")
        return
    
    logger.info("顔認識を開始します")
    last_analysis_time = time.time()
    analysis_interval = 0.2
    
    # 感情の日本語変換辞書
    emotion_dict = {
        'angry': '怒り',
        'disgust': '嫌悪',
        'fear': '恐れ',
        'happy': '喜び',
        'sad': '悲しみ',
        'surprise': '驚き',
        'neutral': '無表情'
    }
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("フレームの取得に失敗しました")
                break
            
            current_time = time.time()
            display_frame = frame.copy()
            
            try:
                if current_time - last_analysis_time >= analysis_interval:
                    result = DeepFace.analyze(
                        frame, 
                        actions=['emotion', 'age', 'gender'],
                        enforce_detection=False,
                        silent=True
                    )
                    
                    if isinstance(result, list):
                        result = result[0]
                    
                    # 結果の取得と日本語変換
                    emotion = result.get('dominant_emotion', 'unknown')
                    emotion_ja = emotion_dict.get(emotion, emotion)
                    age = result.get('age', 'unknown')
                    gender_ja = get_gender_label(result.get('gender', {}))
                    
                    # 背景描画
                    overlay = display_frame.copy()
                    cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
                    
                    # 情報表示
                    display_frame = cv2_putText_ja(display_frame, f'感情: {emotion_ja}', (20, 40), font_path, font_size, (255, 255, 255))
                    display_frame = cv2_putText_ja(display_frame, f'年齢: {age}歳', (20, 80), font_path, font_size, (255, 255, 255))
                    display_frame = cv2_putText_ja(display_frame, f'性別: {gender_ja}', (20, 120), font_path, font_size, (255, 255, 255))
                    
                    last_analysis_time = current_time
                
                # FPS表示
                fps = 1.0 / (time.time() - current_time) if time.time() != current_time else 0
                display_frame = cv2_putText_ja(
                    display_frame, 
                    f'FPS: {fps:.1f}', 
                    (display_frame.shape[1] - 150, display_frame.shape[0] - 30),
                    font_path, 
                    font_size, 
                    (0, 255, 0)
                )
                
                # 終了案内
                display_frame = cv2_putText_ja(
                    display_frame,
                    "終了: Qキー",
                    (20, display_frame.shape[0] - 30),
                    font_path,
                    font_size,
                    (0, 255, 0)
                )
            
            except Exception as e:
                import traceback
                logger.error(f"分析エラー: {str(e)}")
                logger.error(traceback.format_exc())
            
            cv2.imshow('顔認識テスト', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("プログラムを終了します")
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("リアルタイム顔認識テストを開始します...")
    test_face_recognition()