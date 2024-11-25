import cv2

def test_camera():
    # カメラの初期化（0は通常、デフォルトのWebカメラ）
    cap = cv2.VideoCapture(0)
    
    try:
        while True:
            # フレームの取得
            ret, frame = cap.read()
            if not ret:
                print("フレームの取得に失敗しました")
                break
            
            # フレームの表示
            cv2.imshow('Camera Test', frame)
            
            # 'q'キーで終了
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # リソースの解放
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera()