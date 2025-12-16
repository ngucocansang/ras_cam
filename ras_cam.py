import time
import cv2
import numpy as np
import requests
import tensorflow as tf

# =========================
# CONFIG
# =========================
H5_MODEL_PATH = "tomato_model.h5"      # tên file .h5
ESP32_IP = "192.168.4.1"
ESP32_AI_URL = f"http://{ESP32_IP}/api/ai"

CAM_INDEX = 0            # thử 0 trước, nếu lỗi đổi 1
IMG_SIZE = 256           # phải đúng với lúc train
SEND_EVERY_SEC = 2.0     # gửi kết quả mỗi 2 giây

# =========================
# CLASS NAMES (GIỮ NGUYÊN THEO MODEL CỦA BẠN)
# =========================
CLASS_NAMES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

HEALTHY_CLASS_NAME = "Tomato___healthy"

# =========================
# LOAD MODEL
# =========================
print("🔄 Loading model:", H5_MODEL_PATH)
model = tf.keras.models.load_model(H5_MODEL_PATH)
print("✅ Model loaded")

# =========================
# PREPROCESS
# =========================
def preprocess_bgr(frame_bgr):
    """
    BGR -> RGB
    resize
    normalize [0,1]
    NHWC
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    x = rgb.astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=0)  # (1, H, W, 3)
    return x

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError("❌ Cannot open camera. Try CAM_INDEX=1")

print("🎥 Camera running. Press 'q' to quit.")

last_send = 0.0

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Camera read failed")
        time.sleep(0.2)
        continue

    x = preprocess_bgr(frame)

    # inference
    preds = model.predict(x, verbose=0)[0]  # shape (num_classes,)
    idx = int(np.argmax(preds))
    conf = float(preds[idx])
    raw_label = CLASS_NAMES[idx]

    health_status = "healthy" if raw_label == HEALTHY_CLASS_NAME else "unhealthy"

    # =========================
    # SEND TO ESP32
    # =========================
    now = time.time()
    if now - last_send >= SEND_EVERY_SEC:
        payload = {
            "healthy": (health_status == "healthy"),
            "score": conf
        }
        try:
            r = requests.post(ESP32_AI_URL, json=payload, timeout=3)
            print(f"📤 AI → ESP32 | {health_status.upper()} {conf:.3f} | raw={raw_label} | {r.status_code}")
        except Exception as e:
            print("❌ Send failed:", e)
        last_send = now

    # =========================
    # VISUAL DEBUG
    # =========================
    label = f"{health_status.upper()} {conf*100:.1f}%"
    color = (0, 255, 0) if health_status == "healthy" else (0, 0, 255)

    cv2.putText(frame, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, raw_label, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

    cv2.imshow("Plant Health AI (.h5)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
