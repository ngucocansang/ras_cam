import os
import time
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import requests

# =========================
# 0. RAS CONFIG
# =========================
WEIGHTS_PATH = "tomato_disease_model_weights.pth"
IMG_SIZE = 256

# ESP32 AP default IP
ESP32_IP = os.getenv("ESP32_IP", "192.168.4.1")
ESP32_AI_URL = f"http://{ESP32_IP}/api/ai"

# Camera index on Raspberry Pi (try 0 first)
CAM_INDEX = int(os.getenv("CAM_INDEX", "0"))

# Send every N seconds (avoid spamming)
SEND_EVERY_SEC = float(os.getenv("SEND_EVERY_SEC", "2.0"))

# If you are running headless (no monitor), set HEADLESS=1
HEADLESS = os.getenv("HEADLESS", "0") == "1"

# Timeout for HTTP requests
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "3.0"))

# =========================
# 1. CLASS NAMES
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
# 2. DEVICE UTILS
# =========================
def get_default_device():
    # Raspberry Pi usually is CPU only
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def to_device(x, device):
    return x.to(device, non_blocking=True)

# =========================
# 3. MODEL
# =========================
def ConvBlock(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

class CNN_NeuralNet(nn.Module):
    def __init__(self, in_channels, num_diseases):
        super().__init__()

        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(
            ConvBlock(128, 128),
            ConvBlock(128, 128),
        )

        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)

        self.res2 = nn.Sequential(
            ConvBlock(512, 512),
            ConvBlock(512, 512),
        )

        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Linear(512, num_diseases),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

# =========================
# 4. LOAD MODEL
# =========================
def load_trained_model(weights_path: str):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Không tìm thấy file: {weights_path}")

    device = get_default_device()
    print("✅ Device:", device)

    num_classes = len(CLASS_NAMES)
    model = CNN_NeuralNet(in_channels=3, num_diseases=num_classes)

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)

    model = to_device(model, device)
    model.eval()
    print(f"✅ Loaded weights: {weights_path}")
    return model, device

# =========================
# 5. PREPROCESS
# =========================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

def preprocess_frame(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_t = transform(frame_rgb)
    img_t = img_t.unsqueeze(0)
    return img_t

# =========================
# 6. PREDICT (2 states)
# =========================
def predict_frame_binary(frame_bgr, model, device):
    xb = preprocess_frame(frame_bgr)
    xb = to_device(xb, device)

    with torch.no_grad():
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)
        idx = torch.argmax(probs, dim=1).item()
        raw_label = CLASS_NAMES[idx]
        conf = probs[0, idx].item()

    health_status = "healthy" if raw_label == HEALTHY_CLASS_NAME else "unhealthy"
    return health_status, conf, raw_label

# =========================
# 7. SEND TO ESP32
# =========================
def send_to_esp32(health_status: str, conf: float):
    # ESP32 expects: {"healthy": true/false, "score": 0..1}
    payload = {
        "healthy": True if health_status == "healthy" else False,
        "score": float(conf),
    }
    r = requests.post(ESP32_AI_URL, json=payload, timeout=HTTP_TIMEOUT)
    return r.status_code, r.text

# =========================
# 8. WEBCAM + LOOP
# =========================
def run_webcam_and_publish():
    print("ESP32 AI URL:", ESP32_AI_URL)
    print("CAM_INDEX:", CAM_INDEX, "| SEND_EVERY_SEC:", SEND_EVERY_SEC, "| HEADLESS:", HEADLESS)

    model, device = load_trained_model(WEIGHTS_PATH)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("❌ Không mở được camera. Thử CAM_INDEX=0/1/2")
        return

    print("🎥 Running. Press 'q' to quit (if not headless).")

    last_send = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠ Không đọc được frame.")
            time.sleep(0.2)
            continue

        health_status, conf, raw_label = predict_frame_binary(frame, model, device)

        now = time.time()
        if now - last_send >= SEND_EVERY_SEC:
            try:
                code, text = send_to_esp32(health_status, conf)
                print(f"📤 AI -> ESP32 | {health_status} conf={conf:.3f} raw={raw_label} | {code} {text}")
            except Exception as e:
                print(f"❌ Send failed: {e}")
            last_send = now

        # Optional display (disable for headless)
        if not HEADLESS:
            label = f"{health_status.upper()} ({conf*100:.1f}%)"
            color = (0, 255, 0) if health_status == "healthy" else (0, 0, 255)

            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
            cv2.putText(frame, f"raw: {raw_label}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("Plant Health (Binary) - RPi Sender", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if not HEADLESS:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam_and_publish()
