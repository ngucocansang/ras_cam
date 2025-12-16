import os, time, cv2, numpy as np, requests
from tflite_runtime.interpreter import Interpreter

TFLITE_PATH = os.getenv("TFLITE_PATH", "tomato_model.tflite")
ESP32_IP    = os.getenv("ESP32_IP", "192.168.4.1")
URL         = f"http://{ESP32_IP}/api/ai"
CAM_INDEX   = int(os.getenv("CAM_INDEX", "0"))
IMG_SIZE    = int(os.getenv("IMG_SIZE", "256"))
SEND_EVERY  = float(os.getenv("SEND_EVERY_SEC", "2.0"))

CLASS_NAMES = [
    "Tomato___Bacterial_spot","Tomato___Early_blight","Tomato___Late_blight",
    "Tomato___Leaf_Mold","Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite","Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus","Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]
HEALTHY_CLASS_NAME = "Tomato___healthy"

def softmax(x):
    x = x.astype(np.float32)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

def preprocess(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    x = rgb.astype(np.float32) / 255.0
    return np.expand_dims(x, 0)  # NHWC

def main():
    print("Loading:", TFLITE_PATH)
    intr = Interpreter(model_path=TFLITE_PATH, num_threads=2)
    intr.allocate_tensors()
    in0  = intr.get_input_details()[0]
    out0 = intr.get_output_details()[0]
    print("Input:", in0["shape"], in0["dtype"])
    print("Output:", out0["shape"], out0["dtype"])
    print("ESP32:", URL)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {CAM_INDEX}")

    last_send = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("⚠ camera read failed")
            time.sleep(0.2)
            continue

        x = preprocess(frame)

        # Nếu model đòi NCHW thì bật dòng này:
        # x = np.transpose(x, (0,3,1,2))

        intr.set_tensor(in0["index"], x)
        intr.invoke()
        y = intr.get_tensor(out0["index"])[0]

        # y có thể là logits hoặc probs
        probs = softmax(y) if (y.ndim == 1 and (y.max() > 1.0 or y.min() < 0.0)) else y
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        raw = CLASS_NAMES[idx]
        healthy = (raw == HEALTHY_CLASS_NAME)

        now = time.time()
        if now - last_send >= SEND_EVERY:
            payload = {"healthy": bool(healthy), "score": float(conf)}
            try:
                r = requests.post(URL, json=payload, timeout=3)
                print(f"📤 {payload} raw={raw} -> {r.status_code}")
            except Exception as e:
                print("❌ Send failed:", e)
            last_send = now

        # preview (q to quit)
        txt = f"{'HEALTHY' if healthy else 'UNHEALTHY'} {conf*100:.1f}%"
        cv2.putText(frame, txt, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0,255,0) if healthy else (0,0,255), 2)
        cv2.imshow("Planty TFLite", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
