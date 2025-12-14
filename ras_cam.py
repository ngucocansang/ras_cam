import time
import random
import requests

ESP32_IP = "192.168.4.1"                 # IP mặc định AP của ESP32
URL = f"http://{ESP32_IP}/api/ai"

def send(healthy: bool, score: float):
    payload = {"healthy": healthy, "score": float(score)}
    r = requests.post(URL, json=payload, timeout=3)
    print("POST", payload, "->", r.status_code, r.text)

if __name__ == "__main__":
    print("Make sure Raspberry Pi is connected to Wi-Fi: PLANTY_AP")
    while True:
        healthy = random.choice([True, False])
        score = random.random()
        send(healthy, score)
        time.sleep(2)
