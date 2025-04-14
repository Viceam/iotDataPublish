from zoneinfo import ZoneInfo

import pandas as pd
import paho.mqtt.client as mqtt
import time
import json
import os
import random
from datetime import datetime
import numpy as np

# MQTT Broker 配置
broker = "localhost"
port = 1883
topic = "sensor/data"

# CSV 数据路径
csv_file_path = "data/equipment_anomaly_data.csv"
if not os.path.exists(csv_file_path):
    raise FileNotFoundError(f"The file {csv_file_path} does not exist.")

df = pd.read_csv(csv_file_path).head(100)

# 正常 & 异常参数范围和扰动标准差
PARAM_CONFIG = {
    "temperature": {"min": 59.61, "max": 85.92, "normal_std": 3.0, "abnormal_offset": 50},
    "pressure": {"min": 23.39, "max": 56.17, "normal_std": 1.5, "abnormal_offset": 25},
    "vibration": {"min": 0.31, "max": 2.62, "normal_std": 0.1, "abnormal_offset": 2},
    "humidity": {"min": 33.99, "max": 76.39, "normal_std": 2.0, "abnormal_offset": 40}
}

SEND_INTERVAL = 0.3

def generate_value(base, key, is_abnormal):
    config = PARAM_CONFIG[key]
    if is_abnormal:
        # 加入固定偏移 + 大扰动
        noisy = base + random.choice([-1, 1]) * config["abnormal_offset"]
        return float(max(noisy, 0.))
    else:
        noisy = np.random.normal(base, config["normal_std"])
        return float(np.clip(noisy, config["min"], config["max"]))

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to broker")
    else:
        print(f"Failed to connect, return code {rc}")

def on_disconnect(client, userdata, rc):
    print("Disconnected from broker")

def main():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect

    client.connect(broker, port, 60)
    client.loop_start()

    try:
        while True:
            for index, row in df.iterrows():
                is_abnormal = random.random() < 0.05  # 10% 概率触发异常
                data = {
                    "device_id": f"sensor_{index}",
                    "timestamp": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                    "temperature": generate_value(float(row['temperature']), "temperature", is_abnormal),
                    "pressure": generate_value(float(row['pressure']), "pressure", is_abnormal),
                    "vibration": generate_value(float(row['vibration']), "vibration", is_abnormal),
                    "humidity": generate_value(float(row['humidity']), "humidity", is_abnormal),
                    "equipment": row['equipment'],
                    "location": row['location'],
                    "faulty": int(row['faulty']),
                }
                client.publish(topic, json.dumps(data))
                print(f"[{'⚠️' if is_abnormal else 'OK'}] Published: {data}")
                time.sleep(SEND_INTERVAL)
    except KeyboardInterrupt:
        print("Simulation stopped by user.")
    finally:
        client.loop_stop()
        client.disconnect()
        print("Disconnected from broker")

if __name__ == "__main__":
    main()
