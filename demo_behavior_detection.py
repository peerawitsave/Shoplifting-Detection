import os
os.environ['TORCH_HOME'] = './torch_cache'
os.environ['MPLCONFIGDIR'] = './mpl_cache'

import cv2
import torch
import pandas as pd
import numpy as np
import joblib
from collections import deque
import webcolors

# === Utility to get closest readable color name ===
def closest_color(requested_color):
    min_colors = {}
    for hex_value, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(hex_value)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

# === Load YOLOv5 model ===
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
model.conf = 0.3

# === Load ML classifier and scaler ===
classifier = joblib.load('shoplifting_classifier_lgbm.pkl')
scaler = joblib.load('shoplifting_scaler_lgbm.pkl')

# === Class ID settings ===
PERSON_CLASS = 0
BAG_CLASSES = [24, 26, 28, 31, 39, 56, 62, 63, 66, 73]

# === Open video ===
cap = cv2.VideoCapture("demo.mp4")
frame_buffer = deque(maxlen=30)
predictions = []

shoplifting_alert_shown = False
threshold = 0.65

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    df = results.pandas().xyxy[0]

    persons = df[df['class'] == PERSON_CLASS]
    bags = df[df['class'].isin(BAG_CLASSES)]

    num_person = len(persons)
    num_bag = len(bags)
    overlap_count = 0
    unique_classes = df['class'].nunique()

    for _, p in persons.iterrows():
        px = (p['xmin'] + p['xmax']) / 2
        py = (p['ymin'] + p['ymax']) / 2
        for _, b in bags.iterrows():
            bx = (b['xmin'] + b['xmax']) / 2
            by = (b['ymin'] + b['ymax']) / 2
            if abs(px - bx) < 50 and abs(py - by) < 50:
                overlap_count += 1

    frame_buffer.append([num_person, num_bag, overlap_count, unique_classes])
    window = pd.DataFrame(frame_buffer, columns=['num_person', 'num_bag', 'overlap_count', 'unique_classes'])
    features = {
        'num_person': num_person,
        'num_bag': num_bag,
        'overlap_count': overlap_count,
        'unique_classes': unique_classes,
        'person_rolling_mean': window['num_person'].mean(),
        'bag_rolling_mean': window['num_bag'].mean(),
        'overlap_rolling_mean': window['overlap_count'].mean()
    }

    X = scaler.transform([list(features.values())])
    pred = classifier.predict(X)[0]
    predictions.append(pred)

    label = "Shoplifting" if pred == 1 else "Normal"
    color = (0, 0, 255) if pred == 1 else (0, 255, 0)
    cv2.putText(frame, f"Behavior: {label}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # === Draw bounding boxes + clothing color ===
    for _, p in persons.iterrows():
        x1, y1 = int(p['xmin']), int(p['ymin'])
        x2, y2 = int(p['xmax']), int(p['ymax'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Torso region for clothing color
        torso_x1 = x1 + int((x2 - x1) * 0.25)
        torso_x2 = x1 + int((x2 - x1) * 0.75)
        torso_y1 = y1 + int((y2 - y1) * 0.4)
        torso_y2 = y1 + int((y2 - y1) * 0.6)
        torso_crop = frame[torso_y1:torso_y2, torso_x1:torso_x2]

        if torso_crop.size > 0:
            avg_color = np.mean(torso_crop.reshape(-1, 3), axis=0)
            try:
                color_name = closest_color(avg_color.astype(int))
            except:
                color_name = "Unknown"
            cv2.putText(frame, f"Clothing: {color_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # === Ratio calculation ===
    if len(predictions) > 360:
        predictions = predictions[-360:]
    shop_ratio = predictions.count(1) / len(predictions)
    cv2.putText(frame, f"Shoplifting Ratio: {shop_ratio:.2%}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    # === Real-time shoplifting alert with pause ===
    if shop_ratio >= threshold and not shoplifting_alert_shown:
        shoplifting_alert_shown = True
        cv2.putText(frame, "SHOPLIFTING DETECTED", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
        cv2.imshow("Shoplifting Detection", frame)
        print("🔴 PAUSED: Shoplifting detected. Press any key to continue...")
        cv2.waitKey(0)

    cv2.imshow("Shoplifting Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# === Final decision output ===
ratio = predictions.count(1) / len(predictions) if predictions else 0
final_decision = "SHOPLIFTING DETECTED" if ratio >= threshold else "NORMAL BEHAVIOR"
final_frame = np.zeros((300, 600, 3), dtype=np.uint8)
final_color = (0, 0, 255) if "SHOPLIFTING" in final_decision else (0, 255, 0)

cv2.putText(final_frame, final_decision, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, final_color, 4)
cv2.imshow("Final Decision", final_frame)
print(f"\n🔍 Final Behavior Detection: {final_decision} ({ratio*100:.1f}% suspicious frames)")
cv2.waitKey(0)
cv2.destroyAllWindows()
