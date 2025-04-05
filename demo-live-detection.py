import os
os.environ['TORCH_HOME'] = './torch_cache'
os.environ['MPLCONFIGDIR'] = './mpl_cache'

import cv2
import torch
import pandas as pd
import numpy as np
import joblib
from collections import deque

# === Load YOLOv5 model ===
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
model.conf = 0.3

# === Load ML classifier and scaler ===
classifier = joblib.load('shoplifting_classifier_lgbm.pkl')
scaler = joblib.load('shoplifting_scaler_lgbm.pkl')

# === Class ID settings ===
PERSON_CLASS = 0
BAG_CLASSES = [24, 26, 28, 31, 39, 56, 62, 63, 66, 73]

# === Open webcam (0 = default camera) ===
cap = cv2.VideoCapture(0)
frame_buffer = deque(maxlen=30)
predictions = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    df = results.pandas().xyxy[0]

    persons = df[df['class'] == PERSON_CLASS]
    items = df[df['class'] != PERSON_CLASS]  # Treat everything else as shopliftable item

    num_person = len(persons)
    num_bag = len(df[df['class'].isin(BAG_CLASSES)])
    overlap_count = 0
    unique_classes = df['class'].nunique()

    for _, p in persons.iterrows():
        px = (p['xmin'] + p['xmax']) / 2
        py = (p['ymin'] + p['ymax']) / 2
        for _, i in items.iterrows():
            ix = (i['xmin'] + i['xmax']) / 2
            iy = (i['ymin'] + i['ymax']) / 2
            if abs(px - ix) < 50 and abs(py - iy) < 50:
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

    # === Draw bounding boxes around entire person ===
    for _, p in persons.iterrows():
        x1, y1 = int(p['xmin']), int(p['ymin'])
        x2, y2 = int(p['xmax']), int(p['ymax'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # === Shoplifting ratio display and alert ===
    if len(predictions) > 360:
        predictions = predictions[-360:]
    shop_ratio = predictions.count(1) / len(predictions)
    cv2.putText(frame, f"Shoplifting Ratio: {shop_ratio:.2%}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    # === Real-time alert if ratio exceeds 65% ===
    if shop_ratio >= 0.65:
        cv2.putText(frame, "⚠️ SHOPLIFTING ALERT!", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    cv2.imshow("Shoplifting Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# === Final Decision Output ===
ratio = predictions.count(1) / len(predictions) if predictions else 0
threshold = 0.6
final_decision = "SHOPLIFTING DETECTED" if ratio >= threshold else "NORMAL BEHAVIOR"

final_frame = np.zeros((300, 600, 3), dtype=np.uint8)
color = (0, 0, 255) if "SHOPLIFTING" in final_decision else (0, 255, 0)
cv2.putText(final_frame, final_decision, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)

cv2.imshow("Final Decision", final_frame)
print(f"\n🔍 Final Behavior Detection: {final_decision} ({ratio*100:.1f}% suspicious frames)")
cv2.waitKey(0)
cv2.destroyAllWindows()
