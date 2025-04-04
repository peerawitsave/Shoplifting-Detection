import os
os.environ['TORCH_HOME'] = './torch_cache'
os.environ['MPLCONFIGDIR'] = './mpl_cache'

import cv2
import torch
import pandas as pd
import numpy as np
import joblib
from collections import deque
from time import time

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
model.conf = 0.3

classifier = joblib.load('shoplifting_classifier_lgbm.pkl')
scaler = joblib.load('shoplifting_scaler_lgbm.pkl')

PERSON_CLASS = 0
BAG_CLASSES = [24, 26, 28, 31, 39, 56, 62, 63, 66, 73]

cap = cv2.VideoCapture("demo.mp4")
frame_buffer = deque(maxlen=30)
predictions = []

start_time = time()
frame_counter = 0

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

    if len(predictions) > 360:
        predictions = predictions[-360:]
    shop_ratio = predictions.count(1) / len(predictions)
    cv2.putText(frame, f"Shoplifting Ratio: {shop_ratio:.2%}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    cv2.imshow("Shoplifting Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

ratio = predictions.count(1) / len(predictions) if predictions else 0
final_decision = "SHOPLIFTING DETECTED" if ratio > 0.3 else "NORMAL BEHAVIOR"
final_frame = np.zeros((300, 600, 3), dtype=np.uint8)
color = (0, 0, 255) if "SHOPLIFTING" in final_decision else (0, 255, 0)

cv2.putText(final_frame, final_decision, (50, 150), cv2.FONT_HERSHEY_SIMPLEX,
            1.5, color, 4)

cv2.imshow("Final Decision", final_frame)
print(f"\n🔍 Final Behavior Detection: {final_decision} ({ratio*100:.1f}% suspicious frames)")
cv2.waitKey(0)
cv2.destroyAllWindows()
