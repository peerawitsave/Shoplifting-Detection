import os
import torch
from PIL import Image

os.environ['TORCH_HOME'] = './torch_cache'
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.25
model.iou = 0.45

img_dir = 'datasets/raw/images'
label_dir = 'datasets/raw/labels'
os.makedirs(label_dir, exist_ok=True)

for img_name in os.listdir(img_dir):
    if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    img_path = os.path.join(img_dir, img_name)
    results = model(img_path)

    h, w = Image.open(img_path).size[::-1]
    label_path = os.path.join(label_dir, img_name.rsplit('.', 1)[0] + '.txt')

    with open(label_path, 'w') as f:
        for *box, conf, cls in results.xywh[0]:
            cls = int(cls.item())
            x_center, y_center, bw, bh = [v.item() for v in box]

            x_center /= w
            y_center /= h
            bw /= w
            bh /= h

            f.write(f"{cls} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")

print("✅ Auto-labeling complete. Labels saved to datasets/raw/labels/")
