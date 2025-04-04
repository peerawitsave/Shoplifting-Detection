import os
import pandas as pd

label_dir = 'datasets/raw/labels'
image_dir = 'datasets/raw/images'
output_csv = 'behavior_features.csv'

PERSON_CLASS = '0'
BAG_CLASSES = ['24', '26', '28', '31', '39', '56', '62', '63', '66', '73']

data = []

def iou(boxA, boxB):
    # Simple IOU for normalized center format boxes (x, y, w, h)
    ax, ay, aw, ah = boxA
    bx, by, bw, bh = boxB

    ax1, ay1, ax2, ay2 = ax - aw / 2, ay - ah / 2, ax + aw / 2, ay + ah / 2
    bx1, by1, bx2, by2 = bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    boxA_area = aw * ah
    boxB_area = bw * bh
    union_area = boxA_area + boxB_area - inter_area

    return inter_area / union_area if union_area else 0

for label_file in sorted(os.listdir(label_dir)):
    if not label_file.endswith('.txt'):
        continue

    filepath = os.path.join(label_dir, label_file)
    with open(filepath, 'r') as f:
        lines = f.readlines()

    num_person = 0
    num_bag = 0
    overlap_count = 0
    suspicious_overlap_density = 0
    class_ids = []
    boxes = []

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        class_id, x, y, w, h = parts
        class_ids.append(int(class_id))
        boxes.append((class_id, [float(x), float(y), float(w), float(h)]))

        if class_id == PERSON_CLASS:
            num_person += 1
        if class_id in BAG_CLASSES:
            num_bag += 1

    # Count overlaps between persons and bags
    for i, (cls1, box1) in enumerate(boxes):
        for j, (cls2, box2) in enumerate(boxes):
            if i != j and cls1 == PERSON_CLASS and cls2 in BAG_CLASSES:
                if iou(box1, box2) > 0.1:
                    overlap_count += 1
                    suspicious_overlap_density += iou(box1, box2)

    filename = label_file.replace('.txt', '.jpg')
    is_shoplifting = int('shoplifting' in filename.lower() and num_person > 0)

    # Additional derived features
    person_to_bag_ratio = num_person / num_bag if num_bag > 0 else 0
    is_crowded = int(num_person >= 4)

    data.append({
        'filename': filename,
        'num_person': num_person,
        'num_bag': num_bag,
        'overlap_count': overlap_count,
        'suspicious_overlap_density': suspicious_overlap_density,
        'person_to_bag_ratio': person_to_bag_ratio,
        'unique_classes': len(set(class_ids)),
        'is_crowded': is_crowded,
        'is_shoplifting': is_shoplifting
    })

df = pd.DataFrame(data)
df.to_csv(output_csv, index=False)
print(f"✅ Done. Enhanced features saved to {output_csv}")
