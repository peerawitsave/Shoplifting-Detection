import os
import pandas as pd

label_dir = 'datasets/raw/labels'
image_dir = 'datasets/raw/images'
output_csv = 'behavior_features.csv'

PERSON_CLASS = '0'
BAG_CLASSES = ['24', '26', '28', '31', '39', '56', '62', '63', '66', '73']  # expand as needed

data = []

for label_file in os.listdir(label_dir):
    if not label_file.endswith('.txt'):
        continue

    filepath = os.path.join(label_dir, label_file)
    with open(filepath, 'r') as f:
        lines = f.readlines()

    num_person = 0
    num_bag = 0
    overlap_count = 0
    class_ids = []

    boxes = []

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        class_id, x, y, w, h = parts
        class_ids.append(int(class_id))
        bbox = [float(x), float(y), float(w), float(h)]
        boxes.append((class_id, bbox))

        if class_id == PERSON_CLASS:
            num_person += 1
        if class_id in BAG_CLASSES:
            num_bag += 1

    for i, (cls1, box1) in enumerate(boxes):
        for j, (cls2, box2) in enumerate(boxes):
            if i != j and cls1 == PERSON_CLASS and cls2 in BAG_CLASSES:
                dx = abs(box1[0] - box2[0])
                dy = abs(box1[1] - box2[1])
                if dx < 0.1 and dy < 0.1:
                    overlap_count += 1

    filename = label_file.replace('.txt', '.jpg')
    is_shoplifting = int('shoplifting' in filename.lower() and num_person > 0)

    data.append({
        'filename': filename,
        'num_person': num_person,
        'num_bag': num_bag,
        'overlap_count': overlap_count,
        'unique_classes': len(set(class_ids)),
        'is_shoplifting': is_shoplifting
    })

df = pd.DataFrame(data)
df.to_csv(output_csv, index=False)
print(f"✅ Done. Features saved to {output_csv}")
