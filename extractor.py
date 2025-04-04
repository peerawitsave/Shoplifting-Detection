import cv2
import os

video_root = 'datasets/cleaned'
categories = ['Normal', 'Shoplifting']
output_dir = 'datasets/images'

os.makedirs(output_dir, exist_ok=True)

frame_interval = 5  # Save every 5th frame
img_count = 0

for label_id, category in enumerate(categories):
    folder_path = os.path.join(video_root, category)
    video_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]

    for video in video_files:
        video_path = os.path.join(folder_path, video)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"[❌] Cannot open video: {video_path}")
            continue

        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_num % frame_interval == 0:
                out_name = f"{category.lower()}_{img_count:05}.jpg"
                out_path = os.path.join(output_dir, out_name)
                cv2.imwrite(out_path, frame)
                img_count += 1
            frame_num += 1

        cap.release()

print(f"✅ Done! Extracted {img_count} frames into '{output_dir}'")
