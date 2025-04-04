import os
import subprocess

video_root = 'datasets'
categories = ['Normal', 'Shoplifting']
output_dir = os.path.join(video_root, 'cleaned')

os.makedirs(output_dir, exist_ok=True)

for category in categories:
    category_path = os.path.join(video_root, category)
    output_category = os.path.join(output_dir, category)
    os.makedirs(output_category, exist_ok=True)

    for video_name in os.listdir(category_path):
        if not video_name.lower().endswith('.mp4'):
            continue

        input_path = os.path.join(category_path, video_name)
        output_path = os.path.join(output_category, video_name)

        print(f"🎞️ Re-encoding: {video_name} → {output_path}")
        cmd = [
            'ffmpeg', '-y',
            '-i', input_path,
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-c:a', 'aac',
            output_path
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed: {video_name} → {e}")
