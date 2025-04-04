import os

base_path = 'datasets/Shoplifting'
videos = os.listdir(base_path)

for i, filename in enumerate(videos):
    if filename.lower().endswith('.mp4'):
        new_name = f'shoplifting_{i:03}.mp4'
        os.rename(os.path.join(base_path, filename), os.path.join(base_path, new_name))
        print(f"✅ Renamed: {filename} → {new_name}")
