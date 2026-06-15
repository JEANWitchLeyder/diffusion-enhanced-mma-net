import cv2
import os
import numpy as np

seq = "0_Road029_Trim001_frames"

img_dir = f"dataset/VIL100/JPEGImages/{seq}"
mask_dir = f"dataset/output/VIL100/60_lr0.001deay1e-6_sgd/{seq}"
out_dir = f"demo_output/{seq}"

os.makedirs(out_dir, exist_ok=True)

frames = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
video_frames = []

for name in frames:
    img_path = os.path.join(img_dir, name)
    mask_path = os.path.join(mask_dir, name.replace(".jpg", ".png"))

    if not os.path.exists(mask_path):
        continue

    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)

    if img is None or mask is None:
        continue

    if img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    lane_area = np.any(mask > 0, axis=2)

    color_mask = np.zeros_like(img)
    color_mask[lane_area] = mask[lane_area]

    overlay = cv2.addWeighted(img, 0.75, color_mask, 1.2, 0)

    out_path = os.path.join(out_dir, name)
    cv2.imwrite(out_path, overlay)
    video_frames.append(overlay)

if video_frames:
    h, w, _ = video_frames[0].shape
    video_path = f"demo_output/{seq}_demo.mp4"
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))

    for frame in video_frames:
        writer.write(frame)

    writer.release()
    print(f"Demo video saved: {video_path}")
    print(f"Overlay frames saved in: {out_dir}")
else:
    print("No frames processed.")