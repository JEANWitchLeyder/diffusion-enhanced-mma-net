import os
import cv2
import numpy as np

seq = "0_Road029_Trim001_frames"

img_dir = f"dataset/VIL100/JPEGImages/{seq}"
mask_dir = f"dataset/output/VIL100/60_lr0.001deay1e-6_sgd/{seq}"
out_dir = f"demo_output/{seq}_anchor_refined"

os.makedirs(out_dir, exist_ok=True)

frames = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
video_frames = []


def refine_lane(binary_mask, min_points=80):
    ys, xs = np.where(binary_mask > 0)

    if len(xs) < min_points:
        return np.zeros_like(binary_mask)

    coeffs = np.polyfit(ys, xs, deg=2)

    y_min, y_max = ys.min(), ys.max()
    y_new = np.linspace(y_min, y_max, num=300)
    x_new = np.polyval(coeffs, y_new)

    refined = np.zeros_like(binary_mask)
    h, w = binary_mask.shape
    points = []

    for x, y in zip(x_new, y_new):
        x = int(round(x))
        y = int(round(y))
        if 0 <= x < w and 0 <= y < h:
            points.append((x, y))

    if len(points) > 1:
        cv2.polylines(refined, [np.array(points, dtype=np.int32)], False, 255, thickness=8)

    return refined


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

    red_lane = (mask[:, :, 2] > 50) & (mask[:, :, 1] < 100)
    green_lane = (mask[:, :, 1] > 50) & (mask[:, :, 2] < 100)

    refined_red = refine_lane(red_lane.astype(np.uint8))
    refined_green = refine_lane(green_lane.astype(np.uint8))

    refined_mask = np.zeros_like(img)
    refined_mask[refined_red > 0] = (0, 0, 255)
    refined_mask[refined_green > 0] = (0, 255, 0)

    overlay = cv2.addWeighted(img, 0.75, refined_mask, 1.0, 0)

    out_path = os.path.join(out_dir, name)
    cv2.imwrite(out_path, overlay)
    video_frames.append(overlay)

if video_frames:
    h, w, _ = video_frames[0].shape
    video_path = f"demo_output/{seq}_anchor_refined_demo.mp4"

    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))

    for frame in video_frames:
        writer.write(frame)

    writer.release()

    print(f"Anchor-refined demo saved: {video_path}")
    print(f"Frames saved in: {out_dir}")
else:
    print("No frames processed.")
    print(f"Checked image dir: {img_dir}")
    print(f"Checked mask dir: {mask_dir}")
