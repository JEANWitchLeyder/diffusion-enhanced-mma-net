import os
import cv2
import numpy as np

IMG_ROOT = "dataset/VIL100/JPEGImages"
MASK_ROOT = "dataset/output/VIL100/60_lr0.001deay1e-6_sgd"
OUT_ROOT = "demo_output/anchor_refined_all"

os.makedirs(OUT_ROOT, exist_ok=True)


def extract_candidates(mask):
    h, w = mask.shape[:2]
    binary = np.any(mask > 40, axis=2).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    candidates = []

    for lab in range(1, num_labels):
        area = stats[lab, cv2.CC_STAT_AREA]
        if area < 120:
            continue

        ys, xs = np.where(labels == lab)

        if len(xs) < 120:
            continue

        y_top = ys.min()
        y_bottom = ys.max()
        y_span = y_bottom - y_top

        if y_span < 0.22 * h:
            continue

        # Lane must reach near the lower road area.
        if y_bottom < 0.60 * h:
            continue

        bottom_region = ys > (y_bottom - 25)
        if np.sum(bottom_region) < 10:
            continue

        x_bottom = np.median(xs[bottom_region])

        # Reject candidates too close to far horizon only.
        if x_bottom < 0 or x_bottom >= w:
            continue

        # Fit stable line: x = ay + b
        try:
            coeffs = np.polyfit(ys, xs, deg=1)
        except Exception:
            continue

        a, b = coeffs
        x_at_bottom = a * (h - 1) + b
        x_at_mid = a * int(h * 0.55) + b

        # Reject lines that explode outside the image.
        if not (-0.25 * w <= x_at_bottom <= 1.25 * w):
            continue
        if not (-0.25 * w <= x_at_mid <= 1.25 * w):
            continue

        side = "left" if x_at_bottom < w / 2 else "right"

        score = (
            area * 0.4
            + y_span * 2.0
            + y_bottom * 1.5
            - abs(x_at_bottom - w / 2) * 0.05
        )

        candidates.append({
            "side": side,
            "coeffs": coeffs,
            "score": score,
            "y_top": y_top,
            "y_bottom": y_bottom,
            "area": area
        })

    return candidates


def draw_lane(img, candidate, color):
    h, w = img.shape[:2]
    a, b = candidate["coeffs"]

    y1 = int(candidate["y_bottom"])
    y2 = int(candidate["y_top"])

    y_values = np.linspace(y1, y2, 120)
    points = []

    for y in y_values:
        x = int(round(a * y + b))
        y = int(round(y))

        if 0 <= x < w and 0 <= y < h:
            points.append((x, y))

    if len(points) > 1:
        cv2.polylines(img, [np.array(points, dtype=np.int32)], False, color, thickness=8)

    return img


def process_sequence(seq):
    img_dir = os.path.join(IMG_ROOT, seq)
    mask_dir = os.path.join(MASK_ROOT, seq)

    if not os.path.isdir(img_dir) or not os.path.isdir(mask_dir):
        return

    frames = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
    video_frames = []

    seq_out_dir = os.path.join(OUT_ROOT, seq)
    os.makedirs(seq_out_dir, exist_ok=True)

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

        candidates = extract_candidates(mask)

        lefts = [c for c in candidates if c["side"] == "left"]
        rights = [c for c in candidates if c["side"] == "right"]

        left = max(lefts, key=lambda c: c["score"]) if lefts else None
        right = max(rights, key=lambda c: c["score"]) if rights else None

        refined_mask = np.zeros_like(img)

        if left is not None:
            refined_mask = draw_lane(refined_mask, left, (0, 0, 255))

        if right is not None:
            refined_mask = draw_lane(refined_mask, right, (0, 255, 0))

        overlay = cv2.addWeighted(img, 0.78, refined_mask, 1.0, 0)

        cv2.imwrite(os.path.join(seq_out_dir, name), overlay)
        video_frames.append(overlay)

    if video_frames:
        h, w, _ = video_frames[0].shape
        video_path = os.path.join(OUT_ROOT, f"{seq}_anchor_refined_demo.mp4")

        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))

        for frame in video_frames:
            writer.write(frame)

        writer.release()
        print(f"Saved: {video_path}")


seqs = sorted([
    d for d in os.listdir(MASK_ROOT)
    if os.path.isdir(os.path.join(MASK_ROOT, d))
])

for seq in seqs:
    process_sequence(seq)

print("Done.")
