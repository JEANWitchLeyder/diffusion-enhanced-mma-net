import numpy as np
import cv2


def diffusion_refine(pred, steps=5):
    """
    Lightweight diffusion-inspired refinement for lane probability masks.

    pred shape: [T, C, H, W]
    T = number of frames
    C = number of classes/lane instances
    H, W = image height and width

    This function simulates iterative denoising:
    1. Convert probabilities into smooth masks.
    2. Apply repeated Gaussian smoothing.
    3. Preserve strongest lane responses.
    4. Normalize probabilities again.

    This is not a full trainable DDPM. It is a practical diffusion-inspired
    refinement module for a semester research prototype.
    """

    refined = pred.copy()

    for _ in range(steps):
        for t in range(refined.shape[0]):
            for c in range(1, refined.shape[1]):  # skip background class
                lane_prob = refined[t, c]

                smooth = cv2.GaussianBlur(lane_prob, (5, 5), 0)

                refined[t, c] = 0.7 * lane_prob + 0.3 * smooth

        refined = np.clip(refined, 1e-8, 1.0)
        refined = refined / np.sum(refined, axis=1, keepdims=True)

    return refined