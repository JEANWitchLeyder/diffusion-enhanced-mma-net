import torch


def cold_diffusion_enhance(frames, steps=4, gamma=0.75, strength=0.22, dark_threshold=0.45):
    """
    Controlled cold-diffusion-inspired enhancement.

    Goal:
    - Enhance only dark/low-visibility regions.
    - Avoid over-enhancing bright road edges, barriers, and shoulders.
    - Reduce false-positive lane detections caused by global enhancement.

    frames: Tensor [T, C, H, W], expected range 0-1.
    """

    frames = torch.clamp(frames, 0.0, 1.0)

    # Convert to approximate luminance
    r = frames[:, 0:1, :, :]
    g = frames[:, 1:2, :, :]
    b = frames[:, 2:3, :, :]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b

    # Mask only genuinely dark regions
    dark_mask = (luminance < dark_threshold).float()

    # Smooth mask to avoid harsh transitions
    dark_mask = torch.nn.functional.avg_pool2d(
        dark_mask,
        kernel_size=7,
        stride=1,
        padding=3
    )

    # Gamma target: brighten dark regions
    target = torch.pow(frames, gamma)

    enhanced = frames.clone()

    for _ in range(steps):
        alpha = strength / steps
        update = (target - enhanced) * dark_mask
        enhanced = enhanced + alpha * update
        enhanced = torch.clamp(enhanced, 0.0, 1.0)

    return enhanced