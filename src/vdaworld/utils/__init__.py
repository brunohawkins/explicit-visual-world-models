import numpy as np
import cv2


def time_freq(frames, thresh=20):
    import time as _time

    n_frames = frames.shape[0]

    if n_frames <= 1:
        return np.zeros_like(frames[0], dtype=np.uint8)

    _t0 = _time.perf_counter()

    # Compute per-frame absolute differences directly in uint8 using cv2.absdiff,
    # then sum channels into a single uint8 magnitude — avoids allocating large
    # int16 intermediates (~1.5 GB for 150 frames at 1024x576).
    h, w = frames.shape[1], frames.shape[2]
    from tqdm import tqdm
    motion_mag = np.empty((n_frames - 1, h, w), dtype=np.uint8)
    for i in tqdm(range(n_frames - 1), desc="MHI diff", leave=False):
        diff = cv2.absdiff(frames[i + 1], frames[i])          # uint8 (H, W, 3)
        motion_mag[i] = np.sum(diff, axis=-1, dtype=np.uint16).clip(0, 255).astype(np.uint8)
    _t1 = _time.perf_counter()
    print(f"  [time_freq] diff+motion_mag: {_t1 - _t0:.3f}s")

    # We blur the 1-channel magnitude frames directly (much faster)
    blurred_motion = np.array(
        [cv2.GaussianBlur(frame, (11, 11), 0) for frame in motion_mag]
    )
    _t2 = _time.perf_counter()
    print(f"  [time_freq] gaussian blur ({len(motion_mag)} frames): {_t2 - _t1:.3f}s")

    # In RGB, motion > thresh across any channel meant motion > thresh.
    # Here motion is the sum, so threshold roughly scales with channels
    any_motion = blurred_motion > (thresh)
    _t3 = _time.perf_counter()
    print(f"  [time_freq] threshold: {_t3 - _t2:.3f}s")

    frequency = (
        np.sum(any_motion, axis=0) / n_frames
    )  # Normalize by the number of frames
    time_of_last_motion = np.zeros_like(frequency)
    for i in range(any_motion.shape[0]):
        motion = any_motion[i]
        time_of_last_motion[motion] = i / n_frames  # Normalize by the number of frames
    _t4 = _time.perf_counter()
    print(f"  [time_freq] frequency + time_of_last_motion loop: {_t4 - _t3:.3f}s")

    alpha = np.any(any_motion, axis=0)

    hue = 2.0 / 3.0 + time_of_last_motion / 3.0  # Map frequency to hue (blue to red)
    saturation = np.clip(1 - frequency * 2, 0, 1)  # Map frequency to saturation
    value = np.ones_like(frequency)  # Full brightness

    hsv_image = np.stack([hue * 360.0, saturation, value], axis=-1).astype(np.float32)
    rgb_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB) * 255.0
    rgb_image[~alpha] = 0  # Set non-motion areas to black
    _t5 = _time.perf_counter()
    print(f"  [time_freq] HSV->RGB + compose: {_t5 - _t4:.3f}s")

    return rgb_image.astype(np.uint8)


def label_image(image, text=None, color=None, border_thickness=None):
    if text is not None:
        # choose bold font
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 3
        font_thickness = 4
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_x = 10
        text_y = text_size[1] + 10
        cv2.putText(
            image,
            text,
            (text_x, text_y),
            font,
            font_scale,
            color if color is not None else (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )
    if color is not None and border_thickness is not None:
        image = cv2.copyMakeBorder(
            image,
            border_thickness,
            border_thickness,
            border_thickness,
            border_thickness,
            cv2.BORDER_CONSTANT,
            value=color,
        )
    return image
