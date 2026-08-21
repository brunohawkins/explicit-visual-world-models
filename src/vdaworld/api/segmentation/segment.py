import logging
from contextlib import nullcontext

import numpy as np
import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model, build_sam3_video_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.sam3_video_predictor import Sam3VideoPredictor

logger = logging.getLogger(__name__)

# Distinct colors for up to 16 tracked objects (RGB).
_PALETTE = np.array([
    [255,   0,   0], [  0, 255,   0], [  0,   0, 255], [255, 255,   0],
    [255,   0, 255], [  0, 255, 255], [255, 128,   0], [128,   0, 255],
    [  0, 128, 255], [255,   0, 128], [  0, 255, 128], [128, 255,   0],
    [128, 128,   0], [  0, 128, 128], [128,   0, 128], [255, 128, 128],
], dtype=np.uint8)


_model = None
_processor = None


def _get_processor() -> Sam3Processor:
    global _model, _processor
    if _processor is None:
        _model = build_sam3_image_model()
        _model.eval()
        _processor = Sam3Processor(_model)
    return _processor


def _autocast_context():
    if torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def segment(image: Image.Image, object_prompt: str) -> np.ndarray:
    """
    Core logic leveraging SAM3 from isolated conda environment.
    """
    processor = _get_processor()
    with torch.inference_mode(), _autocast_context():
        inference_state = processor.set_image(image)
        output = processor.set_text_prompt(state=inference_state, prompt=object_prompt)

    masks = output["masks"]
    return masks.squeeze(1).detach().cpu().numpy()


_CANDIDATE_PROMPTS = ["object", "thing", "item", "block"]


def segment_video(video_path: str, object_prompt: str) -> list:
    """
    Segment all objects matching object_prompt across a video using SAM3 video tracking.

    Tries object_prompt first, then falls back through _CANDIDATE_PROMPTS if nothing
    is detected. Lowers score_threshold_detection to 0.1 to handle simulation frames
    that score below the default 0.5 threshold.

    Returns:
        List of (H, W, 3) uint8 RGB frames where each tracked object is painted
        with a solid, distinct colour against a black background.
    """
    predictor = Sam3VideoPredictor()
    # Lower the detection threshold — simulation renders score below the default 0.5.
    predictor.model.score_threshold_detection = 0.1

    prompts_to_try = [object_prompt] + [p for p in _CANDIDATE_PROMPTS if p != object_prompt]

    for prompt in prompts_to_try:
        frames = _try_segment_video(predictor, video_path, prompt)
        if frames:
            logger.info("segment_video: prompt %r succeeded with %d frames", prompt, len(frames))
            return frames
        logger.warning("segment_video: prompt %r found nothing, trying next", prompt)

    return []


def _try_segment_video(predictor: Sam3VideoPredictor, video_path: str, object_prompt: str) -> list:
    session_result = predictor.start_session(resource_path=video_path)
    session_id = session_result["session_id"]

    try:
        prompt_result = predictor.add_prompt(session_id=session_id, frame_idx=0, text=object_prompt)
        prompt_outputs = prompt_result.get("outputs") or {}
        for k, v in prompt_outputs.items():
            if hasattr(v, "shape"):
                if v.size > 0:
                    logger.info("  add_prompt[%s]: shape=%s min=%.3f max=%.3f",
                                k, v.shape, float(v.min()), float(v.max()))
                else:
                    logger.warning("  add_prompt[%s]: shape=%s (empty)", k, v.shape)

        frame_outputs: dict[int, dict] = {}
        for item in predictor.propagate_in_video(
            session_id=session_id,
            propagation_direction="forward",
            start_frame_idx=None,
            max_frame_num_to_track=None,
        ):
            frame_idx = item["frame_index"]
            outputs = item["outputs"]
            if outputs is not None:
                frame_outputs[frame_idx] = outputs
    finally:
        predictor.close_session(session_id)

    logger.info("propagation(%r): %d frames with outputs", object_prompt, len(frame_outputs))
    if not frame_outputs:
        return []

    first = next(iter(frame_outputs.values()))
    for k, v in first.items():
        if hasattr(v, "shape") and v.size > 0:
            logger.info("  first frame[%s]: shape=%s min=%.3f max=%.3f",
                        k, v.shape, float(v.min()), float(v.max()))

    _, H, W = first["out_binary_masks"].shape
    all_obj_ids = sorted({
        int(oid)
        for out in frame_outputs.values()
        for oid in out["out_obj_ids"]
    })
    logger.info("tracked obj_ids: %s", all_obj_ids)
    if not all_obj_ids:
        return []

    color_map = {oid: _PALETTE[i % len(_PALETTE)] for i, oid in enumerate(all_obj_ids)}

    frames = []
    for frame_idx in sorted(frame_outputs.keys()):
        out = frame_outputs[frame_idx]
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        for obj_id, mask in zip(out["out_obj_ids"], out["out_binary_masks"]):
            canvas[mask > 0] = color_map[int(obj_id)]
        frames.append(canvas)

    return frames
