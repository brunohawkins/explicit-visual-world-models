import os
import sys
import torch
import numpy as np

# Force VGGT to exist
vggt_path = os.environ.get("VGGT_PATH")
if vggt_path not in sys.path:
    sys.path.insert(0, vggt_path)
os.environ["VGGT_PATH"] = vggt_path

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16

_MODEL_CACHE = None


def get_model():
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        import logging

        logging.info("Loading VGGT model from pretrained weights into GPU RAM...")
        _MODEL_CACHE = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
        _MODEL_CACHE.eval()
    return _MODEL_CACHE


def estimate(temp_dir: str):
    """
    Core logic leveraging VGGT from isolated conda environment.
    Reads image_000.png from temp_dir, runs inference, returns dictionary with pts3d and intrinsics.
    """
    model = get_model()

    image_names = [
        os.path.join(temp_dir, path)
        for path in os.listdir(temp_dir)
        if path.endswith((".png", ".jpg"))
    ]
    image_names.sort()

    images = load_and_preprocess_images(image_names).to(device)

    # Get original image size for resizing the output back
    import cv2

    orig_img = cv2.imread(image_names[0])
    orig_h, orig_w = orig_img.shape[:2]

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:]
    )
    extrinsic[0, :] = torch.eye(4)[None, :3].to(device)

    # Use first depth slice like in Landgen
    depth_slice = predictions["depth"].squeeze(0)
    ext_slice = extrinsic.squeeze(0)
    int_slice = intrinsic.squeeze(0)

    point_map_by_unprojection = unproject_depth_map_to_point_map(
        depth_slice, ext_slice, int_slice
    )
    point_map_by_unprojection[:, :, :, 1:] *= -1

    # Extract mean distance natively in PyTorch to avoid CPU roundtrips early
    xyz = point_map_by_unprojection.reshape((-1, 3))

    # We must ensure `xyz` is a torch Tensor, but `unproject_depth_map_to_point_map`
    # from vggt.utils.geometry actually might be returning a numpy array implicitly in the backend.
    if isinstance(xyz, np.ndarray):
        mean_dist = np.mean(np.linalg.norm(xyz, axis=-1))
        # Keep things numpy since it was apparently converted downstream
        pts3d = point_map_by_unprojection / mean_dist
        intrinsics_np = intrinsic[0, 0].detach().cpu().numpy()
    else:
        mean_dist = torch.mean(torch.linalg.norm(xyz, dim=-1))
        pts3d = (point_map_by_unprojection / mean_dist).detach().cpu().numpy()
        intrinsics_np = intrinsic[0, 0].detach().cpu().numpy()

    # Scale the unprojected coordinates and intrinsics back to original image size
    # VGGT image size is found in the points themselves

    if pts3d.ndim == 4 and pts3d.shape[0] == 1:
        pts3d = pts3d[0]

    curr_h, curr_w = pts3d.shape[:2]

    # Scale intrinsics
    scale_x = orig_w / curr_w
    scale_y = orig_h / curr_h
    intrinsics_np[0, 0] *= scale_x
    intrinsics_np[1, 1] *= scale_y
    intrinsics_np[0, 2] *= scale_x
    intrinsics_np[1, 2] *= scale_y

    # Resize points using cv2
    pts3d = pts3d.astype(np.float32)
    pts3d = cv2.resize(pts3d, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    return {
        "pts3d": pts3d,
        "intrinsics": intrinsics_np,
    }
