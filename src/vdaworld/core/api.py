import numpy as np
import sys
import logging
import io
import os
import requests
from PIL import Image, ImageDraw, ImageFont
import trimesh

from vdaworld.utils.cache import DiskCache, _arr_bytes, _hash_inputs
from vdaworld.utils.error_handling import APIInternalError

logger = logging.getLogger(__name__)


def _service_url(env_var: str, default_port: int) -> str:
    """Microservice base URL, overridable via env for multi-job port isolation."""
    url = os.environ.get(env_var)
    if url:
        return url.rstrip("/")
    return f"http://127.0.0.1:{default_port}"


def _rasterize_template_polygons(
    template: list[np.ndarray],
    cx: float,
    cy: float,
    theta: float,
    size: tuple[int, int],
) -> np.ndarray:
    """Rasterize caller-supplied template polygons at ``(cx, cy, theta)``."""
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

    img = Image.new("1", size, 0)
    draw = ImageDraw.Draw(img)
    offset = np.array([cx, cy], dtype=float)
    for poly in template:
        local = np.asarray(poly, dtype=float)
        world = local @ rot.T + offset
        draw.polygon([tuple(pt) for pt in world], fill=1)
    return np.array(img, dtype=bool)


def _as_vec2(name: str, value) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (2,):
        raise ValueError(f"{name} must have shape (2,), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values")
    return arr


def _as_scalar(name: str, value, *, nonnegative: bool = False) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be a finite scalar")
    if nonnegative and scalar < 0:
        raise ValueError(f"{name} must be non-negative")
    return scalar


def _draw_segment_overlay(image: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """
    Draw a colored semi-transparent fill, solid border, and centered index number
    for each mask onto a copy of *image*. Returns the overlay as an np.ndarray.
    """
    import matplotlib.pyplot as plt
    from scipy.ndimage import binary_dilation

    overlay = image.copy()
    colors = plt.get_cmap("hsv", max(masks.shape[0], 1))
    pil_overlay = Image.fromarray(overlay)
    draw = ImageDraw.Draw(pil_overlay)

    for i, mask in enumerate(masks):
        color = np.array(colors(i)[:3]) * 255
        color_u8 = tuple(int(c) for c in color)

        # Semi-transparent fill
        overlay[mask] = (overlay[mask] * 0.5 + color * 0.5).astype(np.uint8)

        # Solid border: dilate the mask and draw only the ring pixels
        border = binary_dilation(mask, iterations=2) & ~mask
        overlay[border] = color_u8

        # Centered index label
        ys, xs = np.where(mask)
        if len(xs) > 0:
            cx, cy = int(xs.mean()), int(ys.mean())
            label = str(i)
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20
                )
            except Exception:
                font = ImageFont.load_default()
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text(
                (cx - tw // 2, cy - th // 2), label, fill=(255, 255, 255), font=font
            )

    # Merge the PIL draw layer (text/border) back with overlay numpy array
    overlay = np.array(pil_overlay)
    # Re-apply border pixels from overlay (PIL draw overwrites the numpy border)
    for i, mask in enumerate(masks):
        color = np.array(colors(i)[:3]) * 255
        color_u8 = tuple(int(c) for c in color)
        border = binary_dilation(mask, iterations=2) & ~mask
        overlay[border] = color_u8

    return overlay


class WorldAPI:
    """
    Simulation API containing various computer vision, 3D geometry, and physics utility methods.
    """

    def __init__(
        self,
        output_dir: str = None,
        cache_dir: str | None = None,
        api_calls_dir: str | None = None,
    ):
        self.assets = {}
        self.output_dir = output_dir
        self.api_call_idx = 0
        if api_calls_dir:
            self.api_calls_dir = api_calls_dir
            os.makedirs(api_calls_dir, exist_ok=True)
        elif output_dir:
            self.api_calls_dir = os.path.join(output_dir, "api_calls")
            os.makedirs(self.api_calls_dir, exist_ok=True)
        else:
            self.api_calls_dir = None
        self._cache: DiskCache | None = DiskCache(cache_dir) if cache_dir else None

    def _set_stage_dir(self, stage_dir: str | None) -> None:
        """Redirect api-call logging to *stage_dir*/api_calls/ and reset the call index."""
        if stage_dir:
            self.api_calls_dir = os.path.join(stage_dir, "api_calls")
            os.makedirs(self.api_calls_dir, exist_ok=True)
        else:
            self.api_calls_dir = None
        self.api_call_idx = 0

    def _log_api_call(
        self,
        tool_name: str,
        image: np.ndarray | None = None,
        result_data: dict | None = None,
        overlay_image=None,
        extra_files: dict[str, bytes] | None = None,
        file_bundle: dict[str, bytes] | None = None,
    ) -> tuple[str | None, dict[str, bytes]]:
        """Save per-call artefacts to a numbered subdirectory.

        Pass ``file_bundle`` on a cache hit to replay a previously-saved set of
        files instead of regenerating them.  Pass ``extra_files`` on a fresh
        computation to include pre-built binary artefacts (e.g. .glb, .obj).

        Returns ``(folder_path, bundle)`` where *bundle* maps every filename
        written to its raw bytes — callers should persist this to the cache so
        it can be replayed on future cache hits.  Returns ``(None, {})`` when
        logging is disabled.
        """
        if not self.api_calls_dir:
            self.api_call_idx += 1
            return None, {}

        import json
        from PIL import Image as _PILImage

        folder_name = f"{self.api_call_idx}_{tool_name}"
        folder_path = os.path.join(self.api_calls_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        self.api_call_idx += 1

        if file_bundle is not None:
            # Cache-hit path: replay pre-built files verbatim.
            for fname, data in file_bundle.items():
                with open(os.path.join(folder_path, fname), "wb") as fh:
                    fh.write(data)
            return folder_path, file_bundle

        # Fresh-computation path: build files from inputs.
        bundle: dict[str, bytes] = {}

        if overlay_image is not None:
            buf = io.BytesIO()
            _PILImage.fromarray(overlay_image).save(buf, format="PNG")
            bundle["overlay.png"] = buf.getvalue()
        elif image is not None:
            buf = io.BytesIO()
            _PILImage.fromarray(image).save(buf, format="PNG")
            bundle["input.png"] = buf.getvalue()

        if result_data is not None:
            bundle["metadata.json"] = json.dumps(result_data, indent=4).encode()

        for fname, data in (extra_files or {}).items():
            bundle[fname] = data

        for fname, data in bundle.items():
            with open(os.path.join(folder_path, fname), "wb") as fh:
                fh.write(data)

        return folder_path, bundle

    def estimate_intrinsics(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate intrinsic parameters of camera for image.

        Arguments:
            image (np.ndarray)
                Image of shape `[h, w, 3]`. Dtype of `np.uint8`

        Returns:
            intrinsics (np.ndarray)
                Matrix of camera intrinsics. [[fx, 0, cx],[0, fy, cy], [0, 0, 1]]. All parameters given in pixels.
        """
        _cache_key = _hash_inputs(_arr_bytes(image)) if self._cache else None
        if self._cache and _cache_key:
            _cached = self._cache.get_pickle("estimate_3d_points", _cache_key)
            if _cached is not None:
                _, intrinsics_matrix = _cached
                self.assets["last_intrinsics"] = intrinsics_matrix
                logger.info("Cache hit: estimate_intrinsics()")
                if self.api_calls_dir:
                    _bundle = self._cache.get_log_bundle("estimate_intrinsics", _cache_key)
                    if _bundle:
                        self._log_api_call(tool_name="intrinsics", file_bundle=_bundle)
                    else:
                        _, _bundle = self._log_api_call(
                            tool_name="intrinsics",
                            image=image,
                            result_data={"intrinsics": intrinsics_matrix.tolist(), "cache_hit": True},
                        )
                        self._cache.set_log_bundle("estimate_intrinsics", _cache_key, _bundle)
                return intrinsics_matrix

        if "last_intrinsics" not in self.assets:
            self.estimate_3d_points(image)

        intrinsics_matrix = self.assets["last_intrinsics"]

        result_data = {"intrinsics": intrinsics_matrix.tolist()}
        if self.api_calls_dir:
            _, _bundle = self._log_api_call(
                tool_name="intrinsics",
                image=image,
                result_data=result_data,
            )
            if self._cache and _cache_key:
                self._cache.set_log_bundle("estimate_intrinsics", _cache_key, _bundle)
        return intrinsics_matrix

    def segment(self, image: np.ndarray, text_prompt: str) -> np.ndarray:
        """
        An open-vocabulary segmentation method for all instances of a single object in an image.
        Best practice is to use this method to segment all instances of a broad category of object (e.g. "block" or "bottle") rather than a specific object (e.g. "red block") since the model may not be able to differentiate between specific prompts.

        Arguments:
            image (np.ndarray)
                Image of shape `[h, w, 3]`. Dtype of `np.uint8`
            text_prompt (str)
                Name of object to be segmented.

        Returns:
            segmentation (np.ndarray)
                `np.ndarray` of dtype `bool` corresponding to all instances of an object found in the image.
                Array has shape of `[N, h, w]`, representing the binary segmentation mask for any `N` instances of that single object.
                Items are ordered by probability of being a relevant object.
                If no objects are found, raises an AssertionError.
        """
        import requests
        import io
        from PIL import Image
        import sys

        # 1. Agent Input Validation
        try:
            if not isinstance(image, np.ndarray):
                raise TypeError(
                    f"`image` must be a numpy.ndarray, got {type(image).__name__}"
                )
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(
                    f"`image` must have shape [h, w, 3], got {image.shape}"
                )
            if image.dtype != np.uint8:
                raise ValueError(f"`image` must have dtype uint8, got {image.dtype}")
            if not isinstance(text_prompt, str):
                raise TypeError(
                    f"`text_prompt` must be a string, got {type(text_prompt).__name__}"
                )
        except Exception as agent_err:
            print(f"[!] Agent Tool Error in segment(): {agent_err}", file=sys.stderr)
            logging.error(f"Agent Tool Error in segment(): {agent_err}")
            if hasattr(self, "api_calls_dir") and self.api_calls_dir:
                self._log_api_call(
                    tool_name="segment",
                    image=(
                        image
                        if isinstance(image, np.ndarray)
                        else np.zeros((1, 1, 3), dtype=np.uint8)
                    ),
                    result_data={
                        "error": True,
                        "prompt": (
                            str(text_prompt) if "text_prompt" in locals() else "unknown"
                        ),
                        "error_type": "Agent Input Validation",
                        "message": str(agent_err),
                    },
                )
            raise agent_err

        # 2. Internal Execution
        _cache_key = (
            _hash_inputs(_arr_bytes(image), text_prompt.encode())
            if self._cache
            else None
        )
        if self._cache and _cache_key:
            _cached = self._cache.get_pickle("segment", _cache_key)
            if _cached is not None:
                logger.info("Cache hit: segment(%r)", text_prompt)
                masks = _cached
                if self.api_calls_dir:
                    _bundle = self._cache.get_log_bundle("segment", _cache_key)
                    if _bundle:
                        self._log_api_call(tool_name="segment", file_bundle=_bundle)
                    else:
                        _, _bundle = self._log_api_call(
                            tool_name="segment",
                            image=image,
                            result_data={
                                "prompt": text_prompt,
                                "number_of_masks": int(masks.shape[0]),
                                "shape": list(masks.shape),
                                "cache_hit": True,
                            },
                            overlay_image=_draw_segment_overlay(image, masks),
                        )
                        self._cache.set_log_bundle("segment", _cache_key, _bundle)
                return masks

        try:
            pil_img = Image.fromarray(image)
            img_buffer = io.BytesIO()
            pil_img.save(img_buffer, format="PNG")
            img_buffer.seek(0)

            response = requests.post(
                _service_url("VDA_SEGMENT_URL", 5001) + "/segment",
                files={"image": ("image.png", img_buffer, "image/png")},
                data={"object_prompt": text_prompt},
            )

            if response.status_code != 200:
                raise APIInternalError(
                    f"Segmentation service returned status {response.status_code}: {response.text}"
                )

            masks = np.load(io.BytesIO(response.content))

            # Empty result is an expected condition the VLM should handle.
            if masks.shape[0] == 0:
                raise AssertionError(f"No instances found for '{text_prompt}'")

        except (AssertionError, APIInternalError):
            raise
        except Exception as internal_err:
            logging.error("Internal segment backend error: %s", internal_err)
            if hasattr(self, "api_calls_dir") and self.api_calls_dir:
                self._log_api_call(
                    tool_name="segment",
                    image=image,
                    result_data={
                        "error": True,
                        "prompt": text_prompt,
                        "error_type": "Internal System Error",
                        "message": str(internal_err),
                    },
                )
            raise APIInternalError(
                f"Internal error in segment(): {internal_err}"
            ) from internal_err

        if self._cache and _cache_key:
            self._cache.set_pickle("segment", _cache_key, masks)

        if self.api_calls_dir:
            _, _bundle = self._log_api_call(
                tool_name="segment",
                image=image,
                result_data={
                    "prompt": text_prompt,
                    "number_of_masks": int(masks.shape[0]),
                    "shape": list(masks.shape),
                },
                overlay_image=_draw_segment_overlay(image, masks),
            )
            if self._cache and _cache_key:
                self._cache.set_log_bundle("segment", _cache_key, _bundle)

        return masks

    def estimate_3d_points(self, image: np.ndarray) -> np.ndarray:
        """
        An estimator of the 3d location of each pixel in the coordinate system of the camera for the image.
        Uses a robust pre-trained depth estimator to back-project depth maps into unified 3D point layouts.

        Arguments:
            image (np.ndarray)
                Image of shape `[h, w, 3]`. Dtype of `np.uint8`

        Returns:
            pts3d (np.ndarray)
                Shape `[h, w, 3]`. Dtype of `np.float32`.
                3D point map in the OpenGL camera coordinate system:
                +X right, +Y up, -Z forward.
                Values are in metres.
        """
        import io
        import sys

        from PIL import Image

        # 1. Agent Input Validation
        try:
            if not isinstance(image, np.ndarray):
                raise TypeError(
                    f"`image` must be a numpy.ndarray, got {type(image).__name__}"
                )
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(
                    f"`image` must have shape [h, w, 3], got {image.shape}"
                )
            if image.dtype != np.uint8:
                raise ValueError(f"`image` must have dtype uint8, got {image.dtype}")
        except Exception as agent_err:
            print(
                f"[!] Agent Tool Error in estimate_3d_points(): {agent_err}",
                file=sys.stderr,
            )
            logging.error(f"Agent Tool Error in estimate_3d_points(): {agent_err}")
            if hasattr(self, "api_calls_dir") and self.api_calls_dir:
                self._log_api_call(
                    tool_name="estimate_3d_points",
                    image=(
                        image
                        if isinstance(image, np.ndarray)
                        else np.zeros((1, 1, 3), dtype=np.uint8)
                    ),
                    result_data={
                        "error": True,
                        "error_type": "Agent Input Validation",
                        "message": str(agent_err),
                    },
                )
            raise agent_err

        # 2. Internal Execution
        # ----------------------------------------------------------------
        # Backend toggle — comment/uncomment exactly one line:
        # _backend = self._depth_backend_vggt   # VGGT: relative depth, mean-normalised
        _backend = self._depth_backend_moge  # MoGe: metric (v2) or mean-normalised (v1)
        # ----------------------------------------------------------------
        _cache_key = _hash_inputs(_arr_bytes(image)) if self._cache else None
        if self._cache and _cache_key:
            _cached = self._cache.get_pickle("estimate_3d_points", _cache_key)
            if _cached is not None:
                logger.info("Cache hit: estimate_3d_points()")
                pts3d, intrinsics = _cached
                self.assets["last_intrinsics"] = intrinsics
                if self.api_calls_dir:
                    _bundle = self._cache.get_log_bundle("estimate_3d_points", _cache_key)
                    if _bundle:
                        self._log_api_call(tool_name="estimate_3d_points", file_bundle=_bundle)
                    else:
                        try:
                            import matplotlib.pyplot as plt
                            depth = -pts3d[..., 2]
                            depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
                            colored_depth = (plt.get_cmap("inferno")(depth_norm)[..., :3] * 255).astype(np.uint8)
                            overlay = (image * 0.5 + colored_depth * 0.5).astype(np.uint8)
                        except Exception:
                            overlay = image.copy()
                        _, _bundle = self._log_api_call(
                            tool_name="estimate_3d_points",
                            image=image,
                            result_data={
                                "shape": list(pts3d.shape),
                                "mean_z": float(np.mean(pts3d[..., 2])),
                                "intrinsics_saved": True,
                                "cache_hit": True,
                            },
                            overlay_image=overlay,
                            extra_files={"points.obj": _point_cloud_obj_bytes(pts3d)},
                        )
                        self._cache.set_log_bundle("estimate_3d_points", _cache_key, _bundle)
                return pts3d

        try:
            pts3d, intrinsics = _backend(image)
            self.assets["last_intrinsics"] = intrinsics

        except Exception as internal_err:
            logging.error("Internal depth backend error: %s", internal_err)
            if hasattr(self, "api_calls_dir") and self.api_calls_dir:
                self._log_api_call(
                    tool_name="estimate_3d_points",
                    image=image,
                    result_data={
                        "error": True,
                        "error_type": "Internal System Error",
                        "message": str(internal_err),
                    },
                )
            raise APIInternalError(
                f"Internal error in estimate_3d_points(): {internal_err}"
            ) from internal_err

        if self._cache and _cache_key:
            self._cache.set_pickle("estimate_3d_points", _cache_key, (pts3d, intrinsics))

        if self.api_calls_dir:
            try:
                import matplotlib.pyplot as plt
                depth = -pts3d[..., 2]
                depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
                colored_depth = (plt.get_cmap("inferno")(depth_norm)[..., :3] * 255).astype(np.uint8)
                overlay = (image * 0.5 + colored_depth * 0.5).astype(np.uint8)
            except Exception:
                overlay = image.copy()

            _, _bundle = self._log_api_call(
                tool_name="estimate_3d_points",
                image=image,
                result_data={
                    "shape": list(pts3d.shape),
                    "mean_z": float(np.mean(pts3d[..., 2])),
                    "intrinsics_saved": bool("last_intrinsics" in self.assets),
                },
                overlay_image=overlay,
                extra_files={"points.obj": _point_cloud_obj_bytes(pts3d)},
            )
            if self._cache and _cache_key:
                self._cache.set_log_bundle("estimate_3d_points", _cache_key, _bundle)

        return pts3d

    def _depth_backend_vggt(self, image: np.ndarray) -> tuple:
        """Call the VGGT microservice (port 5003) and return ``(pts3d, intrinsics)``."""
        import io
        import requests
        from PIL import Image as PILImage

        pil_img = PILImage.fromarray(image)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)

        response = requests.post(
            _service_url("VDA_DEPTH_VGGT_URL", 5003) + "/estimate_3d_points",
            files={"image": ("image.png", buf, "image/png")},
        )
        if response.status_code != 200:
            raise APIInternalError(
                f"VGGT backend failed with status {response.status_code}: {response.text}"
            )

        result = np.load(io.BytesIO(response.content))
        return result["pts3d"], result["intrinsics"]

    def _depth_backend_moge(self, image: np.ndarray) -> tuple:
        """Call the MoGe microservice (port 5004) and return ``(pts3d, intrinsics)``.

        The MoGe server converts OpenCV → OpenGL coordinates and handles
        normalisation internally; outputs are directly compatible with VGGT.
        See ``api/depth/moge_estimator.py`` for full details.
        """
        import io
        import requests
        from PIL import Image as PILImage

        pil_img = PILImage.fromarray(image)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)

        response = requests.post(
            _service_url("VDA_DEPTH_URL", 5004) + "/estimate_3d_points",
            files={"image": ("image.png", buf, "image/png")},
        )
        if response.status_code != 200:
            raise APIInternalError(
                f"MoGe backend failed with status {response.status_code}: {response.text}"
            )

        result = np.load(io.BytesIO(response.content))
        return result["pts3d"], result["intrinsics"]

    def fit_2d_primitive(self, points: np.ndarray, shape_class: str) -> dict:
        """
        Fits a 2D geometric primitive (rectangle or circle) to a set of 2D points.
        Leverages OpenCV's `minEnclosingCircle` and `minAreaRect` for robust fitting.

        Arguments:
            points (np.ndarray):
                An `[N, 2]` array of `[x, y]` points.
            shape_class (str):
                The type of shape to fit. Must be 'rectangle' or 'circle'.

        Returns:
            dict:
                A dictionary containing the fitted shape parameters:
                - For 'rectangle': {'center': [x, y], 'size': [width, height], 'angle': angle_in_degrees}
                - For 'circle': {'center': [x, y], 'radius': r}
        """
        import cv2

        if not isinstance(points, np.ndarray):
            raise TypeError("Input 'points' must be a numpy array.")
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Input 'points' must be an array of shape [N, 2].")
        if points.shape[0] < 3:
            raise ValueError("Cannot fit a shape to fewer than 3 points.")
        if not isinstance(shape_class, str) or shape_class not in [
            "rectangle",
            "circle",
        ]:
            raise ValueError("shape_class must be either 'rectangle' or 'circle'.")

        _cache_key = (
            _hash_inputs(_arr_bytes(points), shape_class.encode())
            if self._cache else None
        )
        if self._cache and _cache_key:
            _cached = self._cache.get_pickle("fit_2d_primitive", _cache_key)
            if _cached is not None:
                logger.info("Cache hit: fit_2d_primitive(%r)", shape_class)
                if self.api_calls_dir:
                    _bundle = self._cache.get_log_bundle("fit_2d_primitive", _cache_key)
                    if _bundle:
                        self._log_api_call(tool_name=f"fit_2d_{shape_class}", file_bundle=_bundle)
                    else:
                        _, _bundle = self._log_api_call(
                            tool_name=f"fit_2d_{shape_class}",
                            result_data={**_cached, "cache_hit": True},
                        )
                        self._cache.set_log_bundle("fit_2d_primitive", _cache_key, _bundle)
                return _cached

        points_cv = points.astype(np.float32)

        if shape_class == "circle":
            (x, y), radius = cv2.minEnclosingCircle(points_cv)
            result = {"center": [float(x), float(y)], "radius": float(radius)}
        elif shape_class == "rectangle":
            center, size, angle = cv2.minAreaRect(points_cv)
            result = {
                "center": [float(center[0]), float(center[1])],
                "size": [float(size[0]), float(size[1])],
                "angle": float(angle),
            }
        else:
            raise ValueError(f"Unknown shape class {shape_class}")

        if self._cache and _cache_key:
            self._cache.set_pickle("fit_2d_primitive", _cache_key, result)

        if self.api_calls_dir:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            ax.scatter(points[:, 0], points[:, 1], c="blue", s=10, label="points")

            if shape_class == "circle":
                ax.add_patch(plt.Circle(
                    result["center"], result["radius"],
                    color="r", fill=False, linewidth=2, label="fitted circle",
                ))
            elif shape_class == "rectangle":
                rect = cv2.boxPoints((
                    (result["center"][0], result["center"][1]),
                    (result["size"][0], result["size"][1]),
                    result["angle"],
                ))
                rect = np.vstack((rect, rect[0]))
                ax.plot(rect[:, 0], rect[:, 1], color="r", linewidth=2, label="fitted rectangle")

            ax.set_aspect("equal")
            ax.legend()
            fig_buf = io.BytesIO()
            fig.savefig(fig_buf, format="png")
            plt.close(fig)

            _, _bundle = self._log_api_call(
                tool_name=f"fit_2d_{shape_class}",
                result_data=result,
                extra_files={"overlay.png": fig_buf.getvalue()},
            )
            if self._cache and _cache_key:
                self._cache.set_log_bundle("fit_2d_primitive", _cache_key, _bundle)

        return result

    def fit_pose_to_mask(
        self,
        mask: np.ndarray,
        template: list[np.ndarray],
        n_angles: int = 180,
    ) -> dict:
        """
        Recover a known shape's pose from a binary segmentation mask.

        Given a mask and caller-supplied template polygons in a local frame,
        finds the encoded body origin ``(cx, cy)`` and orientation ``theta``
        that maximise template-vs-mask IoU over a full 360° angle sweep plus
        local translation refinement. Templates need not be centroid-centred.

        ``theta`` is applied as a counter-clockwise rotation to the template's
        local vertices before placing the centroid at ``(cx, cy)``. The caller
        must render and interpret poses in the same convention.

        Arguments:
            mask (np.ndarray):
                ``(H, W)`` boolean or 0/1 array — the object's segmentation mask.
            template (list[np.ndarray]):
                List of ``(N, 2)`` float arrays — polygon vertices in a local
                frame centred at the template origin, in pixel units.
            n_angles (int):
                Number of angles sampled uniformly in ``[0, 2π)``.

        Returns:
            dict:
                Pose fields ``cx``, ``cy``, ``theta``, overlap fields ``iou``
                and ``dice``, ``mask_area``, plus ``valid`` and ``confidence``.
                ``theta`` is in radians (CCW). ``confidence`` is the fitted IoU.
                An empty mask preserves the legacy centre/pose/IoU fallback
                fields but returns ``valid=False`` and ``confidence=0.0``;
                callers must check validity rather than silently using the
                frame centre as an observation.
        """
        from vdaworld.core.simulator import ActionConditionedSimulatorBase

        array = np.asarray(mask)
        if array.ndim != 2:
            raise ValueError(
                f"mask must be a 2D boolean or 0/1 array, got shape {array.shape}"
            )
        simulator = ActionConditionedSimulatorBase(
            frame_size=(int(array.shape[1]), int(array.shape[0])),
            api=None,
        )
        return simulator.fit_pose_to_mask(array, template, n_angles=n_angles)

    def largest_connected_component(
        self,
        mask: np.ndarray,
        min_area: int = 1,
        connectivity: int = 8,
    ) -> np.ndarray:
        """Return the largest connected foreground region in ``mask``."""
        from vdaworld.core.simulator import ActionConditionedSimulatorBase

        return ActionConditionedSimulatorBase.largest_connected_component(
            mask,
            min_area=min_area,
            connectivity=connectivity,
        )

    def clamp_step(
        self,
        prev_pos: np.ndarray,
        target_pos: np.ndarray,
        max_step: float,
    ) -> np.ndarray:
        """
        Clamp a target position to a maximum displacement from a previous position.

        Arguments:
            prev_pos (np.ndarray):
                ``(2,)`` float array — starting position.
            target_pos (np.ndarray):
                ``(2,)`` float array — desired next position.
            max_step (float):
                Non-negative scalar — largest allowed Euclidean displacement.

        Returns:
            np.ndarray:
                ``(2,)`` float array. If ``target_pos`` is within ``max_step``
                of ``prev_pos`` it is returned unchanged; otherwise the result
                lies on the line from ``prev_pos`` to ``target_pos`` exactly
                ``max_step`` away from ``prev_pos``. A zero displacement returns
                ``target_pos`` without raising.
        """
        prev = _as_vec2("prev_pos", prev_pos)
        target = _as_vec2("target_pos", target_pos)
        max_step = _as_scalar("max_step", max_step, nonnegative=True)

        delta = target - prev
        dist = float(np.linalg.norm(delta))
        if dist == 0.0 or dist <= max_step:
            return target.copy()
        return prev + delta * (max_step / dist)

    def resolve_contact(
        self,
        body_points_world: np.ndarray,
        body_center: np.ndarray,
        pusher_pos: np.ndarray,
        pusher_radius: float,
        k_trans: float,
        k_rot: float,
    ) -> dict:
        """
        Resolve 2-D point-sampled body penetration against a circular pusher.

        For every supplied body point inside the pusher radius, this computes
        the penetration-depth correction pointing away from the pusher. The
        mean correction contributes translation, and the mean moment of those
        corrections about ``body_center`` contributes rotation.

        Arguments:
            body_points_world (np.ndarray):
                ``(M, 2)`` float array — sampled body points in world
                coordinates. ``M=0`` returns zero motion without raising.
            body_center (np.ndarray):
                ``(2,)`` float array — centre used for the rotational moment.
            pusher_pos (np.ndarray):
                ``(2,)`` float array — centre of the circular pusher.
            pusher_radius (float):
                Non-negative scalar — pusher radius in the same units as the
                points.
            k_trans (float):
                Finite scalar — multiplier applied to the mean penetration
                correction.
            k_rot (float):
                Finite scalar — multiplier applied to the normalized mean
                rotational moment.

        Returns:
            dict:
                ``{"d_pos": np.ndarray, "d_theta": float}`` where ``d_pos``
                has shape ``(2,)`` and ``d_theta`` is in radians. If no supplied
                point penetrates the pusher, both outputs are zero.
        """
        points = np.asarray(body_points_world, dtype=float)
        if points.size == 0:
            points = np.empty((0, 2), dtype=float)
        elif points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(
                f"body_points_world must have shape (M, 2), got {points.shape}"
            )
        if not np.all(np.isfinite(points)):
            raise ValueError("body_points_world must contain finite values")

        center = _as_vec2("body_center", body_center)
        pusher = _as_vec2("pusher_pos", pusher_pos)
        radius = _as_scalar("pusher_radius", pusher_radius, nonnegative=True)
        k_trans = _as_scalar("k_trans", k_trans)
        k_rot = _as_scalar("k_rot", k_rot)

        zero = {"d_pos": np.zeros(2, dtype=float), "d_theta": 0.0}
        if points.shape[0] == 0 or radius == 0.0:
            return zero

        vecs = points - pusher
        dists = np.linalg.norm(vecs, axis=1)
        penetrating = dists < radius
        if not np.any(penetrating):
            return zero

        contact_points = points[penetrating]
        contact_vecs = vecs[penetrating]
        contact_dists = dists[penetrating]

        eps = np.finfo(float).eps
        normals = np.zeros_like(contact_vecs)
        valid = contact_dists > eps
        normals[valid] = contact_vecs[valid] / contact_dists[valid, None]
        if np.any(~valid):
            fallback = center - pusher
            fallback_norm = float(np.linalg.norm(fallback))
            if fallback_norm <= eps:
                fallback = np.array([1.0, 0.0], dtype=float)
            else:
                fallback = fallback / fallback_norm
            normals[~valid] = fallback

        depths = radius - contact_dists
        corrections = normals * depths[:, None]
        mean_correction = corrections.mean(axis=0)
        d_pos = k_trans * mean_correction

        offsets = contact_points - center
        moments = offsets[:, 0] * corrections[:, 1] - offsets[:, 1] * corrections[:, 0]
        moment_scale = float(np.mean(np.sum(offsets * offsets, axis=1)))
        if moment_scale <= eps:
            d_theta = 0.0
        else:
            d_theta = float(k_rot * moments.mean() / moment_scale)

        return {"d_pos": np.asarray(d_pos, dtype=float), "d_theta": d_theta}

    def forward_kinematics(
        self,
        base_pos: np.ndarray,
        angles: np.ndarray,
        link_lengths: np.ndarray,
    ) -> dict:
        """
        Compute planar serial-chain forward kinematics for any link count.

        Angles are interpreted cumulatively: each link angle is relative to the
        previous link's frame, and the first link is relative to the positive
        x-axis.

        Arguments:
            base_pos (np.ndarray):
                ``(2,)`` float array — base position of the chain.
            angles (np.ndarray):
                ``(N,)`` float array — relative joint angles in radians.
            link_lengths (np.ndarray):
                ``(N,)`` float array — non-negative link lengths.

        Returns:
            dict:
                ``{"joints": np.ndarray, "end_effector": np.ndarray}`` where
                ``joints`` has shape ``(N + 1, 2)`` and includes ``base_pos`` as
                the first row, and ``end_effector`` has shape ``(2,)``.
                ``N=0`` returns the base as the only joint and end effector.
        """
        base = _as_vec2("base_pos", base_pos)
        angles = np.asarray(angles, dtype=float)
        lengths = np.asarray(link_lengths, dtype=float)
        if angles.ndim != 1:
            raise ValueError(f"angles must have shape (N,), got {angles.shape}")
        if lengths.ndim != 1:
            raise ValueError(
                f"link_lengths must have shape (N,), got {lengths.shape}"
            )
        if angles.shape[0] != lengths.shape[0]:
            raise ValueError(
                "angles and link_lengths must have the same length, "
                f"got {angles.shape[0]} and {lengths.shape[0]}"
            )
        if not np.all(np.isfinite(angles)):
            raise ValueError("angles must contain finite values")
        if not np.all(np.isfinite(lengths)):
            raise ValueError("link_lengths must contain finite values")
        if np.any(lengths < 0):
            raise ValueError("link_lengths must be non-negative")

        joints = np.empty((angles.shape[0] + 1, 2), dtype=float)
        joints[0] = base
        heading = 0.0
        for idx, (angle, length) in enumerate(zip(angles, lengths), start=1):
            heading += float(angle)
            joints[idx] = joints[idx - 1] + length * np.array(
                [np.cos(heading), np.sin(heading)], dtype=float
            )

        return {"joints": joints, "end_effector": joints[-1].copy()}

    def collide_occupancy(
        self,
        pos: np.ndarray,
        step: np.ndarray,
        occupancy_mask: np.ndarray,
        radius: float,
    ) -> np.ndarray:
        """
        Move a circular footprint through a boolean occupancy grid with sliding.

        The method applies the requested displacement one axis at a time. Along
        each axis it sweeps in small grid-unit increments and accepts motion up
        to the first colliding candidate, which prevents thin occupied cells
        from being skipped by a large step. Coordinates use ``(x, y)`` order;
        ``occupancy_mask`` is indexed as ``occupancy_mask[y, x]``.

        Arguments:
            pos (np.ndarray):
                ``(2,)`` float array — starting centre position ``(x, y)``.
            step (np.ndarray):
                ``(2,)`` float array — requested displacement ``(dx, dy)``.
            occupancy_mask (np.ndarray):
                ``(H, W)`` boolean or 0/1 array — true values are obstacles.
            radius (float):
                Non-negative scalar — footprint radius in grid-cell units.

        Returns:
            np.ndarray:
                ``(2,)`` float array — new centre position after axis-separated
                collision handling. If the starting position already collides,
                it is returned unchanged.
        """
        pos = _as_vec2("pos", pos)
        step = _as_vec2("step", step)
        mask = np.asarray(occupancy_mask).astype(bool)
        if mask.ndim != 2 or mask.shape[0] == 0 or mask.shape[1] == 0:
            raise ValueError(
                f"occupancy_mask must have non-empty shape (H, W), got {mask.shape}"
            )
        radius = _as_scalar("radius", radius, nonnegative=True)

        height, width = mask.shape

        def collides(center: np.ndarray) -> bool:
            x, y = float(center[0]), float(center[1])
            if (
                x - radius < -0.5
                or x + radius > width - 0.5
                or y - radius < -0.5
                or y + radius > height - 0.5
            ):
                return True

            x0 = max(0, int(np.floor(x - radius - 0.5)))
            x1 = min(width - 1, int(np.ceil(x + radius + 0.5)))
            y0 = max(0, int(np.floor(y - radius - 0.5)))
            y1 = min(height - 1, int(np.ceil(y + radius + 0.5)))
            ys, xs = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
            occupied = mask[y0 : y1 + 1, x0 : x1 + 1]
            if not np.any(occupied):
                return False

            dx = np.maximum(np.abs(xs.astype(float) - x) - 0.5, 0.0)
            dy = np.maximum(np.abs(ys.astype(float) - y) - 0.5, 0.0)
            return bool(np.any(occupied & (dx * dx + dy * dy <= radius * radius)))

        if collides(pos):
            return pos.copy()

        def move_axis(current: np.ndarray, axis: int, amount: float) -> np.ndarray:
            if amount == 0.0:
                return current
            n_steps = max(1, int(np.ceil(abs(amount) / 0.25)))
            delta = amount / n_steps
            moved = current.copy()
            for _ in range(n_steps):
                candidate = moved.copy()
                candidate[axis] += delta
                if collides(candidate):
                    break
                moved = candidate
            return moved

        new_pos = move_axis(pos.copy(), 0, float(step[0]))
        new_pos = move_axis(new_pos, 1, float(step[1]))
        return new_pos

    def predict_ground_plane(
        self, points: np.ndarray, num_iterations=100, distance_threshold=0.05
    ) -> tuple:
        """
        Predicts the ground plane from a point cloud.

        Arguments:
            points (np.ndarray)
                Point map of shape (N, 3) representing the point cloud.
            num_iterations (int)
                Number of RANSAC iterations.
            distance_threshold (float)
                Maximum distance to be considered an inlier.

        Returns:
            tuple of:
                - best_plane_model (tuple): A tuple (np.ndarray, np.ndarray) representing the best-fit plane, with the normal pointing up from the surface.
                                        `normal` is a numpy array of shape (3,).
                                        `point` is a numpy array on the plane of shape (3,).
                - best_inliers (np.ndarray): A boolean array of shape (N,) indicating the inlier points.
        """
        _cache_key = (
            _hash_inputs(
                _arr_bytes(points),
                str(num_iterations).encode(),
                str(distance_threshold).encode(),
            )
            if self._cache
            else None
        )
        if self._cache and _cache_key:
            _cached = self._cache.get_pickle("predict_ground_plane", _cache_key)
            if _cached is not None:
                logger.info("Cache hit: predict_ground_plane()")
                if self.api_calls_dir:
                    _bundle = self._cache.get_log_bundle("predict_ground_plane", _cache_key)
                    if _bundle:
                        self._log_api_call(tool_name="predict_ground_plane", file_bundle=_bundle)
                    else:
                        best_plane_model, full_inliers = _cached
                        _, _bundle = self._log_api_call(
                            tool_name="predict_ground_plane",
                            result_data={
                                "normal": [float(n) for n in best_plane_model[0]],
                                "point": [float(p) for p in best_plane_model[1]],
                                "inliers_percentage": float(np.sum(full_inliers) / len(points)),
                                "cache_hit": True,
                            },
                        )
                        self._cache.set_log_bundle("predict_ground_plane", _cache_key, _bundle)
                return _cached

        if points.shape[0] > 4_000:
            points_mask = np.random.choice(points.shape[0], 4_000, replace=False)
            filtered_points = points[points_mask]
        else:
            filtered_points = points

        best_plane_model = None
        best_inliers = None
        max_inliers = -1

        if len(filtered_points) < 3:
            raise ValueError("At least 3 points are required to fit a plane.")

        for _ in range(num_iterations):
            random_indices = np.random.choice(len(filtered_points), 3, replace=False)
            sample_points = filtered_points[random_indices]

            p1, p2, p3 = sample_points
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm == 0:
                continue
            normal = normal / norm

            distances = np.abs(np.dot(filtered_points - p1, normal))
            inliers = distances < distance_threshold
            num_inliers = np.sum(inliers)

            if num_inliers > max_inliers:
                max_inliers = num_inliers
                best_plane_model = (normal, p1)
                best_inliers = inliers

        if best_inliers is not None and np.sum(best_inliers) > 0:
            inlier_points = filtered_points[best_inliers]
            centroid = np.mean(inlier_points, axis=0)
            centered_points = inlier_points - centroid

            _, _, vh = np.linalg.svd(centered_points)
            refined_normal = vh[2, :]

            # Ensure normal points upwards (assuming z is up conceptually or whatever assumption)
            if refined_normal[2] < 0:
                refined_normal = -refined_normal

            best_plane_model = (refined_normal, centroid)

        # Re-broadcast best_inliers against FULL points array (not just the 4,000 sub-sampled array)
        if best_plane_model is not None:
            full_distances = np.abs(
                np.dot(points - best_plane_model[1], best_plane_model[0])
            )
            full_inliers = full_distances < distance_threshold

            if self.api_calls_dir:
                import trimesh

                pts_inliers = points[full_inliers]
                if len(pts_inliers) > 0:
                    min_bounds = pts_inliers.min(axis=0)
                    max_bounds = pts_inliers.max(axis=0)
                    center = (min_bounds + max_bounds) / 2.0
                    extents = max_bounds - min_bounds
                    # create a simple box or plane
                    dx, dy, dz = extents
                    plane_mesh = trimesh.creation.box(
                        extents=[max(dx, 0.01), max(dy, 0.01), 0.01]
                    )

                    # rotate the box to align with the normal
                    normal = best_plane_model[0]
                    up = np.array([0, 0, 1])
                    if np.linalg.norm(np.cross(up, normal)) > 1e-6:
                        axis = np.cross(up, normal)
                        axis = axis / np.linalg.norm(axis)
                        angle = np.arccos(np.clip(np.dot(up, normal), -1.0, 1.0))
                        rot_mat = trimesh.transformations.rotation_matrix(angle, axis)
                        plane_mesh.apply_transform(rot_mat)

                    plane_mesh.apply_translation(best_plane_model[1])
                    plane_mesh.visual.vertex_colors = [0, 255, 0, 100]

                    pc = trimesh.points.PointCloud(points)
                    scene = trimesh.Scene([pc, plane_mesh])

                    result_data = {
                        "normal": [float(n) for n in best_plane_model[0]],
                        "point": [float(p) for p in best_plane_model[1]],
                        "inliers_percentage": float(np.sum(full_inliers) / len(points)),
                    }

                    _, _bundle = self._log_api_call(
                        tool_name="predict_ground_plane",
                        result_data=result_data,
                        extra_files={"scene.glb": scene.export(file_type="glb")},
                    )
                    if self._cache and _cache_key:
                        self._cache.set_log_bundle("predict_ground_plane", _cache_key, _bundle)

            if self._cache and _cache_key:
                self._cache.set_pickle(
                    "predict_ground_plane", _cache_key, (best_plane_model, full_inliers)
                )
            return best_plane_model, full_inliers

        fallback = (np.array([0, 1, 0]), np.array([0, 0, 0])), np.ones(
            points.shape[0], dtype=bool
        )
        if self._cache and _cache_key:
            self._cache.set_pickle("predict_ground_plane", _cache_key, fallback)
        return fallback

    def fit_3d_primitive(
        self, point_cloud: np.ndarray, shape_class: str, axis: np.ndarray = None
    ) -> dict:
        """
        Fit a 3D primitive to a provided point cloud.

        Arguments:
            point_cloud (np.ndarray)
                An Nx3 array of points in 3D space.
            shape_class (str)
                The type of shape to fit ('cuboid', 'sphere', 'plane', 'cylinder').
            axis (np.ndarray, optional)
                For 'cylinder' shape_class, a 3D vector indicating the cylinder's axis direction.

        Returns:
            Fitted parameters (dict)
                The structure of the dictionary depends on the shape class:
                    - 'cuboid': {'center': [x, y, z], 'size': [s_x, s_y, s_z], 'rotation': scipy.spatial.transform.Rotation}
                    - 'sphere': {'center': [x, y, z], 'radius': r}
                    - 'plane': {'normal': [nx, ny, nz], 'd': d}
                    - 'cylinder': {'center': [x, y, z], 'radius': r, 'height': h, 'axis': [ax, ay, az]}
        """
        from pyransac3d import Cuboid, Sphere

        _axis_bytes = axis.tobytes() if axis is not None else b"none"
        _cache_key = (
            _hash_inputs(_arr_bytes(point_cloud), shape_class.encode(), _axis_bytes)
            if self._cache
            else None
        )
        if self._cache and _cache_key:
            _cached = self._cache.get_pickle("fit_3d_primitive", _cache_key)
            if _cached is not None:
                logger.info("Cache hit: fit_3d_primitive(%r)", shape_class)
                if self.api_calls_dir:
                    _bundle = self._cache.get_log_bundle("fit_3d_primitive", _cache_key)
                    if _bundle:
                        self._log_api_call(tool_name=f"fit_3d_{shape_class}", file_bundle=_bundle)
                    else:
                        _log_result = {k: v for k, v in _cached.items() if k != "rotation"}
                        if "rotation" in _cached:
                            _log_result["rotation"] = _cached["rotation"].as_matrix().tolist()
                        _, _bundle = self._log_api_call(
                            tool_name=f"fit_3d_{shape_class}",
                            result_data={"args": _log_result, "shape": shape_class, "cache_hit": True},
                        )
                        self._cache.set_log_bundle("fit_3d_primitive", _cache_key, _bundle)
                return _cached

        if shape_class == "cuboid":
            thresh = 0.07 * point_cloud.std(axis=0).mean()
            params = Cuboid().fit(point_cloud, thresh=thresh, maxIteration=2_000)
            best_planes, inliers = params
            inliers_points = point_cloud[inliers]
            center, size, rot, local_inliers, min_coords, max_coords = (
                self._fit_cuboid_from_corner(best_planes, inliers_points)
            )
            result = {
                "center": [float(c) for c in center],
                "size": [float(s) for s in size],
                "rotation": rot,
            }
            if self.api_calls_dir:
                import trimesh
                import matplotlib.pyplot as plt

                # --- GLB: inliers green, outliers grey, fitted box red ---
                outlier_mask = np.ones(len(point_cloud), dtype=bool)
                outlier_mask[inliers] = False
                inlier_colors = np.tile([0, 200, 0, 200], (len(inliers), 1))
                outlier_colors = np.tile([150, 150, 150, 120], (outlier_mask.sum(), 1))
                pc_in = trimesh.points.PointCloud(
                    point_cloud[inliers], colors=inlier_colors
                )
                pc_out = trimesh.points.PointCloud(
                    point_cloud[outlier_mask], colors=outlier_colors
                )
                mesh = trimesh.creation.box(extents=result["size"])
                rot_4x4 = np.eye(4)
                rot_4x4[:3, :3] = rot.as_matrix()
                mesh.apply_transform(rot_4x4)
                mesh.apply_translation(result["center"])
                mesh.visual.vertex_colors = [255, 0, 0, 150]
                scene = trimesh.Scene([pc_in, pc_out, mesh])

                # --- Axis-projection plots with percentile markers ---
                axis_labels = ["Local X", "Local Y", "Local Z"]
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                for i, (ax, label) in enumerate(zip(axes, axis_labels)):
                    ax.hist(local_inliers[:, i], bins=60, color="steelblue", alpha=0.75)
                    ax.axvline(
                        min_coords[i],
                        color="red",
                        linestyle="--",
                        linewidth=1.5,
                        label=f"2nd pct ({min_coords[i]:.3f})",
                    )
                    ax.axvline(
                        max_coords[i],
                        color="orange",
                        linestyle="--",
                        linewidth=1.5,
                        label=f"98th pct ({max_coords[i]:.3f})",
                    )
                    ax.set_title(f"Inlier projections — {label}")
                    ax.set_xlabel("Local coordinate (m)")
                    ax.set_ylabel("Count")
                    ax.legend(fontsize=7)
                fig.suptitle(
                    f"Cuboid fit  |  inliers {len(inliers)}/{len(point_cloud)}"
                    f"  ({100*len(inliers)/len(point_cloud):.1f}%)",
                    fontsize=11,
                )
                fig.tight_layout()

                log_data = {
                    "args": {
                        "center": result["center"],
                        "size": result["size"],
                        "rotation": rot.as_matrix().tolist(),
                    },
                    "shape": "cuboid",
                    "n_inliers": int(len(inliers)),
                    "n_total": int(len(point_cloud)),
                    "inlier_fraction": float(len(inliers) / len(point_cloud)),
                    "local_axis_extents": {
                        "X": [float(min_coords[0]), float(max_coords[0])],
                        "Y": [float(min_coords[1]), float(max_coords[1])],
                        "Z": [float(min_coords[2]), float(max_coords[2])],
                    },
                    "ransac_thresh": float(thresh),
                }

                proj_buf = io.BytesIO()
                fig.savefig(proj_buf, format="png", dpi=120)
                plt.close(fig)
                _, _bundle = self._log_api_call(
                    tool_name="fit_3d_cuboid",
                    result_data=log_data,
                    extra_files={
                        "scene.glb": scene.export(file_type="glb"),
                        "axis_projections.png": proj_buf.getvalue(),
                    },
                )
                if self._cache and _cache_key:
                    self._cache.set_log_bundle("fit_3d_primitive", _cache_key, _bundle)
        elif shape_class == "sphere":
            thresh = 0.3 * point_cloud.std(axis=0).mean()
            params = Sphere().fit(point_cloud, thresh=thresh)
            result = {
                "center": [float(c) for c in params[0]],
                "radius": float(params[1]),
            }
            if self.api_calls_dir:
                import trimesh

                mesh = trimesh.creation.icosphere(
                    radius=result["radius"], subdivisions=3
                )
                mesh.apply_translation(result["center"])
                mesh.visual.vertex_colors = [0, 255, 0, 150]
                pc = trimesh.points.PointCloud(point_cloud)
                scene = trimesh.Scene([pc, mesh])
                _, _bundle = self._log_api_call(
                    tool_name="fit_3d_sphere",
                    result_data={"args": result, "shape": "sphere"},
                    extra_files={"scene.glb": scene.export(file_type="glb")},
                )
                if self._cache and _cache_key:
                    self._cache.set_log_bundle("fit_3d_primitive", _cache_key, _bundle)
        elif shape_class == "cylinder":
            if axis is None:
                raise ValueError("Axis must be provided for cylinder fitting.")

            # 1. Project points onto plane orthogonal to axis
            axis = axis / np.linalg.norm(axis)
            point_cloud_on_plane = (
                point_cloud - np.dot(point_cloud, axis[:, None]) * axis
            )

            # 2. Fit circle using scikit-image RANSAC
            from skimage.measure import ransac, CircleModel

            # Arbitrary coordinate frame on the plane to reduce to 2D
            base = np.array([1, 0, 0]) if np.abs(axis[0]) < 0.9 else np.array([0, 1, 0])
            u = np.cross(axis, base)
            u /= np.linalg.norm(u)
            v = np.cross(axis, u)

            # Project 3D points in the plane to 2D coordinates
            u_coords = np.dot(point_cloud_on_plane, u)
            v_coords = np.dot(point_cloud_on_plane, v)
            points_2d = np.column_stack((u_coords, v_coords))

            # Threshold relative to spread in the circle's 2D frame, ignoring
            # axis-direction variance entirely.
            thresh_2d = max(0.05 * points_2d.std(), 1e-3)
            model_robust, inliers = ransac(
                points_2d,
                CircleModel,
                min_samples=3,
                residual_threshold=thresh_2d,
                max_trials=1000,
            )

            # Recover 3D center
            cx_2d, cy_2d, radius = model_robust.params
            c_3d = cx_2d * u + cy_2d * v

            # Determine height from inliers projected onto the axis.
            # Reject the extreme 1% at each end so outliers that leaked
            # through the circle RANSAC cannot dominate the extent.
            inlier_3d = point_cloud[inliers]
            projections_on_axis = np.dot(inlier_3d, axis)
            min_proj = np.percentile(projections_on_axis, 1)
            max_proj = np.percentile(projections_on_axis, 99)

            height = max_proj - min_proj
            center_3d = c_3d + ((min_proj + max_proj) / 2.0) * axis

            result = {
                "center": [float(c) for c in center_3d],
                "radius": float(radius),
                "height": float(height),
                "axis": [float(a) for a in axis],
            }
            if self.api_calls_dir:
                import trimesh

                mesh = trimesh.creation.cylinder(
                    radius=result["radius"], height=result["height"]
                )
                # align direction (z-axis) to "axis"
                norm_axis = np.array(result["axis"])
                z_axis = np.array([0, 0, 1])
                if np.linalg.norm(np.cross(z_axis, norm_axis)) > 1e-6:
                    rot_axis = np.cross(z_axis, norm_axis)
                    rot_axis = rot_axis / np.linalg.norm(rot_axis)
                    angle = np.arccos(np.clip(np.dot(z_axis, norm_axis), -1.0, 1.0))
                    rot_mat = trimesh.transformations.rotation_matrix(angle, rot_axis)
                    mesh.apply_transform(rot_mat)
                elif np.dot(z_axis, norm_axis) < 0:
                    rot_mat = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
                    mesh.apply_transform(rot_mat)

                mesh.apply_translation(result["center"])
                mesh.visual.vertex_colors = [0, 0, 255, 150]
                pc = trimesh.points.PointCloud(point_cloud)
                scene = trimesh.Scene([pc, mesh])
                _, _bundle = self._log_api_call(
                    tool_name="fit_3d_cylinder",
                    result_data={"args": result, "shape": "cylinder"},
                    extra_files={"scene.glb": scene.export(file_type="glb")},
                )
                if self._cache and _cache_key:
                    self._cache.set_log_bundle("fit_3d_primitive", _cache_key, _bundle)
        elif shape_class == "plane":
            best_plane, inliers = self.predict_ground_plane(point_cloud)
            result = {
                "normal": [float(n) for n in best_plane[0]],
                "d": float(-np.dot(best_plane[0], best_plane[1])),
            }
            if self.api_calls_dir:
                _, _bundle = self._log_api_call(
                    tool_name="fit_3d_plane",
                    result_data={"args": result, "shape": "plane"},
                )
                if self._cache and _cache_key:
                    self._cache.set_log_bundle("fit_3d_primitive", _cache_key, _bundle)
        else:
            raise ValueError(f"Unknown shape class: {shape_class}")

        if self._cache and _cache_key:
            self._cache.set_pickle("fit_3d_primitive", _cache_key, result)
        return result

    def estimate_3d_mesh(
        self, image: np.ndarray, mask: np.ndarray, max_vertices: int = 10000
    ) -> trimesh.Trimesh:
        """
        Estimate a 3D mesh from an image crop using SAM3D and align it to the world space.
        Uses a pretrained model to regress a full 3D object from 2D and then aligns back-projected geometry limits using `estimate_3d_points`.

        Arguments:
            image (np.ndarray): Image of shape [h, w, 3] with dtype np.uint8.
            mask (np.ndarray): Binary mask of shape [h, w] with dtype bool.
                This describes exactly the silhouette of the object you want to mesh in `image`.
            max_vertices (int, optional): The limit on the detail density. Defaults to 10k.

        Returns:
            mesh (trimesh.Trimesh): An aligned Manifold 3D Trimesh mesh.
        """
        try:
            if not isinstance(image, np.ndarray):
                raise TypeError(
                    f"`image` must be a numpy.ndarray, got {type(image).__name__}"
                )
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(
                    f"`image` must have shape [h, w, 3], got {image.shape}"
                )
            if image.dtype != np.uint8:
                raise ValueError(f"`image` must have dtype uint8, got {image.dtype}")
            if not isinstance(mask, (np.ndarray, bool)):
                raise TypeError("`mask` must be a boolean numpy array")
        except Exception as agent_err:
            print(
                f"[!] Agent Tool Error in estimate_3d_mesh(): {agent_err}",
                file=sys.stderr,
            )
            if hasattr(self, "api_calls_dir") and self.api_calls_dir:
                self._log_api_call(
                    tool_name="estimate_3d_mesh",
                    image=(
                        image
                        if isinstance(image, np.ndarray)
                        else np.zeros((1, 1, 3), dtype=np.uint8)
                    ),
                    result_data={
                        "error": True,
                        "error_type": "Agent Input Validation",
                        "message": str(agent_err),
                    },
                )
            raise agent_err

        _cache_key = (
            _hash_inputs(
                _arr_bytes(image), _arr_bytes(mask), str(max_vertices).encode()
            )
            if self._cache
            else None
        )
        if self._cache and _cache_key:
            _cached_mesh = self._cache.get_glb("estimate_3d_mesh", _cache_key)
            if _cached_mesh is not None:
                logger.info("Cache hit: estimate_3d_mesh()")
                if self.api_calls_dir:
                    _bundle = self._cache.get_log_bundle("estimate_3d_mesh", _cache_key)
                    if _bundle:
                        self._log_api_call(tool_name="estimate_3d_mesh", file_bundle=_bundle)
                    else:
                        _, _bundle = self._log_api_call(
                            tool_name="estimate_3d_mesh",
                            image=image,
                            result_data={
                                "vertices": len(_cached_mesh.vertices),
                                "faces": len(_cached_mesh.faces),
                                "bbox_extents": [float(m) for m in _cached_mesh.extents],
                                "cache_hit": True,
                            },
                        )
                        self._cache.set_log_bundle("estimate_3d_mesh", _cache_key, _bundle)
                return _cached_mesh

        try:
            pil_img = Image.fromarray(image)
            img_buffer = io.BytesIO()
            pil_img.save(img_buffer, format="PNG")
            img_buffer.seek(0)

            pil_mask = Image.fromarray((mask * 255).astype(np.uint8))
            mask_buffer = io.BytesIO()
            pil_mask.save(mask_buffer, format="PNG")
            mask_buffer.seek(0)

            response = requests.post(
                _service_url("VDA_MESH_URL", 5005) + "/estimate_3d_mesh",
                files={
                    "image": ("image.png", img_buffer, "image/png"),
                    "mask": ("mask.png", mask_buffer, "image/png"),
                },
                data={"max_vertices": max_vertices},
            )

            if response.status_code != 200:
                raise APIInternalError(
                    f"Mesh service returned status {response.status_code}: {response.text}"
                )

            mesh = trimesh.load(
                file_obj=io.BytesIO(response.content), file_type="glb", force="mesh"
            )

            # Align mesh back to VGGT 3D geometrical planes via point correlations
            mesh = self._align_mesh_to_pts3d(mesh, image, mask)

        except (APIInternalError, AssertionError):
            raise
        except Exception as internal_err:
            logging.error("Internal mesh backend error: %s", internal_err)
            if hasattr(self, "api_calls_dir") and self.api_calls_dir:
                self._log_api_call(
                    tool_name="estimate_3d_mesh",
                    image=image,
                    result_data={
                        "error": True,
                        "error_type": "Internal System Error",
                        "message": str(internal_err),
                    },
                )
            raise APIInternalError(
                f"Internal error in estimate_3d_mesh(): {internal_err}"
            ) from internal_err

        if self._cache and _cache_key:
            self._cache.set_glb("estimate_3d_mesh", _cache_key, mesh)

        if self.api_calls_dir:
            _, _bundle = self._log_api_call(
                tool_name="estimate_3d_mesh",
                image=image,
                result_data={
                    "vertices": len(mesh.vertices),
                    "faces": len(mesh.faces),
                    "bbox_extents": [float(m) for m in mesh.extents],
                },
            )
            if self._cache and _cache_key:
                self._cache.set_log_bundle("estimate_3d_mesh", _cache_key, _bundle)

        return mesh

    def _align_mesh_to_pts3d(
        self, mesh: trimesh.Trimesh, image: np.ndarray, mask: np.ndarray
    ) -> trimesh.Trimesh:
        """
        Duct-tapes Sam3D topological scales directly against the generated point-cloud values from VGGT.
        """
        pts3d = self.estimate_3d_points(image)
        pts3d_masked = pts3d[mask]
        mesh_vertices = mesh.vertices

        # fix scale of SAM3D points to match pts3d
        extent_pts3d_y = pts3d_masked[:, 1].max() - pts3d_masked[:, 1].min()
        extent_mesh_y = mesh_vertices[:, 1].max() - mesh_vertices[:, 1].min()
        scale_factor = extent_pts3d_y / extent_mesh_y

        mesh_vertices = mesh_vertices * scale_factor
        mesh.vertices = mesh_vertices

        z_min = mesh.vertices[:, 2].min() - 1.0
        ray_origins = np.column_stack(
            [
                pts3d_masked[:, 0],
                pts3d_masked[:, 1],
                np.full(pts3d_masked.shape[0], z_min),
            ]
        )
        ray_directions = np.tile([0, 0, 1], (pts3d_masked.shape[0], 1))

        # 2. Find intersections
        locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins=ray_origins, ray_directions=ray_directions
        )

        if len(locations) > 0:
            dtype = [("ray_idx", int), ("z", float)]
            hits = np.empty(len(locations), dtype=dtype)
            hits["ray_idx"] = index_ray
            hits["z"] = locations[:, 2]

            sorted_idx = np.argsort(hits, order=["ray_idx", "z"])
            hits = hits[sorted_idx[::-1]]

            unique_rays, first_hit_idx = np.unique(hits["ray_idx"], return_index=True)
            mesh_z_intersections = hits["z"][first_hit_idx]
            target_z = pts3d_masked[unique_rays, 2]

            z_diff = mesh_z_intersections - target_z
            translation_z = np.mean(z_diff)
            mesh.vertices[:, 2] -= translation_z
        else:
            mean_pts3d = pts3d_masked.mean(axis=0)
            mean_mesh = mesh.vertices.mean(axis=0)
            translation = mean_mesh - mean_pts3d
            mesh.vertices -= translation

        return mesh

    def _fit_cuboid_from_corner(self, planes: np.ndarray, inliers: np.ndarray) -> tuple:
        normals = planes[:, :3]
        d_coeffs = planes[:, 3]
        corner_point = np.linalg.solve(normals, -d_coeffs)

        rotation_matrix = normals.T
        rotation_matrix[:, 0] /= np.linalg.norm(rotation_matrix[:, 0])
        rotation_matrix[:, 1] /= np.linalg.norm(rotation_matrix[:, 1])
        rotation_matrix[:, 2] /= np.linalg.norm(rotation_matrix[:, 2])

        if np.linalg.det(rotation_matrix) < 0:
            rotation_matrix[:, 2] *= -1

        # pyransac3d returns plane normals in an arbitrary order, so the column
        # assignment (local X/Y/Z) is not deterministic.  Reorder columns so each
        # local axis is assigned to the world axis it is most aligned with.  This
        # ensures an axis-aligned box produces near-identity rotation rather than a
        # spurious 90-degree permutation.
        abs_R = np.abs(rotation_matrix)
        col_order = np.argmax(abs_R, axis=0)  # world-axis each local axis is closest to
        if len(set(col_order)) == 3:  # only reorder if assignment is unambiguous
            rotation_matrix = rotation_matrix[:, np.argsort(col_order)]
            # Flip any column that points opposite to its world axis
            for i in range(3):
                if rotation_matrix[i, i] < 0:
                    rotation_matrix[:, i] *= -1
            # Re-enforce right-handedness after flips
            if np.linalg.det(rotation_matrix) < 0:
                rotation_matrix[:, 2] *= -1

        centered_inliers = inliers - corner_point
        local_inliers = np.dot(centered_inliers, rotation_matrix)

        # Trim the extreme 2% at each end per axis so outliers that leaked
        # through the plane RANSAC cannot inflate the fitted extents.
        min_coords = np.percentile(local_inliers, 1, axis=0)
        max_coords = np.percentile(local_inliers, 99, axis=0)

        size = max_coords - min_coords

        local_center = (min_coords + max_coords) / 2.0
        global_center = corner_point + np.dot(rotation_matrix, local_center)

        from scipy.spatial.transform import Rotation

        rot = Rotation.from_matrix(rotation_matrix)

        return global_center, size, rot, local_inliers, min_coords, max_coords

    def polygon_from_mask(
        self, mask: np.ndarray, simplification_factor: float = 0.005
    ) -> np.ndarray:
        """
        Generates a simplified 2D polygon from a binary mask by finding and
        approximating its contour.

        Arguments:
            mask (np.ndarray):
                A 2D boolean numpy array. Assumes the mask has been cleaned
                and contains one contiguous object.
            simplification_factor (float):
                Controls the precision of the polygon approximation. Smaller values
                result in more vertices and higher detail. Value is relative to
                the contour's perimeter.

        Returns:
            np.ndarray:
                An array of shape `[k, 2]` representing the polygon's vertex
                coordinates `[x, y]`.
        """
        import cv2

        if not isinstance(mask, np.ndarray):
            raise TypeError("Input 'mask' must be a numpy array.")
        if mask.ndim != 2:
            raise ValueError(
                f"Input 'mask' must be a 2D array, but got {mask.ndim} dimensions."
            )
        if mask.dtype != bool:
            raise TypeError(f"Input 'mask' must have dtype bool, but got {mask.dtype}.")
        if (
            not isinstance(simplification_factor, (int, float))
            or simplification_factor < 0
        ):
            raise ValueError("simplification_factor must be a non-negative number.")

        _cache_key = (
            _hash_inputs(_arr_bytes(mask.view(np.uint8)), str(simplification_factor).encode())
            if self._cache else None
        )
        if self._cache and _cache_key:
            _cached = self._cache.get_pickle("polygon_from_mask", _cache_key)
            if _cached is not None:
                logger.info("Cache hit: polygon_from_mask()")
                if self.api_calls_dir:
                    _bundle = self._cache.get_log_bundle("polygon_from_mask", _cache_key)
                    if _bundle:
                        self._log_api_call(tool_name="generate_polygon_from_mask", file_bundle=_bundle)
                    else:
                        _, _bundle = self._log_api_call(
                            tool_name="generate_polygon_from_mask",
                            result_data={"points_count": int(_cached.shape[0]), "cache_hit": True},
                        )
                        self._cache.set_log_bundle("polygon_from_mask", _cache_key, _bundle)
                return _cached

        if not np.any(mask):
            return np.array([], dtype=int).reshape(0, 2)

        mask_uint8 = mask.astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return np.array([], dtype=int).reshape(0, 2)

        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)
        epsilon = simplification_factor * perimeter
        approx_polygon = cv2.approxPolyDP(contour, epsilon, True)

        result = approx_polygon.squeeze(axis=1)

        if self._cache and _cache_key:
            self._cache.set_pickle("polygon_from_mask", _cache_key, result)

        if self.api_calls_dir and result.shape[0] > 0:
            h, w = mask.shape
            vis = np.zeros((h, w, 3), dtype=np.uint8)
            vis[mask] = [128, 128, 128]
            pts = result.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(vis, [pts], True, (255, 0, 0), 2)
            for x, y in result:
                cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)

            _, _bundle = self._log_api_call(
                tool_name="generate_polygon_from_mask",
                image=vis,
                result_data={"points_count": int(result.shape[0])},
            )
            if self._cache and _cache_key:
                self._cache.set_log_bundle("polygon_from_mask", _cache_key, _bundle)

        return result

    def ground_plane_to_w2c(self, ground_plane: tuple) -> np.ndarray:
        """
        Converts a ground plane model to a world-to-camera transformation matrix.

        Args:
            ground_plane (tuple): A tuple (normal, point) representing the ground plane.
                                    `normal` is a (3,) numpy array representing the plane normal.
                                    `point` is a (3,) numpy array representing a point on the plane.

        Returns:
            np.ndarray: A 4x4 transformation matrix that transforms points from world coordinates to camera coordinates.
        """
        plane_normal, plane_point = ground_plane
        if plane_normal[1] < 0:
            plane_normal = -plane_normal

        # Establish a robust world coordinate system.
        # The world's +Z axis is the "up" direction normal to the table.
        world_z_axis = plane_normal

        # To find the world's X axis, we project the camera's X axis ([1, 0, 0])
        # onto the newly found ground plane. This makes the world orientation intuitive.
        cam_x_axis = np.array([1.0, 0.0, 0.0])
        # Use Gram-Schmidt to get an orthogonal vector
        world_x_axis = cam_x_axis - np.dot(cam_x_axis, world_z_axis) * world_z_axis
        world_x_axis /= np.linalg.norm(world_x_axis)

        # The world's Y axis is found via the cross product to form a right-handed system.
        world_y_axis = np.cross(world_z_axis, world_x_axis)

        # The rotation matrix from world to camera coordinates is composed of these axes.
        # Note: The columns of a change-of-basis matrix are the new basis vectors.
        # Here we want the inverse (camera to world), so we use the rows.
        rotation_cam_to_world = np.array([world_x_axis, world_y_axis, world_z_axis])

        # The full transform from camera to world coordinates
        cam_to_world_transform = np.eye(4)
        cam_to_world_transform[:3, :3] = rotation_cam_to_world
        cam_to_world_transform[:3, 3] = -rotation_cam_to_world @ plane_point

        # We store the inverse for projecting from our world back to the camera for rendering
        world_to_cam_transform = np.linalg.inv(cam_to_world_transform)

        _cache_key = (
            _hash_inputs(
                _arr_bytes(np.asarray(plane_normal)),
                _arr_bytes(np.asarray(plane_point)),
            )
            if self._cache else None
        )
        if self._cache and _cache_key:
            _cached = self._cache.get_pickle("ground_plane_to_w2c", _cache_key)
            if _cached is not None:
                logger.info("Cache hit: ground_plane_to_w2c()")
                if self.api_calls_dir:
                    _bundle = self._cache.get_log_bundle("ground_plane_to_w2c", _cache_key)
                    if _bundle:
                        self._log_api_call(tool_name="ground_plane_to_w2c", file_bundle=_bundle)
                    else:
                        _, _bundle = self._log_api_call(
                            tool_name="ground_plane_to_w2c",
                            result_data={"w2c_matrix": _cached.tolist(), "cache_hit": True},
                        )
                        self._cache.set_log_bundle("ground_plane_to_w2c", _cache_key, _bundle)
                return _cached

        if self._cache and _cache_key:
            self._cache.set_pickle("ground_plane_to_w2c", _cache_key, world_to_cam_transform)

        if self.api_calls_dir:
            _, _bundle = self._log_api_call(
                tool_name="ground_plane_to_w2c",
                result_data={"w2c_matrix": world_to_cam_transform.tolist()},
            )
            if self._cache and _cache_key:
                self._cache.set_log_bundle("ground_plane_to_w2c", _cache_key, _bundle)

        return world_to_cam_transform

    def _view_proj_matrices(
        self, w2c: np.ndarray, intrinsics: np.ndarray, frame_size: tuple
    ) -> tuple:
        """
        Computes the view and projection matrices for rendering.

        Args:
            w2c (np.ndarray): A 4x4 world-to-camera transformation matrix.
            intrinsics (np.ndarray): A 3x3 camera intrinsic matrix.
            frame_size (tuple): The size of the frame as (width, height).

        Returns:
            tuple (np.ndarray, np.ndarray): A view matrix and projection matrix for rendering (16-element 1D arrays matching pybullet's convention).
        """
        cam_to_world = np.linalg.inv(w2c)

        # OpenGL view matrix convention
        # It needs a "look at" point, a camera position, and an up vector.
        cam_pos = cam_to_world[:3, 3]
        target_pos = cam_to_world @ np.array([0, 0, -1, 1])
        target_pos = target_pos[:3] / target_pos[3]
        up_vector = cam_to_world[:3, 1]

        f = target_pos - cam_pos
        f = f / np.linalg.norm(f)
        s = np.cross(f, up_vector)
        s = s / np.linalg.norm(s)
        u = np.cross(s, f)

        view_matrix = np.eye(4)
        view_matrix[0, :3] = s
        view_matrix[1, :3] = u
        view_matrix[2, :3] = -f
        view_matrix[0, 3] = -np.dot(s, cam_pos)
        view_matrix[1, 3] = -np.dot(u, cam_pos)
        view_matrix[2, 3] = np.dot(f, cam_pos)

        # PyBullet's projection matrix from intrinsics
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        w, h = frame_size
        n, f_val = 0.01, 10.0  # Heuristic near_plane, far_plane

        left = -cx / fx * n
        right = (w - cx) / fx * n
        bottom = -(h - cy) / fy * n
        top = cy / fy * n

        projection_matrix = np.zeros((4, 4))
        projection_matrix[0, 0] = (2.0 * n) / (right - left)
        projection_matrix[1, 1] = (2.0 * n) / (top - bottom)
        projection_matrix[0, 2] = (right + left) / (right - left)
        projection_matrix[1, 2] = (top + bottom) / (top - bottom)
        projection_matrix[2, 2] = -(f_val + n) / (f_val - n)
        projection_matrix[2, 3] = -(2.0 * f_val * n) / (f_val - n)
        projection_matrix[3, 2] = -1.0

        # PyBullet compute*Matrix returns 16-element tuples in column-major order.
        # We flatten the transposed 4x4 matrix to replicate this behavior exactly.
        return view_matrix.T.flatten(), projection_matrix.T.flatten()


class GeometryOnlyWorldAPI(WorldAPI):
    """WorldAPI variant with segmentation disabled for generated simulators.

    Action-conditioned prompts route segmentation through Gemini codegen tools.
    Generated simulator code may still use WorldAPI for non-segmentation
    geometry helpers in ``fit(image_A, image_B)``, but should not call
    ``self.api.segment``.
    """

    def segment(self, image: np.ndarray, text_prompt: str) -> np.ndarray:
        raise RuntimeError(
            "WorldAPI.segment is disabled for action-conditioned generated simulators. "
            "Use segment_image_with_gemini / segment_trajectory_with_gemini during "
            "code generation, then bake the resulting parsing logic into fit()."
        )


def _point_cloud_obj_bytes(pts3d: np.ndarray) -> bytes:
    """Return the Wavefront OBJ representation of *pts3d* as raw bytes."""
    flat = pts3d.reshape(-1, 3)
    norms = np.linalg.norm(flat, axis=-1)
    valid = flat[norms > 0.0]
    header = (
        "# 3-D point cloud — OpenGL camera coords (+X right, +Y up, -Z fwd)\n"
        f"# {valid.shape[0]:,} points ({pts3d.shape[0]}x{pts3d.shape[1]} image)\n"
    ).encode()
    buf = io.BytesIO()
    np.savetxt(buf, valid, fmt="v %.6f %.6f %.6f")
    return header + buf.getvalue()


def _write_point_cloud_obj(pts3d: np.ndarray, path: str) -> None:
    """Write a point cloud to a Wavefront OBJ file.

    Args:
        pts3d: ``(H, W, 3)`` float32 array of 3-D points in the OpenGL camera
            coordinate system (+X right, +Y up, -Z forward).  Points whose
            magnitude is zero are treated as masked / invalid and are omitted.
        path: Destination file path (created or overwritten).
    """
    flat = pts3d.reshape(-1, 3)

    # Omit zero-magnitude points (masked out by depth backends).
    norms = np.linalg.norm(flat, axis=-1)
    valid = flat[norms > 0.0]

    with open(path, "w", encoding="utf-8") as f:
        f.write("# 3-D point cloud — OpenGL camera coords (+X right, +Y up, -Z fwd)\n")
        f.write(
            f"# {valid.shape[0]:,} points ({pts3d.shape[0]}x{pts3d.shape[1]} image)\n"
        )
        # Build all vertex lines in one vectorised pass to avoid a Python loop
        # over potentially ~600 k points.
        np.savetxt(f, valid, fmt="v %.6f %.6f %.6f")
