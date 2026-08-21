import numpy as np


class SimulatorBase:
    def __init__(self, frame_size=(1024, 576), api=None, fps=30):
        # Define state variables
        self.t = 0

        # Provided parameters
        self.frame_size = frame_size
        self.api = api
        self.fps = fps

        self._debug_view: np.ndarray | None = None

    def log_debug_view(self, image: np.ndarray) -> None:
        """Store a debug image retrievable via the view_debug tool.

        Call this from fit(), render_frame(), or update_simulation() to expose
        an auxiliary view — e.g. a component-only render, annotated overlay, or
        intermediate state — to help diagnose something confusing or in need of
        refinement. Only the last image passed is retained.

        Arguments:
            image (np.ndarray[np.uint8])
                RGB image of shape (H, W, 3) and dtype uint8.
        """
        self._debug_view = image

    def render_frame(self):
        """
        Render the next frame of the simulation.
        """
        raise NotImplementedError()

    def update_simulation(self, dt: float):
        """
        Update the simulation by one timestep dt.

        Arguments:
            dt (float)
                Time step to update the simulation by.
        """
        raise NotImplementedError()

    def fit(self, image: np.ndarray):
        """
        Fit the parameters of the scene to the provided image.

        Arguments:
            image (np.ndarray[np.uint8])
                image showing physical process to be simulated. Video has size [H, W, 3] corresponding to the frame height, width, and three color channels.
        """
        raise NotImplementedError()

    def reset(self):
        """
        Reset the simulation to its initial state.
        """
        self.t = 0

    def __next__(self):
        """
        Return the next frame and advance by one timestep.

        Calls render_frame() for the current state, then advances via
        update_simulation(dt=1/fps).

        Returns:
            frame (np.ndarray[np.uint8])
                Rendered simulation frame of shape (height, width, 3).
        """
        frame = self.render_frame()
        self.update_simulation(dt=1 / self.fps)
        self.t += 1 / self.fps

        target_height = self.frame_size[1]
        target_width = self.frame_size[0]
        height = frame.shape[0]
        width = frame.shape[1]
        assert (height, width) == (
            target_height,
            target_width,
        ), f"Expected frame size {target_height}x{target_width}, got {height}x{width}"
        if len(frame.shape) == 3:
            assert (
                frame.shape[2] == 3
            ), "Expected frame to have 3 color channels as it has 3 dimensions"
        else:
            assert (
                len(frame.shape) == 2
            ), f"Expected frame to have 2 dimensions (height, width) or 3 dimensions (height, width, channels). Got {len(frame.shape)} dimensions instead."
        assert frame.dtype == np.uint8

        return frame

    def run_simulation(self, n_frames, out_arr=None):
        frames = []
        for i in range(n_frames):
            frame = next(self)
            if out_arr is not None:
                np.copyto(out_arr[i], frame)
            else:
                frames.append(frame)
        return out_arr if out_arr is not None else frames

    def __iter__(self):
        return self

    def __del__(self):
        pass


class ActionConditionedSimulatorBase(SimulatorBase):
    """Base class for action-conditioned simulators driven by an external planner.

    Subclasses (generated per-environment by the VLM) implement:

    * :meth:`update`         — apply one action step (replaces parent's
                               time-passive ``update_simulation(dt)``).
    * :meth:`fit`            — set ``self.state`` from start image and
                               ``self.target_state`` from goal image.
    * :meth:`stage_cost`     — optional generic trajectory cost after each action.
    * :meth:`terminal_cost`  — scalar planner cost ``self.state`` → ``self.target_state``.
    * :meth:`render_frame`   — inherited contract (uint8 HWC).

    Contract for planners (e.g. :class:`vdaworld.core.cem.CEM`): the only
    mutable per-rollout state lives in ``self.state``. Snapshot via
    ``copy.deepcopy(sim.state)`` before a rollout and restore by assigning
    back. Subclasses that need other mutable state (e.g. physics-engine
    bodies) must either repack it into ``self.state`` or implement custom
    save/restore.
    """

    def __init__(self, frame_size=(65, 65), api=None, fps=30):
        super().__init__(frame_size=frame_size, api=api, fps=fps)
        self.state = None
        self.target_state = None

    def update(self, a):
        """Apply one action step. Subclasses implement."""
        raise NotImplementedError()

    def fit(self, image_A, image_B):
        """Set ``self.state`` from image_A and ``self.target_state`` from image_B."""
        raise NotImplementedError()

    # ------------------------------------------------------------------
    # Always-available local 2-D toolbox
    #
    # These wrappers deliberately live on the base class so generated code can
    # use the same deterministic perception/dynamics primitives with api=None
    # in both P1 and P2.  Object identity, action semantics, state design, and
    # parameter selection remain the generated simulator's responsibility.

    @staticmethod
    def clean_mask(
        mask: np.ndarray,
        *,
        min_area: int = 1,
        open_radius: int = 0,
        close_radius: int = 1,
        largest: bool = True,
    ) -> np.ndarray:
        """Clean a binary mask and optionally retain its largest component."""
        from vdaworld.core.local_2d import clean_mask

        return clean_mask(
            mask,
            min_area=min_area,
            open_radius=open_radius,
            close_radius=close_radius,
            largest=largest,
        )

    @staticmethod
    def select_connected_component(
        mask: np.ndarray,
        *,
        min_area: int = 1,
        reference_centroid=None,
        expected_area: float | None = None,
        connectivity: int = 8,
    ) -> dict:
        """Select one component using optional centroid and area expectations."""
        from vdaworld.core.local_2d import select_connected_component

        return select_connected_component(
            mask,
            min_area=min_area,
            reference_centroid=reference_centroid,
            expected_area=expected_area,
            connectivity=connectivity,
        )

    @staticmethod
    def mask_from_appearance(
        image: np.ndarray,
        model: dict,
        *,
        min_area: int = 1,
        open_radius: int = 0,
        close_radius: int = 1,
        largest: bool = True,
    ) -> np.ndarray:
        """Apply a robust Lab appearance model emitted by derive_visual_model."""
        from vdaworld.core.local_2d import mask_from_appearance

        return mask_from_appearance(
            image,
            model,
            min_area=min_area,
            open_radius=open_radius,
            close_radius=close_radius,
            largest=largest,
        )

    @staticmethod
    def foreground_from_border(
        image: np.ndarray,
        *,
        lab_distance: float = 12.0,
        border_width: int = 3,
        min_area: int = 1,
        open_radius: int = 0,
        close_radius: int = 1,
        largest: bool = False,
    ) -> np.ndarray:
        """Segment foreground relative to the robust median border colour."""
        from vdaworld.core.local_2d import foreground_from_border

        return foreground_from_border(
            image,
            lab_distance=lab_distance,
            border_width=border_width,
            min_area=min_area,
            open_radius=open_radius,
            close_radius=close_radius,
            largest=largest,
        )

    @staticmethod
    def component_geometry(
        mask: np.ndarray,
        *,
        simplify_px: float = 1.0,
        max_vertices: int = 64,
    ) -> dict:
        """Return area, centroid, bounds, PCA angle, and simplified contour."""
        from vdaworld.core.local_2d import component_geometry

        return component_geometry(
            mask,
            simplify_px=simplify_px,
            max_vertices=max_vertices,
        )

    @staticmethod
    def template_from_mask(
        mask: np.ndarray,
        *,
        simplify_px: float = 1.0,
        max_vertices: int = 64,
    ) -> list[np.ndarray]:
        """Derive a centroid-centred rigid polygon template from a mask."""
        from vdaworld.core.local_2d import template_from_mask

        return template_from_mask(
            mask,
            simplify_px=simplify_px,
            max_vertices=max_vertices,
        )

    @staticmethod
    def calibrate_component_geometry(
        masks,
        *,
        min_area: int = 1,
        open_radius: int = 0,
        close_radius: int = 0,
        largest: bool = True,
        simplify_px: float = 1.0,
        max_vertices: int = 64,
        orientation_anisotropy_threshold: float = 0.05,
    ) -> dict:
        """Calibrate robust rigid/component geometry from multiple masks."""
        from vdaworld.core.local_2d import calibrate_component_geometry

        return calibrate_component_geometry(
            masks,
            min_area=min_area,
            open_radius=open_radius,
            close_radius=close_radius,
            largest=largest,
            simplify_px=simplify_px,
            max_vertices=max_vertices,
            orientation_anisotropy_threshold=orientation_anisotropy_threshold,
        )

    @staticmethod
    def fit_articulated_chain(
        mask: np.ndarray,
        base_xy: np.ndarray,
        link_lengths: np.ndarray,
        *,
        initial_angles: np.ndarray | None = None,
        relative_angles: bool = True,
        n_starts: int = 128,
        prior_weight: float = 0.02,
    ) -> dict:
        """Fit a fixed-length planar serial chain to a binary link mask."""
        from vdaworld.core.local_2d import fit_articulated_chain

        return fit_articulated_chain(
            mask,
            base_xy,
            link_lengths,
            initial_angles=initial_angles,
            relative_angles=relative_angles,
            n_starts=n_starts,
            prior_weight=prior_weight,
        )

    @staticmethod
    def calibrate_articulated_chain_geometry(
        masks,
        base_xy: np.ndarray,
        link_lengths: np.ndarray,
        **kwargs,
    ) -> dict:
        """Calibrate fixed chain geometry from several diverse masks."""
        from vdaworld.core.local_2d import calibrate_articulated_chain_geometry

        return calibrate_articulated_chain_geometry(
            masks,
            base_xy,
            link_lengths,
            **kwargs,
        )

    @staticmethod
    def clamp_step(
        previous: np.ndarray,
        target: np.ndarray,
        max_step: float,
    ) -> np.ndarray:
        """Move from ``previous`` toward ``target`` by at most ``max_step``."""
        from vdaworld.core.local_2d import clamp_step

        return clamp_step(previous, target, max_step)

    @staticmethod
    def estimate_observed_velocity(
        current_positions: np.ndarray,
        previous_observed_positions: np.ndarray,
        elapsed_seconds: float,
        periodic,
    ) -> np.ndarray:
        """Estimate observed-to-observed velocity under declared topology."""
        from vdaworld.core.local_2d import estimate_observed_velocity

        return estimate_observed_velocity(
            current_positions,
            previous_observed_positions,
            elapsed_seconds,
            periodic,
        )

    @staticmethod
    def linear_dynamics_step(
        state: np.ndarray,
        action: np.ndarray,
        gain: np.ndarray,
        *,
        bias: np.ndarray | None = None,
        max_delta: float | np.ndarray | None = None,
    ) -> np.ndarray:
        """Apply ``x_next = x + gain @ action + bias`` with optional clipping."""
        from vdaworld.core.local_2d import linear_dynamics_step

        return linear_dynamics_step(
            state,
            action,
            gain,
            bias=bias,
            max_delta=max_delta,
        )

    @staticmethod
    def damped_dynamics_step(
        position: np.ndarray,
        velocity: np.ndarray,
        action: np.ndarray,
        gain: np.ndarray,
        *,
        damping,
        dt: float = 1.0,
        max_velocity=None,
        position_low=None,
        position_high=None,
        periodic=False,
    ) -> dict[str, np.ndarray]:
        """Apply a damped update with optional bounds and periodic dimensions."""
        from vdaworld.core.local_2d import damped_dynamics_step

        return damped_dynamics_step(
            position,
            velocity,
            action,
            gain,
            damping=damping,
            dt=dt,
            max_velocity=max_velocity,
            position_low=position_low,
            position_high=position_high,
            periodic=periodic,
        )

    @staticmethod
    def planar_push_step(**kwargs) -> dict:
        """Advance an arbitrary polygon body pushed by a controlled circle."""
        from vdaworld.core.local_2d import planar_push_step

        return planar_push_step(**kwargs)

    @staticmethod
    def convex_decompose_polygon(
        polygon: np.ndarray,
        *,
        tolerance: float = 1e-9,
    ) -> list[np.ndarray]:
        """Decompose a simple local-coordinate polygon into convex pieces."""
        from vdaworld.core.local_2d import convex_decompose_polygon

        return convex_decompose_polygon(polygon, tolerance=tolerance)

    @staticmethod
    def largest_connected_component(
        mask: np.ndarray,
        min_area: int = 1,
        connectivity: int = 8,
    ) -> np.ndarray:
        """Return the largest connected foreground component as a boolean mask.

        This deterministic cleanup helper is available even when ``api=None``.
        It prevents isolated threshold noise or unrelated regions from shifting
        mask centroids and pose estimates.
        """
        from scipy.ndimage import label

        binary = np.asarray(mask).astype(bool)
        if binary.ndim != 2:
            raise ValueError(
                f"mask must be a 2D boolean or 0/1 array, got shape {binary.shape}"
            )
        if isinstance(min_area, bool) or int(min_area) <= 0:
            raise ValueError(f"min_area must be a positive integer, got {min_area!r}")
        if connectivity not in {4, 8}:
            raise ValueError("connectivity must be 4 or 8")
        structure = (
            np.ones((3, 3), dtype=np.uint8)
            if connectivity == 8
            else np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        )
        labels, count = label(binary, structure=structure)
        if count <= 0:
            return np.zeros_like(binary)
        areas = np.bincount(labels.reshape(-1))
        areas[0] = 0
        component = int(np.argmax(areas))
        if int(areas[component]) < int(min_area):
            return np.zeros_like(binary)
        return labels == component

    def fit_pose_to_mask(
        self,
        mask: np.ndarray,
        template: list[np.ndarray],
        n_angles: int = 180,
    ) -> dict:
        """Fit a known asymmetric 2D template to a binary mask locally.

        This helper is always available, including when ``self.api is None``.
        It searches a full 360-degree rotation sweep and a small translation
        neighbourhood, returning the pose with maximum template/mask IoU.
        Template polygons use local ``(x, y)`` pixel coordinates. They need not
        be centroid-centred: the helper accounts for the template's rasterized
        centroid while preserving its encoded body origin at local ``(0, 0)``.
        """
        from vdaworld.core.api import _rasterize_template_polygons

        binary = np.asarray(mask).astype(bool)
        if binary.ndim != 2:
            raise ValueError(
                f"mask must be a 2D boolean or 0/1 array, got shape {binary.shape}"
            )
        if isinstance(n_angles, bool) or int(n_angles) <= 0:
            raise ValueError(f"n_angles must be a positive integer, got {n_angles!r}")
        polygons = [np.asarray(poly, dtype=float) for poly in template]
        if not polygons:
            raise ValueError("template must contain at least one polygon")
        for index, polygon in enumerate(polygons):
            if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
                raise ValueError(
                    "template polygons must each have shape (N, 2) with N >= 3; "
                    f"polygon {index} has shape {polygon.shape}"
                )
            if not np.all(np.isfinite(polygon)):
                raise ValueError(f"template polygon {index} contains non-finite values")

        h, w = binary.shape
        ys, xs = np.where(binary)
        if len(xs) == 0:
            return {
                "cx": w / 2.0,
                "cy": h / 2.0,
                "theta": 0.0,
                "iou": 0.0,
                "valid": False,
                "confidence": 0.0,
            }

        observed_centroid = np.array([float(xs.mean()), float(ys.mean())])
        min_xy = np.min(np.concatenate(polygons, axis=0), axis=0)
        max_xy = np.max(np.concatenate(polygons, axis=0), axis=0)
        margin = 4
        local_w = max(8, int(np.ceil(max_xy[0] - min_xy[0])) + 2 * margin + 1)
        local_h = max(8, int(np.ceil(max_xy[1] - min_xy[1])) + 2 * margin + 1)
        local_origin = np.array(
            [margin - min_xy[0], margin - min_xy[1]], dtype=float
        )
        local_mask = _rasterize_template_polygons(
            polygons,
            float(local_origin[0]),
            float(local_origin[1]),
            0.0,
            (local_w, local_h),
        )
        local_ys, local_xs = np.where(local_mask)
        if len(local_xs) == 0:
            raise ValueError("template rasterizes to an empty mask")
        template_centroid = np.array(
            [
                float(local_xs.mean()) - local_origin[0],
                float(local_ys.mean()) - local_origin[1],
            ]
        )

        best_theta, best_iou, best_dice = 0.0, -1.0, -1.0
        best_cx, best_cy = float(observed_centroid[0]), float(observed_centroid[1])
        angle_step = 2.0 * np.pi / int(n_angles)
        for theta in np.linspace(
            0.0, 2.0 * np.pi, int(n_angles), endpoint=False
        ):
            cosine, sine = np.cos(theta), np.sin(theta)
            rotation = np.array([[cosine, -sine], [sine, cosine]])
            origin = observed_centroid - rotation @ template_centroid
            for dx in (-2.0, -1.0, 0.0, 1.0, 2.0):
                for dy in (-2.0, -1.0, 0.0, 1.0, 2.0):
                    candidate_cx = float(origin[0] + dx)
                    candidate_cy = float(origin[1] + dy)
                    candidate = _rasterize_template_polygons(
                        polygons,
                        candidate_cx,
                        candidate_cy,
                        float(theta),
                        (w, h),
                    )
                    intersection = int(np.logical_and(candidate, binary).sum())
                    union = int(np.logical_or(candidate, binary).sum())
                    candidate_area = int(candidate.sum())
                    iou = float(intersection / union) if union else 0.0
                    dice = float(
                        2.0 * intersection / max(1, candidate_area + int(binary.sum()))
                    )
                    if iou > best_iou:
                        best_iou = iou
                        best_dice = dice
                        best_theta = float(theta)
                        best_cx = candidate_cx
                        best_cy = candidate_cy

        for theta in np.linspace(
            best_theta - angle_step,
            best_theta + angle_step,
            9,
            endpoint=True,
        ):
            cosine, sine = np.cos(theta), np.sin(theta)
            rotation = np.array([[cosine, -sine], [sine, cosine]])
            origin = observed_centroid - rotation @ template_centroid
            for dx in np.linspace(-2.0, 2.0, 9):
                for dy in np.linspace(-2.0, 2.0, 9):
                    candidate_cx = float(origin[0] + dx)
                    candidate_cy = float(origin[1] + dy)
                    candidate = _rasterize_template_polygons(
                        polygons,
                        candidate_cx,
                        candidate_cy,
                        float(theta),
                        (w, h),
                    )
                    intersection = int(np.logical_and(candidate, binary).sum())
                    union = int(np.logical_or(candidate, binary).sum())
                    candidate_area = int(candidate.sum())
                    iou = float(intersection / union) if union else 0.0
                    dice = float(
                        2.0 * intersection / max(1, candidate_area + int(binary.sum()))
                    )
                    if iou > best_iou:
                        best_iou = iou
                        best_dice = dice
                        best_theta = float(theta % (2.0 * np.pi))
                        best_cx = candidate_cx
                        best_cy = candidate_cy

        return {
            "cx": best_cx,
            "cy": best_cy,
            "theta": best_theta,
            "iou": best_iou,
            "dice": best_dice,
            "mask_area": int(binary.sum()),
            "valid": True,
            "confidence": float(np.clip(best_iou, 0.0, 1.0)),
        }

    def _overrides_planning_method(self, name: str) -> bool:
        method = getattr(type(self), name, None)
        base_method = getattr(ActionConditionedSimulatorBase, name, None)
        return method is not None and method is not base_method

    def terminal_cost(self):
        """Terminal planner cost from ``self.state`` to ``self.target_state``.

        New generated simulators should implement this method. The fallback keeps
        older generated simulators with only ``loss_to_target`` runnable.
        """
        if self._overrides_planning_method("loss_to_target"):
            return self.loss_to_target()
        raise NotImplementedError()

    def loss_to_target(self):
        """Legacy alias for ``terminal_cost``."""
        if self._overrides_planning_method("terminal_cost"):
            return self.terminal_cost()
        raise NotImplementedError()

    def stage_cost(self, action, step_index):
        """Optional generic trajectory cost evaluated after an action update.

        The default is side-effect-free and contributes no cost. Subclasses may
        override it to express path-dependent preferences without changing the
        terminal planning objective.
        """
        return 0.0

    def planning_objective(self):
        """Planner-facing scalar cost, preferring ``terminal_cost`` when present."""
        if self._overrides_planning_method("terminal_cost"):
            return self.terminal_cost()
        if self._overrides_planning_method("loss_to_target"):
            return self.loss_to_target()
        raise NotImplementedError()

    def update_simulation(self, dt):
        raise NotImplementedError(
            "ActionConditionedSimulatorBase replaces update_simulation(dt) with "
            "update(a). Override update(a) in subclasses; do not call "
            "update_simulation directly or via __next__."
        )

    def __next__(self):
        raise NotImplementedError(
            "ActionConditionedSimulatorBase does not support iteration. "
            "Drive it via vdaworld.core.cem.CEM, or call update(a) + "
            "render_frame() in your own loop."
        )
