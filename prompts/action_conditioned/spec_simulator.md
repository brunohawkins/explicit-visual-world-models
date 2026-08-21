```python
import numpy as np


class SimulatorBase:
    """(For reference — your simulator inherits from ActionConditionedSimulatorBase below.)"""

    def __init__(self, frame_size=(1024, 576), api=None, fps=30):
        self.t = 0
        self.frame_size = frame_size
        self.api = api
        self.fps = fps

    def render_frame(self):
        """Render the next frame of the simulation as uint8 (H, W, 3)."""
        raise NotImplementedError()

    def log_debug_view(self, image):
        """Store an auxiliary debug image, if the harness exposes debug viewing."""


class ActionConditionedSimulatorBase(SimulatorBase):
    """Base class for action-conditioned simulators driven by an external planner."""

    def __init__(self, frame_size=(65, 65), api=None, fps=30):
        super().__init__(frame_size=frame_size, api=api, fps=fps)
        self.state = None         # set by fit(); mutated by update(); deep-copied by the planner
        self.target_state = None  # set by fit(); read by terminal_cost(); never mutated by update()

    def update(self, a):
        """Apply one action step. Must mutate self.state; must not mutate self.target_state."""
        raise NotImplementedError()

    def fit(self, image_A, image_B):
        """Parse current movable state from image_A; parse target_state from image_B.

        target_state should use the same state variables that update(a) moves
        and the terminal planning objective terminal_cost() scores. WorldAPI
        (non-segmentation geometry helpers) may be available here; rollout
        methods should use the ordinary state values produced by fit.
        """
        raise NotImplementedError()

    def fit_pose_to_mask(self, mask, template, n_angles=180):
        """Fit an asymmetric polygon template to a local binary mask.

        Returns {"cx", "cy", "theta", "iou", "dice", "mask_area", "valid",
        "confidence"} after origin-aware rotation/translation refinement. Empty
        masks preserve legacy centre fields but return valid=False; never treat
        that centre as an observation. This inherited deterministic helper
        works when self.api is None.
        """

    def select_connected_component(
        self,
        mask,
        min_area=1,
        reference_centroid=None,
        expected_area=None,
        connectivity=8,
    ):
        """Select one coherent region using optional temporal/area expectations.

        Returns a boolean mask plus valid/confidence/area/centroid and component
        counts. Never average disconnected regions.
        """

    def largest_connected_component(self, mask, min_area=1, connectivity=8):
        """Legacy largest-only cleanup; prefer select_connected_component."""

    def estimate_observed_velocity(
        self,
        current_positions,
        previous_observed_positions,
        elapsed_seconds,
        periodic,
    ):
        """Observed-to-observed velocity over real elapsed time.

        Wrapped deltas apply only to dimensions declared periodic; bounded and
        other nonperiodic dimensions use raw deltas.
        """

    def stage_cost(self, action, step_index):
        """Optional side-effect-free cost for intermediate rollout states.

        The planner uses this only when an explicit diagnostic/controller
        profile enables stage costs. Keep the default zero unless data supports
        generic path constraints, action realizability, or stability penalties.
        """

    def terminal_cost(self):
        """Terminal planner cost / planning objective over explicit state; lower is better.

        CEM minimises this after rolling out candidate action sequences.
        The objective may include structured guidance, but it should have no
        side effects and should not call segmentation/geometry APIs or roll out actions.
        """
        raise NotImplementedError()
```
