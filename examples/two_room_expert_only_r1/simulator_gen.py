import numpy as np
import cv2

class GeneratedSimulator(ActionConditionedSimulatorBase):
    def __init__(self, frame_size=(224, 224), api=None, fps=10):
        super().__init__(frame_size=frame_size, api=api, fps=fps)
        self.params = {
            "gain_x": 5.0,
            "gain_y": 5.0,
            "radius": 5.0,
            "wall_margin": 1.0,
        }
        self.state = {"x": 100.0, "y": 100.0}
        self.target_state = {"x": 100.0, "y": 100.0}
        self.wall_mask = None
        self.wall_x = 112.0
        self.gap_y = 35.0

    def _extract_walls(self, img):
        if img is None:
            return None
        is_black = (img[:, :, 0] < 50) & (img[:, :, 1] < 50) & (img[:, :, 2] < 50)
        return is_black.astype(np.uint8)

    def _extract_red_pos(self, img):
        if img is None:
            return 112.0, 112.0
        r = img[:, :, 0].astype(float)
        g = img[:, :, 1].astype(float)
        b = img[:, :, 2].astype(float)
        red_score = r - 0.5 * (g + b)
        red_score = np.maximum(red_score, 0)
        total = np.sum(red_score)
        if total < 1e-3:
            return float(img.shape[1] // 2), float(img.shape[0] // 2)
        
        ys, xs = np.indices(red_score.shape)
        x_c = np.sum(xs * red_score) / total
        y_c = np.sum(ys * red_score) / total
        return float(x_c), float(y_c)

    def fit(self, image_A, image_B, prev_state=None):
        xA, yA = self._extract_red_pos(image_A)
        xB, yB = self._extract_red_pos(image_B)
        self.state = {"x": xA, "y": yA}
        self.target_state = {"x": xB, "y": yB}
        self.wall_mask = self._extract_walls(image_A)
        
        if self.wall_mask is not None:
            h, w = self.wall_mask.shape
            mid_x = w // 2
            col_sums = np.sum(self.wall_mask, axis=0)
            divider_x = mid_x - 10 + np.argmax(col_sums[mid_x-10:mid_x+10])
            col = self.wall_mask[:, divider_x]
            gap_indices = np.where((col == 0) & (np.arange(h) > 15) & (np.arange(h) < h - 15))[0]
            if len(gap_indices) > 0:
                self.wall_x = float(divider_x)
                self.gap_y = float(np.mean(gap_indices))

    def _is_valid(self, x, y):
        if self.wall_mask is None:
            return True
        h, w = self.wall_mask.shape
        check_r = float(self.params["radius"] - self.params["wall_margin"])
        r_int = int(np.ceil(check_r))
        ix_min = max(0, int(np.floor(x - r_int)))
        ix_max = min(w - 1, int(np.ceil(x + r_int)))
        iy_min = max(0, int(np.floor(y - r_int)))
        iy_max = min(h - 1, int(np.ceil(y + r_int)))
        if ix_min > ix_max or iy_min > iy_max:
            return False
        sub_mask = self.wall_mask[iy_min:iy_max+1, ix_min:ix_max+1]
        if not np.any(sub_mask == 1):
            return True
        ys, xs = np.ogrid[iy_min:iy_max+1, ix_min:ix_max+1]
        dist_sq = (xs - x)**2 + (ys - y)**2
        if np.any((dist_sq <= check_r**2) & (sub_mask == 1)):
            return False
        return True

    def update(self, a):
        dx = a[0] * self.params["gain_x"]
        dy = a[1] * self.params["gain_y"]
        
        n_steps = max(1, int(np.ceil(max(abs(dx), abs(dy)) * 2)))
        step_dx = dx / n_steps
        step_dy = dy / n_steps
        
        x, y = self.state["x"], self.state["y"]
        for _ in range(n_steps):
            nx, ny = x + step_dx, y + step_dy
            if self._is_valid(nx, ny):
                x, y = nx, ny
            else:
                if self._is_valid(x + step_dx, y):
                    x = x + step_dx
                elif self._is_valid(x, y + step_dy):
                    y = y + step_dy
        self.state["x"], self.state["y"] = x, y

    def terminal_cost(self) -> float:
        x, y = self.state["x"], self.state["y"]
        tx, ty = self.target_state["x"], self.target_state["y"]
        
        # Geodesic path cost through central wall gap
        if (x - self.wall_x) * (tx - self.wall_x) < 0:
            d1 = np.sqrt((x - self.wall_x)**2 + (y - self.gap_y)**2)
            d2 = np.sqrt((tx - self.wall_x)**2 + (ty - self.gap_y)**2)
            return float(d1 + d2)
        else:
            dx = x - tx
            dy = y - ty
            return float(np.sqrt(dx*dx + dy*dy))

    def render_frame(self) -> np.ndarray:
        h, w = self.frame_size[1], self.frame_size[0]
        canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
        if self.wall_mask is not None and self.wall_mask.shape == (h, w):
            canvas[self.wall_mask == 1] = [0, 0, 0]
        
        cx, cy = int(round(self.state["x"])), int(round(self.state["y"]))
        cv2.circle(canvas, (cx, cy), int(self.params["radius"]), (255, 0, 0), -1)
        return canvas
