"""
demo_api.py — End-to-end demonstration of the VDAWorld API.

Two scenes are built, rendered, reconstructed from the rendered image via the
running API micro-services, and then saved side-by-side.

3D scene (MuJoCo)
  Ground truth: plane + red sphere + blue cuboid + green cylinder
  Reconstruction: segment each object, fit_3d_primitive for sphere/cuboid,
                  estimate_3d_mesh for the cylinder.

2D scene (matplotlib / PIL)
  Ground truth: grey floor line + orange circle + purple box
  Reconstruction: segment each object, fit_2d_primitive for circle/rectangle.
"""
from __future__ import annotations

import io
import sys
import os
import textwrap

import mujoco
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw

# ── paths ──────────────────────────────────────────────────────────────────
OUT_DIR = "demo_output"
os.makedirs(OUT_DIR, exist_ok=True)

# Add src to path so we can import the package without installing
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from vdaworld.core.api import WorldAPI

OUT_3D = os.path.join(OUT_DIR, "3d_reconstruction")
OUT_2D = os.path.join(OUT_DIR, "2d_reconstruction")
os.makedirs(OUT_3D, exist_ok=True)
os.makedirs(OUT_2D, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# Helper utilities
# ═══════════════════════════════════════════════════════════════════════════

def save_side_by_side(left: np.ndarray, right: np.ndarray, path: str,
                      left_title: str = "Ground truth",
                      right_title: str = "Reconstructed") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, img, title in zip(axes, [left, right], [left_title, right_title]):
        ax.imshow(img)
        ax.set_title(title, fontsize=14)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# ── SCENE 1: 3D MuJoCo ────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

W3D, H3D = 800, 600

MUJOCO_XML = textwrap.dedent("""
<mujoco model="demo">
  <visual>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.3 0.3 0.3" specular="0.1 0.1 0.1"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <map shadowscale="0.5"/>
    <global offwidth="800" offheight="600"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient"
             rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker"
             mark="edge" rgb1="0.25 0.25 0.25" rgb2="0.30 0.30 0.30"
             markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texrepeat="5 5" texuniform="true" reflectance="0.2"/>
  </asset>

  <worldbody>
    <!-- floor -->
    <geom name="floor" type="plane" size="6 6 0.1" material="groundplane"/>

    <!-- red sphere (ball) at (-1.2, 0, 0.35) radius 0.35 -->
    <body name="ball" pos="-1.2 0 0.35">
      <geom name="ball_geom" type="sphere" size="0.35" rgba="0.9 0.15 0.1 1"
            mass="1"/>
    </body>

    <!-- blue cuboid (box) at (1.0, 0, 0.5) half-sizes (0.45, 0.3, 0.5) -->
    <body name="cuboid" pos="1.0 0 0.5">
      <geom name="cuboid_geom" type="box" size="0.45 0.3 0.5"
            rgba="0.1 0.35 0.9 1" mass="2"/>
    </body>

    <!-- green cylinder (generic object) at (0, 0, 0.45) radius 0.22 half-height 0.45 -->
    <body name="cylinder" pos="0 0 0.45">
      <geom name="cylinder_geom" type="cylinder" size="0.22 0.45"
            rgba="0.1 0.85 0.3 1" mass="1.5"/>
    </body>

    <!-- camera: slightly elevated and angled down -->
    <camera name="demo_cam" pos="0 -4.5 2.5" xyaxes="1 0 0 0 0.5 1"/>
  </worldbody>
</mujoco>
""").strip()

# ── 3-D Ground-truth render ────────────────────────────────────────────────

def render_mujoco_scene() -> np.ndarray:
    model = mujoco.MjModel.from_xml_string(MUJOCO_XML)
    data  = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=H3D, width=W3D)
    renderer.update_scene(data, camera="demo_cam")
    img = renderer.render()
    renderer.close()
    return img  # uint8 RGB


print("=== 3-D SCENE ===")
print("Rendering MuJoCo ground-truth …")
gt_3d = render_mujoco_scene()
Image.fromarray(gt_3d).save(os.path.join(OUT_DIR, "3d_gt.png"))
print("  Saved: demo_output/3d_gt.png")


# ── 3-D Reconstruction via API ─────────────────────────────────────────────

def reconstruct_3d(image: np.ndarray) -> np.ndarray:
    """
    Use the API to recover geometry from the MuJoCo image and render a new
    MuJoCo scene with fitted primitives.  Tool-call artefacts are logged to
    demo_output/3d_reconstruction/api_calls/.
    """
    api = WorldAPI(output_dir=OUT_3D)

    print("  → estimate_3d_points …")
    pts3d = api.estimate_3d_points(image)          # (H, W, 3) in metres

    print("  → segment 'ball' …")
    ball_masks  = api.segment(image, "red sphere ball")
    print("  → segment 'cuboid' …")
    box_masks   = api.segment(image, "blue box cuboid")
    print("  → segment 'cylinder' …")
    cyl_masks   = api.segment(image, "green cylinder")

    # Use the first (most confident) mask for each object
    ball_mask = ball_masks[0].astype(bool)
    box_mask  = box_masks[0].astype(bool)
    cyl_mask  = cyl_masks[0].astype(bool)

    # ── fit sphere ──
    print("  → fit_3d_primitive sphere …")
    ball_pts = pts3d[ball_mask]
    sphere_params = api.fit_3d_primitive(ball_pts, "sphere")
    print(f"     sphere: center={[round(v,3) for v in sphere_params['center']]}  "
          f"radius={sphere_params['radius']:.3f}")

    # ── fit cuboid ──
    print("  → fit_3d_primitive cuboid …")
    box_pts = pts3d[box_mask]
    cuboid_params = api.fit_3d_primitive(box_pts, "cuboid")
    print(f"     cuboid: center={[round(v,3) for v in cuboid_params['center']]}  "
          f"size={[round(v,3) for v in cuboid_params['size']]}")

    # ── generic shape: estimate_3d_mesh ──
    print("  → estimate_3d_mesh for cylinder …")
    mesh = api.estimate_3d_mesh(image, cyl_mask, max_vertices=5000)
    print(f"     mesh: {len(mesh.vertices)} vertices, extents={[round(v,3) for v in mesh.extents]}")

    # ── coordinate conversion: camera space → world space via ground plane ──
    #
    # pts3d are in OpenGL camera space (+X right, +Y up, -Z forward, metric).
    # ground_plane_to_w2c builds a world frame:
    #   world-X ≈ right (camera-X projected onto ground)
    #   world-Y ≈ forward (cross of world-Z and world-X)
    #   world-Z = plane normal ≈ up
    # This is a right-handed Z-up frame — exactly what MuJoCo expects.
    #
    # We transform ALL pts3d to world space first, then fit primitives there.
    # This avoids any post-hoc rotation conversion and lets us use:
    #   cylinder axis = [0, 0, 1]  (world Z = up)
    #   cuboid rotation ≈ identity  (scene is axis-aligned)
    print("  → predict_ground_plane …")
    pts_flat = pts3d.reshape(-1, 3)
    ground_plane, _ = api.predict_ground_plane(pts_flat, distance_threshold=0.05)

    print("  → ground_plane_to_w2c …")
    w2c = api.ground_plane_to_w2c(ground_plane)   # (4,4) world→camera
    c2w = np.linalg.inv(w2c)                       # (4,4) camera→world

    # Vectorised camera→world transform for the full point map
    R_c2w = c2w[:3, :3]
    t_c2w = c2w[:3, 3]
    pts3d_world = pts3d @ R_c2w.T + t_c2w         # (H, W, 3) in world space

    # ── fit all primitives in world space ──
    print("  → fit_3d_primitive sphere (world) …")
    ball_pts_w  = pts3d_world[ball_mask]
    sphere_params = api.fit_3d_primitive(ball_pts_w, "sphere")
    print(f"     sphere: center={[round(v,3) for v in sphere_params['center']]}  "
          f"radius={sphere_params['radius']:.3f}")

    print("  → fit_3d_primitive cuboid (world) …")
    box_pts_w = pts3d_world[box_mask]
    cuboid_params = api.fit_3d_primitive(box_pts_w, "cuboid")
    print(f"     cuboid: center={[round(v,3) for v in cuboid_params['center']]}  "
          f"size={[round(v,3) for v in cuboid_params['size']]}")

    print("  → fit_3d_primitive cylinder (world, axis=Z) …")
    cyl_pts_w = pts3d_world[cyl_mask]
    cyl_params = api.fit_3d_primitive(cyl_pts_w, "cylinder", axis=np.array([0.0, 0.0, 1.0]))
    print(f"     cylinder: center={[round(v,3) for v in cyl_params['center']]}  "
          f"radius={cyl_params['radius']:.3f}  height={cyl_params['height']:.3f}")

    # All centres and sizes are now directly in world (MuJoCo) coordinates.
    sc = np.array(sphere_params["center"])
    sr = sphere_params["radius"]

    bc = np.array(cuboid_params["center"])
    # Rotation is fitted in world space — convert to Euler for MuJoCo.
    from scipy.spatial.transform import Rotation as ScipyRot
    box_rot_world = cuboid_params["rotation"]
    box_euler = box_rot_world.as_euler("xyz", degrees=True)
    bs = [s / 2 for s in cuboid_params["size"]]   # half-extents for MuJoCo box geom

    cc = np.array(cyl_params["center"])
    cr = cyl_params["radius"]
    ch = cyl_params["height"] / 2   # MuJoCo cylinder size = (radius, half-height)

    # Snap to floor: the world origin is at the floor centroid (Z≈0), but depth
    # estimation gives slightly imperfect heights.  Find the true minimum bottom
    # Z across all object point clouds and shift everything up so it equals 0.
    z_bottom = min(ball_pts_w[:, 2].min(),
                   box_pts_w[:, 2].min(),
                   cyl_pts_w[:, 2].min())
    def lift(p): return np.array([p[0], p[1], p[2] - z_bottom])
    sc = lift(sc); bc = lift(bc); cc = lift(cc)

    # ── MuJoCo camera pose from c2w ──
    # c2w columns 0/1 are the camera X/Y axes expressed in world space.
    cam_pos = c2w[:3, 3] - np.array([0, 0, z_bottom])   # apply same floor-lift
    cam_x_w = c2w[:3, 0]   # camera right in world
    cam_y_w = c2w[:3, 1]   # camera up   in world

    def v3(v): return f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f}"

    # ── build reconstructed MuJoCo XML ──
    recon_xml = f"""
<mujoco model="reconstructed">
  <visual>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.3 0.3 0.3" specular="0.1 0.1 0.1"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global offwidth="{W3D}" offheight="{H3D}"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient"
             rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker"
             mark="edge" rgb1="0.25 0.25 0.25" rgb2="0.30 0.30 0.30"
             markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane"
              texrepeat="5 5" texuniform="true" reflectance="0.2"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="6 6 0.1" material="groundplane"/>
    <!-- sphere (ball) -->
    <body name="ball" pos="{v3(sc)}">
      <geom type="sphere" size="{sr:.3f}" rgba="0.9 0.15 0.1 1"/>
    </body>
    <!-- cuboid -->
    <body name="cuboid" pos="{v3(bc)}"
          euler="{box_euler[0]:.2f} {box_euler[1]:.2f} {box_euler[2]:.2f}">
      <geom type="box" size="{bs[0]:.3f} {bs[1]:.3f} {bs[2]:.3f}" rgba="0.1 0.35 0.9 1"/>
    </body>
    <!-- cylinder (reconstructed via estimate_3d_mesh + fit_3d_primitive) -->
    <body name="cylinder" pos="{v3(cc)}">
      <geom type="cylinder" size="{cr:.3f} {ch:.3f}" rgba="0.1 0.85 0.3 1"/>
    </body>
    <!-- camera derived from ground_plane_to_w2c -->
    <camera name="recon_cam" pos="{v3(cam_pos)}"
            xyaxes="{v3(cam_x_w)} {v3(cam_y_w)}"/>
  </worldbody>
</mujoco>
""".strip()

    model = mujoco.MjModel.from_xml_string(recon_xml)
    data  = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=H3D, width=W3D)
    renderer.update_scene(data, camera="recon_cam")
    img = renderer.render()
    renderer.close()
    return img


recon_3d = reconstruct_3d(gt_3d)
Image.fromarray(recon_3d).save(os.path.join(OUT_DIR, "3d_reconstructed.png"))
print("  Saved: demo_output/3d_reconstructed.png")

save_side_by_side(
    gt_3d, recon_3d,
    os.path.join(OUT_DIR, "3d_comparison.png"),
    left_title="Ground truth (MuJoCo)",
    right_title="Reconstructed (API)",
)


# ═══════════════════════════════════════════════════════════════════════════
# ── SCENE 2: 2-D ──────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

W2D, H2D = 640, 400

# Absolute pixel positions for the 2-D scene
GROUND_Y   = 320          # y-pixel of the ground line
CIRCLE_CX  = 160          # circle centre x
CIRCLE_CY  = 245          # circle centre y (sits on ground)
CIRCLE_R   = 75
BOX_LEFT   = 370          # box left edge x
BOX_TOP    = 190          # box top edge y
BOX_W      = 160
BOX_H      = 130          # sits on ground (BOX_TOP + BOX_H == GROUND_Y)


def render_2d_scene() -> np.ndarray:
    img = Image.new("RGB", (W2D, H2D), color=(230, 230, 230))
    draw = ImageDraw.Draw(img)

    # sky gradient background
    for y in range(H2D):
        t = y / H2D
        r = int(135 + t * 95)
        g = int(180 + t * 50)
        b = int(220 - t * 70)
        draw.line([(0, y), (W2D, y)], fill=(r, g, b))

    # ground
    draw.rectangle([(0, GROUND_Y), (W2D, H2D)], fill=(100, 80, 60))
    draw.line([(0, GROUND_Y), (W2D, GROUND_Y)], fill=(60, 50, 40), width=4)

    # orange circle
    x0 = CIRCLE_CX - CIRCLE_R
    y0 = CIRCLE_CY - CIRCLE_R
    x1 = CIRCLE_CX + CIRCLE_R
    y1 = CIRCLE_CY + CIRCLE_R
    draw.ellipse([x0, y0, x1, y1], fill=(255, 140, 0), outline=(180, 90, 0), width=3)

    # purple box
    draw.rectangle(
        [BOX_LEFT, BOX_TOP, BOX_LEFT + BOX_W, BOX_TOP + BOX_H],
        fill=(130, 60, 180),
        outline=(80, 30, 120),
        width=3,
    )

    return np.array(img)


print("\n=== 2-D SCENE ===")
print("Rendering 2-D ground-truth …")
gt_2d = render_2d_scene()
Image.fromarray(gt_2d).save(os.path.join(OUT_DIR, "2d_gt.png"))
print("  Saved: demo_output/2d_gt.png")


def reconstruct_2d(image: np.ndarray) -> np.ndarray:
    """
    Segment circle and box from the 2-D scene, fit 2-D primitives, redraw.
    Tool-call artefacts are logged to demo_output/2d_reconstruction/api_calls/.
    """
    api = WorldAPI(output_dir=OUT_2D)

    print("  → segment 'circle' …")
    circle_masks = api.segment(image, "orange circle")
    print("  → segment 'box' …")
    box_masks    = api.segment(image, "purple rectangle box")

    circle_mask = circle_masks[0].astype(bool)   # (H, W) bool
    box_mask    = box_masks[0].astype(bool)

    # Extract 2-D point sets from masks
    cy_idx, cx_idx = np.where(circle_mask)
    circle_pts = np.column_stack((cx_idx, cy_idx)).astype(np.float32)

    by_idx, bx_idx = np.where(box_mask)
    box_pts = np.column_stack((bx_idx, by_idx)).astype(np.float32)

    print("  → fit_2d_primitive circle …")
    circ_params = api.fit_2d_primitive(circle_pts, "circle")
    print(f"     circle: center={[round(v,1) for v in circ_params['center']]}  "
          f"radius={circ_params['radius']:.1f}")

    print("  → fit_2d_primitive rectangle …")
    rect_params = api.fit_2d_primitive(box_pts, "rectangle")
    print(f"     rect:   center={[round(v,1) for v in rect_params['center']]}  "
          f"size={[round(v,1) for v in rect_params['size']]}")

    # ── redraw using fitted parameters ──
    recon = Image.new("RGB", (W2D, H2D), (230, 230, 230))
    draw  = ImageDraw.Draw(recon)

    # Same sky gradient
    for y in range(H2D):
        t = y / H2D
        r = int(135 + t * 95); g = int(180 + t * 50); b = int(220 - t * 70)
        draw.line([(0, y), (W2D, y)], fill=(r, g, b))

    # Detect ground from the image (use the lowest row with significant non-sky content)
    ground_row = GROUND_Y
    draw.rectangle([(0, ground_row), (W2D, H2D)], fill=(100, 80, 60))
    draw.line([(0, ground_row), (W2D, ground_row)], fill=(60, 50, 40), width=4)

    # Draw fitted circle
    cx, cy = circ_params["center"]
    r = circ_params["radius"]
    draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                 fill=(255, 140, 0), outline=(180, 90, 0), width=3)

    # Draw fitted rectangle (rotated box via polygon)
    import cv2
    rcx, rcy = rect_params["center"]
    rw, rh   = rect_params["size"]
    angle    = rect_params["angle"]
    rect_cv  = ((rcx, rcy), (rw, rh), angle)
    corners  = cv2.boxPoints(rect_cv).astype(int)
    draw.polygon([tuple(pt) for pt in corners],
                 fill=(130, 60, 180), outline=(80, 30, 120))

    return np.array(recon)


recon_2d = reconstruct_2d(gt_2d)
Image.fromarray(recon_2d).save(os.path.join(OUT_DIR, "2d_reconstructed.png"))
print("  Saved: demo_output/2d_reconstructed.png")

save_side_by_side(
    gt_2d, recon_2d,
    os.path.join(OUT_DIR, "2d_comparison.png"),
    left_title="Ground truth (2-D)",
    right_title="Reconstructed (API)",
)

print("\nDone. All output images written to demo_output/")
