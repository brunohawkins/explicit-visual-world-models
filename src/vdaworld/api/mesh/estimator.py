import os
import sys
import logging
import numpy as np
from trimesh import Trimesh
import torch

SAM3D_PATH = os.environ.get("SAM3D_PATH")
if SAM3D_PATH not in sys.path:
    sys.path.append(SAM3D_PATH)
if f"{SAM3D_PATH}/notebook" not in sys.path:
    sys.path.append(f"{SAM3D_PATH}/notebook")

from inference import Inference
from sam3d_objects.utils.visualization import SceneVisualizer
import manifold3d

_MODEL_CACHE = None
logger = logging.getLogger(__name__)


def get_model():
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        import logging

        logging.info("Loading SAM3D model...")
        tag = "hf"
        config_path = f"{SAM3D_PATH}/checkpoints/{tag}/pipeline.yaml"
        _MODEL_CACHE = Inference(config_path, compile=False)
    return _MODEL_CACHE


def make_manifold(mesh, refine_steps=0):
    if not isinstance(mesh, Trimesh):
        raise ValueError("Input must be a trimesh.Trimesh object")
    verts = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
    faces = np.ascontiguousarray(mesh.faces, dtype=np.uint32)
    m_mesh = manifold3d.Mesh(vert_properties=verts, tri_verts=faces)
    m_obj = manifold3d.Manifold(m_mesh)
    if refine_steps > 0:
        m_obj = m_obj.refine(refine_steps)
    out_mesh_data = m_obj.to_mesh()
    if len(out_mesh_data.vert_properties) == 0:
        return mesh
    return Trimesh(
        vertices=out_mesh_data.vert_properties,
        faces=out_mesh_data.tri_verts,
        process=True,
    )


def remove_interior_surfaces(mesh: Trimesh):
    components = mesh.split(only_watertight=False)
    if len(components) == 1:
        return mesh
    components = sorted(components, key=lambda m: m.volume, reverse=True)
    return components[0]


def estimate(image: np.ndarray, mask: np.ndarray, max_vertices: int = 10000) -> Trimesh:
    inference = get_model()
    # Mask is expected as np bool array or uint8
    output = inference(image, mask, seed=42)

    mesh = output["glb"]
    mesh_vertices = mesh.vertices
    mesh_to_splat = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    mesh_vertices = mesh_vertices @ mesh_to_splat
    verts = (
        SceneVisualizer.object_pointcloud(
            points_local=torch.tensor(mesh_vertices)
            .float()
            .unsqueeze(0)
            .to(output["rotation"].device),
            quat_l2c=output["rotation"],
            trans_l2c=output["translation"],
            scale_l2c=output["scale"],
        )
        .points_list()[0]
        .detach()
        .cpu()
        .numpy()
    )
    coordinate_transform_mat = np.array(
        [[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32
    )
    verts = verts @ coordinate_transform_mat
    mesh.vertices = verts

    mesh = remove_interior_surfaces(mesh)
    try:
        mesh = mesh.simplify_quadric_decimation(
            face_count=min(max_vertices, len(mesh.faces))
        )
    except ModuleNotFoundError as exc:
        if exc.name != "fast_simplification":
            raise
        logger.warning(
            "Skipping mesh simplification because optional dependency "
            "'fast_simplification' is not installed."
        )
    mesh = make_manifold(mesh, refine_steps=0)

    return mesh
