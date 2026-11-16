from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pyrender
import trimesh


ROOT = Path(__file__).parent / "artifacts/scenegen_zen_room_test"


def camera_pose(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    backward = eye - target
    backward /= np.linalg.norm(backward)
    right = np.cross(np.array([0.0, 0.0, 1.0]), backward)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = backward
    pose[:3, 3] = eye
    return pose


def render(path: Path, output: Path, angle: float = 0.0) -> None:
    source = trimesh.load(path, force="scene")
    bounds = source.bounds
    center = bounds.mean(axis=0)
    extent = bounds[1] - bounds[0]
    radius = max(float(np.linalg.norm(extent)), 0.5)

    scene = pyrender.Scene.from_trimesh_scene(
        source,
        bg_color=np.array([236, 233, 226, 255]),
        ambient_light=np.array([0.7, 0.7, 0.7, 1.0]),
    )
    eye = center + radius * np.array(
        [1.35 * np.cos(angle), 1.35 * np.sin(angle), 0.8]
    )
    scene.add(
        pyrender.PerspectiveCamera(yfov=np.deg2rad(42.0)),
        pose=camera_pose(eye, center),
    )
    light_pose = camera_pose(eye + np.array([0.0, 0.0, radius]), center)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), pose=light_pose)
    renderer = pyrender.OffscreenRenderer(900, 700)
    color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()
    iio.imwrite(output, color)


render(ROOT / "zen_room_combined.glb", ROOT / "combined_front.png", angle=-1.15)
render(ROOT / "zen_room_combined.glb", ROOT / "combined_side.png", angle=0.2)
for asset in sorted((ROOT / "assets").glob("*.glb")):
    render(asset, ROOT / f"preview_{asset.stem}.png", angle=-1.0)
