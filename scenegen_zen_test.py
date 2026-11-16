import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

os.environ["ATTN_BACKEND"] = "xformers"
os.environ["SPCONV_ALGO"] = "native"

from scenegen.pipelines import SceneGenImageToScenePipeline
from scenegen.utils.grounding_sam import plot_segmentation, segment
from scenegen.utils.inference_scene import run_scene


ROOT = Path("/opt/scenegen/SceneGen")
INPUT = ROOT / "test_inputs/zen_room_anchor.png"
OUTPUT = ROOT / "test_outputs/zen_room"

# Boxes are xyxy coordinates for the 1300x720 source image.
BOXES = [
    [480, 25, 655, 205],     # hanging lamp
    [345, 405, 790, 525],    # wooden bench
    [785, 100, 1025, 525],   # plant and basket
    [45, 515, 1145, 690],    # rug
]
LABELS = ["lamp", "bench", "plant", "rug"]


def export_individual_glbs(scene: trimesh.Scene, output_dir: Path) -> list[dict]:
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    exported = []
    for index, node_name in enumerate(scene.graph.nodes_geometry):
        transform, geometry_name = scene.graph.get(node_name)
        geometry = scene.geometry[geometry_name].copy()
        geometry.apply_transform(transform)
        safe_name = LABELS[index] if index < len(LABELS) else f"asset_{index:02d}"
        path = assets_dir / f"{index:02d}_{safe_name}.glb"
        geometry.export(path)
        exported.append(
            {
                "index": index,
                "label": safe_name,
                "node": str(node_name),
                "geometry": str(geometry_name),
                "path": str(path),
                "vertices": int(len(geometry.vertices)),
                "faces": int(len(geometry.faces)),
                "bounds": geometry.bounds.tolist(),
            }
        )
    return exported


def main() -> None:
    started = time.time()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    image = Image.open(INPUT).convert("RGB")

    sam = build_sam2(
        "configs/sam2/sam2_hiera_l.yaml",
        str(ROOT / "checkpoints/sam2-hiera-large/sam2_hiera_large.pt"),
    )
    predictor = SAM2ImagePredictor(sam)
    detections = segment(predictor, image, boxes=[BOXES], polygon_refinement=False)
    segmentation = plot_segmentation(image, detections)
    segmentation.save(OUTPUT / "segmentation.png")

    del predictor, sam
    torch.cuda.empty_cache()

    pipeline = SceneGenImageToScenePipeline.from_pretrained(
        str(ROOT / "checkpoints/scenegen")
    )
    pipeline.cuda()

    scene = run_scene(
        pipeline,
        [image],
        [segmentation],
        seed=42,
        ss_num_inference_steps=20,
        ss_cfg_strength=3.0,
        slat_num_inference_steps=15,
        slat_cfg_strength=5.0,
        order="order",
        positions_type="last",
        simplify=0.95,
        texture_size=512,
    )

    combined_path = OUTPUT / "zen_room_combined.glb"
    scene.export(combined_path)
    assets = export_individual_glbs(scene, OUTPUT)
    report = {
        "input": str(INPUT),
        "combined_glb": str(combined_path),
        "elapsed_seconds": round(time.time() - started, 2),
        "gpu": torch.cuda.get_device_name(0),
        "seed": 42,
        "boxes": dict(zip(LABELS, BOXES)),
        "asset_count": len(assets),
        "assets": assets,
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
