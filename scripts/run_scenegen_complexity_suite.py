#!/usr/bin/env python3
"""Benchmark SceneGen as scene object count increases."""

from __future__ import annotations

import gc
import json
import os
import time
from contextlib import redirect_stdout
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
from scenegen.utils.inference_scene import split_rgb_mask


ROOT = Path("/opt/scenegen/SceneGen")
INPUT = ROOT / "complexity_inputs"
OUTPUT = ROOT / "complexity_outputs"

SCENES = [
    {
        "id": "scene_06",
        "labels": ["floor_lamp", "sofa", "coffee_table", "area_rug", "potted_plant", "armchair"],
        "boxes": [[140, 115, 365, 590], [255, 310, 925, 565], [325, 465, 830, 645], [0, 505, 1235, 715], [870, 60, 1110, 550], [0, 340, 210, 640]],
    },
    {
        "id": "scene_09",
        "labels": ["shelving_unit", "tool_chest", "floor_lamp", "workbench", "stool", "area_rug", "potted_plant", "storage_cabinet", "wooden_crate"],
        "boxes": [[115, 120, 275, 565], [0, 400, 185, 650], [265, 190, 445, 575], [330, 355, 855, 555], [475, 430, 630, 590], [85, 535, 1145, 715], [805, 130, 995, 550], [990, 125, 1190, 600], [1065, 495, 1275, 685]],
    },
    {
        "id": "scene_12",
        "labels": ["bookcase", "armchair_left", "sofa", "coffee_table", "armchair_right", "floor_lamp", "potted_plant", "media_console", "side_table", "area_rug", "ottoman", "storage_chest"],
        "boxes": [[170, 140, 335, 535], [30, 385, 305, 630], [325, 340, 780, 540], [385, 450, 725, 595], [755, 375, 1015, 575], [585, 140, 830, 445], [780, 135, 1020, 430], [975, 275, 1275, 615], [245, 410, 350, 540], [35, 495, 1125, 685], [785, 500, 1010, 655], [1110, 510, 1278, 690]],
    },
]


def signature(geometry) -> tuple[int, int]:
    return len(geometry.vertices), len(geometry.faces)


def export_labeled_assets(scene, generated_assets, labels, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    source_signatures = []
    for index, asset in enumerate(generated_assets):
        geom = asset.to_geometry() if isinstance(asset, trimesh.Scene) else asset
        source_signatures.append((index, signature(geom)))

    unused = set(range(len(source_signatures)))
    exported = []
    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph.get(node_name)
        geometry = scene.geometry[geometry_name].copy()
        sig = signature(geometry)
        candidates = [index for index in unused if source_signatures[index][1] == sig]
        index = candidates[0] if candidates else min(unused)
        unused.remove(index)
        geometry.apply_transform(transform)
        label = labels[index]
        path = output_dir / f"{index:02d}_{label}.glb"
        geometry.export(path)
        exported.append(
            {
                "index": index,
                "label": label,
                "path": str(path),
                "vertices": sig[0],
                "faces": sig[1],
                "bounds": geometry.bounds.tolist(),
            }
        )
    return sorted(exported, key=lambda item: item["index"])


def clear_cuda():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    suite_started = time.perf_counter()

    sam_started = time.perf_counter()
    sam = build_sam2("configs/sam2/sam2_hiera_l.yaml", str(ROOT / "checkpoints/sam2-hiera-large/sam2_hiera_large.pt"))
    predictor = SAM2ImagePredictor(sam)
    prepared = {}
    for config in SCENES:
        started = time.perf_counter()
        image = Image.open(INPUT / f"{config['id']}.png").convert("RGB")
        detections = segment(predictor, image, boxes=[config["boxes"]], polygon_refinement=False)
        segmentation = plot_segmentation(image, detections)
        scene_dir = OUTPUT / config["id"]
        scene_dir.mkdir(exist_ok=True)
        segmentation.save(scene_dir / "segmentation.png")
        prepared[config["id"]] = (image, segmentation)
        config["segmentation_seconds"] = round(time.perf_counter() - started, 3)
    sam_total = time.perf_counter() - sam_started
    del predictor, sam
    clear_cuda()

    load_started = time.perf_counter()
    pipeline = SceneGenImageToScenePipeline.from_pretrained(str(ROOT / "checkpoints/scenegen"))
    pipeline.cuda()
    pipeline_load_seconds = time.perf_counter() - load_started

    results_report = []
    for config in SCENES:
        clear_cuda()
        image, segmentation = prepared[config["id"]]
        instance_rgbs, instance_masks, scene_rgbs = split_rgb_mask(image, segmentation, order="order")
        run = {
            "id": config["id"],
            "requested_objects": len(config["labels"]),
            "labels": config["labels"],
            "segmentation_seconds": config["segmentation_seconds"],
            "settings": {"ss_steps": 20, "slat_steps": 15, "simplify": 0.95, "texture_size": 512, "seed": 42},
        }
        started = time.perf_counter()
        try:
            generated = pipeline.run_scene(
                image=instance_rgbs,
                mask_image=instance_masks,
                scene_image=scene_rgbs[0],
                seed=42,
                sparse_structure_sampler_params={"steps": 20, "cfg_strength": 3.0, "cfg_interval": [0.0, 1.0], "rescale_t": 1.0},
                slat_sampler_params={"steps": 15, "cfg_strength": 5.0, "cfg_interval": [0.5, 1.0], "rescale_t": 3.0},
                positions_type="last",
                simplify=0.95,
                texture_size=512,
            )
            run["generation_seconds"] = round(time.perf_counter() - started, 3)
            run["peak_vram_mib"] = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
            run["status"] = "completed"
            scene_dir = OUTPUT / config["id"]
            generated["scene"].export(scene_dir / "combined.glb")
            run["objects"] = export_labeled_assets(generated["scene"], generated["assets"], config["labels"], scene_dir / "assets")
            (scene_dir / "run.json").write_text(json.dumps(run, indent=2))
            del generated
        except torch.OutOfMemoryError as exc:
            run["generation_seconds"] = round(time.perf_counter() - started, 3)
            run["peak_vram_mib"] = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
            run["status"] = "oom"
            run["error"] = str(exc)
            clear_cuda()
        except Exception as exc:
            run["generation_seconds"] = round(time.perf_counter() - started, 3)
            run["status"] = "error"
            run["error"] = repr(exc)
        results_report.append(run)
        (OUTPUT / "benchmark.partial.json").write_text(json.dumps(results_report, indent=2))

    report = {
        "pipeline_load_seconds": round(pipeline_load_seconds, 3),
        "sam_total_seconds": round(sam_total, 3),
        "total_seconds": round(time.perf_counter() - suite_started, 3),
        "gpu": torch.cuda.get_device_name(0),
        "runs": results_report,
    }
    (OUTPUT / "benchmark.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
