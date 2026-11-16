import json
import shutil
from pathlib import Path

import numpy as np
import trimesh


ROOT = Path(__file__).parent / "artifacts/scenegen_zen_room_test"
SOURCE_ASSETS = ROOT / "assets"
PCG_ASSETS = ROOT / "pcg_assets"
OUTPUT = ROOT / "scene.json"

# Visual inspection of the generated geometry shows that trimesh's generated
# geometry order is plant, lamp, bench, rug (not the original prompt order).
OBJECTS = [
    {
        "id": "plant_01",
        "label": "potted_plant",
        "category": "vegetation.indoor.potted_plant",
        "source": "00_lamp.glb",
        "output": "plant_01.glb",
        "anchor": "floor",
        "segmentation_id": 3,
        "segmentation_color_rgb": [0, 0, 255],
        "box_px": [785, 100, 1025, 525],
        "confidence": 0.93,
        "pcg_tags": ["vegetation", "indoor", "prop", "floor_supported"],
    },
    {
        "id": "lamp_01",
        "label": "hanging_lamp",
        "category": "lighting.ceiling.pendant",
        "source": "01_bench.glb",
        "output": "lamp_01.glb",
        "anchor": "ceiling",
        "segmentation_id": 1,
        "segmentation_color_rgb": [255, 0, 0],
        "box_px": [480, 25, 655, 205],
        "confidence": 0.91,
        "pcg_tags": ["lighting", "indoor", "ceiling_supported"],
    },
    {
        "id": "bench_01",
        "label": "wooden_bench",
        "category": "furniture.seating.bench",
        "source": "02_plant.glb",
        "output": "bench_01.glb",
        "anchor": "floor",
        "segmentation_id": 2,
        "segmentation_color_rgb": [0, 255, 0],
        "box_px": [345, 405, 790, 525],
        "confidence": 0.96,
        "pcg_tags": ["furniture", "seating", "wood", "floor_supported"],
    },
    {
        "id": "rug_01",
        "label": "area_rug",
        "category": "surface.floor.rug",
        "source": "03_rug.glb",
        "output": "rug_01.glb",
        "anchor": "floor_surface",
        "segmentation_id": 4,
        "segmentation_color_rgb": [255, 255, 0],
        "box_px": [45, 515, 1145, 690],
        "confidence": 0.97,
        "pcg_tags": ["surface", "rug", "walkable_overlay", "floor_supported"],
    },
]

# Single-image metric calibration. Kept explicit so Unreal can override it.
ASSUMED_BENCH_WIDTH_M = 2.0
ROOM_HALF_EXTENT_SOURCE = 1.05


def round_vec(values, digits=4):
    return [round(float(value), digits) for value in values]


def source_to_unreal_cm(point, meters_per_source_unit, floor_y):
    """SceneGen RH Y-up -> Unreal LH Z-up: X=-source Z, Y=source X."""
    x, y, z = point
    return np.array(
        [
            -z * meters_per_source_unit * 100.0,
            x * meters_per_source_unit * 100.0,
            (y - floor_y) * meters_per_source_unit * 100.0,
        ]
    )


def load_mesh(path):
    scene = trimesh.load(path, force="scene")
    return scene.to_geometry()


def main():
    PCG_ASSETS.mkdir(parents=True, exist_ok=True)
    loaded = []
    for spec in OBJECTS:
        mesh = load_mesh(SOURCE_ASSETS / spec["source"])
        loaded.append((spec, mesh, mesh.bounds.copy()))

    bench_bounds = next(bounds for spec, _, bounds in loaded if spec["id"] == "bench_01")
    bench_width_source = float(bench_bounds[1, 0] - bench_bounds[0, 0])
    meters_per_source_unit = ASSUMED_BENCH_WIDTH_M / bench_width_source

    rug_bounds = next(bounds for spec, _, bounds in loaded if spec["id"] == "rug_01")
    floor_y = float(np.mean(rug_bounds[:, 1]))
    lamp_bounds = next(bounds for spec, _, bounds in loaded if spec["id"] == "lamp_01")
    ceiling_y = float(lamp_bounds[1, 1])
    room_height_m = (ceiling_y - floor_y) * meters_per_source_unit
    half_m = ROOM_HALF_EXTENT_SOURCE * meters_per_source_unit

    scene_objects = []
    corrections = []
    for spec, mesh, bounds in loaded:
        center = bounds.mean(axis=0)
        if spec["anchor"] == "ceiling":
            raw_pivot = np.array([center[0], bounds[1, 1], center[2]])
            corrected_pivot = raw_pivot.copy()
            corrected_pivot[1] = ceiling_y
        elif spec["anchor"] == "floor_surface":
            raw_pivot = np.array([center[0], floor_y, center[2]])
            corrected_pivot = raw_pivot.copy()
        else:
            raw_pivot = np.array([center[0], bounds[0, 1], center[2]])
            corrected_pivot = raw_pivot.copy()
            corrected_pivot[1] = floor_y

        correction_source = corrected_pivot - raw_pivot
        raw_location_cm = source_to_unreal_cm(raw_pivot, meters_per_source_unit, floor_y)
        location_cm = source_to_unreal_cm(corrected_pivot, meters_per_source_unit, floor_y)

        # Export a metric, locally pivoted GLB. Orientation is baked into its vertices.
        local_mesh = mesh.copy()
        local_mesh.apply_translation(-raw_pivot)
        local_mesh.apply_scale(meters_per_source_unit)
        local_path = PCG_ASSETS / spec["output"]
        local_mesh.export(local_path)

        source_dimensions_cm = (bounds[1] - bounds[0]) * meters_per_source_unit * 100.0
        # Match the target mapping Xue=-Zsg, Yue=Xsg, Zue=Ysg.
        dimensions_cm = source_dimensions_cm[[2, 0, 1]]
        world_bounds_min_cm = source_to_unreal_cm(bounds[0] + correction_source, meters_per_source_unit, floor_y)
        world_bounds_max_cm = source_to_unreal_cm(bounds[1] + correction_source, meters_per_source_unit, floor_y)
        collision_min = np.minimum(world_bounds_min_cm, world_bounds_max_cm)
        collision_max = np.maximum(world_bounds_min_cm, world_bounds_max_cm)

        correction_cm = location_cm - raw_location_cm
        if np.linalg.norm(correction_cm) > 0.01:
            corrections.append(
                {
                    "objectId": spec["id"],
                    "type": "snap_to_anchor",
                    "translationCm": round_vec(correction_cm),
                    "reason": f"Snap generated {spec['anchor']} pivot to inferred room surface",
                }
            )

        scene_objects.append(
            {
                "id": spec["id"],
                "label": spec["label"],
                "category": spec["category"],
                "asset": {
                    "uri": f"pcg_assets/{spec['output']}",
                    "format": "glb",
                    "units": "meters",
                    "textures": "embedded",
                    "orientationBaked": True,
                },
                "transform": {
                    "locationCm": round_vec(location_cm),
                    "rotationEulerDeg": [0.0, 0.0, 0.0],
                    "scale": [1.0, 1.0, 1.0],
                },
                "rawGeneratedTransform": {
                    "locationCm": round_vec(raw_location_cm),
                    "sourcePivot": round_vec(raw_pivot),
                },
                "dimensionsCm": round_vec(dimensions_cm),
                "anchor": {"type": spec["anchor"], "surfaceId": "ceiling_01" if spec["anchor"] == "ceiling" else "floor_01"},
                "collision": {
                    "type": "aabb",
                    "minCm": round_vec(collision_min),
                    "maxCm": round_vec(collision_max),
                },
                "sourceSegmentation": {
                    "instanceId": spec["segmentation_id"],
                    "colorRgb": spec["segmentation_color_rgb"],
                    "boxPixelsXyxy": spec["box_px"],
                    "labelConfidence": spec["confidence"],
                    "method": "SAM2 box prompt plus geometry inspection",
                },
                "pcg": {
                    "enabled": True,
                    "spawnMode": "fixed_seed_asset",
                    "tags": spec["pcg_tags"],
                    "allowReplacement": True,
                    "placementPriority": 100,
                },
            }
        )

    width_cm = half_m * 200.0
    depth_cm = width_cm
    height_cm = room_height_m * 100.0
    wall_thickness_cm = 10.0
    room_surfaces = [
        {
            "id": "floor_01",
            "type": "floor",
            "transform": {"locationCm": [0.0, 0.0, 0.0], "rotationEulerDeg": [0.0, 0.0, 0.0]},
            "dimensionsCm": [round(depth_cm, 3), round(width_cm, 3), 2.0],
            "normal": [0.0, 0.0, 1.0],
            "pcgTags": ["ground", "walkable", "indoor", "spawn_surface"],
        },
        {
            "id": "ceiling_01",
            "type": "ceiling",
            "transform": {"locationCm": [0.0, 0.0, round(height_cm, 3)], "rotationEulerDeg": [0.0, 0.0, 0.0]},
            "dimensionsCm": [round(depth_cm, 3), round(width_cm, 3), 2.0],
            "normal": [0.0, 0.0, -1.0],
            "pcgTags": ["ceiling", "indoor", "attachment_surface"],
        },
        {
            "id": "wall_back_01",
            "type": "wall",
            "transform": {"locationCm": [round(half_m * 100, 3), 0.0, round(height_cm / 2, 3)], "rotationEulerDeg": [0.0, 0.0, 0.0]},
            "dimensionsCm": [wall_thickness_cm, round(width_cm, 3), round(height_cm, 3)],
            "normal": [-1.0, 0.0, 0.0],
            "pcgTags": ["wall", "indoor", "attachment_surface"],
        },
        {
            "id": "wall_left_01",
            "type": "wall",
            "transform": {"locationCm": [0.0, round(-half_m * 100, 3), round(height_cm / 2, 3)], "rotationEulerDeg": [0.0, 0.0, 0.0]},
            "dimensionsCm": [round(depth_cm, 3), wall_thickness_cm, round(height_cm, 3)],
            "normal": [0.0, 1.0, 0.0],
            "pcgTags": ["wall", "indoor", "attachment_surface"],
        },
        {
            "id": "wall_right_01",
            "type": "wall",
            "transform": {"locationCm": [0.0, round(half_m * 100, 3), round(height_cm / 2, 3)], "rotationEulerDeg": [0.0, 0.0, 0.0]},
            "dimensionsCm": [round(depth_cm, 3), wall_thickness_cm, round(height_cm, 3)],
            "normal": [0.0, -1.0, 0.0],
            "pcgTags": ["wall", "indoor", "attachment_surface"],
        },
    ]

    payload = {
        "$schema": "pcg_scene.schema.json",
        "schemaVersion": "1.0.0",
        "sceneId": "zen_room_scenegen_test_001",
        "source": {
            "image": "../wavespeed_relaxed_zen_room_v1/anchor.png",
            "imageSizePixels": [1300, 720],
            "generator": "SceneGen",
            "seed": 42,
            "checkpoint": "haoningwu/SceneGen",
        },
        "coordinateSystem": {
            "target": "UnrealEngine",
            "handedness": "left",
            "upAxis": "+Z",
            "forwardAxis": "+X",
            "rightAxis": "+Y",
            "units": "centimeters",
            "sourceToTarget": "Xue=-Zsg, Yue=Xsg, Zue=Ysg-floorY",
        },
        "metricCalibration": {
            "metersPerSceneGenUnit": round(meters_per_source_unit, 6),
            "method": "known_size_prior",
            "reference": {"objectId": "bench_01", "assumedWidthMeters": ASSUMED_BENCH_WIDTH_M},
            "confidence": 0.62,
            "userOverrideRecommended": True,
        },
        "room": {
            "type": "box_room_open_front",
            "dimensionsCm": [round(depth_cm, 3), round(width_cm, 3), round(height_cm, 3)],
            "surfaces": room_surfaces,
            "openings": [
                {
                    "id": "opening_right_01",
                    "surfaceId": "wall_right_01",
                    "type": "window_or_glazed_door",
                    "sourceBoxPixelsXyxy": [1150, 0, 1300, 720],
                    "geometryStatus": "unresolved_from_single_view",
                    "enabledForBooleanCut": False,
                    "confidence": 0.58,
                },
                {
                    "id": "opening_front_01",
                    "type": "open_room_front",
                    "geometryStatus": "room_prior",
                    "enabledForBooleanCut": True,
                    "confidence": 0.9,
                },
            ],
            "inference": {
                "method": "single_image_box_room_prior",
                "confidence": 0.64,
                "requiresMetricCalibration": True,
            },
        },
        "objects": scene_objects,
        "initialConditions": {
            "deterministicSeed": 42,
            "groundSurfaceId": "floor_01",
            "collisionPolicy": "reject_overlap_then_project_to_anchor",
            "placementRules": [
                {"tag": "floor_supported", "operation": "snap_bottom_to_surface", "surfaceTag": "ground", "offsetCm": 0.0},
                {"tag": "ceiling_supported", "operation": "snap_top_to_surface", "surfaceTag": "ceiling", "offsetCm": 0.0},
                {"tag": "walkable_overlay", "operation": "conform_to_surface", "surfaceTag": "ground", "offsetCm": 0.5},
            ],
            "pcgInputs": {
                "fixedObjects": [obj["id"] for obj in scene_objects],
                "spawnSurfaces": ["floor_01"],
                "exclusionVolumesFromObjects": True,
                "wallAttachmentSurfaces": ["wall_back_01", "wall_left_01", "wall_right_01"],
                "unresolvedOpenings": ["opening_right_01"],
            },
        },
        "postprocess": {
            "correctionsApplied": corrections,
            "orientation": "baked_into_each_metric_glb",
            "textures": "embedded_in_each_glb",
        },
        "quality": {
            "objectsGenerated": 4,
            "objectsLabeled": 4,
            "roomGeometryGeneratedByModel": False,
            "roomGeometryInferredByPostprocess": True,
            "warnings": [
                "Absolute scale is inferred from an assumed 2.0 m bench width.",
                "Wall geometry is a box-room prior, not direct SceneGen output.",
                "The right-side glazed opening needs multi-view or user calibration before boolean cutting.",
            ],
        },
    }

    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
