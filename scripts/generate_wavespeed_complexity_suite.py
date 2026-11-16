#!/usr/bin/env python3
"""Generate controlled interior-scene complexity inputs with WaveSpeed."""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


API_ROOT = "https://api.wavespeed.ai/api/v3"
MODEL = "wavespeed-ai/qwen-image/text-to-image-2512"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "wavespeed_scenegen_complexity_suite"

COMMON = (
    "Photorealistic interior architectural visualization, simple rectangular room, "
    "all objects fully visible and spatially separated, no object occluding another, "
    "eye-level wide shot, 24mm lens, deep focus, soft diffuse daylight, even exposure, "
    "physically plausible PBR materials, fixed clean geometry. No people, no animals, "
    "no text, no logo, no mirror, no clutter, no extra furniture, no duplicated objects. "
)

SCENES = [
    {
        "id": "scene_06",
        "seed": 61006,
        "object_count": 6,
        "labels": ["sofa", "coffee_table", "area_rug", "floor_lamp", "potted_plant", "armchair"],
        "prompt": COMMON + (
            "A calm modern living room containing exactly six movable objects: exactly one beige "
            "three-seat sofa against the back wall, one rectangular oak coffee table in front, one "
            "large blue-gray area rug under the table, one black floor lamp in the back-left corner, "
            "one tall potted ficus in the back-right corner, and one rust-colored armchair on the left."
        ),
    },
    {
        "id": "scene_09",
        "seed": 91009,
        "object_count": 9,
        "labels": [
            "workbench", "stool", "shelving_unit", "tool_chest", "floor_lamp",
            "area_rug", "potted_plant", "storage_cabinet", "wooden_crate"
        ],
        "prompt": COMMON + (
            "A relaxed artisan studio containing exactly nine large distinct objects: one long wooden "
            "workbench against the back wall, one round stool in front of it, one tall open shelving "
            "unit on the left wall, one red metal tool chest, one industrial floor lamp, one flat woven "
            "area rug, one tall potted plant, one closed storage cabinet on the right wall, and one wooden "
            "crate on the floor. Keep generous visible gaps between every object; shelves are empty."
        ),
    },
    {
        "id": "scene_12",
        "seed": 121012,
        "object_count": 12,
        "labels": [
            "sofa", "armchair_left", "armchair_right", "coffee_table", "side_table",
            "bookcase", "media_console", "floor_lamp", "area_rug", "potted_plant",
            "ottoman", "storage_chest"
        ],
        "prompt": COMMON + (
            "A spacious warm loft lounge containing exactly twelve clearly separated objects: one tan "
            "sofa centered at the back, two matching green armchairs separated on its left and right, one "
            "oval coffee table, one small side table, one tall bookcase on the left wall, one low media "
            "console on the right wall, one arc floor lamp, one large cream area rug, one tall potted palm, "
            "one round ottoman, and one wooden storage chest. Every item is fully visible with clean gaps."
        ),
    },
    {
        "id": "scene_09_repeat",
        "repeat_of": "scene_09",
        "seed": 91009,
        "object_count": 9,
        "labels": [
            "workbench", "stool", "shelving_unit", "tool_chest", "floor_lamp",
            "area_rug", "potted_plant", "storage_cabinet", "wooden_crate"
        ],
        "prompt": COMMON + (
            "A relaxed artisan studio containing exactly nine large distinct objects: one long wooden "
            "workbench against the back wall, one round stool in front of it, one tall open shelving "
            "unit on the left wall, one red metal tool chest, one industrial floor lamp, one flat woven "
            "area rug, one tall potted plant, one closed storage cabinet on the right wall, and one wooden "
            "crate on the floor. Keep generous visible gaps between every object; shelves are empty."
        ),
    },
]


def api(method: str, path: str, payload: dict | None = None) -> dict:
    key = os.environ.get("WAVESPEED_API_KEY")
    if not key:
        raise SystemExit("WAVESPEED_API_KEY is required")
    request = urllib.request.Request(
        f"{API_ROOT}/{path.lstrip('/')}",
        data=json.dumps(payload).encode() if payload is not None else None,
        method=method,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"WaveSpeed HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc


def response_data(response: dict) -> dict:
    return response.get("data", response)


def task_id(response: dict) -> str:
    data = response_data(response)
    return data.get("id") or data.get("prediction_id")


def output_url(result: dict) -> str:
    outputs = result.get("outputs") or result.get("output")
    if isinstance(outputs, str):
        return outputs
    if isinstance(outputs, list) and outputs:
        return outputs[0] if isinstance(outputs[0], str) else outputs[0]["url"]
    raise RuntimeError(f"No output in {result}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for scene in SCENES:
        submitted_at = time.perf_counter()
        response = api(
            "POST",
            MODEL,
            {"prompt": scene["prompt"], "size": "1280*720", "seed": scene["seed"], "output_format": "png"},
        )
        jobs.append({**scene, "task_id": task_id(response), "submitted_at": submitted_at})
        print(f"submitted {scene['id']} {jobs[-1]['task_id']}", flush=True)

    pending = {job["task_id"]: job for job in jobs}
    while pending:
        for prediction_id, job in list(pending.items()):
            result = response_data(api("GET", f"predictions/{prediction_id}/result"))
            status = str(result.get("status", "")).lower()
            if status in {"completed", "succeeded", "success"}:
                completed_at = time.perf_counter()
                url = output_url(result)
                path = OUT / f"{job['id']}.png"
                with urllib.request.urlopen(url, timeout=300) as response:
                    path.write_bytes(response.read())
                job["latency_seconds"] = round(completed_at - job["submitted_at"], 3)
                job["file"] = path.name
                job["sha256"] = sha256(path)
                job["result_timings"] = result.get("timings")
                del pending[prediction_id]
                print(f"completed {job['id']} in {job['latency_seconds']}s", flush=True)
            elif status in {"failed", "error", "cancelled", "canceled"}:
                raise RuntimeError(f"{job['id']} failed: {result}")
        if pending:
            time.sleep(5)

    manifest = {
        "suite": "wavespeed_scenegen_complexity_suite_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "request": {"size": "1280*720", "output_format": "png"},
        "scenes": [{k: v for k, v in job.items() if k != "submitted_at"} for job in jobs],
        "security": {"api_key_stored": False},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "count": len(jobs)}, indent=2))


if __name__ == "__main__":
    main()
