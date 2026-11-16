#!/usr/bin/env python3
"""Generate a simple, low-entropy image-to-video consistency test with WaveSpeed."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


API_ROOT = "https://api.wavespeed.ai/api/v3"
ANCHOR_MODEL = "wavespeed-ai/qwen-image/text-to-image-2512"
VIDEO_MODEL = "wavespeed-ai/wan-2.2/image-to-video"
SEED = 260927
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "wavespeed_relaxed_zen_room_v1"

ANCHOR_PROMPT = (
    "Photorealistic calm minimalist meditation room, a simple rectangular room with "
    "smooth warm off-white plaster walls and a continuous pale oak floor. Exactly one "
    "low oak bench centered against the back wall, exactly one large plain oatmeal rug "
    "centered on the floor, exactly one tall potted olive tree in the back-right corner, "
    "and exactly one round white paper pendant lamp overhead. Empty clean room, no windows, "
    "no doors visible, no shelves, no small objects. Soft diffuse daylight, even exposure, "
    "eye-level wide shot, 24mm lens, deep focus, physically plausible materials, quiet and "
    "relaxed atmosphere, fixed architectural geometry, high spatial coherence. No people, "
    "no animals, no text, no clutter, no extra furniture."
)

VIDEO_PROMPT = (
    "A rigid, completely static empty meditation room. The camera makes a very slow, smooth "
    "clockwise orbit of about 35 degrees around the center of the rug, at constant eye-level "
    "height, constant 24mm focal length and constant exposure. The walls, floor, low bench, "
    "rug, planter, olive tree and pendant lamp remain exactly fixed, unchanged and stationary. "
    "Photorealistic, deep focus, stable geometry, gentle diffuse lighting."
)

NEGATIVE_PROMPT = (
    "people, animals, wind, moving leaves, moving objects, object duplication, extra furniture, "
    "appearing objects, disappearing objects, morphing geometry, bending walls, changing room "
    "layout, doors, windows, cuts, transitions, camera jump, zoom, dolly, focal length change, "
    "exposure change, flicker, motion blur, shallow depth of field, text, logo, watermark"
)


def api(method: str, path: str, payload: dict | None = None) -> dict:
    key = os.environ.get("WAVESPEED_API_KEY")
    if not key:
        raise SystemExit("WAVESPEED_API_KEY is required")
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        f"{API_ROOT}/{path.lstrip('/')}",
        data=data,
        method=method,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"WaveSpeed HTTP {exc.code}: {detail}") from exc


def task_id(response: dict) -> str:
    data = response.get("data", response)
    value = data.get("id") or data.get("prediction_id")
    if not value:
        raise RuntimeError(f"No task id in response: {response}")
    return value


def wait_for(task: str, timeout_seconds: int = 1800) -> dict:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        result = api("GET", f"predictions/{task}/result")
        data = result.get("data", result)
        status = str(data.get("status", "")).lower()
        if status in {"completed", "succeeded", "success"}:
            return data
        if status in {"failed", "error", "cancelled", "canceled"}:
            raise RuntimeError(f"Task {task} ended as {status}: {result}")
        print(f"{task}: {status or 'pending'}", flush=True)
        time.sleep(8)
    raise TimeoutError(f"Timed out waiting for {task}")


def first_output(result: dict) -> str:
    outputs = result.get("outputs") or result.get("output")
    if isinstance(outputs, str):
        return outputs
    if isinstance(outputs, list) and outputs:
        item = outputs[0]
        return item if isinstance(item, str) else item.get("url")
    raise RuntimeError(f"No output URL in result: {result}")


def download(url: str, path: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "WorldSculpt-test/1.0"})
    with urllib.request.urlopen(request, timeout=300) as response, path.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def run_ffmpeg(video: Path) -> list[Path]:
    frame_dir = OUT / "frames"
    frame_dir.mkdir(exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
            "-vf", "fps=12/5", "-frames:v", "12",
            str(frame_dir / "frame_%02d.png"),
        ],
        check=True,
    )
    frames = sorted(frame_dir.glob("frame_*.png"))
    if len(frames) != 12:
        raise RuntimeError(f"Expected 12 frames, got {len(frames)}")
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-framerate", "1", "-i", str(frame_dir / "frame_%02d.png"),
            "-vf", "scale=640:-2,tile=4x3:padding=8:margin=8:color=white", "-frames:v", "1",
            str(OUT / "contact_sheet.png"),
        ],
        check=True,
    )
    return frames


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    print("Submitting anchor image...", flush=True)
    anchor_response = api(
        "POST",
        ANCHOR_MODEL,
        {"prompt": ANCHOR_PROMPT, "size": "1280*720", "seed": SEED, "output_format": "png"},
    )
    anchor_task = task_id(anchor_response)
    anchor_result = wait_for(anchor_task)
    anchor_url = first_output(anchor_result)
    anchor_path = OUT / "anchor.png"
    download(anchor_url, anchor_path)

    print("Submitting 5-second image-to-video test...", flush=True)
    video_response = api(
        "POST",
        VIDEO_MODEL,
        {
            "image": anchor_url,
            "prompt": VIDEO_PROMPT,
            "negative_prompt": NEGATIVE_PROMPT,
            "resolution": "720p",
            "duration": 5,
            "seed": SEED,
        },
    )
    video_task = task_id(video_response)
    video_result = wait_for(video_task)
    video_path = OUT / "orbit_5s_720p.mp4"
    download(first_output(video_result), video_path)
    frames = run_ffmpeg(video_path)

    files = [anchor_path, video_path, OUT / "contact_sheet.png", *frames]
    manifest = {
        "dataset_id": "wavespeed_relaxed_zen_room_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Low-entropy image-to-video consistency test for WorldSculpt preprocessing",
        "anchor": {
            "model": ANCHOR_MODEL,
            "task_id": anchor_task,
            "seed": SEED,
            "requested_size": "1280*720",
            "prompt": ANCHOR_PROMPT,
        },
        "video": {
            "model": VIDEO_MODEL,
            "task_id": video_task,
            "seed": SEED,
            "resolution": "720p",
            "duration_seconds": 5,
            "prompt": VIDEO_PROMPT,
            "negative_prompt": NEGATIVE_PROMPT,
        },
        "frames": {
            "count": len(frames),
            "sampling_fps": 2.4,
            "calibrated_camera_poses": False,
            "files": [str(path.relative_to(OUT)).replace("\\", "/") for path in frames],
        },
        "files": {
            str(path.relative_to(OUT)).replace("\\", "/"): sha256(path) for path in files
        },
        "security": {"api_key_stored": False, "rotation_recommended": True},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "anchor_task": anchor_task, "video_task": video_task}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
