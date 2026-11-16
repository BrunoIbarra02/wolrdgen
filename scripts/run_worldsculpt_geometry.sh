#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   source config/worldsculpt-a10g.env
#   INPUT_ROOT=/data/input OUTPUT_ROOT=/data/output CKPT_ROOT=/opt/scenegen/WorldSculpt/pretrained \
#     scripts/run_worldsculpt_geometry.sh scene_00001

: "${INPUT_ROOT:?Set INPUT_ROOT}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
: "${CKPT_ROOT:?Set CKPT_ROOT}"
: "${1:?Pass a scene name}"

SCENE="$1"
WORLD_ROOT="${WORLD_ROOT:-/opt/scenegen/WorldSculpt}"
CASE_ROOT="$OUTPUT_ROOT/$SCENE"
SCENE_DIR="$INPUT_ROOT/$SCENE"
SS_DIR="$CKPT_ROOT/ss_ft64_mv_lora_ibr_texverse"
SHAPE_DIR="$CKPT_ROOT/shape_ft1024_mv_lora_ibr_texverse_fixedmem05"

: "${GPU:=0}"
: "${SAMPLER:=official}"
: "${SS_STEP:=15000}"
: "${SHAPE_STEP:=15000}"
: "${VIEWS:=all}"
: "${MAX_VIEWS:=8}"
: "${VIEW_SELECT:=fps}"
: "${RESOLUTION:=1024}"
: "${SEED:=42}"
: "${MAX_NUM_TOKENS:=32768}"
: "${MIN_MASK_RATIO:=0.001}"
: "${MAX_CROP_RATIO:=3.0}"
: "${MASK_FIT_PCT:=95}"
: "${FACE_BUDGET:=250000}"
: "${GLB_DECIMATION:=100000}"
: "${RENDER:=0}"

test -f "$SCENE_DIR/transforms.json"
test -f "$SS_DIR/config.json"
test -f "$SHAPE_DIR/config.json"
mkdir -p "$CASE_ROOT"

cd "$WORLD_ROOT"
python prepare_crops_scene.py \
  --scene_dir "$SCENE_DIR" \
  --case_root "$CASE_ROOT" \
  --crop_resolution 1024 \
  --save_alignments \
  --alpha_erode_kernel 0 \
  --alpha_erode_iters 0 \
  --min_mask_ratio "$MIN_MASK_RATIO" \
  --max_crop_ratio "$MAX_CROP_RATIO" \
  --mask_fit_scale \
  --mask_fit_pct "$MASK_FIT_PCT" \
  --anchor_frame anchor

CUDA_VISIBLE_DEVICES="$GPU" python reconstruct_batch.py \
  --case_root "$CASE_ROOT" \
  --views "$VIEWS" \
  --max_views "$MAX_VIEWS" \
  --view_select "$VIEW_SELECT" \
  --resolution "$RESOLUTION" \
  --seed "$SEED" \
  --sampler "$SAMPLER" \
  --max_num_tokens "$MAX_NUM_TOKENS" \
  --no-ema --no_glb --low_vram \
  --ss_config "$SS_DIR/config.json" \
  --ss_ckpt_dir "$SS_DIR/ckpts" \
  --ss_step "$SS_STEP" \
  --shape_config "$SHAPE_DIR/config.json" \
  --shape_ckpt_dir "$SHAPE_DIR/ckpts" \
  --shape_step "$SHAPE_STEP"

render_flag=(--no_render)
if [[ "$RENDER" == "1" ]]; then
  render_flag=()
fi

CUDA_VISIBLE_DEVICES="$GPU" python compose_scene.py \
  --case_root "$CASE_ROOT" \
  --recon_dir "$CASE_ROOT/_recon" \
  --face_budget "$FACE_BUDGET" \
  --glb_decimation "$GLB_DECIMATION" \
  --output_dir "$CASE_ROOT/_scene" \
  --normal \
  "${render_flag[@]}"

echo "Scene ready at $CASE_ROOT/_scene/scene.glb"
