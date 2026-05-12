#!/usr/bin/env bash

set -euo pipefail


ROOT_DIR="path/to/compbench_images"
T2I_COMP_CODE_ROOT="your-t2i-compbench/T2I-CompBench"


## Attribute Binding:
cd "${T2I_COMP_CODE_ROOT}/BLIPvqa_eval/"
if [[ ! -f "$ROOT_DIR/texture_val/annotation_blip/vqa_result.json" ]]; then
  echo "[RUN] texture_val BLIP_vqa.py"
  python BLIP_vqa.py --out_dir "$ROOT_DIR/texture_val"
else
  echo "[SKIP] texture_val already has vqa_result.json"
fi

if [[ ! -f "$ROOT_DIR/color_val/annotation_blip/vqa_result.json" ]]; then
  echo "[RUN] color_val BLIP_vqa.py"
  python BLIP_vqa.py --out_dir "$ROOT_DIR/color_val"
else
  echo "[SKIP] color_val already has vqa_result.json"
fi

if [[ ! -f "$ROOT_DIR/shape_val/annotation_blip/vqa_result.json" ]]; then
  echo "[RUN] shape_val BLIP_vqa.py"
  python BLIP_vqa.py --out_dir "$ROOT_DIR/shape_val"
else
  echo "[SKIP] shape_val already has vqa_result.json"
fi

## 2D-spatial
cd "${T2I_COMP_CODE_ROOT}/UniDet_eval"
if [[ ! -f "$ROOT_DIR/spatial_val/labels/annotation_obj_detection_2d/vqa_result.json" ]]; then
  echo "[RUN] spatial_val 2D_spatial_eval.py"
  python 2D_spatial_eval.py --outpath "$ROOT_DIR/spatial_val"
else
  echo "[SKIP] spatial_val already has vqa_result.json"
fi

# numeracy
cd "${T2I_COMP_CODE_ROOT}/UniDet_eval"
if [[ ! -f "$ROOT_DIR/numeracy_val/labels/annotation_num/vqa_result.json" ]]; then
  echo "[RUN] numeracy_val numeracy_eval.py"
  python numeracy_eval.py --outpath "$ROOT_DIR/numeracy_val"
else
  echo "[SKIP] numeracy_val already has vqa_result.json"
fi

## 3d spatial
cd "${T2I_COMP_CODE_ROOT}/UniDet_eval"
if [[ ! -f "$ROOT_DIR/3d_spatial_val/labels/annotation_obj_detection_3d/vqa_result.json" ]]; then
  echo "[RUN] 3d_spatial_val 3D_spatial_eval.py"
  python 3D_spatial_eval.py --outpath "$ROOT_DIR/3d_spatial_val"
else
  echo "[SKIP] 3d_spatial_val already has vqa_result.json"
fi

## Non-Spatial Relationship
cd "${T2I_COMP_CODE_ROOT}"
if [[ ! -f "$ROOT_DIR/non_spatial_val/annotation_clip/vqa_result.json" ]]; then
  echo "[RUN] non_spatial_val CLIP_similarity.py"
  python CLIPScore_eval/CLIP_similarity.py --outpath "$ROOT_DIR/non_spatial_val"
else
  echo "[SKIP] non_spatial_val already has vqa_result.json"
fi

# 3-in-1 for Complex Compositions
cd "${T2I_COMP_CODE_ROOT}/BLIPvqa_eval/"
if [[ ! -f "$ROOT_DIR/complex_val/annotation_blip/vqa_result.json" ]]; then
  echo "[RUN] complex_val BLIP_vqa.py"
  python BLIP_vqa.py --out_dir "$ROOT_DIR/complex_val"
else
  echo "[SKIP] complex_val BLIP_vqa.py already done"
fi

cd "${T2I_COMP_CODE_ROOT}/UniDet_eval"
if [[ ! -f "$ROOT_DIR/complex_val/labels/annotation_obj_detection_2d/vqa_result.json" ]]; then
  echo "[RUN] complex_val 2D_spatial_eval.py"
  python 2D_spatial_eval.py --outpath "$ROOT_DIR/complex_val"
else
  echo "[SKIP] complex_val 2D_spatial_eval.py already done"
fi

cd "${T2I_COMP_CODE_ROOT}"
if [[ ! -f "$ROOT_DIR/complex_val/annotation_clip/vqa_result.json" ]]; then
  echo "[RUN] complex_val CLIP_similarity.py"
  python CLIPScore_eval/CLIP_similarity.py --outpath "$ROOT_DIR/complex_val"
else
  echo "[SKIP] complex_val CLIP_similarity.py already done"
fi

cd "${T2I_COMP_CODE_ROOT}/3_in_1_eval/"
if [[ ! -f "$ROOT_DIR/complex_val/annotation_3_in_1/vqa_score.txt" ]]; then
  echo "[RUN] complex_val 3_in_1.py"
  python "3_in_1.py" --outpath "$ROOT_DIR/complex_val"
else
  echo "[SKIP] complex_val 3_in_1.py already done"
fi
