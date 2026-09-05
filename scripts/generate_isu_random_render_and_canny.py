"""
Generate random ISU samples, render them with Blender, and export Canny+depth maps.

Usage example:
    python scripts/generate_isu_random_render_and_canny.py \
        --num-samples 1000 \
        --scene isu/blender/scenes/scene_v3.blend \
        --features-config configs/isu_features.json \
        --output-dir results/scene_v3_random_1000
"""

import argparse
import inspect
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from PIL import Image


#############################################################################
# EXTEND PYTHON PATH WITH THIS CODE TO KEEP THE SCRIPT IN A SEPARATE FOLDER #
#############################################################################
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
#############################################################################

from isu.features import FeatureHandler
from isu.model.blender_contentinput import BlenderContentInput
from isu.model.run_blender import run_blender


try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

try:
    from skimage.feature import canny as skimage_canny  # type: ignore
except Exception:
    skimage_canny = None

try:
    import torch as _torch
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator as _SamAMG
    _SAM_AVAILABLE = True
except Exception:
    _SAM_AVAILABLE = False

_SAM_CHECKPOINT = Path(__file__).resolve().parents[1] / "isu/utils/SegmentAnyRGBD-main/sam_vit_h_4b8939.pth"
_SAM_MODEL_TYPE = "vit_h"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate random ISU renders and Canny maps.")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples to generate.")
    parser.add_argument(
        "--features-config",
        type=str,
        default="configs/isu_features.json",
        help="Feature configuration JSON path.",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="isu/blender/scenes/scene_v3.blend",
        help="Blend scene file path, relative to repo root or absolute.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/scene_v3_random_1000",
        help="Output root folder.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--timeout-sec", type=int, default=220, help="Per-render timeout in seconds.")
    parser.add_argument("--canny-low", type=int, default=100, help="Lower Canny threshold.")
    parser.add_argument("--canny-high", type=int, default=200, help="Upper Canny threshold.")
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately when a sample fails.",
    )
    return parser.parse_args()


def make_dirs(output_root: Path) -> Dict[str, Path]:
    paths = {
        "root": output_root,
        "json": output_root / "json_inputs",
        "images": output_root / "images",
        "depth": output_root / "depth",
        "depth_exr": output_root / "depth" / "exr",
        "depth_png": output_root / "depth" / "png",
        "depth_vis": output_root / "depth" / "vis",
        "seg": output_root / "seg",
        "seg_png": output_root / "seg",
        "canny": output_root / "canny",
        "instance_seg": output_root / "instance_seg",
        "sad_rgb": output_root / "sad" / "rgb",
        "sad_depth": output_root / "sad" / "depth",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


# Segmentation class palette: class_id -> (R, G, B)
SEG_PALETTE: Dict[int, tuple] = {
    0:  (20,  20,  20),   # unlabeled    — near black
    1:  (220, 30,  60),   # human        — bold crimson
    2:  (30,  100, 200),  # phone        — deep blue
    3:  (210, 160, 0),    # suitcase     — dark amber
    4:  (30,  150, 80),   # baby_seat    — forest green
    5:  (220, 100, 0),    # safety_belt  — deep orange
    6:  (160, 30,  80),   # beverage     — dark magenta
    7:  (90,  60,  200),  # car_interior — indigo
    8:  (50,  130, 180),  # exterior     — steel blue
    9:  (130, 50,  160),  # blanket      — deep violet
    10: (100, 60,  30),   # seat         — dark brown
}

SEG_CLASS_NAMES: Dict[int, str] = {
    0: "unlabeled",
    1: "human",
    2: "phone",
    3: "suitcase",
    4: "baby_seat",
    5: "safety_belt",
    6: "beverage",
    7: "car_interior",
    8: "exterior",
    9: "blanket",
    10: "seat",
}

# Instance segmentation palette (19 classes, matches INST_COLORS order in driver_rgb.py)
INST_PALETTE: Dict[int, tuple] = {
    0:  (1,   1,   1),    # background
    1:  (220, 30,  60),   # driver
    2:  (30,  100, 200),  # passenger_codriver
    3:  (210, 160, 0),    # passenger_rear_left
    4:  (30,  150, 80),   # passenger_rear_right
    5:  (220, 100, 0),    # driver_seatbelt
    6:  (130, 50,  160),  # codriver_seatbelt
    7:  (140, 80,  255),  # rear_left_seatbelt
    8:  (50,  130, 180),  # rear_right_seatbelt
    9:  (160, 30,  80),   # phone_codriver
    10: (200, 100, 0),    # cola_bottle
    11: (0,   160, 100),  # cola_can
    12: (0,   140, 200),  # suitcase
    13: (200, 0,   90),   # baby_seat
    14: (0,   100, 60),   # baby
    15: (200, 200, 0),    # driver_phone
    16: (38,  15,  3),    # seat
    17: (28,  15,  150),  # car_interior
    18: (8,   63,  122),  # exterior
}

INST_CLASS_NAMES: Dict[int, str] = {
    0:  "background",
    1:  "driver",
    2:  "passenger_codriver",
    3:  "passenger_rear_left",
    4:  "passenger_rear_right",
    5:  "driver_seatbelt",
    6:  "codriver_seatbelt",
    7:  "rear_left_seatbelt",
    8:  "rear_right_seatbelt",
    9:  "phone_codriver",
    10: "cola_bottle",
    11: "cola_can",
    12: "suitcase",
    13: "baby_seat",
    14: "baby",
    15: "driver_phone",
    16: "seat",
    17: "car_interior",
    18: "exterior",
}


# Fixed palette for SAD — indexed by area rank so colors are consistent across images
_SAD_PALETTE = [
    (255,  80,  80),  # 0 – largest region (usually background/car)
    ( 80, 140, 255),  # 1
    ( 80, 220,  80),  # 2
    (255, 200,  50),  # 3
    (200,  80, 255),  # 4
    ( 80, 220, 220),  # 5
    (255, 140,  50),  # 6
    (140, 255,  80),  # 7
    (255,  80, 180),  # 8
    ( 80, 180, 255),  # 9
    (220, 220,  80),  # 10
    (255, 120, 120),  # 11
    (120, 255, 120),  # 12
    (120, 120, 255),  # 13
    (255, 255, 100),  # 14
    (180, 100, 255),  # 15
    (100, 255, 200),  # 16
    (255, 180, 100),  # 17
    (100, 200, 255),  # 18
    (200, 255, 100),  # 19
    (255, 100, 200),  # 20
    (150, 150, 255),  # 21
    (255, 150, 150),  # 22
    (150, 255, 150),  # 23
]


def _colorize_sam_masks(image_np: np.ndarray, masks: list) -> np.ndarray:
    """Overlay SAM masks as semi-transparent colored regions using a fixed palette by area rank."""
    overlay = image_np.copy().astype(np.float32)
    for rank, m in enumerate(sorted(masks, key=lambda x: x["area"], reverse=True)):
        color = np.array(_SAD_PALETTE[rank % len(_SAD_PALETTE)], dtype=np.float32)
        overlay[m["segmentation"]] = overlay[m["segmentation"]] * 0.4 + color * 0.6
    return overlay.astype(np.uint8)


def _load_sam_generator():
    if not _SAM_AVAILABLE:
        return None
    if not _SAM_CHECKPOINT.exists():
        print(f"[WARN] SAM checkpoint not found: {_SAM_CHECKPOINT} — skipping SAD")
        return None
    device = "cuda" if _torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading SAM ({_SAM_MODEL_TYPE}) on {device}…")
    sam = sam_model_registry[_SAM_MODEL_TYPE](checkpoint=str(_SAM_CHECKPOINT))
    sam.to(device=device)
    return _SamAMG(
        sam,
        points_per_side=8,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        min_mask_region_area=3000,
        crop_n_layers=0,
    )


def _run_sam_on_image(generator, img_path: Path, out_path: Path) -> bool:
    """Run SAM automatic mask generator and save colorized overlay PNG."""
    try:
        img = np.array(Image.open(img_path).convert("RGB"))
        masks = generator.generate(img)
        colored = _colorize_sam_masks(img, masks)
        Image.fromarray(colored, mode="RGB").save(str(out_path))
        return True
    except Exception as e:
        print(f"[WARN] SAD failed on {img_path.name}: {e}")
        return False


def build_sample_name(index: int) -> str:
    return f"sample_{index:05d}"


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def clean_content_for_json(content_input: BlenderContentInput) -> Dict[str, Any]:
    """
    Create a clean JSON representation that only includes relevant features.
    Omits fields that are None or not applicable based on parent feature state.
    """
    data = content_input.model_dump()
    
    # Remove None values
    data = {k: v for k, v in data.items() if v is not None}
    
    # Remove dependent features when parent is "NO"
    if data.get("suitcase") == "NO":
        data.pop("suitcase_color", None)
        data.pop("suitcase_location", None)
        data.pop("suitcase_pose", None)
    
    if data.get("baby_seat") == "NO":
        data.pop("baby_seat_orientation", None)
        data.pop("baby", None)
    
    if data.get("passenger_codriver") == "NO":
        data.pop("passenger_codriver_tshirt_color", None)
        data.pop("passenger_codriver_emotion", None)
        data.pop("codriver_safety_belt", None)
        data.pop("passenger_codriver_head_angle", None)
    
    if data.get("passenger_back_seat_left") == "NO":
        data.pop("passenger_rear_left_safety_belt", None)
        data.pop("passenger_rear_left_tshirt_color", None)
        data.pop("passenger_rear_left_emotion", None)
        data.pop("passenger_rear_left_head_angle", None)
    
    if data.get("passenger_back_seat_right") == "NO":
        data.pop("passenger_rear_right_safety_belt", None)
        data.pop("passenger_rear_right_tshirt_color", None)
        data.pop("passenger_rear_right_emotion", None)
        data.pop("passenger_rear_right_head_angle", None)
    
    if data.get("phone_codriver_seat") == "NO":
        data.pop("phone_codriver_seat_color", None)
    
    if data.get("driver_safety_belt") == "NO":
        data.pop("driver_safety_belt", None)
    
    return data


def generate_canny_map(image_path: Path, canny_path: Path, low: int, high: int) -> str:
    if cv2 is not None:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Failed to read image for canny: {image_path}")
        edges = cv2.Canny(image, low, high)
        ok = cv2.imwrite(str(canny_path), edges)
        if not ok:
            raise RuntimeError(f"Failed to write canny image: {canny_path}")
        return "cv2"

    if skimage_canny is not None:
        gray = np.array(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0
        low_n = max(0.0, min(1.0, low / 255.0))
        high_n = max(0.0, min(1.0, high / 255.0))
        edges = skimage_canny(gray, low_threshold=low_n, high_threshold=high_n)
        edge_img = (edges.astype(np.uint8) * 255)
        Image.fromarray(edge_img, mode="L").save(canny_path)
        return "skimage"

    raise RuntimeError(
        "No Canny backend available. Install one of: opencv-python (or opencv-python-headless) or scikit-image."
    )


def apply_constraints(content_input: BlenderContentInput) -> BlenderContentInput:
    # --- Suitcase ---
    if content_input.suitcase == "NO":
        content_input.suitcase_color = None
        content_input.suitcase_location = None
        content_input.suitcase_pose = None

    # --- Baby seat: occupies co-driver seat exclusively ---
    if content_input.baby_seat == "NO":
        content_input.baby_seat_orientation = None
        content_input.baby = None
    else:
        content_input.phone_codriver_seat = "NO"
        content_input.colabottle_codriver_seat = "NO"
        content_input.colacan_codriver_seat = "NO"
        content_input.passenger_codriver = "NO"
        content_input.codriver_safety_belt = "NO"
        if content_input.suitcase_location == "CO_DRIVER_SEAT":
            content_input.suitcase_location = "REAR_SEAT"

    # --- Co-driver seat: passenger blocks items and vice-versa ---
    if content_input.passenger_codriver == "YES":
        content_input.phone_codriver_seat = "NO"
        content_input.colabottle_codriver_seat = "NO"
        content_input.colacan_codriver_seat = "NO"
        if content_input.suitcase_location == "CO_DRIVER_SEAT":
            content_input.suitcase_location = "REAR_SEAT"
    elif (
        content_input.phone_codriver_seat == "YES"
        or content_input.colabottle_codriver_seat == "YES"
        or content_input.colacan_codriver_seat == "YES"
        or content_input.suitcase_location == "CO_DRIVER_SEAT"
    ):
        # Items in co-driver seat block passenger
        content_input.passenger_codriver = "NO"
        content_input.codriver_safety_belt = "NO"
        content_input.passenger_codriver_tshirt_color = None
        content_input.passenger_codriver_emotion = None
    else:
        # No blocking items, keep passenger_codriver as sampled (don't force to NO)
        if content_input.passenger_codriver == "NO":
            content_input.codriver_safety_belt = "NO"
            content_input.passenger_codriver_tshirt_color = None
            content_input.passenger_codriver_emotion = None

    if content_input.suitcase_location == "CO_DRIVER_SEAT" and (
        content_input.phone_codriver_seat == "YES"
        or content_input.colabottle_codriver_seat == "YES"
        or content_input.colacan_codriver_seat == "YES"
        or content_input.passenger_codriver == "YES"
    ):
        content_input.suitcase_location = "REAR_SEAT"

    if content_input.suitcase_location != "REAR_SEAT":
        content_input.suitcase_pose = None

    if content_input.phone_codriver_seat == "NO":
        content_input.phone_codriver_seat_color = None

    # --- Rear passengers: seatbelt + appearance only if passenger present ---
    if content_input.passenger_back_seat_left != "YES":
        content_input.passenger_back_seat_left = "NO"
        content_input.passenger_rear_left_safety_belt = "NO"
        content_input.passenger_rear_left_tshirt_color = None
        content_input.passenger_rear_left_emotion = None

    if content_input.passenger_back_seat_right != "YES":
        content_input.passenger_back_seat_right = "NO"
        content_input.passenger_rear_right_safety_belt = "NO"
        content_input.passenger_rear_right_tshirt_color = None
        content_input.passenger_rear_right_emotion = None

    return content_input


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    repo_root = Path(parentdir).resolve()
    features_config = (repo_root / args.features_config).resolve()
    scene_path = (repo_root / args.scene).resolve() if not Path(args.scene).is_absolute() else Path(args.scene)
    output_root = (repo_root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)

    if not features_config.exists():
        raise FileNotFoundError(f"Feature config not found: {features_config}")
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene file not found: {scene_path}")

    dirs = make_dirs(output_root)

    # Load SAM once for all samples
    sam_generator = None  # Disabled for faster generation
    if sam_generator is None:
        print("[INFO] SAD will be skipped (SAM disabled for faster generation)")

    feature_handler = FeatureHandler.from_json(str(features_config))

    records: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    started = time.time()

    print(f"[INFO] Starting generation of {args.num_samples} samples")
    print(f"[INFO] Scene: {scene_path}")
    print(f"[INFO] Output: {output_root}")

    for i in range(args.num_samples):
        sample_name = build_sample_name(i)
        json_path = dirs["json"] / f"{sample_name}.json"

        try:
            sampled = feature_handler.sample_feature_scores()
            feature_values = feature_handler.get_feature_values_dict(
                ordinal_feature_scores=sampled.ordinal,
                categorical_feature_indices=sampled.categorical,
                continuous_feature_values=sampled.continuous,
            )

            content_input = BlenderContentInput.model_validate(feature_values)
            apply_constraints(content_input)
            
            save_json(json_path, content_input.model_dump())

            image_path = Path(
                run_blender(
                    json_path=str(json_path),
                    image_output_dir=str(dirs["images"]),
                    depth_output_dir=str(dirs["depth"]),
                    seg_output_dir=str(dirs["seg"]),
                    canny_output_dir=str(dirs["canny"]),
                    instance_seg_output_dir=str(dirs["instance_seg"]),
                    blend_file=str(scene_path),
                    timeout_sec=args.timeout_sec,
                )
            )
            depth_path = dirs["depth_exr"] / f"{json_path.stem}_depth.exr"
            if not depth_path.exists():
                raise RuntimeError(f"Depth map missing after render: {depth_path}")
            
            # Convert EXR depth to PNG
            depth_png_path = dirs["depth_png"] / f"{json_path.stem}_depth.png"
            try:
                import OpenEXR
                import Imath
                
                exr_file = OpenEXR.InputFile(str(depth_path))
                dw = exr_file.header()['dataWindow']
                width = dw.max.x - dw.min.x + 1
                height = dw.max.y - dw.min.y + 1
                
                # Read Z channel (depth)
                if 'Z' in exr_file.header()['channels']:
                    z_channel = exr_file.channel('Z')
                    depth_array = np.frombuffer(z_channel, dtype=np.float32).reshape(height, width)
                else:
                    # If no Z channel, try to read first available channel
                    channels = exr_file.header()['channels'].keys()
                    first_channel = list(channels)[0]
                    z_channel = exr_file.channel(first_channel)
                    depth_array = np.frombuffer(z_channel, dtype=np.float32).reshape(height, width)
                
                # Normalize to 16-bit
                depth_min = depth_array.min()
                depth_max = depth_array.max()
                if depth_max > depth_min:
                    depth_normalized = ((depth_array - depth_min) / (depth_max - depth_min) * 65535).astype(np.uint16)
                else:
                    depth_normalized = (depth_array * 65535).astype(np.uint16)
                
                # Save as 16-bit grayscale PNG
                Image.fromarray(depth_normalized, mode='I;16').save(str(depth_png_path))

                # Colorized depth visualization using magma colormap
                depth_vis_path = dirs["depth_vis"] / f"{json_path.stem}_depth_vis.png"
                import matplotlib as _mpl
                depth_float = depth_normalized.astype(np.float32) / 65535.0
                _cmap = _mpl.colormaps["magma"]
                depth_rgb = (_cmap(depth_float)[:, :, :3] * 255).astype(np.uint8)
                Image.fromarray(depth_rgb, mode="RGB").save(str(depth_vis_path))

                # Keep EXR file in exr/ directory and reference PNG
                depth_path = depth_png_path
            except Exception as e:
                print(f"[WARN] EXR to PNG conversion failed: {e}. Keeping EXR file.")

            # Segmentation: Blender wrote a flat-emission PNG directly to seg/
            # Freestyle is disabled in render_segmentation so output is correct Cycles orientation
            seg_png_path = dirs["seg_png"] / f"{json_path.stem}_seg.png"
            if seg_png_path.exists():
                seg_path_final = seg_png_path
            else:
                print(f"[WARN] Segmentation PNG missing: {seg_png_path}")
                seg_path_final = None

            # Canny: prefer Blender GT (Freestyle geometry edges), fall back to CV
            canny_path = dirs["canny"] / f"{json_path.stem}_canny.png"
            if canny_path.exists():
                canny_backend = "blender_freestyle"
            else:
                # Fallback: CV-based Canny on RGB image
                canny_path = dirs["canny"] / f"{image_path.stem}_canny.png"
                canny_backend = generate_canny_map(
                    image_path=image_path,
                    canny_path=canny_path,
                    low=args.canny_low,
                    high=args.canny_high,
                )

            def _rel(p) -> str:
                """Return path relative to output_root, or None if p is None."""
                if p is None:
                    return None
                try:
                    return str(Path(p).relative_to(output_root))
                except ValueError:
                    return str(p)

            # Instance segmentation: PNG written directly by Blender
            instance_seg_path = dirs["instance_seg"] / f"{json_path.stem}_instance_seg.png"
            instance_seg_path_final = None
            if instance_seg_path.exists():
                instance_seg_path_final = instance_seg_path
            else:
                print(f"[WARN] Instance segmentation PNG missing: {instance_seg_path}")

            record = {
                "sample_index": i,
                "sample_name": sample_name,
                "json_path": _rel(json_path),
                "image_path": _rel(image_path),
                "depth_path": _rel(depth_path),
                "seg_path": _rel(seg_path_final),
                "canny_path": _rel(canny_path),
                "canny_backend": canny_backend,
                "instance_seg_path": _rel(instance_seg_path_final),
                "sad_rgb_path": None,
                "sad_depth_path": None,
                "params": content_input.model_dump(),
            }

            # SAD: run SAM on RGB and depth vis
            if sam_generator is not None:
                sad_rgb_path   = dirs["sad_rgb"]   / f"{sample_name}_sad_rgb.png"
                sad_depth_path = dirs["sad_depth"] / f"{sample_name}_sad_depth.png"
                depth_vis_path = dirs["depth_vis"] / f"{sample_name}_depth_vis.png"

                if _run_sam_on_image(sam_generator, image_path, sad_rgb_path):
                    record["sad_rgb_path"] = _rel(sad_rgb_path)
                if depth_vis_path.exists() and _run_sam_on_image(sam_generator, depth_vis_path, sad_depth_path):
                    record["sad_depth_path"] = _rel(sad_depth_path)

            records.append(record)

            if (i + 1) % 10 == 0 or (i + 1) == args.num_samples:
                print(f"[INFO] Completed {i + 1}/{args.num_samples}")

        except Exception as e:
            failure = {
                "sample_index": i,
                "sample_name": sample_name,
                "json_path": str(Path(json_path).relative_to(output_root)),
                "error": str(e),
            }
            failures.append(failure)
            print(f"[WARN] Sample {i} failed: {e}")
            if args.stop_on_error:
                raise

    elapsed = time.time() - started
    summary = {
        "num_requested": args.num_samples,
        "num_success": len(records),
        "num_failed": len(failures),
        "elapsed_sec": round(elapsed, 2),
        "seed": args.seed,
        "scene": Path(scene_path).name,
        "features_config": Path(features_config).name,
        "canny_thresholds": {"low": args.canny_low, "high": args.canny_high},
        "image_resolution": "960x540",
        "render_engine": "Cycles",
    }

    manifest = {
        "summary": summary,
        "records": records,
        "failures": failures,
    }

    manifest_path = output_root / "sample_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Write segmentation class map once
    class_map = [
        {"id": cid, "name": SEG_CLASS_NAMES[cid], "color_rgb": list(SEG_PALETTE[cid])}
        for cid in sorted(SEG_CLASS_NAMES)
    ]
    with open(output_root / "seg_class_map.json", "w", encoding="utf-8") as f:
        json.dump(class_map, f, indent=2)

    # Write instance segmentation class map once
    inst_class_map = [
        {"id": cid, "name": INST_CLASS_NAMES[cid], "color_rgb": list(INST_PALETTE[cid])}
        for cid in sorted(INST_CLASS_NAMES)
    ]
    with open(output_root / "instance_seg_class_map.json", "w", encoding="utf-8") as f:
        json.dump(inst_class_map, f, indent=2)

    print("[INFO] Done")
    print(f"[INFO] Manifest: {manifest_path}")
    print(f"[INFO] Success: {len(records)} / {args.num_samples}")
    print(f"[INFO] Failed: {len(failures)}")


if __name__ == "__main__":
    main()
