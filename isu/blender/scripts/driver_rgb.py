import sys
import os
from pathlib import Path
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.append(HERE)          
# Or if utils.py is one dir up:
# sys.path.append(os.path.dirname(HERE))
PROJECT_ROOT = Path(__file__).resolve().parents[3]
TEXTURE_ROOT = PROJECT_ROOT / "isu" / "blender" / "assets" / "human_texture"

import json
from utils import *
from smpl.utils import *
import bpy
import platform
import time
t0 = time.perf_counter()

# === Set up Cycles render engine ===
context = bpy.context
scene = context.scene
scene.render.engine = 'CYCLES'
scene.render.use_persistent_data = True
cycles = scene.cycles
# cycles.use_fast_gi = True
cycles.use_fast_gi = False
# cycles.sample_clamp_indirect = 2.0
cycles.samples = 8
cycles.use_adaptive_sampling = True
cycles.adaptive_threshold = 0.02

# cycles.max_bounces = 4
# cycles.diffuse_bounces = 2
# cycles.glossy_bounces = 2
# cycles.transmission_bounces = 4
# cycles.volume_bounces = 0
prefs = context.preferences.addons['cycles'].preferences

# set render device to CPU if --force-cpu is passed, otherwise use GPU if available
def refresh_devices():
    try:
        prefs.refresh_devices()
    except Exception:
        pass

def set_cpu():
    refresh_devices()
    for d in getattr(prefs, "devices", []):
        d.use = False  # disable all devices
    prefs.compute_device_type = 'NONE'
    cycles.device = 'CPU'

def try_set_gpu(device_types):
    for dev_type in device_types:
        try:
            prefs.compute_device_type = dev_type
        except TypeError:
            continue

        refresh_devices()
        gpu_found = False
        devices = getattr(prefs, "devices", [])
        print("Available devices:", [getattr(d, "name", "") for d in devices])
        for d in devices:
            # Enable only non-CPU devices
            if getattr(d, "type", "") != 'CPU':
                d.use = True
                gpu_found = True
            else:
                d.use = False

        if gpu_found:
            print("gpu_found")
            cycles.device = 'GPU'
            # ---------  Denoiser ----------
            cycles.use_denoising = True
            try:
                cycles.denoiser = 'OPTIX'
                print("Using OptiX GPU denoiser")
            except (AttributeError, TypeError, ValueError):
                cycles.denoiser = 'OPENIMAGEDENOISE'
                print("OptiX denoiser not available, using OIDN (CPU)")
            # ----------------------------------------
            return True
    return False

refresh_devices()

if "--force-cpu" in sys.argv:
    set_cpu()
else:
    system = platform.system()
    print("system: ", system)

    if system == "Darwin":
        # macOS 
        #if not try_set_gpu(['METAL']):
        set_cpu()
    else:
        if not try_set_gpu(['OPTIX', 'CUDA', 'HIP', 'ONEAPI']):
            set_cpu()

print(f"Render device: {cycles.device}")

# === Parse input args, Load config ===
objs = bpy.data.objects
cols = bpy.data.collections

argv = sys.argv
args = argv[argv.index("--") + 1:]
json_path = args[0]
image_path = args[1]
depth_path = Path(args[2]).resolve() if len(args) > 2 and not args[2].startswith('--') else None

seg_path = None
if "--seg-path" in args:
    _seg_idx = args.index("--seg-path")
    seg_path = Path(args[_seg_idx + 1]).resolve()

canny_path = None
if "--canny-path" in args:
    _canny_idx = args.index("--canny-path")
    canny_path = Path(args[_canny_idx + 1]).resolve()

instance_seg_path = None
if "--instance-seg-path" in args:
    _iseg_idx = args.index("--instance-seg-path")
    instance_seg_path = Path(args[_iseg_idx + 1]).resolve()


# Segmentation class pass indices
PASS_INDEX = {
    "human":        1,  # driver, all passengers, baby
    "phone":        2,  # all phones (driver hand + co-driver seat)
    "suitcase":     3,
    "baby_seat":    4,
    "safety_belt":  5,
    "beverage":     6,  # cola bottle + cola can
}


def setup_render_outputs(depth_target, seg_target=None):
    """Set up compositor nodes for depth output."""
    vl = scene.view_layers[0]
    if depth_target is not None:
        vl.use_pass_z = True

    if hasattr(scene, "node_tree"):
        scene.use_nodes = True
        node_tree = scene.node_tree
    else:
        node_tree = scene.compositing_node_group
        if node_tree is None:
            node_tree = bpy.data.node_groups.new("RenderOutputs", 'CompositorNodeTree')
            scene.compositing_node_group = node_tree

    nodes = node_tree.nodes
    links = node_tree.links
    nodes.clear()

    render_layers = nodes.new(type='CompositorNodeRLayers')

    # --- Depth branch ---
    if depth_target is not None:
        normalize = nodes.new(type='CompositorNodeNormalize')
        links.new(render_layers.outputs['Depth'], normalize.inputs[0])

        out_depth = nodes.new(type='CompositorNodeOutputFile')
        if hasattr(out_depth, "file_output_items"):
            out_depth.file_output_items.clear()
            out_depth.file_output_items.new('FLOAT', 'Depth')
            out_depth.directory = str(depth_target.parent)
            out_depth.file_name = depth_target.stem
            out_depth.format.file_format = 'OPEN_EXR_MULTILAYER'
            links.new(normalize.outputs[0], out_depth.inputs['Depth'])
        else:
            out_depth.base_path = str(depth_target.parent)
            out_depth.file_slots[0].path = depth_target.stem + "_"
            out_depth.format.file_format = 'OPEN_EXR'
            out_depth.format.color_mode = 'BW'
            out_depth.format.color_depth = '16'
            links.new(normalize.outputs[0], out_depth.inputs[0])


def render_instance_seg(instance_seg_target: Path, scene_config: dict):
    """Per-instance color render: each object/group gets a unique distinct color."""
    # 19 bold distinct colors (Blender linear space) — one per instance slot
    INST_COLORS = [
        (0.004, 0.004, 0.004, 1.0),  #  0: background / car
        (0.706, 0.013, 0.057, 1.0),  #  1: driver
        (0.013, 0.148, 0.588, 1.0),  #  2: passenger codriver
        (0.678, 0.378, 0.0,   1.0),  #  3: passenger rear left
        (0.013, 0.329, 0.086, 1.0),  #  4: passenger rear right
        (0.706, 0.148, 0.0,   1.0),  #  5: driver seatbelt
        (0.247, 0.033, 0.378, 1.0),  #  6: codriver seatbelt
        (0.329, 0.148, 1.0,   1.0),  #  7: rear left seatbelt
        (0.033, 0.247, 0.478, 1.0),  #  8: rear right seatbelt
        (0.378, 0.013, 0.092, 1.0),  #  9: phone (codriver seat)
        (0.478, 0.190, 0.0,   1.0),  # 10: cola bottle
        (0.0,   0.378, 0.247, 1.0),  # 11: cola can
        (0.0,   0.329, 0.533, 1.0),  # 12: suitcase
        (0.533, 0.0,   0.216, 1.0),  # 13: baby seat
        (0.0,   0.247, 0.148, 1.0),  # 14: baby
        (0.533, 0.533, 0.0,   1.0),  # 15: driver phone (in hand)
        (0.148, 0.057, 0.013, 1.0),  # 16: seat         — dark brown
        (0.109, 0.057, 0.588, 1.0),  # 17: car_interior — indigo
        (0.033, 0.247, 0.478, 1.0),  # 18: exterior     — steel blue
    ]

    # Create materials
    inst_mats = {}
    for _idx, _rgba in enumerate(INST_COLORS):
        _mat = bpy.data.materials.new(name=f"_inst_{_idx}")
        _mat.use_nodes = True
        _nt = _mat.node_tree
        _nt.nodes.clear()
        _emit = _nt.nodes.new('ShaderNodeEmission')
        _emit.inputs['Color'].default_value = _rgba
        _emit.inputs['Strength'].default_value = 1.0
        _out = _nt.nodes.new('ShaderNodeOutputMaterial')
        _nt.links.new(_emit.outputs['Emission'], _out.inputs['Surface'])
        inst_mats[_idx] = _mat

    # Build object-name → instance index map
    inst_map = {}
    _gender = scene_config.get("driver_gender", "MALE")

    # Driver
    inst_map["SMPLX-mesh-male" if _gender == "MALE" else "SMPLX-mesh-female"] = 1

    # Driver phone (in hand)
    if scene_config.get("driver_phone") == "YES":
        inst_map["phone_male" if _gender == "MALE" else "phone_female"] = 15

    # Passenger codriver
    if scene_config.get("passenger_codriver") == "YES":
        _c = cols.get("Passenger_codriver")
        if _c:
            for _o in _c.objects:
                if _o.type == 'MESH': inst_map[_o.name] = 2

    # Passenger rear left
    if scene_config.get("passenger_back_seat_left") == "YES":
        _c = cols.get("Passenger_rear_left")
        if _c:
            for _o in _c.objects:
                if _o.type == 'MESH': inst_map[_o.name] = 3

    # Passenger rear right
    if scene_config.get("passenger_back_seat_right") == "YES":
        _c = cols.get("Passenger_rear_right")
        if _c:
            for _o in _c.objects:
                if _o.type == 'MESH': inst_map[_o.name] = 4

    # Driver seatbelt
    if scene_config.get("driver_safety_belt") == "YES":
        _c = cols.get(f"safety_belt_driver_{_gender.lower()}")
        if _c:
            for _o in _c.objects:
                if _o.type == 'MESH': inst_map[_o.name] = 5

    # Codriver seatbelt
    if scene_config.get("codriver_safety_belt") == "YES":
        _c = cols.get("safety_belt_codriver")
        if _c:
            for _o in _c.objects:
                if _o.type == 'MESH': inst_map[_o.name] = 6

    # Rear left seatbelt
    if scene_config.get("passenger_rear_left_safety_belt") == "YES":
        _c = cols.get("safety_belt_rear_left")
        if _c:
            for _o in _c.objects:
                if _o.type == 'MESH': inst_map[_o.name] = 7

    # Rear right seatbelt
    if scene_config.get("passenger_rear_right_safety_belt") == "YES":
        _c = cols.get("safety_belt_rear_right")
        if _c:
            for _o in _c.objects:
                if _o.type == 'MESH': inst_map[_o.name] = 8

    # Phone on codriver seat
    if scene_config.get("phone_codriver_seat") == "YES":
        _pcolor = scene_config.get("phone_codriver_seat_color", "BLACK").lower()
        _po = objs.get(f"phone_{_pcolor}")
        if _po: inst_map[_po.name] = 9

    # Cola bottle
    if scene_config.get("colabottle_codriver_seat") == "YES":
        _c = cols.get("colabottle")
        if _c:
            for _o in _c.objects:
                if _o.type == 'MESH': inst_map[_o.name] = 10

    # Cola can
    if scene_config.get("colacan_codriver_seat") == "YES":
        _c = cols.get("colacan")
        if _c:
            for _o in _c.objects:
                if _o.type == 'MESH': inst_map[_o.name] = 11

    # Suitcase
    if scene_config.get("suitcase") == "YES":
        _so = objs.get("suitcase")
        if _so: inst_map[_so.name] = 12

    # Baby seat
    if scene_config.get("baby_seat") == "YES":
        _bso = objs.get("baby_seat")
        if _bso: inst_map[_bso.name] = 13

    # Baby body (Plane = blanket, keep as background)
    if scene_config.get("baby") == "YES":
        _bc = cols.get("baby")
        if _bc:
            for _o in _bc.objects:
                if _o.type == 'MESH' and _o.name != "Plane":
                    inst_map[_o.name] = 14

    # Car: glass/window meshes → exterior class (18), seats → seat class (16), everything else → car_interior class (17)
    _WINDOW_MESH_NAMES = {
        "polymsh65_SUB1",
        "polymsh62_SUB0",
        "polymsh_detached29_SUB3",
    }
    _SEAT_MESH_NAMES = {
        "polymsh259_SUB3",
        "polymsh259_SUB15",
    }
    for _car_cn in ["car_part1", "car_part2", "car_mock", "car_imported"]:
        _car_col = cols.get(_car_cn)
        if _car_col:
            for _o in _car_col.objects:
                if _o.type == 'MESH':
                    if ("glass" in _o.name.lower() or "window" in _o.name.lower()
                            or _o.name in _WINDOW_MESH_NAMES):
                        inst_map[_o.name] = 18
                    elif _o.name in _SEAT_MESH_NAMES:
                        inst_map[_o.name] = 16
                    else:
                        inst_map[_o.name] = 17

    # Override ALL mesh materials
    for _obj in bpy.data.objects:
        if _obj.type != 'MESH':
            continue
        _idx = inst_map.get(_obj.name, 0)
        _obj.data.materials.clear()
        _obj.data.materials.append(inst_mats[_idx])

    # Disable all lights
    for _obj in bpy.data.objects:
        if _obj.type == 'LIGHT':
            _obj.hide_render = True

    # Disable Freestyle (may be left on from a prior render_gt_canny call)
    scene.render.use_freestyle = False
    scene.view_layers[0].use_freestyle = False

    # Near-black world (background)
    _world = scene.world
    if _world and _world.node_tree:
        _bg = _world.node_tree.nodes.get("Background")
        if _bg:
            _bg.inputs[0].default_value = (0.004, 0.004, 0.004, 1.0)
            _bg.inputs[1].default_value = 1.0

    # Disable compositor
    if hasattr(scene, "compositing_node_group"):
        scene.compositing_node_group = None
    else:
        scene.use_nodes = False

    scene.render.filepath = str(instance_seg_target)
    scene.render.image_settings.file_format = 'PNG'
    bpy.ops.render.render(write_still=True)


def render_gt_canny(canny_target: Path):
    """Freestyle render pass: ground-truth geometry edges (silhouettes + creases).

    Replaces CV-based Canny with mesh-topology-derived edges that are invariant
    to lighting, texture, and shadows.
    """
    # Enable Freestyle on scene + view layer
    scene.render.use_freestyle = True
    vl = scene.view_layers[0]
    vl.use_freestyle = True
    fs = vl.freestyle_settings

    # Clear any existing line sets and create a single clean one
    while len(fs.linesets) > 0:
        fs.linesets.remove(fs.linesets[0])
    ls = fs.linesets.new("GT_Edges")
    ls.select_silhouette = True
    ls.select_crease = True
    ls.select_border = True
    ls.select_contour = True
    ls.select_edge_mark = False
    ls.select_suggestive_contour = False
    ls.select_material_boundary = True

    # White lines, 1.5 px
    ls.linestyle.color = (1.0, 1.0, 1.0)
    ls.linestyle.thickness = 1.5

    # Invisible surfaces: black emission so only Freestyle lines are visible
    _black_mat = bpy.data.materials.new(name="_gt_canny_bg")
    _black_mat.use_nodes = True
    _nt = _black_mat.node_tree
    _nt.nodes.clear()
    _emit = _nt.nodes.new('ShaderNodeEmission')
    _emit.inputs['Color'].default_value = (0.0, 0.0, 0.0, 1.0)
    _emit.inputs['Strength'].default_value = 0.0
    _out = _nt.nodes.new('ShaderNodeOutputMaterial')
    _nt.links.new(_emit.outputs['Emission'], _out.inputs['Surface'])

    for _obj in bpy.data.objects:
        if _obj.type == 'MESH':
            _obj.data.materials.clear()
            _obj.data.materials.append(_black_mat)

    # Disable all lights
    for _obj in bpy.data.objects:
        if _obj.type == 'LIGHT':
            _obj.hide_render = True

    # Black world background
    _world = scene.world
    if _world and _world.node_tree:
        _bg = _world.node_tree.nodes.get("Background")
        if _bg:
            _bg.inputs[1].default_value = 0.0

    # Disable persistent render data (depth-pass compositor may have cached transform state)
    scene.render.use_persistent_data = False

    # Disable compositor (clear stale depth-pass nodes before Freestyle render)
    if hasattr(scene, "compositing_node_group"):
        scene.compositing_node_group = None
    else:
        scene.use_nodes = False

    # Force camera perspective render - find and set the active camera
    camera_obj = None
    # First, try to use the currently set camera
    if scene.camera and scene.camera.type == 'CAMERA':
        camera_obj = scene.camera
    else:
        # Otherwise, find the first camera in the scene
        for obj in bpy.data.objects:
            if obj.type == 'CAMERA':
                camera_obj = obj
                break
    
    if camera_obj:
        print(f"[CANNY] Using camera: {camera_obj.name}")
        scene.camera = camera_obj
        # Ensure view layer is updated to use this camera
        context.view_layer.update()
        # Try to set view to camera (may not be available in headless mode)
        try:
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    for space in area.spaces:
                        if space.type == 'VIEW_3D':
                            space.region_3d.view_perspective = 'CAMERA'
        except (AttributeError, RuntimeError):
            pass  # Headless rendering mode, no screen available
    else:
        print("[CANNY] WARNING: No camera found in scene!")

    # Render
    scene.render.filepath = str(canny_target)
    scene.render.image_settings.file_format = 'PNG'
    bpy.ops.render.render(write_still=True)


def render_segmentation(seg_target: Path):
    """Second render pass: flat emission materials per semantic class → colorized PNG."""
    # Linear-space RGB colors per class (Blender uses linear internally)
    SEG_COLORS = {
        0:  (0.004, 0.004, 0.004, 1.0),  # unlabeled    — near black
        1:  (0.706, 0.013, 0.057, 1.0),  # human        — bold crimson
        2:  (0.013, 0.148, 0.588, 1.0),  # phone        — deep blue
        3:  (0.678, 0.378, 0.0,   1.0),  # suitcase     — dark amber
        4:  (0.013, 0.329, 0.086, 1.0),  # baby_seat    — forest green
        5:  (0.706, 0.148, 0.0,   1.0),  # safety_belt  — deep orange
        6:  (0.378, 0.013, 0.092, 1.0),  # beverage     — dark magenta
        7:  (0.109, 0.057, 0.588, 1.0),  # car_interior — indigo
        8:  (0.033, 0.247, 0.478, 1.0),  # exterior     — steel blue
        9:  (0.247, 0.033, 0.378, 1.0),  # blanket      — deep violet
        10: (0.148, 0.057, 0.013, 1.0),  # seat         — dark brown
    }

    # Create flat emission materials
    seg_mats = {}
    for cid, rgba in SEG_COLORS.items():
        mat = bpy.data.materials.new(name=f"_seg_flat_{cid}")
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()
        emit_node = nt.nodes.new('ShaderNodeEmission')
        emit_node.inputs['Color'].default_value = rgba
        emit_node.inputs['Strength'].default_value = 1.0
        out_node = nt.nodes.new('ShaderNodeOutputMaterial')
        nt.links.new(emit_node.outputs['Emission'], out_node.inputs['Surface'])
        seg_mats[cid] = mat

    # Build object-name → class_id map
    seg_class_map = {}
    for _n in ["SMPLX-mesh-male", "SMPLX-mesh-female",
               "SMPLX-mesh-male.001", "SMPLX-mesh-male.002", "SMPLX-mesh-male.003"]:
        seg_class_map[_n] = 1
    _bc = cols.get("baby")
    if _bc:
        for _o in _bc.objects:
            if _o.type == 'MESH':
                # Plane = blanket; all other baby meshes = human
                if _o.name == "Plane":
                    seg_class_map[_o.name] = 9
                else:
                    seg_class_map[_o.name] = 1
    for _n in ["phone_male", "phone_female", "phone_black", "phone_white"]:
        seg_class_map[_n] = 2
    seg_class_map["suitcase"] = 3
    seg_class_map["baby_seat"] = 4
    for _cn in ["safety_belt_driver_male", "safety_belt_driver_female",
                "safety_belt_baby_back", "safety_belt_baby_front",
                "safety_belt_codriver", "safety_belt_rear_left", "safety_belt_rear_right"]:
        _col = cols.get(_cn)
        if _col:
            for _o in _col.objects:
                if _o.type == 'MESH':
                    seg_class_map[_o.name] = 5
    for _cn in ["colabottle", "colacan"]:
        _col = cols.get(_cn)
        if _col:
            for _o in _col.objects:
                if _o.type == 'MESH':
                    seg_class_map[_o.name] = 6
    # Car: glass/window meshes → exterior class (8), everything else → car_interior class (7)
    _WINDOW_MESH_NAMES = {
        "polymsh65_SUB1",
        "polymsh62_SUB0",
        "polymsh_detached29_SUB3",
    }
    _SEAT_MESH_NAMES = {
        "polymsh259_SUB3",
        "polymsh259_SUB15",
    }
    for _car_cn in ["car_part1", "car_part2", "car_mock", "car_imported"]:
        _car_col = cols.get(_car_cn)
        if _car_col:
            for _o in _car_col.objects:
                if _o.type == 'MESH':
                    if ("glass" in _o.name.lower() or "window" in _o.name.lower()
                            or _o.name in _WINDOW_MESH_NAMES):
                        seg_class_map[_o.name] = 8
                    elif _o.name in _SEAT_MESH_NAMES:
                        seg_class_map[_o.name] = 10
                    else:
                        seg_class_map[_o.name] = 7

    # Override ALL mesh materials with flat colors
    for _obj in bpy.data.objects:
        if _obj.type != 'MESH':
            continue
        cid = seg_class_map.get(_obj.name, 0)
        _obj.data.materials.clear()
        _obj.data.materials.append(seg_mats[cid])

    # Disable all lights
    for _obj in bpy.data.objects:
        if _obj.type == 'LIGHT':
            _obj.hide_render = True

    # Disable Freestyle (may be left on from a prior render_gt_canny call)
    scene.render.use_freestyle = False
    scene.view_layers[0].use_freestyle = False

    # Zero world background — but keep env color visible through windows
    _world = scene.world
    if _world and _world.node_tree:
        _bg = _world.node_tree.nodes.get("Background")
        if _bg:
            _bg.inputs[0].default_value = (0.216, 0.592, 0.827, 1.0)  # sky blue
            _bg.inputs[1].default_value = 1.0  # keep visible through windows

    # Disable compositor (avoid re-writing depth EXR)
    if hasattr(scene, "compositing_node_group"):
        scene.compositing_node_group = None
    else:
        scene.use_nodes = False

    # Render directly to PNG
    scene.render.filepath = str(seg_target)
    scene.render.image_settings.file_format = 'PNG'
    bpy.ops.render.render(write_still=True)
with open(json_path, "r") as f:
    config = json.load(f)

# Exposure: fixed bright default (was previously randomised via env_light_discrete)
exposure_compensation = config.get("exposure_compensation", 1.5)
scene.view_settings.exposure = exposure_compensation

# === Set up lights (derived from env — interior lights off by default) ===
objs["light_front"].data.energy = 0.0
objs["light_back_left"].data.energy = 0.0
objs["light_back_right"].data.energy = 0.0

# === Load and set environment ===
EXR_FOLDER = os.path.join(HERE, "..", "assets", "env_texture")
ENV_MAP = {
    "GARAGE":      ("garage.exr",      95,  (-0.5, 0, 0)),
    "URBAN":       ("urban.exr",       135, (0, 0, 0)),
    "HIGHWAY":     ("highway.exr",     142, (0, 0, 0)),
    "NATURE":      ("nature.exr",      80,  (0, 0, 0.05)),
    "COUNTRYSIDE": ("countryside.exr", 270,  (0, 0, 0.05)),
}

env = config.get("env", "URBAN")
strength = config.get("env_strength", 1.0)
cols["external_light"].hide_render = True
env_file, z_rot, transform = ENV_MAP.get(env)

if env == "GARAGE":
    cols["external_light"].hide_render = False
    for obj in cols["external_light"].objects:
        if obj.type == 'LIGHT':
            obj.data.energy = strength * 1000

z_rot_rad = deg2rad(z_rot)
selected_exr_path = os.path.join(EXR_FOLDER, env_file)
if not os.path.exists(selected_exr_path):
    raise FileNotFoundError(f"Environment texture not found: {selected_exr_path}")

world = scene.world
nt = world.node_tree
nodes = nt.nodes

mapping = nodes.get("Mapping")
env_tex = nodes.get("Environment Texture")
background = nodes.get("Background")

env_tex.image = bpy.data.images.load(selected_exr_path, check_existing=True)
mapping.inputs['Rotation'].default_value[2] = z_rot_rad
mapping.inputs['Location'].default_value = transform
background.inputs['Strength'].default_value = strength

# === Set up Driver ===
gender = config.get("driver_gender", "MALE")
height_m = 1.80   # fixed default
weight_kg = 85    # fixed default

# === Set up gender ===
obj_female = objs['SMPLX-mesh-female']
obj_male = objs['SMPLX-mesh-male']
if gender == "MALE":
    obj_male.hide_render = False
    obj_female.hide_render = True
    obj = obj_male
elif gender == "FEMALE":
    obj_male.hide_render = True
    obj_female.hide_render = False
    obj = obj_female

TEXTURE_MAP = {
    "MALE": {
        "WHITE": "0white_male.png",
        "BLACK": "1black_male.png",
    },
    "FEMALE": {
        "WHITE": "0white_female.png",
        "BLACK": "1black_female.png",
    }
}
driver_tshirt_color = config.get("driver_tshirt_color")
tex_name = TEXTURE_MAP[gender][driver_tshirt_color]
texture_path = TEXTURE_ROOT / gender.lower() / tex_name
smplx_set_texture(obj, texture_path)

# === phone ===
phone_driver = config.get("driver_phone", "NO")
phone_female = objs.get("phone_female")
phone_male = objs.get("phone_male")
phone_male.hide_render = True
phone_female.hide_render = True

if phone_driver == "NO":
    scene.frame_set(1)

elif phone_driver == "YES":
    if gender == "MALE":
        phone_male.hide_render = False
    else:
        phone_female.hide_render = False

    scene.frame_set(200)

# === emotion ===
emotion = config.get("driver_emotion", "HAPPY")
if emotion == "HAPPY":
    apply_expression_from_emotion("HAPPY", obj= obj)
    # if gender == "FEMALE":
    #   apply_expression_from_pkl("isu/blender/assets/smplx_gt/trainset_3dpeople_adults_bfh/10071_w_Mia_0_0.pkl", obj= obj)
elif emotion == "SERIOUS":
    if gender == "MALE":    
        apply_expression_from_pkl("isu/blender/assets/smplx_gt/trainset_3dpeople_adults_bfh/10186_m_Carlos_0_0.pkl", obj= obj)
    elif gender == "FEMALE":
        apply_expression_from_pkl("isu/blender/assets/smplx_gt/trainset_3dpeople_adults_bfh/10015_w_Myriam_0_0.pkl", obj= obj)
else:
    reset_expression_and_jaw(obj= obj)

armature, mesh = get_arm_and_mesh(obj)

# === height, weight ===
smplx_measurements_to_shape(
    obj=mesh,
    height_m=height_m,
    weight_kg=weight_kg,
    gender=gender
)
smplx_set_poseshapes(obj)  

# === Phone on co-driver seat ===
phone_codriver_seat = config.get("phone_codriver_seat", "NO")
phone_codriver_seat_color = config.get("phone_codriver_seat_color", "BLACK")
objs["phone_white"].hide_render = True
objs["phone_black"].hide_render = True
if phone_codriver_seat == "YES":
    objs[f"phone_{phone_codriver_seat_color.lower()}"].hide_render = False

# === suitcase === 
suitcase= config.get("suitcase")
if suitcase == "NO":
    cols["suitcase"].hide_render = True
else:  # YES
    cols["suitcase"].hide_render = False
    suitcase_location = config.get("suitcase_location")
    suitcase_pose = config.get("suitcase_pose")
    if suitcase_location == "CO_DRIVER_SEAT":
        objs["suitcase"].location = (-0.88, -0.04, 0.67)
    elif suitcase_location == "REAR_SEAT":
        if suitcase_pose == "UPRIGHT":
            objs["suitcase"].location = (-0.5, 1.02, 0.67)
        elif suitcase_pose == "FLAT":
            objs["suitcase"].location = (-0.35, 0.82, 0.67)
            objs["suitcase"].rotation_euler = (deg2rad(-90), 0, deg2rad(90))
    suitcase_color = config.get("suitcase_color")
    colors = {
        "ANTRACITE":  (0.06, 0.06, 0.06, 1.0),
        "RED":    (0.8, 0.0, 0.0, 1.0),
        "YELLOW": (0.75, 0.54, 0.33, 1.0),
    }

    recolor_red_to("XI_LU_Hardshell_Carry_on_mat", suitcase_color, colors[suitcase_color])

# == safety belt (driver) ===
driver_safety_belt = config.get("driver_safety_belt", "NO")
cols["safety_belt_driver_male"].hide_render = True
cols["safety_belt_driver_female"].hide_render = True
if driver_safety_belt == "YES":
    cols[f"safety_belt_driver_{gender.lower()}"].hide_render = False

# == cola on co-driver seat ===
colabottle_codriver_seat = config.get("colabottle_codriver_seat")
colacan_codriver_seat = config.get("colacan_codriver_seat")
cols["colabottle"].hide_render = True
cols["colacan"].hide_render = True
if colabottle_codriver_seat == "YES":
    cols["colabottle"].hide_render = False
if colacan_codriver_seat == "YES":
    cols["colacan"].hide_render = False

# == baby seat ===
baby_seat = config.get("baby_seat")
baby_seat_orientation = config.get("baby_seat_orientation")
#baby_seat_safety_belt = config.get("baby_seat_safety_belt")
baby = config.get("baby")
cols["baby_seat"].hide_render = True
cols["baby"].hide_render = True
if baby_seat == "YES":
    cols["baby_seat"].hide_render = False
    if baby_seat_orientation == "REAR_FACING":
        objs["baby_seat"].location = (-0.917, -0.144, 0.46)
        objs["baby_seat"].rotation_euler = (deg2rad(90), 0, deg2rad(180))
    elif baby_seat_orientation == "FRONT_FACING":
        objs["baby_seat"].location = (-0.917, -0.144, 0.46)
        objs["baby_seat"].rotation_euler = (deg2rad(90), 0, 0)
    # if baby_seat_orientation == "REAR_FACING":
    #     objs["baby_seat"].location = (-0.81, -0.23, 0.63)
    #     objs["baby_seat"].rotation_euler = (deg2rad(75), 0, deg2rad(184))
    # elif baby_seat_orientation == "FRONT_FACING":
    #     objs["baby_seat"].location = (-0.81, -0.05, 0.6)
    #     objs["baby_seat"].rotation_euler = (deg2rad(67), 0, deg2rad(4))
    # if baby_seat_safety_belt == "YES":
    #     if baby_seat_orientation == "REAR_FACING":
    #         cols["safety_belt_baby_back"].hide_render = False
    #     elif baby_seat_orientation == "FRONT_FACING":
    #         cols["safety_belt_baby_front"].hide_render = False

    if baby == "YES":
        cols["baby"].hide_render = False

# === passenger_codriver (co-driver seat occupant) ===
passenger_front = config.get("passenger_codriver", "NO")
passenger_codriver_col = cols.get("Passenger_codriver")
if passenger_codriver_col is not None:
    passenger_codriver_col.hide_render = True
    if passenger_front == "YES":
        passenger_codriver_col.hide_render = False
        for obj in passenger_codriver_col.objects:
            if "SMPLX" in obj.name and obj.type == 'MESH':
                p_tshirt = config.get("passenger_codriver_tshirt_color", "WHITE")
                smplx_set_texture(obj, TEXTURE_ROOT / "male" / TEXTURE_MAP["MALE"][p_tshirt])
                p_emotion = config.get("passenger_codriver_emotion", "HAPPY")
                if p_emotion == "SERIOUS":
                    apply_expression_from_pkl(
                        "isu/blender/assets/smplx_gt/trainset_3dpeople_adults_bfh/10186_m_Carlos_0_0.pkl",
                        obj=obj
                    )
                else:
                    apply_expression_from_emotion("HAPPY", obj=obj)
                # Apply configurable codriver head angle (positive = look right)
                head_angle = config.get("passenger_codriver_head_angle", 45)
                apply_head_rotation(obj, yaw_degrees=head_angle)
                # Use manual arm pose from Blender scene (no Python override)

# === passenger_back_seat_left ===
passenger_back_left = config.get("passenger_back_seat_left", "NO")
passenger_back_left_col = cols.get("Passenger_rear_left")
if passenger_back_left_col is not None:
    passenger_back_left_col.hide_render = True
    if passenger_back_left == "YES":
        passenger_back_left_col.hide_render = False
        for obj in passenger_back_left_col.objects:
            if "SMPLX" in obj.name and obj.type == 'MESH':
                p_tshirt = config.get("passenger_rear_left_tshirt_color", "WHITE")
                smplx_set_texture(obj, TEXTURE_ROOT / "male" / TEXTURE_MAP["MALE"][p_tshirt])
                p_emotion = config.get("passenger_rear_left_emotion", "HAPPY")
                if p_emotion == "SERIOUS":
                    apply_expression_from_pkl(
                        "isu/blender/assets/smplx_gt/trainset_3dpeople_adults_bfh/10186_m_Carlos_0_0.pkl",
                        obj=obj
                    )
                else:
                    apply_expression_from_emotion("HAPPY", obj=obj)
                # Apply configurable rear left head angle (negative = look left, positive = look right)
                head_angle = config.get("passenger_rear_left_head_angle", -45)
                apply_head_rotation(obj, yaw_degrees=head_angle)

# === passenger_back_seat_right ===
passenger_back_right = config.get("passenger_back_seat_right", "NO")
passenger_back_right_col = cols.get("Passenger_rear_right")
if passenger_back_right_col is not None:
    passenger_back_right_col.hide_render = True
    if passenger_back_right == "YES":
        passenger_back_right_col.hide_render = False
        for obj in passenger_back_right_col.objects:
            if "SMPLX" in obj.name and obj.type == 'MESH':
                p_tshirt = config.get("passenger_rear_right_tshirt_color", "WHITE")
                smplx_set_texture(obj, TEXTURE_ROOT / "male" / TEXTURE_MAP["MALE"][p_tshirt])
                p_emotion = config.get("passenger_rear_right_emotion", "HAPPY")
                # Apply configurable rear right head angle (positive = look right)
                head_angle = config.get("passenger_rear_right_head_angle", 45)
                apply_head_rotation(obj, yaw_degrees=head_angle)
                if p_emotion == "SERIOUS":
                    apply_expression_from_pkl(
                        "isu/blender/assets/smplx_gt/trainset_3dpeople_adults_bfh/10186_m_Carlos_0_0.pkl",
                        obj=obj
                    )
                else:
                    apply_expression_from_emotion("HAPPY", obj=obj)

# === passenger seatbelts ===
cols["safety_belt_codriver"].hide_render = True
cols["safety_belt_rear_left"].hide_render = True
cols["safety_belt_rear_right"].hide_render = True
if config.get("codriver_safety_belt", "NO") == "YES" and passenger_front == "YES":
    cols["safety_belt_codriver"].hide_render = False
if config.get("passenger_rear_left_safety_belt", "NO") == "YES" and passenger_back_left == "YES":
    cols["safety_belt_rear_left"].hide_render = False
if config.get("passenger_rear_right_safety_belt", "NO") == "YES" and passenger_back_right == "YES":
    cols["safety_belt_rear_right"].hide_render = False

# === Assign pass indices for segmentation ===
# Human: driver meshes
for _obj_name in ["SMPLX-mesh-male", "SMPLX-mesh-female",
                   "SMPLX-mesh-male.001", "SMPLX-mesh-male.002", "SMPLX-mesh-male.003"]:
    _o = objs.get(_obj_name)
    if _o and _o.type == 'MESH':
        _o.pass_index = PASS_INDEX["human"]
# Human: baby
_baby_col = cols.get("baby")
if _baby_col:
    for _o in _baby_col.objects:
        if _o.type == 'MESH':
            _o.pass_index = PASS_INDEX["human"]
# Phone (driver hand + co-driver seat)
for _obj_name in ["phone_male", "phone_female", "phone_black", "phone_white"]:
    _o = objs.get(_obj_name)
    if _o and _o.type == 'MESH':
        _o.pass_index = PASS_INDEX["phone"]
# Suitcase
_o = objs.get("suitcase")
if _o and _o.type == 'MESH':
    _o.pass_index = PASS_INDEX["suitcase"]
# Baby seat
_o = objs.get("baby_seat")
if _o and _o.type == 'MESH':
    _o.pass_index = PASS_INDEX["baby_seat"]
# Safety belts
for _col_name in ["safety_belt_driver_male", "safety_belt_driver_female",
                   "safety_belt_baby_back", "safety_belt_baby_front",
                   "safety_belt_codriver", "safety_belt_rear_left", "safety_belt_rear_right"]:
    _col = cols.get(_col_name)
    if _col:
        for _o in _col.objects:
            if _o.type == 'MESH':
                _o.pass_index = PASS_INDEX["safety_belt"]
# Beverages
for _col_name in ["colabottle", "colacan"]:
    _col = cols.get(_col_name)
    if _col:
        for _o in _col.objects:
            if _o.type == 'MESH':
                _o.pass_index = PASS_INDEX["beverage"]

print(f"[TIME] Environment setup finished in {time.perf_counter() - t0:.3f}s")
t1 = time.perf_counter()
# === RENDERING ===
context.view_layer.update()
if depth_path is not None:
    setup_render_outputs(depth_path)
scene.render.filepath = image_path

print(f"[TIME] scene update finished in {time.perf_counter() - t1:.3f}s")
t2 = time.perf_counter()

bpy.ops.render.render(write_still=True)

if depth_path is not None:
    if not depth_path.exists():
        depth_candidates = sorted(depth_path.parent.glob(f"{depth_path.stem}*.exr"))
        if depth_candidates:
            os.replace(str(depth_candidates[-1]), str(depth_path))
        else:
            raise RuntimeError(f"Depth output not found for target: {depth_path}")

if seg_path is not None:
    render_segmentation(seg_path)
    if not seg_path.exists():
        raise RuntimeError(f"Segmentation render not found: {seg_path}")

if canny_path is not None:
    render_gt_canny(canny_path)
    if not canny_path.exists():
        raise RuntimeError(f"GT Canny render not found: {canny_path}")

if instance_seg_path is not None:
    render_instance_seg(instance_seg_path, config)

print(f"[TIME] Rendering finished in {time.perf_counter() - t2:.3f}s")

#exit blender
bpy.ops.wm.quit_blender()
sys.exit()