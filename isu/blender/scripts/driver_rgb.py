import sys
import os
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.append(HERE)          
# Or if utils.py is one dir up:
# sys.path.append(os.path.dirname(HERE))

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
json_path = argv[argv.index("--") + 1]
image_path = argv[argv.index("--") + 2]
with open(json_path, "r") as f:
    config = json.load(f)

exposure_compensation = config.get("exposure_compensation", 0.0)
scene.view_settings.exposure = exposure_compensation

# === Set up lights ===
objs["light_front"].data.energy = config.get("light_front", 0.0)
objs["light_back_left"].data.energy = config.get("light_back_left", 0.0)
objs["light_back_right"].data.energy = config.get("light_back_right", 0.0)

# === Load and set environment ===
EXR_FOLDER = "isu/blender/assets/env_texture"
ENV_MAP = {
    "GARAGE":      ("garage.exr",      95,  (-0.5, 0, 0)),
    "URBAN":       ("urban.exr",       100, (0.8, 0, 0.1)),
    "HIGHWAY":     ("highway.exr",     150, (-0.8, 0, 0)),
    "NATURE":      ("nature.exr",      80,  (0, 0, 0.05)),
    "COUNTRYSIDE": ("countryside.exr", 90,  (0, 0, 0.05)),
}

env = config.get("env", "GARAGE")
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

world = scene.world
nt = world.node_tree
nodes = nt.nodes

mapping = nodes.get("Mapping")
env_tex = nodes.get("Environment Texture")
background = nodes.get("Background")
mapping.inputs['Rotation'].default_value[2] = z_rot_rad
mapping.inputs['Location'].default_value = transform
background.inputs['Strength'].default_value = strength

# === Set up Driver ===
gender = config.get("gender", None)
height_m = config.get("height_m", None)
weight_kg = config.get("weight_kg", None)

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
texture_path = os.path.join("isu/blender/assets/human_texture", gender.lower(), tex_name)
smplx_set_texture(obj, texture_path)

# === phone ===
phone_driver = config.get("phone_driver", "NO")
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

    frame = config.get("phone_driver_pose_frame", 200)
    scene.frame_set(frame)

# === emotion ===
emotion = config.get("emotion", "HAPPY")
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
            suitcase_rotation = config.get("suitcase_rotation")
            objs["suitcase"].rotation_euler = (deg2rad(-90), 0, deg2rad(suitcase_rotation))
    suitcase_color = config.get("suitcase_color")
    colors = {
        "ANTRACITE":  (0.06, 0.06, 0.06, 1.0),
        "RED":    (0.8, 0.0, 0.0, 1.0),
        "YELLOW": (0.75, 0.54, 0.33, 1.0),
    }

    recolor_red_to("XI_LU_Hardshell_Carry_on_mat", suitcase_color, colors[suitcase_color])

# == safety belt ===
safety_belt = config.get("safety_belt", "NO")
cols[f"safety_belt_male"].hide_render = True
cols[f"safety_belt_female"].hide_render = True
if safety_belt == "YES":
    cols[f"safety_belt_{gender.lower()}"].hide_render = False

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
        objs["baby_seat"].location = (-0.81, -0.22, 0.46)
        objs["baby_seat"].rotation_euler = (deg2rad(90), 0, deg2rad(180))
    elif baby_seat_orientation == "FRONT_FACING":
        objs["baby_seat"].location = (-0.81, -0.086, 0.46)
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

print(f"[TIME] Environment setup finished in {time.perf_counter() - t0:.3f}s")
t1 = time.perf_counter()
# === RENDERING ===
context.view_layer.update()
scene.render.filepath = image_path

print(f"[TIME] scene update finished in {time.perf_counter() - t1:.3f}s")
t2 = time.perf_counter()

bpy.ops.render.render(write_still=True)

print(f"[TIME] Rendering finished in {time.perf_counter() - t2:.3f}s")

#exit blender
bpy.ops.wm.quit_blender()
sys.exit()