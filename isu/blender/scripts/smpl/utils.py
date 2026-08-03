import bpy
from mathutils import Vector, Quaternion
import json
import numpy as np
import os
import math
import pickle
from typing import Dict, Tuple, Optional

SMPLX_JOINT_NAMES = [
    'pelvis','left_hip','right_hip','spine1','left_knee','right_knee','spine2','left_ankle','right_ankle','spine3', 'left_foot','right_foot','neck','left_collar','right_collar','head','left_shoulder','right_shoulder','left_elbow', 'right_elbow','left_wrist','right_wrist',
    'jaw','left_eye_smplhf','right_eye_smplhf','left_index1','left_index2','left_index3','left_middle1','left_middle2','left_middle3','left_pinky1','left_pinky2','left_pinky3','left_ring1','left_ring2','left_ring3','left_thumb1','left_thumb2','left_thumb3','right_index1','right_index2','right_index3','right_middle1','right_middle2','right_middle3','right_pinky1','right_pinky2','right_pinky3','right_ring1','right_ring2','right_ring3','right_thumb1','right_thumb2','right_thumb3'
]
NUM_SMPLX_JOINTS = len(SMPLX_JOINT_NAMES)
NUM_SMPLX_BODYJOINTS = 21
NUM_SMPLX_HANDJOINTS = 15

def get_arm_and_mesh(obj):
    if obj.type == 'MESH':
        return obj.parent, obj
    raise RuntimeError("select the SMPL-X armature or mesh")

def reset_expression_and_jaw(obj):
    arm, mesh = get_arm_and_mesh(obj)
    # zero Exp***
    if mesh.data.shape_keys:
        for kb in mesh.data.shape_keys.key_blocks:
            if kb.name.startswith("Exp"):
                kb.value = 0.0
    # zero jaw pose
    jaw = arm.pose.bones.get("jaw")
    if jaw:
        jaw.rotation_mode = 'QUATERNION'
        jaw.rotation_quaternion = Quaternion()  # identity
    print("[OK] expression + jaw reset to neutral.")

def apply_expression_from_pkl(filepath, obj):
    arm, mesh = get_arm_and_mesh(obj)
    with open(filepath, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    exp = np.array(data.get("expression", []), dtype=float).tolist()

    # write to Shape Keys named Exp000, Exp001, ...(auto truncate/pad with zeros)
    if not mesh.data.shape_keys:
        raise RuntimeError("No Shape Keys for current mesh.")
    kb = mesh.data.shape_keys.key_blocks
    i = 0
    while True:
        name = f"Exp{i:03d}"
        if name not in kb: break
        # print(exp)
        kb[name].value = float(exp[0][i]) if i < len(exp) else 0.0
        i += 1
    print(f"[OK] write to {min(len(exp), i)} expression coefficients.")

def apply_expression_from_emotion(emotion, obj):
    arm, mesh = get_arm_and_mesh(obj)
    # with open(filepath, "rb") as f:
    #     data = pickle.load(f, encoding="latin1")
    # exp = np.array(data.get("expression", []), dtype=float).tolist()
    if emotion == "HAPPY":
        exp = [ [5.0, 0.0, 1.0, 0.0, 5.0, -2.5] + [0.0]*24 ]  # smile
    elif emotion == "SERIOUS":
        exp = [ [0.0]*30 ]  # neutral

    # write to Shape Keys named Exp000, Exp001, ...(auto truncate/pad with zeros)
    if not mesh.data.shape_keys:
        raise RuntimeError("No Shape Keys for current mesh.")
    kb = mesh.data.shape_keys.key_blocks
    i = 0
    while True:
        name = f"Exp{i:03d}"
        if name not in kb: break
        # print(exp)
        kb[name].value = float(exp[0][i]) if i < len(exp) else 0.0
        i += 1
    print(f"[OK] write to {min(len(exp), i)} expression coefficients.")

_MEAS2BETAS_CACHE = {"female": None, "male": None, "neutral": None}
def _load_meas2betas(gender: str):
    """Load (A, B) for height/weight → betas and cache them."""
    if _MEAS2BETAS_CACHE.get(gender) is not None:
        return _MEAS2BETAS_CACHE[gender]

    path = os.path.dirname(os.path.realpath(__file__))
    reg_path = os.path.join(path, "data", f"smplx_measurements_to_betas_{gender.lower()}.json")
    with open(reg_path) as f:
        data = json.load(f)
    A = np.asarray(data["A"]).reshape(-1, 2)   # (num_betas, 2)
    B = np.asarray(data["B"]).reshape(-1, 1)   # (num_betas, 1)
    _MEAS2BETAS_CACHE[gender] = (A, B)
    return A, B

# Module-level cache (persists across calls while the add-on is loaded)
_J_REGRESSOR_CACHE: Dict[str, Dict[str, Optional[Tuple[np.ndarray, np.ndarray]]]] = {
    "female": {"10": None, "300": None, "300_lh": None},
    "male":   {"10": None, "300": None, "300_lh": None},
    "neutral":{"10": None, "300": None, "300_lh": None},
}

def _load_regressor(gender: str, betas_key: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the betas->joints regressor and template joints for the given gender and betas key.
    betas_key ∈ {"10", "300", "300_lh"}.
    """
    if betas_key not in {"10", "300", "300_lh"}:
        raise ValueError(f"No betas-to-joints regressor for betas key '{betas_key}'")

    prefix = ""
    suffix = ""
    if betas_key == "300":
        suffix = "_300"
    elif betas_key == "300_lh":
        prefix = "lh_"
        suffix = "_300"

    path = os.path.dirname(os.path.realpath(__file__))
    regressor_path = os.path.join(path, "data", f"smplx_betas_to_joints_{prefix}{gender.lower()}{suffix}.json")

    with open(regressor_path, "r") as f:
        data = json.load(f)

    betasJ_regr = np.asarray(data["betasJ_regr"])
    template_J  = np.asarray(data["template_J"])
    return betasJ_regr, template_J

def _get_regressor(gender: str, betas_key: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get a cached regressor (and template_J). If absent, load and cache it.
    """
    cached = _J_REGRESSOR_CACHE.get(gender, {}).get(betas_key)
    if cached is None:
        # Ensure gender exists in cache
        if gender not in _J_REGRESSOR_CACHE:
            _J_REGRESSOR_CACHE[gender] = {"10": None, "300": None, "300_lh": None}
        betasJ_regr, template_J = _load_regressor(gender, betas_key)
        _J_REGRESSOR_CACHE[gender][betas_key] = (betasJ_regr, template_J)
        return betasJ_regr, template_J
    return cached

def smplx_measurements_to_shape(
    obj: bpy.types.Object,
    *,
    height_m: float = None,
    weight_kg: float = None,
    gender: str = None,
    update_joints: bool = True,
):
    # Compute betas from measurements
    A, B = _load_meas2betas(gender)
    height_cm = float(height_m) * 100.0
    v_root = float(weight_kg) ** (1.0 / 3.0)
    meas = np.asarray([[height_cm], [v_root]])  # (2,1)
    betas = (A @ meas + B).reshape(-1)          # (num_betas,)

    # Apply betas to Shape### keys
    bpy.ops.object.mode_set(mode='OBJECT')
    kblocks = obj.data.shape_keys.key_blocks
    for i, value in enumerate(betas):
        name = f"Shape{i:03d}"
        if name not in kblocks:
            # If the model has fewer betas than the regressor predicts, just stop.
            break
        kb = kblocks[name]
        kb.value = float(value)

    if update_joints:
        smplx_update_joint_locations(obj, gender)

def smplx_update_joint_locations(obj: bpy.types.Object, gender: str) -> np.ndarray:
    if obj is None or obj.type != 'MESH':
        raise TypeError("smplx_update_joint_locations: 'obj' must be a Mesh object.")

    armature = obj.parent
    if armature is None or armature.type != 'ARMATURE':
        raise ValueError("smplx_update_joint_locations: mesh must have a parent Armature.")

    # --- Collect betas from shape keys ---
    if not obj.data.shape_keys or not obj.data.shape_keys.key_blocks:
        raise RuntimeError("smplx_update_joint_locations: object has no shape keys (betas).")

    betas = [kb.value for kb in obj.data.shape_keys.key_blocks
             if kb.name.startswith("Shape")]
    num_betas = len(betas)
    if num_betas not in (10, 300):
        raise RuntimeError(f"Unsupported number of betas: {num_betas} (expected 10 or 300).")

    betas = np.asarray(betas, dtype=np.float32)
    version = str(obj.get("smplx_version", "")).lower()

    if num_betas == 10:
        betas_key = "10"
    else:  
        betas_key = "300_lh" if version == "locked_head" else "300"

    # Compute joint locations: (N x B) @ (B,) + (N x 3) -> (N x 3)
    betas_to_joints, template_j = _get_regressor(gender, betas_key)
    joint_locations = (betas_to_joints @ betas) + template_j  # shape: (NUM_SMPLX_JOINTS, 3)

    # --- Edit bones in the armature ---
    ctx = bpy.context
    prev_active = ctx.view_layer.objects.active
    prev_mode = prev_active.mode if prev_active else 'OBJECT'

    def _set_active(ob):
        ctx.view_layer.objects.active = ob

    try:
        # Ensure mesh is in OBJECT mode (so we can freely switch to armature edit)
        _set_active(obj)
        if ctx.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        # Enter armature EDIT mode
        _set_active(armature)
        if ctx.object.mode != 'EDIT':
            bpy.ops.object.mode_set(mode='EDIT')

        # Update each bone's position
        for idx in range(NUM_SMPLX_JOINTS):
            joint_name = SMPLX_JOINT_NAMES[idx]
            try:
                bone = armature.data.edit_bones[joint_name]
            except KeyError:
                # Bone not found; skip gracefully
                print(f"[SMPL-X] Warning: Bone '{joint_name}' not found; skipping.")
                continue

            # Reset bone and place at joint location (convert from SMPL-X to Blender coords)
            bone.head = (0.0, 0.0, 0.0)
            bone.tail = (0.0, 0.0, 0.1)

            j = joint_locations[idx]  # (x, y, z) in SMPL-X
            # Convert SMPL-X (x, y, z) -> Blender (x, -z, y)
            bone_start = Vector((float(j[0]), float(-j[2]), float(j[1])))
            bone.translate(bone_start)

    finally:
        # Leave EDIT mode
        if ctx.object and ctx.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        # Restore previous active object and its mode
        if prev_active is not None:
            _set_active(prev_active)
            try:
                if prev_mode != 'OBJECT':
                    bpy.ops.object.mode_set(mode=prev_mode)
            except Exception:
                # If restoring exact mode fails (e.g., not supported for that object),
                # at least ensure we're in OBJECT mode.
                try:
                    bpy.ops.object.mode_set(mode='OBJECT')
                except Exception:
                    pass

    return joint_locations

def smplx_set_texture(obj: bpy.types.Object, texture: str) -> bool:
    """
    Set (or clear) the base-color texture on the first material of a mesh object.

    Args:
        obj: The mesh object to modify.
        texture: One of:
            - 'NONE' to remove the image texture node and unlink it.
            - 'UV_GRID' or 'COLOR_GRID' (generated images).
            - A filename found in bpy.data.images or located at <this_file_dir>/data/<texture>.
    Returns:
        True on success, False if nothing was changed due to a recoverable issue.
    Raises:
        ValueError for invalid inputs.
    """

    # --- basic checks ---
    if obj is None or obj.type != 'MESH':
        raise ValueError("smplx_set_texture: obj must be a MESH")

    if len(obj.data.materials) == 0 or obj.data.materials[0] is None:
        print(f"[smplx_set_texture] Selected mesh has no material: {obj.name}")
        return False

    mat = obj.data.materials[0]
    if not mat.use_nodes:
        mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # find or create shader node (prefer Principled BSDF)
    node_shader = next((n for n in nodes if n.type.startswith('BSDF')), None)
    if node_shader is None:
        node_shader = nodes.get("Principled BSDF")
    if node_shader is None:
        node_shader = nodes.new(type="ShaderNodeBsdfPrincipled")
        node_shader.location = (0, 0)

    # find existing image texture node (if any)
    node_texture = next((n for n in nodes if n.type == 'TEX_IMAGE'), None)

    # --- handle 'NONE' (remove texture) ---
    if texture == 'NONE':
        if node_texture is not None:
            # unlink output 0 from whatever it feeds
            out_sock = node_texture.outputs[0]
            for link in list(out_sock.links):
                links.remove(link)
            nodes.remove(node_texture)
        # (nothing else to do; shader base color falls back to its default)
        _ensure_viewport_material_mode()
        return True

    # --- ensure we have a texture node ---
    if node_texture is None:
        node_texture = nodes.new(type="ShaderNodeTexImage")
        node_texture.label = "BaseColor Texture"
        node_texture.location = (node_shader.location.x - 400, node_shader.location.y)

    # --- resolve image to assign ---
    if texture in ('UV_GRID', 'COLOR_GRID'):
        if texture not in bpy.data.images:
            bpy.ops.image.new(name=texture, generated_type=texture)
        image = bpy.data.images[texture]
    else:
        texture_path = texture
        if not os.path.isfile(texture_path):
            raise ValueError(f"smplx_set_texture: image file does not exist: {texture_path}")
        image = bpy.data.images.load(texture_path)

    node_texture.image = image

    # --- link texture base color to shader base color if not already ---
    base_color_input = node_shader.inputs.get("Base Color") or node_shader.inputs[0]
    if not any(l.from_node == node_texture and l.to_socket == base_color_input for l in links):
        links.new(node_texture.outputs[0], base_color_input)

    _ensure_viewport_material_mode()
    return True

def _ensure_viewport_material_mode():
    # Switch active 3D viewport to MATERIAL shading (if applicable)
    area = next((a for a in bpy.context.screen.areas if a.type == 'VIEW_3D'), None)
    if area:
        for space in area.spaces:
            if space.type == 'VIEW_3D':
                space.shading.type = 'MATERIAL'
                break



def _rodrigues_to_mat(rotvec: np.ndarray) -> np.ndarray:
    """Rodrigues vector (3,) -> rotation matrix (3,3)."""
    theta = np.linalg.norm(rotvec)
    if theta > 0.0:
        r = (rotvec / theta).reshape(3, 1)
    else:
        # zero rotation -> identity
        return np.eye(3)
    cost = np.cos(theta)
    rrt = r @ r.T
    rx = np.array([[0, -r[2, 0], r[1, 0]],
                   [r[2, 0], 0, -r[0, 0]],
                   [-r[1, 0], r[0, 0], 0]], dtype=float)
    return cost * np.eye(3) + (1.0 - cost) * rrt + np.sin(theta) * rx

def rodrigues_from_pose(armature, bone_name):
    # Use quaternion mode for all bone rotations
    if armature.pose.bones[bone_name].rotation_mode != 'QUATERNION':
        armature.pose.bones[bone_name].rotation_mode = 'QUATERNION'

    quat = armature.pose.bones[bone_name].rotation_quaternion
    (axis, angle) = quat.to_axis_angle()
    rodrigues = axis
    rodrigues.normalize()
    rodrigues = rodrigues * angle
    return rodrigues

def _pose_to_corrective_weights(pose_flat_3n: np.ndarray) -> np.ndarray:
    """
    SMPL-X corrective weights from pose.
    Input:  flat array of length NUM_SMPLX_JOINTS*3 (pelvis first).
    Output: concatenated (R-I).ravel() for all joints except pelvis.
    """
    rod_rots = pose_flat_3n.reshape(-1, 3)
    mats = [_rodrigues_to_mat(r) for r in rod_rots]
    # Skip pelvis (index 0)
    weights = np.concatenate([(m - np.eye(3)).ravel() for m in mats[1:]])
    return weights

def smplx_set_poseshapes(obj_or_armature: bpy.types.Object) -> np.ndarray:
    """
    Compute & set corrective pose shape weights for the current pose.

    Parameters
    ----------
    obj_or_armature : Object
        Either the SMPL-X skinned MESH or its ARMATURE.
    Returns
    -------
    np.ndarray
        The computed pose-corrective weight vector.
    """
    # Resolve mesh/armature pair
    if obj_or_armature.type == 'ARMATURE':
        armature = obj_or_armature
        if not armature.children:
            raise RuntimeError("Armature has no child mesh.")
        obj = armature.children[0]
    elif obj_or_armature.type == 'MESH':
        obj = obj_or_armature
        if obj.parent is None or obj.parent.type != 'ARMATURE':
            raise RuntimeError("Mesh must be parented to an Armature.")
        armature = obj.parent
    else:
        raise TypeError("Pass a SMPL-X MESH or its ARMATURE.")

    # Build Rodrigues pose (3 values per joint)
    pose = np.zeros((NUM_SMPLX_JOINTS * 3,), dtype=float)
    for j_idx, j_name in enumerate(SMPLX_JOINT_NAMES):
        rod = rodrigues_from_pose(armature, j_name)  
        pose[j_idx*3 + 0] = rod[0]
        pose[j_idx*3 + 1] = rod[1]
        pose[j_idx*3 + 2] = rod[2]

    # Compute corrective weights
    poseweights = _pose_to_corrective_weights(pose)

    # Write weights into Pose### keys
    if not obj.data.shape_keys or not obj.data.shape_keys.key_blocks:
        raise RuntimeError("Mesh has no shape keys.")
    kblocks = obj.data.shape_keys.key_blocks
    for idx, w in enumerate(poseweights):
        key_name = f"Pose{idx:03d}"
        if key_name in kblocks:
            kblocks[key_name].value = float(w)
        else:
            # Keep going but warn in console
            print(f"[SMPLX] Missing shape key: {key_name}")

    return poseweights