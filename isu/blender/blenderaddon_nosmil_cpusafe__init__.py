
bl_info = {
    "name": "SMPL-X for Blender",
    "author": "Joachim Tesch, Max Planck Institute for Intelligent Systems",
    "version": (2024, 11, 29),
    "blender": (3, 6, 0),
    "location": "Viewport > Right panel",
    "description": "SMPL-X for Blender",
    "wiki_url": "https://smpl-x.is.tue.mpg.de/",
    "category": "SMPL-X"}

import bpy
from bpy_extras.io_utils import ImportHelper,ExportHelper
from mathutils import Vector, Quaternion
from bpy.props import ( BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty )
from bpy.types import ( PropertyGroup )

import json
from math import radians
import numpy as np
import os
import pickle
import io

# ---- CPU-safe Torch/PKL loader helpers --------------------------------------

try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False

def _to_np(x):
    """Convert torch tensors to numpy on CPU; otherwise just np.array(x)."""
    if _TORCH_AVAILABLE:
        try:
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
        except Exception:
            pass
    return np.array(x)

def smplx_load_pickle_cpu(filepath):
    """
    Robust loader that works on CPU-only machines:
    1) Try torch.load(..., weights_only=True, map_location='cpu')  [safe]
    2) Try safe load with allow-listed global for older checkpoints
    3) Fallback: monkey-patch torch.storage._load_from_bytes so that any
       storages inside a *plain pickle* are remapped to CPU, then use pickle.loads.
    """
    with open(filepath, "rb") as f:
        raw = f.read()

        cpu = torch.device("cpu")

        # 1) Safe path
        try:
            return torch.load(io.BytesIO(raw), map_location=cpu, weights_only=True)
        except Exception:
            pass

        # 2) Safe path with allowlisted global for older pickles
        try:
            from torch import serialization as _ser
            with _ser.safe_globals([torch.storage._load_from_bytes]):
                return torch.load(io.BytesIO(raw), map_location=cpu, weights_only=True)
        except Exception:
            pass

        # 3) Plain-pickle fallback with CPU-forcing monkey patch
        orig_lfb = getattr(torch.storage, "_load_from_bytes", None)

        def _load_from_bytes_cpu(b):
            # ensure nested storages are loaded onto CPU using legacy loader
            return torch.load(io.BytesIO(b), map_location=cpu, weights_only=False)

        try:
            if orig_lfb is not None:
                torch.storage._load_from_bytes = _load_from_bytes_cpu
            # IMPORTANT: do NOT torch.load(...) the top-level here; use pickle.
            return pickle.loads(raw, encoding="latin1")
        finally:
            if orig_lfb is not None:
                torch.storage._load_from_bytes = orig_lfb

    # Torch not available: try plain pickle (may fail if torch storages present)
    return pickle.loads(raw, encoding="latin1")

# -----------------------------------------------------------------------------

# SMPL-X globals
USE_SMPLX_2020 = False
SMPLX_MODELFILE = "smplx_model_20210421.blend"
SMPLX_MODELFILE_300 = "smplx_model_20230302.blend"
SMPLX_MODELFILE_LH = "smplx_model_lh_20230302.blend"
SMPLX_MODELFILE_2020 = "smplx_model_2020_20230227.blend"
SMPLX_JOINT_NAMES = [
    'pelvis','left_hip','right_hip','spine1','left_knee','right_knee','spine2','left_ankle','right_ankle','spine3', 'left_foot','right_foot','neck','left_collar','right_collar','head','left_shoulder','right_shoulder','left_elbow', 'right_elbow','left_wrist','right_wrist',
    'jaw','left_eye_smplhf','right_eye_smplhf','left_index1','left_index2','left_index3','left_middle1','left_middle2','left_middle3','left_pinky1','left_pinky2','left_pinky3','left_ring1','left_ring2','left_ring3','left_thumb1','left_thumb2','left_thumb3','right_index1','right_index2','right_index3','right_middle1','right_middle2','right_middle3','right_pinky1','right_pinky2','right_pinky3','right_ring1','right_ring2','right_ring3','right_thumb1','right_thumb2','right_thumb3'
]
NUM_SMPLX_JOINTS = len(SMPLX_JOINT_NAMES)
NUM_SMPLX_BODYJOINTS = 21
NUM_SMPLX_HANDJOINTS = 15
SHAPEKEY_VALUE_RANGE=5
# End SMPL-X globals

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

def update_corrective_poseshapes(self, context):
    if self.smplx_corrective_poseshapes:
        bpy.ops.object.smplx_set_poseshapes('EXEC_DEFAULT')
    else:
        bpy.ops.object.smplx_reset_poseshapes('EXEC_DEFAULT')

def set_pose_from_rodrigues(armature, bone_name, rodrigues, rodrigues_reference=None):
    rod = Vector((rodrigues[0], rodrigues[1], rodrigues[2]))
    angle_rad = rod.length
    axis = rod.normalized()

    if armature.pose.bones[bone_name].rotation_mode != 'QUATERNION':
        armature.pose.bones[bone_name].rotation_mode = 'QUATERNION'

    quat = Quaternion(axis, angle_rad)

    if rodrigues_reference is None:
        armature.pose.bones[bone_name].rotation_quaternion = quat
    else:
        rod_reference = Vector((rodrigues_reference[0], rodrigues_reference[1], rodrigues_reference[2]))
        rod_result = rod + rod_reference
        angle_rad_result = rod_result.length
        axis_result = rod_result.normalized()
        quat_result = Quaternion(axis_result, angle_rad_result)
        armature.pose.bones[bone_name].rotation_quaternion = quat_result
    return

# Ensure that we have valid slider ranges
def smplx_ensure_valid_shapekey_slider_ranges(skinned_mesh):
    update_slider_ranges = False
    for key_name in ["Shape000", "Exp000", "Pose000"]:
        if key_name in skinned_mesh.data.shape_keys.key_blocks:
            key_block = skinned_mesh.data.shape_keys.key_blocks[key_name]
            if (key_block.slider_min > -SHAPEKEY_VALUE_RANGE) or (key_block.slider_max < SHAPEKEY_VALUE_RANGE):
                update_slider_ranges = True
                break

    if update_slider_ranges:
        for index, key_block in enumerate(skinned_mesh.data.shape_keys.key_blocks):
            if index == 0:
                continue # skip Base shape key

            key_block.slider_min = -SHAPEKEY_VALUE_RANGE
            key_block.slider_max = SHAPEKEY_VALUE_RANGE

# Property groups for UI
class PG_SMPLXProperties(PropertyGroup):

    if USE_SMPLX_2020:
        smplx_version: EnumProperty(
            name = "Version",
            description = "SMPL-X version",
            items = [ ("2020", "2020", "SMPL-X with FLAME 2020 expression blendshapes")]
        )

        smplx_gender: EnumProperty(
            name = "Model",
            description = "SMPL-X model",
            items = [ ("neutral", "Neutral", "")]
        )
    else:
        smplx_version: EnumProperty(
            name = "Version",
            description = "SMPL-X version",
            items = [ ("locked_head", "Locked Head", "Locked head model with removed head bun"), ("v1.1", "v1.1", "") ]
        )

        smplx_gender: EnumProperty(
            name = "Model",
            description = "SMPL-X model",
            items = [ ("female", "Female", ""), ("male", "Male", ""), ("neutral", "Neutral", "")]
        )

    smplx_uv: EnumProperty(
        name = "UV",
        description = "SMPL-X UV version",
        items = [ ("UV_2023", "2023", "Latest UV layout with two eyeball regions"), ("UV_2021", "2021", "Original Blender add-on UV layout") ]
    )

    smplx_texture: EnumProperty(
        name = "",
        description = "SMPL-X model texture",
        items = [ ("NONE", "None", ""), ("smplx_texture_f_2023.png", "Female (UV 2023)", ""), ("smplx_texture_m_2023.png", "Male (UV 2023)", ""), ("smplx_texture_f_alb.png", "Female (UV 2021)", ""), ("smplx_texture_m_alb.png", "Male (UV 2021)", ""), ("smplx_texture_rainbow.png", "Rainbow (UV 2021)", ""), ("UV_GRID", "UV Grid", ""), ("COLOR_GRID", "Color Grid", "") ]
    )

    smplx_corrective_poseshapes: BoolProperty(
        name = "Corrective Pose Shapes",
        description = "Enable/disable corrective pose shapes of SMPL-X model",
        update = update_corrective_poseshapes,
        default = True
    )

    smplx_handpose: EnumProperty(
        name = "",
        description = "SMPL-X hand pose",
        items = [ ("relaxed", "Relaxed", ""), ("flat", "Flat", "") ]
    )

    smplx_height: FloatProperty(name="Target Height [m]", default=1.70, min=1.4, max=2.2)

    smplx_weight: FloatProperty(name="Target Weight [kg]", default=60, min=40, max=110)


class SMPLXAddGender(bpy.types.Operator):
    bl_idname = "scene.smplx_add_gender"
    bl_label = "Add"
    bl_description = ("Add SMPL-X model of selected gender to scene")
    bl_options = {'REGISTER', 'UNDO'}

    uv_2023 = None

    @classmethod
    def poll(cls, context):
        try:
            if (context.active_object is None) or (context.active_object.mode == 'OBJECT'):
                return True
            else:
                return False
        except: return False

    def execute(self, context):
        gender = context.window_manager.smplx_tool.smplx_gender
        print("Adding gender: " + gender)

        path = os.path.dirname(os.path.realpath(__file__))

        if context.window_manager.smplx_tool.smplx_version == "locked_head":
            model_file = SMPLX_MODELFILE_LH
        elif context.window_manager.smplx_tool.smplx_version == "2020":
            model_file = SMPLX_MODELFILE_2020
        else:
            # v1.1
            model_path = os.path.join(path, "data", SMPLX_MODELFILE_300)
            if os.path.exists(model_path):
                model_file = SMPLX_MODELFILE_300
            else:
                model_file = SMPLX_MODELFILE

        objects_path = os.path.join(path, "data", model_file, "Object")
        object_name = "SMPLX-mesh-" + gender

        bpy.ops.wm.append(filename=object_name, directory=str(objects_path))

        # Select imported mesh
        object_name = context.selected_objects[0].name
        bpy.ops.object.select_all(action='DESELECT')
        context.view_layer.objects.active = bpy.data.objects[object_name]
        bpy.data.objects[object_name].select_set(True)
        obj = bpy.context.active_object

        # Set currently selected hand pose
        bpy.ops.object.smplx_set_handpose('EXEC_DEFAULT')

        # Set target UV if needed, default UV in .blend is UV_2021
        uv_version = context.window_manager.smplx_tool.smplx_uv
        print(f"UV map: {uv_version}")
        obj["smplx_uv"] = uv_version

        if uv_version == "UV_2023":
            if self.uv_2023 is None:
                path = os.path.dirname(os.path.realpath(__file__))
                uv_npz_path = os.path.join(path, "data", "smplx_uv_2023.npz")
                with np.load(uv_npz_path) as data:
                    self.uv_2023 = data["uv_coordinates"]

            uv_map = obj.data.uv_layers.active.data
            for i, face in enumerate(obj.data.polygons):
                for j, loop_index in enumerate(face.loop_indices):
                    uv_map[loop_index].uv = self.uv_2023[i * len(face.loop_indices) + j]

        return {'FINISHED'}

class SMPLXSetTexture(bpy.types.Operator):
    bl_idname = "object.smplx_set_texture"
    bl_label = "Set"
    bl_description = ("Set selected texture")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            if (context.object.type == 'MESH'):
                return True
            else:
                return False
        except: return False

    def execute(self, context):
        texture = context.window_manager.smplx_tool.smplx_texture
        print("Setting texture: " + texture)

        obj = bpy.context.object
        if (len(obj.data.materials) == 0) or (obj.data.materials[0] is None):
            self.report({'WARNING'}, "Selected mesh has no material: %s" % obj.name)
            return {'CANCELLED'}

        mat = obj.data.materials[0]
        links = mat.node_tree.links
        nodes = mat.node_tree.nodes

        node_texture = None
        for node in nodes:
            if node.type == 'TEX_IMAGE':
                node_texture = node
                break

        node_shader = None
        for node in nodes:
            if node.type.startswith('BSDF'):
                node_shader = node
                break

        if texture == 'NONE':
            if node_texture is not None:
                for link in node_texture.outputs[0].links:
                    links.remove(link)
                nodes.remove(node_texture)
                node_shader.inputs[0].default_value = node_shader.inputs[0].default_value
        else:
            if node_texture is None:
                node_texture = nodes.new(type="ShaderNodeTexImage")

            if (texture == 'UV_GRID') or (texture == 'COLOR_GRID'):
                if texture not in bpy.data.images:
                    bpy.ops.image.new(name=texture, generated_type=texture)
                image = bpy.data.images[texture]
            else:
                if texture not in bpy.data.images:
                    path = os.path.dirname(os.path.realpath(__file__))
                    texture_path = os.path.join(path, "data", texture)
                    image = bpy.data.images.load(texture_path)
                else:
                    image = bpy.data.images[texture]

            node_texture.image = image

            if len(node_texture.outputs[0].links) == 0:
                links.new(node_texture.outputs[0], node_shader.inputs[0])

        if bpy.context.space_data:
            if bpy.context.space_data.type == 'VIEW_3D':
                bpy.context.space_data.shading.type = 'MATERIAL'

        return {'FINISHED'}

class SMPLXMeasurementsToShape(bpy.types.Operator):
    bl_idname = "object.smplx_measurements_to_shape"
    bl_label = "Measurements To Shape"
    bl_description = ("Calculate and set shape parameters for specified measurements")
    bl_options = {'REGISTER', 'UNDO'}

    betas_regressor = {}
    betas_regressor["female"] = None
    betas_regressor["male"] = None
    betas_regressor["neutral"] = None

    @classmethod
    def poll(cls, context):
        try:
            return ((context.object.type == 'MESH') and (context.object.parent.type == 'ARMATURE'))
        except: return False

    def execute(self, context):
        obj = bpy.context.object
        bpy.ops.object.mode_set(mode='OBJECT')

        for gender in ["female", "male", "neutral"]:
            if self.betas_regressor[gender] is None:
                path = os.path.dirname(os.path.realpath(__file__))
                regressor_path = os.path.join(path, "data", f"smplx_measurements_to_betas_{gender}.json")
                with open(regressor_path) as f:
                    data = json.load(f)
                    self.betas_regressor[gender] = (np.asarray(data["A"]).reshape(-1, 2), np.asarray(data["B"]).reshape(-1, 1))

        gender = obj["smplx_gender"]
        (A, B) = self.betas_regressor[gender]

        height_m = context.window_manager.smplx_tool.smplx_height
        height_cm = height_m * 100.0
        weight_kg = context.window_manager.smplx_tool.smplx_weight

        v_root = pow(weight_kg, 1.0/3.0)
        measurements = np.asarray([[height_cm], [v_root]])
        betas = A @ measurements + B

        num_betas = betas.shape[0]
        for i in range(num_betas):
            name = f"Shape{i:03d}"
            key_block = obj.data.shape_keys.key_blocks[name]
            value = betas[i, 0]

            if value < key_block.slider_min:
                key_block.slider_min = value
            elif value > key_block.slider_max:
                key_block.slider_max = value

            key_block.value = value

        bpy.ops.object.smplx_update_joint_locations('EXEC_DEFAULT')

        return {'FINISHED'}

class SMPLXRandomShape(bpy.types.Operator):
    bl_idname = "object.smplx_random_shape"
    bl_label = "Random"
    bl_description = ("Sets all shape blend shape keys to a random value")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            return context.object.type == 'MESH'
        except: return False

    def execute(self, context):
        obj = bpy.context.object
        bpy.ops.object.mode_set(mode='OBJECT')
        smplx_ensure_valid_shapekey_slider_ranges(obj)
        randomized_betas = 0
        for key_block in obj.data.shape_keys.key_blocks:
            if key_block.name.startswith("Shape"):
                beta = np.random.normal(0.0, 1.0)
                beta = np.clip(beta, -1.0, 1.0)
                key_block.value = beta

                randomized_betas += 1
                if randomized_betas >= 16:
                    break

        bpy.ops.object.smplx_update_joint_locations('EXEC_DEFAULT')

        return {'FINISHED'}

class SMPLXResetShape(bpy.types.Operator):
    bl_idname = "object.smplx_reset_shape"
    bl_label = "Reset"
    bl_description = ("Resets all blend shape keys for shape")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            return context.object.type == 'MESH'
        except: return False

    def execute(self, context):
        obj = bpy.context.object
        bpy.ops.object.mode_set(mode='OBJECT')
        for key_block in obj.data.shape_keys.key_blocks:
            if key_block.name.startswith("Shape"):
                key_block.value = 0.0

        bpy.ops.object.smplx_update_joint_locations('EXEC_DEFAULT')

        return {'FINISHED'}

class SMPLXRandomExpressionShape(bpy.types.Operator):
    bl_idname = "object.smplx_random_expression_shape"
    bl_label = "Random Face Expression"
    bl_description = ("Sets all face expression blend shape keys to a random value")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            return context.object.type == 'MESH'
        except: return False

    def execute(self, context):
        obj = bpy.context.object
        bpy.ops.object.mode_set(mode='OBJECT')
        smplx_ensure_valid_shapekey_slider_ranges(obj)
        for key_block in obj.data.shape_keys.key_blocks:
            if key_block.name.startswith("Exp"):
                key_block.value = np.random.uniform(-2, 2)

        return {'FINISHED'}

class SMPLXResetExpressionShape(bpy.types.Operator):
    bl_idname = "object.smplx_reset_expression_shape"
    bl_label = "Reset"
    bl_description = ("Resets all blend shape keys for face expression")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            return context.object.type == 'MESH'
        except: return False

    def execute(self, context):
        obj = bpy.context.object
        bpy.ops.object.mode_set(mode='OBJECT')
        for key_block in obj.data.shape_keys.key_blocks:
            if key_block.name.startswith("Exp"):
                key_block.value = 0.0

        return {'FINISHED'}

class SMPLXSnapGroundPlane(bpy.types.Operator):
    bl_idname = "object.smplx_snap_ground_plane"
    bl_label = "Snap To Ground Plane"
    bl_description = ("Snaps mesh to the XY ground plane")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            return ((context.object.type == 'MESH') or (context.object.type == 'ARMATURE'))
        except: return False

    def execute(self, context):
        bpy.ops.object.mode_set(mode='OBJECT')

        obj = bpy.context.object
        if obj.type == 'ARMATURE':
            armature = obj
            obj = bpy.context.object.children[0]
        else:
            armature = obj.parent

        depsgraph = context.evaluated_depsgraph_get()
        object_eval = obj.evaluated_get(depsgraph)
        mesh_from_eval = object_eval.to_mesh()

        matrix_world = obj.matrix_world
        vertices_world = [matrix_world @ vertex.co for vertex in mesh_from_eval.vertices]
        z_min = (min(vertices_world, key=lambda item: item.z)).z
        object_eval.to_mesh_clear()

        armature.location.z = armature.location.z - z_min

        return {'FINISHED'}

class SMPLXUpdateJointLocations(bpy.types.Operator):
    bl_idname = "object.smplx_update_joint_locations"
    bl_label = "Update Joint Locations"
    bl_description = ("Update joint locations after shape changes")
    bl_options = {'REGISTER', 'UNDO'}

    j_regressor = {}
    j_regressor["female"] = { "10": None, "300": None, "300_lh": None }
    j_regressor["male"] = { "10": None, "300": None, "300_lh": None }
    j_regressor["neutral"] = { "10": None, "300": None, "300_lh": None }

    @classmethod
    def poll(cls, context):
        try:
            return ((context.object.type == 'MESH') and (context.object.parent.type == 'ARMATURE'))
        except: return False

    def load_regressor(self, gender, betas):
        path = os.path.dirname(os.path.realpath(__file__))
        prefix = ""
        if betas == "10":
            suffix = ""
        elif betas == "300":
            suffix = "_300"
        elif betas == "300_lh":
            suffix = "_300"
            prefix = "lh_"
        else:
            print(f"ERROR: No betas-to-joints regressor for desired beta shapes [{betas}]")
            return (None, None)

        regressor_path = os.path.join(path, "data", f"smplx_betas_to_joints_{prefix}{gender}{suffix}.json")
        with open(regressor_path) as f:
            data = json.load(f)
            return (np.asarray(data["betasJ_regr"]), np.asarray(data["template_J"]))

    def execute(self, context):
        obj = bpy.context.object
        bpy.ops.object.mode_set(mode='OBJECT')

        betas = []
        for key_block in obj.data.shape_keys.key_blocks:
            if key_block.name.startswith("Shape"):
                betas.append(key_block.value)
        num_betas = len(betas)
        betas = np.array(betas)

        for target_betas in ["10", "300", "300_lh"]:
            for gender in ["female", "male", "neutral"]:
                if self.j_regressor[gender][target_betas] is None:
                    self.j_regressor[gender][target_betas] = self.load_regressor(gender, target_betas)

        key = f"{num_betas}"
        if obj["smplx_version"] == "locked_head":
            key += "_lh"
        gender = obj["smplx_gender"]
        (betas_to_joints, template_j) = self.j_regressor[gender][key]
        joint_locations = betas_to_joints @ betas + template_j

        armature = obj.parent
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode='EDIT')

        for index in range(NUM_SMPLX_JOINTS):
            bone = armature.data.edit_bones[SMPLX_JOINT_NAMES[index]]
            bone.head = (0.0, 0.0, 0.0)
            bone.tail = (0.0, 0.0, 0.1)

            joint_location_smplx = joint_locations[index]
            bone_start = Vector( (joint_location_smplx[0], -joint_location_smplx[2], joint_location_smplx[1]) )
            bone.translate(bone_start)

        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.context.view_layer.objects.active = obj

        return {'FINISHED'}

class SMPLXSetPoseshapes(bpy.types.Operator):
    bl_idname = "object.smplx_set_poseshapes"
    bl_label = "Update Pose Shapes"
    bl_description = ("Sets and updates corrective poseshapes for current pose")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            return ( ((context.object.type == 'MESH') and (context.object.parent.type == 'ARMATURE')) or (context.object.type == 'ARMATURE'))
        except: return False

    # Computes rotation matrix through Rodrigues formula
    def rodrigues_to_mat(self, rotvec):
        theta = np.linalg.norm(rotvec)
        r = (rotvec/theta).reshape(3, 1) if theta > 0. else rotvec
        cost = np.cos(theta)
        mat = np.asarray([[0, -r[2], r[1]],
                        [r[2], 0, -r[0]],
                        [-r[1], r[0], 0]], dtype=object)
        return(cost*np.eye(3) + (1-cost)*r.dot(r.T) + np.sin(theta)*mat)

    def rodrigues_to_posecorrective_weight(self, pose):
        joints_posecorrective = NUM_SMPLX_JOINTS
        rod_rots = np.asarray(pose).reshape(joints_posecorrective, 3)
        mat_rots = [self.rodrigues_to_mat(rod_rot) for rod_rot in rod_rots]
        bshapes = np.concatenate([(mat_rot - np.eye(3)).ravel() for mat_rot in mat_rots[1:]])
        return(bshapes)

    def execute(self, context):
        obj = bpy.context.object

        if obj.type == 'ARMATURE':
            armature = obj
            obj = bpy.context.object.children[0]
        else:
            armature = obj.parent

        smplx_ensure_valid_shapekey_slider_ranges(obj)

        pose = [0.0] * (NUM_SMPLX_JOINTS * 3)

        for index in range(NUM_SMPLX_JOINTS):
            joint_name = SMPLX_JOINT_NAMES[index]
            joint_pose = rodrigues_from_pose(armature, joint_name)
            pose[index*3 + 0] = joint_pose[0]
            pose[index*3 + 1] = joint_pose[1]
            pose[index*3 + 2] = joint_pose[2]

        poseweights = self.rodrigues_to_posecorrective_weight(pose)

        for index, weight in enumerate(poseweights):
            obj.data.shape_keys.key_blocks["Pose%03d" % index].value = weight

        context.window_manager.smplx_tool["smplx_corrective_poseshapes"] = True

        return {'FINISHED'}

class SMPLXResetPoseshapes(bpy.types.Operator):
    bl_idname = "object.smplx_reset_poseshapes"
    bl_label = "Reset"
    bl_description = ("Resets corrective poseshapes for current pose")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            return ( ((context.object.type == 'MESH') and (context.object.parent.type == 'ARMATURE')) or (context.object.type == 'ARMATURE'))
        except: return False

    def execute(self, context):
        obj = bpy.context.object

        if obj.type == 'ARMATURE':
            obj = bpy.context.object.children[0]

        for key_block in obj.data.shape_keys.key_blocks:
            if key_block.name.startswith("Pose"):
                key_block.value = 0.0

        return {'FINISHED'}

class SMPLXSetHandpose(bpy.types.Operator):
    bl_idname = "object.smplx_set_handpose"
    bl_label = "Set"
    bl_description = ("Set selected hand pose")
    bl_options = {'REGISTER', 'UNDO'}

    hand_poses = None

    @classmethod
    def poll(cls, context):
        try:
            return ( ((context.object.type == 'MESH') and (context.object.parent.type == 'ARMATURE')) or (context.object.type == 'ARMATURE'))
        except: return False

    def execute(self, context):
        obj = bpy.context.object
        if obj.type == 'MESH':
            armature = obj.parent
        else:
            armature = obj

        if self.hand_poses is None:
            path = os.path.dirname(os.path.realpath(__file__))
            data_path = os.path.join(path, "data", "smplx_handposes.npz")
            with np.load(data_path, allow_pickle=True) as data:
                self.hand_poses = data["hand_poses"].item()

        hand_pose_name = context.window_manager.smplx_tool.smplx_handpose
        print("Setting hand pose: " + hand_pose_name)

        if hand_pose_name not in self.hand_poses:
            self.report({"ERROR"}, f"Desired hand pose not existing: {hand_pose_name}")
            return {"CANCELLED"}

        (left_hand_pose, right_hand_pose) = self.hand_poses[hand_pose_name]

        hand_pose = np.concatenate( (left_hand_pose, right_hand_pose) ).reshape(-1, 3)

        hand_joint_start_index = 1 + NUM_SMPLX_BODYJOINTS + 3
        for index in range(2 * NUM_SMPLX_HANDJOINTS):
            pose_rodrigues = hand_pose[index]
            bone_name = SMPLX_JOINT_NAMES[index + hand_joint_start_index]
            set_pose_from_rodrigues(armature, bone_name, pose_rodrigues)

        if context.window_manager.smplx_tool.smplx_corrective_poseshapes:
            bpy.ops.object.smplx_set_poseshapes('EXEC_DEFAULT')

        return {'FINISHED'}

class SMPLXWritePose(bpy.types.Operator):
    bl_idname = "object.smplx_write_pose"
    bl_label = "Write Pose To Console"
    bl_description = ("Writes SMPL-X flat hand pose thetas to console window")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            return (context.object.type == 'MESH') or (context.object.type == 'ARMATURE')
        except: return False

    def execute(self, context):
        obj = bpy.context.object

        if obj.type == 'MESH':
            armature = obj.parent
        else:
            armature = obj

        pose = [0.0] * (NUM_SMPLX_JOINTS * 3)

        for index in range(NUM_SMPLX_JOINTS):
            joint_name = SMPLX_JOINT_NAMES[index]
            joint_pose = rodrigues_from_pose(armature, joint_name)
            pose[index*3 + 0] = joint_pose[0]
            pose[index*3 + 1] = joint_pose[1]
            pose[index*3 + 2] = joint_pose[2]

        print("\npose = " + str(pose))

        return {'FINISHED'}

class SMPLXResetPose(bpy.types.Operator):
    bl_idname = "object.smplx_reset_pose"
    bl_label = "Reset Pose"
    bl_description = ("Resets pose to default zero pose")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            return ( ((context.object.type == 'MESH') and (context.object.parent.type == 'ARMATURE')) or (context.object.type == 'ARMATURE'))
        except: return False

    def execute(self, context):
        obj = bpy.context.object

        if obj.type == 'MESH':
            armature = obj.parent
        else:
            armature = obj

        for bone in armature.pose.bones:
            if bone.rotation_mode != 'QUATERNION':
                bone.rotation_mode = 'QUATERNION'
            bone.rotation_quaternion = Quaternion()

        bpy.ops.object.smplx_reset_poseshapes('EXEC_DEFAULT')

        return {'FINISHED'}

class SMPLXLoadPose(bpy.types.Operator, ImportHelper):
    bl_idname = "object.smplx_load_pose"
    bl_label = "Load Pose"
    bl_description = ("Load relaxed-hand model pose from file")
    bl_options = {'REGISTER', 'UNDO'}

    filter_glob: StringProperty(
        default="*.pkl",
        options={'HIDDEN'}
    )

    update_shape: BoolProperty(
        name="Update shape parameters",
        description="Update shape parameters using the beta shape information in the loaded file",
        default=True
    )

    hand_pose_relaxed = None

    @classmethod
    def poll(cls, context):
        try:
            return ( ((context.object.type == 'MESH') and (context.object.parent.type == 'ARMATURE')) or (context.object.type == 'ARMATURE'))
        except: return False

    def execute(self, context):
        obj = bpy.context.object

        if obj.type == 'MESH':
            armature = obj.parent
        else:
            armature = obj
            obj = armature.children[0]
            context.view_layer.objects.active = obj

        if self.hand_pose_relaxed is None:
            path = os.path.dirname(os.path.realpath(__file__))
            data_path = os.path.join(path, "data", "smplx_handposes.npz")
            with np.load(data_path, allow_pickle=True) as data:
                hand_poses = data["hand_poses"].item()
                (left_hand_pose, right_hand_pose) = hand_poses["relaxed"]
                self.hand_pose_relaxed = np.concatenate( (left_hand_pose, right_hand_pose) ).reshape(-1, 3)

        print("Loading: " + self.filepath)

        # ---- CPU-safe loading here ----
        try:
            data = smplx_load_pickle_cpu(self.filepath)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to load PKL (CPU-safe): {e}")
            return {'CANCELLED'}

        # Extract fields with robust shapes and Torch→NumPy conversion
        translation = None
        global_orient = None
        body_pose = None
        jaw_pose = None
        left_hand_pose = None
        right_hand_pose = None
        betas = []
        expression = []

        if "transl" in data:
            try:
                translation = _to_np(data["transl"]).reshape(-1)[:3]
            except Exception:
                translation = None

        if "global_orient" in data:
            try:
                global_orient = _to_np(data["global_orient"]).reshape(-1)[:3]
            except Exception:
                global_orient = None

        if "body_pose" in data:
            bp = _to_np(data["body_pose"]).reshape(-1)
            if bp.size != (NUM_SMPLX_BODYJOINTS * 3):
                self.report({'ERROR'}, f"Invalid body pose length: {bp.size} (expected {NUM_SMPLX_BODYJOINTS*3})")
                return {'CANCELLED'}
            body_pose = bp.reshape(NUM_SMPLX_BODYJOINTS, 3)

        if "jaw_pose" in data:
            jp = _to_np(data["jaw_pose"]).reshape(-1)
            jaw_pose = (jp[:3] if jp.size >= 3 else np.zeros(3))

        if "left_hand_pose" in data:
            lhp = _to_np(data["left_hand_pose"]).reshape(-1, 3)
            if lhp.shape[0] < NUM_SMPLX_HANDJOINTS:
                # pad missing joints with zeros
                pad = np.zeros((NUM_SMPLX_HANDJOINTS - lhp.shape[0], 3), dtype=float)
                lhp = np.vstack([lhp, pad])
            left_hand_pose = lhp[:NUM_SMPLX_HANDJOINTS]

        if "right_hand_pose" in data:
            rhp = _to_np(data["right_hand_pose"]).reshape(-1, 3)
            if rhp.shape[0] < NUM_SMPLX_HANDJOINTS:
                pad = np.zeros((NUM_SMPLX_HANDJOINTS - rhp.shape[0], 3), dtype=float)
                rhp = np.vstack([rhp, pad])
            right_hand_pose = rhp[:NUM_SMPLX_HANDJOINTS]

        if "betas" in data:
            betas = _to_np(data["betas"]).reshape(-1).tolist()

        if "expression" in data:
            expression = _to_np(data["expression"]).reshape(-1).tolist()

        # Update shape if selected
        if self.update_shape and (betas is not None):
            bpy.ops.object.mode_set(mode='OBJECT')
            for index, beta in enumerate(betas):
                key_block_name = f"Shape{index:03}"

                if key_block_name in obj.data.shape_keys.key_blocks:
                    obj.data.shape_keys.key_blocks[key_block_name].value = float(beta)
                else:
                    print(f"ERROR: No key block for: {key_block_name}")

            bpy.ops.object.smplx_update_joint_locations('EXEC_DEFAULT')

        if global_orient is not None:
            set_pose_from_rodrigues(armature, "pelvis", global_orient)

        for index in range(NUM_SMPLX_BODYJOINTS):
            pose_rodrigues = body_pose[index]
            bone_name = SMPLX_JOINT_NAMES[index + 1] # body pose starts with left_hip
            set_pose_from_rodrigues(armature, bone_name, pose_rodrigues)

        if jaw_pose is not None:
            set_pose_from_rodrigues(armature, "jaw", jaw_pose)

        # Left hand
        start_name_index = 1 + NUM_SMPLX_BODYJOINTS + 3
        for i in range(0, NUM_SMPLX_HANDJOINTS):
            pose_rodrigues = left_hand_pose[i]
            bone_name = SMPLX_JOINT_NAMES[start_name_index + i]
            pose_relaxed_rodrigues = self.hand_pose_relaxed[i]
            set_pose_from_rodrigues(armature, bone_name, pose_rodrigues, pose_relaxed_rodrigues)

        # Right hand
        start_name_index = 1 + NUM_SMPLX_BODYJOINTS + 3 + NUM_SMPLX_HANDJOINTS
        for i in range(0, NUM_SMPLX_HANDJOINTS):
            pose_rodrigues = right_hand_pose[i]
            bone_name = SMPLX_JOINT_NAMES[start_name_index + i]
            pose_relaxed_rodrigues = self.hand_pose_relaxed[NUM_SMPLX_HANDJOINTS + i]
            set_pose_from_rodrigues(armature, bone_name, pose_rodrigues, pose_relaxed_rodrigues)

        if translation is not None:
            armature.location = (translation[0], -translation[2], translation[1])

        bpy.ops.object.smplx_set_poseshapes('EXEC_DEFAULT')

        for index, exp in enumerate(expression):
            key_block_name = f"Exp{index:03}"

            if key_block_name in obj.data.shape_keys.key_blocks:
                obj.data.shape_keys.key_blocks[key_block_name].value = float(exp)
            else:
                print(f"ERROR: No key block for: {key_block_name}")

        return {'FINISHED'}

class SMPLXAddAnimation(bpy.types.Operator, ImportHelper):
    bl_idname = "object.smplx_add_animation"
    bl_label = "Add Animation"
    bl_description = ("Load AMASS/SMPL-X animation and create animated SMPL-X body")
    bl_options = {'REGISTER', 'UNDO'}

    filter_glob: StringProperty(
        default="*.npz",
        options={'HIDDEN'}
    )

    anim_format: EnumProperty(
        name="Format",
        items=(
            ("AMASS", "AMASS", ""),
            ("SMPL-X", "SMPL-X", ""),
        ),
    )

    rest_position: EnumProperty(
        name="Body rest position",
        items=(
            ("SMPL-X", "SMPL-X", "Use default SMPL-X rest position (feet below the floor)"),
            ("GROUNDED", "Grounded", "Use feet-on-floor rest position"),
        ),
    )

    hand_reference: EnumProperty(
        name="Hand pose reference",
        items=(
            ("FLAT", "Flat", "Use flat hand as hand pose reference"),
            ("RELAXED", "Relaxed", "Use relaxed hand as hand pose reference"),
        ),
    )

    keyframe_corrective_pose_weights: BoolProperty(
        name="Use keyframed corrective pose weights",
        description="Keyframe the weights of the corrective pose shapes for each frame. This increases animation load time and slows down editor real-time playback.",
        default=False
    )

    target_framerate: IntProperty(
        name="Target framerate [fps]",
        description="Target framerate for animation in frames-per-second. Lower values will speed up import time.",
        default=30,
        min = 1,
        max = 120
    )

    hand_pose_relaxed = None

    @classmethod
    def poll(cls, context):
        try:
            return True
        except: return False

    def execute(self, context):

        target_framerate = self.target_framerate

        if self.hand_reference == "RELAXED":
            if self.hand_pose_relaxed is None:
                path = os.path.dirname(os.path.realpath(__file__))
                data_path = os.path.join(path, "data", "smplx_handposes.npz")
                with np.load(data_path, allow_pickle=True) as data:
                    hand_poses = data["hand_poses"].item()
                    (left_hand_pose, right_hand_pose) = hand_poses["relaxed"]
                    self.hand_pose_relaxed = np.concatenate( (left_hand_pose, right_hand_pose) ).reshape(-1, 3)

        print("Loading: " + self.filepath)
        with np.load(self.filepath) as data:
            if ("trans" not in data) or ("gender" not in data) or (("mocap_frame_rate" not in data) and ("mocap_framerate" not in data)) or ("betas" not in data) or ("poses" not in data):
                self.report({"ERROR"}, "Invalid AMASS animation data file")
                return {"CANCELLED"}

            trans = data["trans"]
            gender = str(data["gender"])
            mocap_framerate = int(data["mocap_frame_rate"]) if "mocap_frame_rate" in data else int(data["mocap_framerate"])
            betas = data["betas"]
            poses = data["poses"]

            if mocap_framerate < target_framerate:
                self.report({"ERROR"}, f"Mocap framerate ({mocap_framerate}) below target framerate ({target_framerate})")
                return {"CANCELLED"}

        if (context.active_object is not None):
            bpy.ops.object.mode_set(mode='OBJECT')

        context.window_manager.smplx_tool.smplx_gender = gender
        context.window_manager.smplx_tool.smplx_handpose = "flat"
        bpy.ops.scene.smplx_add_gender()

        obj = context.view_layer.objects.active
        armature = obj.parent

        armature.name = armature.name + "_" + os.path.basename(self.filepath).replace(".npz", "")

        context.scene.render.fps = target_framerate
        context.scene.frame_start = 1

        bpy.ops.object.mode_set(mode='OBJECT')
        for index, beta in enumerate(betas):
            key_block_name = f"Shape{index:03}"

            if key_block_name in obj.data.shape_keys.key_blocks:
                obj.data.shape_keys.key_blocks[key_block_name].value = beta
            else:
                print(f"ERROR: No key block for: {key_block_name}")

        bpy.ops.object.smplx_update_joint_locations('EXEC_DEFAULT')

        height_offset = 0
        if self.rest_position == "GROUNDED":
            bpy.ops.object.smplx_snap_ground_plane('EXEC_DEFAULT')
            height_offset = armature.location[2]

            obj["smplx_bind_pose_height_offset"] = height_offset

            bpy.context.view_layer.objects.active = armature
            armature.select_set(True)
            obj.select_set(True)
            bpy.ops.object.transform_apply(location = True, rotation=False, scale=False)
            armature.select_set(False)

            bpy.ops.object.mode_set(mode='EDIT')
            bone = armature.data.edit_bones["root"]
            bone.head = (0.0, 0.0, 0.0)
            bone.tail = (0.0, 0.0, 0.1)
            bpy.ops.object.mode_set(mode='OBJECT')
            bpy.context.view_layer.objects.active = obj

        step_size = int(mocap_framerate / target_framerate)

        num_frames = trans.shape[0]
        num_keyframes = int(num_frames / step_size)

        if self.keyframe_corrective_pose_weights:
            print(f"Adding pose keyframes with keyframed corrective pose weights: {num_keyframes}")
        else:
            print(f"Adding pose keyframes: {num_keyframes}")

        if len(bpy.data.actions) == 0:
            context.scene.frame_end = num_keyframes
        elif num_keyframes > context.scene.frame_end:
            context.scene.frame_end = num_keyframes

        for index, frame in enumerate(range(0, num_frames, step_size)):
            if (index % 100) == 0:
                print(f"  {index}/{num_keyframes}")
            current_frame = index + 1
            current_pose = poses[frame].reshape(-1, 3)
            current_trans = trans[frame]
            for bone_index, bone_name in enumerate(SMPLX_JOINT_NAMES):
                if bone_name == "pelvis":
                    if self.rest_position == "GROUNDED":
                        current_trans[1] = current_trans[1] - height_offset
                    armature.pose.bones[bone_name].location = Vector((current_trans[0], current_trans[1], current_trans[2]))
                    armature.pose.bones[bone_name].keyframe_insert('location', frame=current_frame)

                pose_rodrigues = current_pose[bone_index]

                if self.hand_reference == "FLAT":
                    set_pose_from_rodrigues(armature, bone_name, pose_rodrigues)
                else:
                    finger_names = ["index", "middle", "pinky", "ring", "thumb"]
                    if not any([x in bone_name for x in finger_names]):
                        set_pose_from_rodrigues(armature, bone_name, pose_rodrigues)
                    else:
                        hand_start_index = 1 + NUM_SMPLX_BODYJOINTS + 3
                        relaxed_hand_joint_index = bone_index - hand_start_index
                        pose_relaxed_rodrigues = self.hand_pose_relaxed[relaxed_hand_joint_index]
                        set_pose_from_rodrigues(armature, bone_name, pose_rodrigues, pose_relaxed_rodrigues)

                armature.pose.bones[bone_name].keyframe_insert('rotation_quaternion', frame=current_frame)

            if self.keyframe_corrective_pose_weights:
                bpy.ops.object.smplx_set_poseshapes('EXEC_DEFAULT')
                for key_block in obj.data.shape_keys.key_blocks:
                    if key_block.name.startswith("Pose"):
                        key_block.keyframe_insert("value", frame=current_frame)

        if self.anim_format == "AMASS":
            bone_name = "root"
            if armature.pose.bones[bone_name].rotation_mode != 'QUATERNION':
                armature.pose.bones[bone_name].rotation_mode = 'QUATERNION'
            armature.pose.bones[bone_name].rotation_quaternion = Quaternion((1.0, 0.0, 0.0), radians(-90))
            armature.pose.bones[bone_name].keyframe_insert('rotation_quaternion', frame=1)

        print(f"  {num_keyframes}/{num_keyframes}")
        context.scene.frame_set(1)

        return {'FINISHED'}

class SMPLXExportAlembic(bpy.types.Operator, ExportHelper):
    bl_idname = "object.smplx_export_alembic"
    bl_label = "Export Alembic ABC"
    bl_description = ("Export as Alembic geometry cache")
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".abc"

    @classmethod
    def poll(cls, context):
        try:
            return (context.object.type == 'MESH')
        except: return False

    def execute(self, context):
        bpy.ops.wm.alembic_export(filepath=self.filepath, selected=True, packuv=False, face_sets=True)
        print("Exported: " + self.filepath)

        return {'FINISHED'}

class SMPLXExportFBX(bpy.types.Operator, ExportHelper):
    bl_idname = "object.smplx_export_fbx"
    bl_label = "Export FBX"
    bl_description = ("Export skinned mesh in FBX format")
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".fbx"

    export_shape_keys: EnumProperty(
        name = "Blend Shapes",
        description = "Blend shape export settings",
        items = [ ("SHAPE_POSECORRECTIVES", "All: Shape + Posecorrectives", "Export shape keys for body shape and pose correctives"),
                  ("SHAPE", "Reduced: Shape space only", "Export only shape keys for body shape"),
                  ("POSECORRECTIVES", "Reduced: Posecorrectives only", "Bake shape and expression into mesh, export only shape keys for pose correctives"),
                  ("NONE", "None: Apply shape space", "Do not export any shape keys, shape keys for body shape will be baked into mesh") ],
    )

    target_format: EnumProperty(
        name="Format",
        items=(
            ("UNITY", "Unity", ""),
            ("UNREAL", "Unreal", ""),
        ),
    )

    @classmethod
    def poll(cls, context):
        try:
            return (context.object.type == 'MESH')
        except: return False

    def execute(self, context):

        obj = bpy.context.object

        armature_original = obj.parent
        skinned_mesh_original = obj

        bpy.ops.object.select_all(action='DESELECT')
        skinned_mesh_original.select_set(True)
        armature_original.select_set(True)
        bpy.context.view_layer.objects.active = skinned_mesh_original
        bpy.ops.object.duplicate()
        skinned_mesh = bpy.context.object
        armature = skinned_mesh.parent

        context.view_layer.objects.active = armature
        armature_offset = Vector(armature.location)
        armature.location = (0, 0, 0)
        bpy.ops.object.mode_set(mode='EDIT')
        for edit_bone in armature.data.edit_bones:
            if edit_bone.name != "root":
                edit_bone.translate(armature_offset)

        bpy.ops.object.mode_set(mode='OBJECT')
        context.view_layer.objects.active = skinned_mesh
        mesh_location = Vector(skinned_mesh.location)
        skinned_mesh.location = mesh_location + armature_offset
        bpy.ops.object.transform_apply(location = True)

        bpy.ops.object.smplx_reset_pose('EXEC_DEFAULT')

        if ( (self.export_shape_keys == 'SHAPE') or (self.export_shape_keys == 'NONE') ):
            print("Removing pose corrective shape keys")
            num_shape_keys = len(skinned_mesh.data.shape_keys.key_blocks.keys())

            current_shape_key_index = 0
            for index in range(0, num_shape_keys):
                bpy.context.object.active_shape_key_index = current_shape_key_index

                if bpy.context.object.active_shape_key is not None:
                    if bpy.context.object.active_shape_key.name.startswith('Pose'):
                        bpy.ops.object.shape_key_remove(all=False)
                    else:
                        current_shape_key_index = current_shape_key_index + 1

        if self.export_shape_keys == 'NONE':
            print("Baking shape and removing shape keys for shape")

            for key_block in skinned_mesh.data.shape_keys.key_blocks:
                if key_block.name.startswith("Pose"):
                    key_block.value = 0.0

            bpy.ops.object.shape_key_add(from_mix=True)
            num_shape_keys = len(skinned_mesh.data.shape_keys.key_blocks.keys())

            bpy.context.object.active_shape_key_index = 0
            for count in range(0, num_shape_keys):
                bpy.ops.object.shape_key_remove(all=False)

        elif self.export_shape_keys == 'POSECORRECTIVES':
            print("Baking shape and expression into Base shape key")

            for key_block in skinned_mesh.data.shape_keys.key_blocks:
                if key_block.name.startswith("Pose"):
                    key_block.value = 0.0

            bpy.ops.object.shape_key_add(from_mix=True)
            bpy.context.object.active_shape_key.name = "ShapeMix"

            bpy.context.object.active_shape_key_index = 0
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.blend_from_shape(shape="ShapeMix", blend=1, add=False)
            bpy.ops.object.mode_set(mode='OBJECT')

            num_shape_keys = len(skinned_mesh.data.shape_keys.key_blocks.keys())
            current_shape_key_index = 1
            for _ in range(1, num_shape_keys):
                bpy.context.object.active_shape_key_index = current_shape_key_index

                if bpy.context.object.active_shape_key is not None:
                    if (bpy.context.object.active_shape_key.name.startswith('Shape') or
                        bpy.context.object.active_shape_key.name.startswith('Exp')):
                        bpy.ops.object.shape_key_remove(all=False)
                    else:
                        current_shape_key_index = current_shape_key_index + 1
            bpy.context.object.active_shape_key_index = 0

        bpy.ops.object.mode_set(mode='OBJECT')

        bpy.ops.object.select_all(action='DESELECT')
        skinned_mesh.select_set(True)
        skinned_mesh.rotation_euler = (radians(-90), 0, 0)
        bpy.context.view_layer.objects.active = skinned_mesh
        bpy.ops.object.transform_apply(location = False, rotation = True, scale = False)
        skinned_mesh.rotation_euler = (radians(90), 0, 0)
        skinned_mesh.select_set(False)

        armature.select_set(True)
        armature.rotation_euler = (radians(-90), 0, 0)
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.transform_apply(location = False, rotation = True, scale = False)
        armature.rotation_euler = (radians(90), 0, 0)

        if self.target_format == "UNREAL":
            armature.scale = (100, 100, 100)
            if armature.animation_data is not None:
                action = armature.animation_data.action
                for fcurve in action.fcurves:
                    if fcurve.data_path.endswith("location"):
                        for keyframe_point in fcurve.keyframe_points:
                            keyframe_point.co[1] = keyframe_point.co[1] * 100
                            keyframe_point.handle_left[1] = keyframe_point.handle_left[1] * 100
                            keyframe_point.handle_right[1] = keyframe_point.handle_right[1] * 100

            bpy.ops.object.transform_apply(location = False, rotation = False, scale = True)

        skinned_mesh.select_set(True)

        gender = skinned_mesh_original["smplx_gender"]

        target_mesh_name = "SMPLX-mesh-%s" % gender
        target_armature_name = "SMPLX-%s" % gender

        if target_mesh_name in bpy.data.objects:
            bpy.data.objects[target_mesh_name].name = "SMPLX-temp-mesh"
        skinned_mesh.name = target_mesh_name

        if target_armature_name in bpy.data.objects:
            bpy.data.objects[target_armature_name].name = "SMPLX-temp-armature"
        armature.name = target_armature_name

        bpy.ops.export_scene.fbx(filepath=self.filepath,
                                 use_selection=True,
                                 apply_scale_options="FBX_SCALE_ALL",
                                 use_custom_props=True,
                                 add_leaf_bones=False,
                                 bake_anim_use_nla_strips=False,
                                 bake_anim_use_all_actions=False,
                                 bake_anim_simplify_factor=0)

        print("Exported: " + self.filepath)

        bpy.ops.object.select_all(action='DESELECT')
        skinned_mesh.select_set(True)
        armature.select_set(True)
        bpy.ops.object.delete()

        bpy.ops.object.select_all(action='DESELECT')
        skinned_mesh_original.select_set(True)
        bpy.context.view_layer.objects.active = skinned_mesh_original

        if "SMPLX-temp-mesh" in bpy.data.objects:
            bpy.data.objects["SMPLX-temp-mesh"].name = target_mesh_name

        if "SMPLX-temp-armature" in bpy.data.objects:
            bpy.data.objects["SMPLX-temp-armature"].name = target_armature_name

        return {'FINISHED'}

class SMPLXExportShape(bpy.types.Operator, ExportHelper):
    bl_idname = "object.smplx_export_shape"
    bl_label = "Export Shape"
    bl_description = ("Export shape beta values in NPZ format")
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".npz"

    @classmethod
    def poll(cls, context):
        try:
            return (context.object.type == 'MESH')
        except: return False

    def execute(self, context):

        obj = bpy.context.object
        armature = obj.parent

        betas = []
        for index in range(300):
            key_block_name = f"Shape{index:03}"

            if key_block_name in obj.data.shape_keys.key_blocks:
                beta = obj.data.shape_keys.key_blocks[key_block_name].value
                betas.append(beta)

        data = {}
        data["gender"] = obj["smplx_gender"]
        data["mocap_frame_rate"] = 30
        data["model"] = "smplx_" + obj["smplx_version"]
        data["betas"] = betas
        data["poses"] = [ [0.0] * 3 * NUM_SMPLX_JOINTS ]
        data["trans"] = [ [0.0, 0.0, 0.0] ]
        data["info"] = "Shape only,default pose"

        bind_pose_height_offset = 0.0
        if "smplx_bind_pose_height_offset" in obj:
            bind_pose_height_offset = obj["smplx_bind_pose_height_offset"]
        else:
            bind_pose_height_offset = armature.location[2]

        data["bind_pose_height_offset"] = bind_pose_height_offset

        np.savez_compressed(self.filepath, **data)
        print("Exported: " + self.filepath)

        return {'FINISHED'}

class SMPLX_PT_Model(bpy.types.Panel):
    bl_label = "SMPL-X Model"
    bl_category = "SMPL-X"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):

        layout = self.layout
        col = layout.column(align=True)
        
        row = col.row(align=True)
        col.prop(context.window_manager.smplx_tool, "smplx_version")
        col.prop(context.window_manager.smplx_tool, "smplx_gender")
        col.prop(context.window_manager.smplx_tool, "smplx_uv")
        col.operator("scene.smplx_add_gender", text="Add")

        col.separator()
        col.label(text="Texture:")
        row = col.row(align=True)
        split = row.split(factor=0.75, align=True)
        split.prop(context.window_manager.smplx_tool, "smplx_texture")
        split.operator("object.smplx_set_texture", text="Set")

class SMPLX_PT_Shape(bpy.types.Panel):
    bl_label = "Shape"
    bl_category = "SMPL-X"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)

        col.prop(context.window_manager.smplx_tool, "smplx_height")
        col.prop(context.window_manager.smplx_tool, "smplx_weight")
        col.operator("object.smplx_measurements_to_shape")
        col.separator()

        row = col.row(align=True)
        split = row.split(factor=0.75, align=True)
        split.operator("object.smplx_random_shape")
        split.operator("object.smplx_reset_shape")
        col.separator()

        col.operator("object.smplx_snap_ground_plane")
        col.separator()

        col.operator("object.smplx_update_joint_locations")
        col.separator()
        row = col.row(align=True)
        split = row.split(factor=0.75, align=True)
        split.operator("object.smplx_random_expression_shape")
        split.operator("object.smplx_reset_expression_shape")

class SMPLX_PT_Pose(bpy.types.Panel):
    bl_label = "Pose"
    bl_category = "SMPL-X"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)

        col.prop(context.window_manager.smplx_tool, "smplx_corrective_poseshapes")
        col.separator()
        col.operator("object.smplx_set_poseshapes")

        col.separator()
        col.label(text="Hand Pose:")
        row = col.row(align=True)
        split = row.split(factor=0.75, align=True)
        split.prop(context.window_manager.smplx_tool, "smplx_handpose")
        split.operator("object.smplx_set_handpose", text="Set")

        col.separator()
        col.operator("object.smplx_write_pose")
        col.separator()
        col.operator("object.smplx_load_pose")

class SMPLX_PT_Animation(bpy.types.Panel):
    bl_label = "Animation"
    bl_category = "SMPL-X"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        col.operator("object.smplx_add_animation")

class SMPLX_PT_Export(bpy.types.Panel):
    bl_label = "Export"
    bl_category = "SMPL-X"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)

        col.operator("object.smplx_export_alembic")
        col.separator()

        col.operator("object.smplx_export_fbx")
        col.separator()

        col.operator("object.smplx_export_shape")
        col.separator()

        row = col.row(align=True)
        row.operator("ed.undo", icon='LOOP_BACK')
        row.operator("ed.redo", icon='LOOP_FORWARDS')
        col.separator()

        (year, month, day) = bl_info["version"]
        col.label(text="Version: %s-%s-%s" % (year, month, day))

classes = [
    PG_SMPLXProperties,
    SMPLXAddGender,
    SMPLXSetTexture,
    SMPLXMeasurementsToShape,
    SMPLXRandomShape,
    SMPLXResetShape,
    SMPLXRandomExpressionShape,
    SMPLXResetExpressionShape,
    SMPLXSnapGroundPlane,
    SMPLXUpdateJointLocations,
    SMPLXSetPoseshapes,
    SMPLXResetPoseshapes,
    SMPLXSetHandpose,
    SMPLXWritePose,
    SMPLXLoadPose,
    SMPLXResetPose,
    SMPLXAddAnimation,
    SMPLXExportAlembic,
    SMPLXExportFBX,
    SMPLXExportShape,
    SMPLX_PT_Model,
    SMPLX_PT_Shape,
    SMPLX_PT_Pose,
    SMPLX_PT_Animation,
    SMPLX_PT_Export
]

def register():
    from bpy.utils import register_class
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.WindowManager.smplx_tool = PointerProperty(type=PG_SMPLXProperties)

def unregister():
    from bpy.utils import unregister_class
    for cls in classes:
        bpy.utils.unregister_class(cls)

    del bpy.types.WindowManager.smplx_tool

if __name__ == "__main__":
    register()

