bl_info = {
    "name": "Robot Left Live IK",
    "author": "OpenAI Codex",
    "version": (0, 1, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Robotics",
    "description": "Drive robot_left Panda joints from a draggable TCP target",
    "category": "Rigging",
}

import math
import sys

import bpy
from bpy.app.handlers import persistent
from bpy.props import BoolProperty
from bpy.types import Operator, Panel
from mathutils import Euler, Matrix, Vector


TCP_NAME = "TCP_robot_left"
HAND_NAME = "fer_hand.002"
JOINT_OBJECTS = [f"fer_link{i}.002" for i in range(1, 8)]
JOINT_BONE = "Bone"
_HANDLER_GUARD = False
_LAST_TCP_STATE = None
_CALIBRATION = None
_CHAIN_CACHE = None


def _joint_objects():
    objs = []
    for name in JOINT_OBJECTS:
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != "ARMATURE":
            return None
        bone = obj.pose.bones.get(JOINT_BONE)
        if bone is None:
            return None
        if bone.rotation_mode != "XYZ":
            bone.rotation_mode = "XYZ"
        objs.append(obj)
    return objs


def _hand_object():
    return bpy.data.objects.get(HAND_NAME)


def _current_q():
    objs = _joint_objects()
    if not objs:
        return None
    return [obj.pose.bones[JOINT_BONE].rotation_euler.y for obj in objs]


def _joint_limits():
    limits = []
    objs = _joint_objects()
    if not objs:
        return None
    for obj in objs:
        bone = obj.pose.bones[JOINT_BONE]
        lo = -math.pi
        hi = math.pi
        for constraint in bone.constraints:
            if constraint.type != "LIMIT_ROTATION":
                continue
            if getattr(constraint, "use_limit_y", False):
                lo = constraint.min_y
                hi = constraint.max_y
                break
        limits.append((lo, hi))
    return limits


def _set_q(q_values):
    objs = _joint_objects()
    if not objs:
        return
    for obj, q in zip(objs, q_values):
        obj.pose.bones[JOINT_BONE].rotation_euler.y = q


def _rot_y(q_value):
    return Euler((0.0, q_value, 0.0), "XYZ").to_matrix().to_4x4()


def _frame_world(obj_name):
    obj = bpy.data.objects.get(obj_name)
    if obj is None or obj.type != "ARMATURE":
        return None
    bone = obj.pose.bones.get(JOINT_BONE)
    if bone is None:
        return None
    return obj.matrix_world.copy() @ bone.matrix.copy()


def _extract_chain():
    global _CHAIN_CACHE
    if _CHAIN_CACHE is not None:
        return _CHAIN_CACHE
    q_values = _current_q()
    hand = _hand_object()
    if q_values is None or hand is None:
        return None
    base = _frame_world("fer_link0.002")
    frames = [_frame_world(f"fer_link{i}.002") for i in range(1, 8)]
    if base is None or any(frame is None for frame in frames):
        return None
    parent = base
    fixed = []
    for frame, q_value in zip(frames, q_values):
        fixed.append(parent.inverted() @ frame @ _rot_y(-q_value))
        parent = frame
    hand_offset = frames[-1].inverted() @ hand.matrix_world.copy()
    _CHAIN_CACHE = {
        "base": base,
        "fixed": fixed,
        "hand_offset": hand_offset,
    }
    return _CHAIN_CACHE


def _calibration():
    global _CALIBRATION
    if _CALIBRATION is not None:
        return _CALIBRATION
    chain = _extract_chain()
    if chain is None:
        return None
    _CALIBRATION = Matrix.Identity(4)
    return _CALIBRATION


def _predicted_hand_matrix(q_values):
    calibration = _calibration()
    chain = _extract_chain()
    if calibration is None or chain is None:
        return None
    transform = chain["base"].copy()
    for fixed, q_value in zip(chain["fixed"], q_values):
        transform @= fixed @ _rot_y(q_value)
    transform @= chain["hand_offset"]
    return calibration @ transform


def _hand_pos(q_values):
    matrix = _predicted_hand_matrix(q_values)
    if matrix is None:
        return None
    return matrix.translation


def _pos_error(q_values, target_pos):
    hand_pos = _hand_pos(q_values)
    if hand_pos is None:
        return None
    return target_pos - hand_pos


def _jacobian(q_values, target_pos, eps=1e-4):
    base_pos = _hand_pos(q_values)
    if base_pos is None:
        return None
    columns = []
    for i in range(len(q_values)):
        perturbed = list(q_values)
        perturbed[i] += eps
        delta_pos = _hand_pos(perturbed)
        columns.append((delta_pos - base_pos) / eps)
    return columns


def _solve_to_target(target_pos, q_start, limits, iterations=48):
    q_values = list(q_start)
    damping = 0.03
    for _ in range(iterations):
        error = _pos_error(q_values, target_pos)
        if error is None:
            break
        if error.length < 1e-4:
            break
        jac = _jacobian(q_values, target_pos)
        if jac is None:
            break
        jjt = Matrix(((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
        for col in jac:
            jjt[0][0] += col.x * col.x
            jjt[0][1] += col.x * col.y
            jjt[0][2] += col.x * col.z
            jjt[1][0] += col.y * col.x
            jjt[1][1] += col.y * col.y
            jjt[1][2] += col.y * col.z
            jjt[2][0] += col.z * col.x
            jjt[2][1] += col.z * col.y
            jjt[2][2] += col.z * col.z
        for i in range(3):
            jjt[i][i] += damping * damping
        try:
            gain = jjt.inverted() @ error
        except Exception:
            break
        dq = [col.dot(gain) for col in jac]
        for i, delta in enumerate(dq):
            lo, hi = limits[i]
            q_values[i] = min(max(q_values[i] + delta, lo), hi)
    return q_values


def _tcp_state(tcp):
    quat = tcp.matrix_world.to_quaternion()
    return (
        tuple(round(v, 7) for v in tcp.matrix_world.translation),
        tuple(round(v, 7) for v in quat),
    )


def ensure_tcp():
    tcp = bpy.data.objects.get(TCP_NAME)
    if tcp is None:
        tcp = bpy.data.objects.new(TCP_NAME, None)
        bpy.context.scene.collection.objects.link(tcp)
    tcp.empty_display_type = "ARROWS"
    tcp.empty_display_size = 0.3
    tcp.show_name = True
    tcp.hide_viewport = False
    tcp.hide_render = False
    tcp.hide_set(False)
    return tcp


def sync_tcp_to_hand():
    global _LAST_TCP_STATE
    hand = _hand_object()
    tcp = ensure_tcp()
    if hand is None or tcp is None:
        return None
    tcp.matrix_world = hand.matrix_world.copy()
    tcp.hide_viewport = False
    tcp.hide_render = False
    tcp.hide_set(False)
    _LAST_TCP_STATE = _tcp_state(tcp)
    return tcp


def reset_calibration():
    global _CALIBRATION, _CHAIN_CACHE
    _CALIBRATION = None
    _CHAIN_CACHE = None


def solve_once():
    global _HANDLER_GUARD, _LAST_TCP_STATE
    if _HANDLER_GUARD:
        return False
    scene = bpy.context.scene
    if not getattr(scene, "robot_left_live_ik_enabled", False):
        return False
    tcp = bpy.data.objects.get(TCP_NAME)
    if tcp is None:
        return False
    state = _tcp_state(tcp)
    if state == _LAST_TCP_STATE:
        return False
    q_start = _current_q()
    limits = _joint_limits()
    if q_start is None or limits is None:
        return False
    _HANDLER_GUARD = True
    try:
        solved = _solve_to_target(tcp.matrix_world.translation.copy(), q_start, limits)
        _set_q(solved)
        bpy.context.view_layer.update()
        _LAST_TCP_STATE = state
    finally:
        _HANDLER_GUARD = False
    return True


@persistent
def robot_left_ik_handler(scene, depsgraph):
    if _HANDLER_GUARD:
        return
    solve_once()


class ROBOTLEFT_OT_enable_live_ik(Operator):
    bl_idname = "robot_left.enable_live_ik"
    bl_label = "Enable Left TCP IK"
    bl_description = "Create the TCP target and enable live Panda IK for robot_left"

    def execute(self, context):
        reset_calibration()
        sync_tcp_to_hand()
        context.scene.robot_left_live_ik_enabled = True
        solve_once()
        self.report({"INFO"}, "Left TCP IK enabled")
        return {"FINISHED"}


class ROBOTLEFT_OT_sync_tcp_to_hand(Operator):
    bl_idname = "robot_left.sync_tcp_to_hand"
    bl_label = "Snap TCP To Hand"
    bl_description = "Recalibrate the TCP target to the current hand pose"

    def execute(self, context):
        reset_calibration()
        sync_tcp_to_hand()
        self.report({"INFO"}, "TCP snapped to the current hand pose")
        return {"FINISHED"}


class VIEW3D_PT_robot_left_live_ik(Panel):
    bl_label = "Robot Left IK"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Robotics"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        layout.prop(scene, "robot_left_live_ik_enabled", text="Enabled")
        layout.operator("robot_left.enable_live_ik")
        layout.operator("robot_left.sync_tcp_to_hand")
        tcp = bpy.data.objects.get(TCP_NAME)
        if tcp is not None:
            layout.label(text=f"TCP: {TCP_NAME}")


CLASSES = (
    ROBOTLEFT_OT_enable_live_ik,
    ROBOTLEFT_OT_sync_tcp_to_hand,
    VIEW3D_PT_robot_left_live_ik,
)


def ensure_handler():
    handlers = bpy.app.handlers.depsgraph_update_post
    if robot_left_ik_handler not in handlers:
        handlers.append(robot_left_ik_handler)


def remove_handler():
    handlers = bpy.app.handlers.depsgraph_update_post
    if robot_left_ik_handler in handlers:
        handlers.remove(robot_left_ik_handler)


def ensure_setup():
    ensure_handler()
    ensure_tcp()
    scene = bpy.context.scene
    if getattr(scene, "robot_left_live_ik_enabled", False):
        reset_calibration()
        sync_tcp_to_hand()


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.robot_left_live_ik_enabled = BoolProperty(
        name="Live IK",
        default=True,
        description="Keep robot_left joints synced to TCP_robot_left",
    )
    ensure_setup()


def unregister():
    remove_handler()
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
    if hasattr(bpy.types.Scene, "robot_left_live_ik_enabled"):
        del bpy.types.Scene.robot_left_live_ik_enabled


if __name__ == "__main__":
    sys.modules.setdefault("robot_left_live_ik_addon", sys.modules[__name__])
    register()
