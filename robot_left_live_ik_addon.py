bl_info = {
    "name": "Robot Left Hybrid IK",
    "author": "OpenAI Codex",
    "version": (0, 3, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Robotics",
    "description": "Drive robot_left with a fast native Panda IK seed plus visible-chain correction",
    "category": "Rigging",
}

import math
import pathlib
import sys

import bpy
from bpy.app.handlers import persistent
from bpy.props import BoolProperty
from bpy.types import Operator, Panel
from mathutils import Matrix, Vector


BACKUP_BLEND = "/home/tp2/Documents/kejia/blender/dual_arm_cable_clip_backup.blend"
TCP_NAME = "TCP_robot_left"
SOLVER_NAME = "Panda_left_solver"
SOLVER_TCP_NAME = "TCP_left_solver"
EE_OBJECT_NAME = "fer_link8.002"
JOINT_BONE = "Bone"
VISIBLE_JOINTS = [f"fer_link{i}.002" for i in range(1, 8)]
SOLVER_BONES = [f"Axis-{i}" for i in range(1, 8)]

_HANDLER_GUARD = False
_LAST_TCP_STATE = None
_Q_OFFSETS = None


def _visible_joint_objects():
    objs = []
    for name in VISIBLE_JOINTS:
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != "ARMATURE":
            return None
        bone = obj.pose.bones.get(JOINT_BONE)
        if bone is None:
            return None
        bone.rotation_mode = "XYZ"
        objs.append(obj)
    return objs


def _visible_q():
    objs = _visible_joint_objects()
    if not objs:
        return None
    return [obj.pose.bones[JOINT_BONE].rotation_euler.y for obj in objs]


def _set_visible_q(q_values):
    objs = _visible_joint_objects()
    if not objs:
        return
    for obj, q_value in zip(objs, q_values):
        obj.pose.bones[JOINT_BONE].rotation_euler.y = q_value


def _visible_limits():
    limits = []
    objs = _visible_joint_objects()
    if not objs:
        return None
    for obj in objs:
        bone = obj.pose.bones[JOINT_BONE]
        lo = -math.pi
        hi = math.pi
        for constraint in bone.constraints:
            if constraint.type == "LIMIT_ROTATION" and getattr(constraint, "use_limit_y", False):
                lo = constraint.min_y
                hi = constraint.max_y
                break
        limits.append((lo, hi))
    return limits


def _visible_ee_matrix():
    obj = bpy.data.objects.get(EE_OBJECT_NAME)
    if obj is None or obj.type != "ARMATURE":
        return None
    bone = obj.pose.bones.get(JOINT_BONE)
    if bone is None:
        return None
    return obj.matrix_world.copy() @ bone.matrix.copy()


def _predicted_visible_ee_matrix(q_values):
    current_q = _visible_q()
    if current_q is None:
        return None
    try:
        _set_visible_q(q_values)
        bpy.context.view_layer.update()
        ee = _visible_ee_matrix()
        return ee.copy() if ee is not None else None
    finally:
        _set_visible_q(current_q)
        bpy.context.view_layer.update()


def _predicted_visible_pos(q_values):
    ee = _predicted_visible_ee_matrix(q_values)
    return None if ee is None else ee.translation.copy()


def _jacobian_pos(q_values, eps=1e-4):
    base = _predicted_visible_pos(q_values)
    if base is None:
        return None
    cols = []
    for i in range(len(q_values)):
        perturbed = list(q_values)
        perturbed[i] += eps
        pos = _predicted_visible_pos(perturbed)
        cols.append((pos - base) / eps)
    return cols


def _solve_position_target(target_pos, q_start, limits, iterations=8):
    q_values = list(q_start)
    damping = 0.035
    for _ in range(iterations):
        pos = _predicted_visible_pos(q_values)
        if pos is None:
            break
        error = target_pos - pos
        if error.length < 1e-4:
            break
        jac = _jacobian_pos(q_values)
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


def _solver_objects():
    return bpy.data.objects.get(SOLVER_NAME), bpy.data.objects.get(SOLVER_TCP_NAME)


def _hide_legacy_helpers():
    for name in ["robot_left_IK_CTRL"]:
        obj = bpy.data.objects.get(name)
        if obj is not None:
            obj.hide_viewport = True
            obj.hide_render = True
            obj.hide_set(True)


def ensure_solver_pair():
    solver, solver_tcp = _solver_objects()
    if solver is None or solver_tcp is None:
        with bpy.data.libraries.load(BACKUP_BLEND, link=False) as (data_from, data_to):
            data_to.objects = ["Panda", "TCP"]
        imported = [obj for obj in data_to.objects if obj is not None]
        if len(imported) != 2:
            return None, None
        for obj in imported:
            bpy.context.scene.collection.objects.link(obj)
        solver = next(obj for obj in imported if obj.type == "ARMATURE")
        solver_tcp = next(obj for obj in imported if obj.type == "EMPTY")
        solver.name = SOLVER_NAME
        solver_tcp.name = SOLVER_TCP_NAME
        if solver.data is not None:
            solver.data.name = f"{SOLVER_NAME}_data"
    solver.hide_viewport = True
    solver.hide_render = True
    solver.hide_set(True)
    solver_tcp.hide_viewport = True
    solver_tcp.hide_render = True
    solver_tcp.hide_set(True)
    solver_tcp.empty_display_type = "ARROWS"
    solver_tcp.empty_display_size = 0.12
    solver_tcp.show_name = False
    limits = _visible_limits() or []
    for bone_name, limit_pair in zip(SOLVER_BONES, limits):
        bone = solver.pose.bones[bone_name]
        bone.rotation_mode = "XYZ"
        bone.lock_ik_x = True
        bone.lock_ik_z = True
        bone.ik_stretch = 0.0
        bone.use_ik_limit_y = True
        bone.ik_min_y = limit_pair[0]
        bone.ik_max_y = limit_pair[1]
    ik = solver.pose.bones["Axis-7"].constraints.get("IK")
    if ik is not None:
        ik.target = solver_tcp
        ik.use_tail = True
        ik.mute = False
    _hide_legacy_helpers()
    return solver, solver_tcp


def _extract_solver_q():
    solver, _ = _solver_objects()
    if solver is None:
        return None
    q_values = []
    for bone_name in SOLVER_BONES:
        pb = solver.pose.bones[bone_name]
        if pb.parent:
            rel = pb.parent.matrix.inverted() @ pb.matrix
            rest = pb.parent.bone.matrix_local.inverted() @ pb.bone.matrix_local
        else:
            rel = pb.matrix.copy()
            rest = pb.bone.matrix_local.copy()
        delta = rest.inverted() @ rel
        q_values.append(delta.to_euler("XYZ").y)
    return q_values


def _align_solver_pair_to_visible():
    global _Q_OFFSETS
    solver, solver_tcp = ensure_solver_pair()
    visible_q = _visible_q()
    ee = _visible_ee_matrix()
    if solver is None or solver_tcp is None or visible_q is None or ee is None:
        return None
    transform = ee @ solver_tcp.matrix_world.inverted()
    solver.matrix_world = transform @ solver.matrix_world
    solver_tcp.matrix_world = transform @ solver_tcp.matrix_world
    bpy.context.view_layer.update()
    solver_q = _extract_solver_q()
    if solver_q is None:
        return None
    _Q_OFFSETS = [vq - sq for vq, sq in zip(visible_q, solver_q)]
    return solver, solver_tcp


def sync_tcp_to_hand():
    global _LAST_TCP_STATE
    tcp = ensure_tcp()
    aligned = _align_solver_pair_to_visible()
    ee = _visible_ee_matrix()
    if aligned is None or ee is None:
        return None
    solver, solver_tcp = aligned
    tcp.matrix_world = ee.copy()
    solver_tcp.matrix_world = tcp.matrix_world.copy()
    bpy.context.view_layer.update()
    _LAST_TCP_STATE = _tcp_state(tcp)
    return tcp


def solve_once():
    global _HANDLER_GUARD, _LAST_TCP_STATE
    if _HANDLER_GUARD:
        return False
    scene = bpy.context.scene
    if not getattr(scene, "robot_left_live_ik_enabled", False):
        return False
    tcp = ensure_tcp()
    solver, solver_tcp = ensure_solver_pair()
    if solver is None or solver_tcp is None or _Q_OFFSETS is None:
        return False
    state = _tcp_state(tcp)
    if state == _LAST_TCP_STATE:
        return False
    _HANDLER_GUARD = True
    try:
        solver_tcp.matrix_world = tcp.matrix_world.copy()
        bpy.context.view_layer.update()
        solver_q = _extract_solver_q()
        limits = _visible_limits()
        if solver_q is None or limits is None:
            return False
        seed_q = [sq + off for sq, off in zip(solver_q, _Q_OFFSETS)]
        solved_q = _solve_position_target(tcp.matrix_world.translation.copy(), seed_q, limits)
        _set_visible_q(solved_q)
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
    bl_description = "Enable fast hybrid IK for robot_left"

    def execute(self, context):
        context.scene.robot_left_live_ik_enabled = True
        tcp = sync_tcp_to_hand()
        if tcp is None:
            self.report({"ERROR"}, "Could not initialize left IK")
            return {"CANCELLED"}
        solve_once()
        self.report({"INFO"}, "Left TCP IK enabled")
        return {"FINISHED"}


class ROBOTLEFT_OT_sync_tcp_to_hand(Operator):
    bl_idname = "robot_left.sync_tcp_to_hand"
    bl_label = "Snap TCP To Hand"
    bl_description = "Snap the TCP control back onto the current robot flange"

    def execute(self, context):
        tcp = sync_tcp_to_hand()
        if tcp is None:
            self.report({"ERROR"}, "Could not snap TCP")
            return {"CANCELLED"}
        self.report({"INFO"}, "TCP snapped to current flange pose")
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
        if bpy.data.objects.get(TCP_NAME) is not None:
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
    _hide_legacy_helpers()


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.robot_left_live_ik_enabled = BoolProperty(
        name="Live IK",
        default=False,
        description="Drive the visible left Panda from a fast hybrid IK rig",
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
