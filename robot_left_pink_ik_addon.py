bl_info = {
    "name": "Robot Left Pink IK",
    "author": "OpenAI Codex",
    "version": (0, 1, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Robotics",
    "description": "Drive robot_left with a Pink IK backend running in the blender Conda environment",
    "category": "Rigging",
}

import json
import subprocess
import sys

import bpy
import mathutils
from bpy.app.handlers import persistent
from bpy.props import BoolProperty
from bpy.types import Operator, Panel


TCP_NAME = "TCP_robot_left"
TCP_MARKER_NAME = "TCP_robot_left_marker"
EE_OBJECT_NAME = "fer_hand.002"
JOINT_BONE = "Bone"
VISIBLE_JOINTS = [f"fer_link{i}.002" for i in range(1, 8)]
TCP_LOCAL_OFFSET = (0.0, 0.1034, 0.0)
BACKEND_CMD = [
    "/home/tp2/anaconda3/envs/blender/bin/python",
    "/home/tp2/Documents/kejia/blender/scripts/pink_backend/pink_ik_server.py",
]

_PROCESS = None
_HANDLER_GUARD = False
_LAST_TCP_STATE = None


def _quat_xyzw(quat):
    return [quat.x, quat.y, quat.z, quat.w]


def _ee_matrix_world():
    obj = bpy.data.objects.get(EE_OBJECT_NAME)
    if obj is None or obj.type != "ARMATURE":
        return None
    bone = obj.pose.bones.get(JOINT_BONE)
    if bone is None:
        return None
    matrix = obj.matrix_world.copy() @ bone.matrix.copy()
    tip_matrix = matrix.copy()
    tip_matrix.translation = matrix @ mathutils.Vector(TCP_LOCAL_OFFSET)
    return tip_matrix


def _visible_q():
    values = []
    for name in VISIBLE_JOINTS:
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != "ARMATURE":
            return None
        bone = obj.pose.bones.get(JOINT_BONE)
        if bone is None:
            return None
        bone.rotation_mode = "XYZ"
        values.append(float(bone.rotation_euler.y))
    return values


def _set_visible_q(q_values):
    for name, q_value in zip(VISIBLE_JOINTS, q_values):
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != "ARMATURE":
            continue
        bone = obj.pose.bones.get(JOINT_BONE)
        if bone is None:
            continue
        bone.rotation_mode = "XYZ"
        bone.rotation_euler.y = q_value


def _tcp_state(tcp):
    quat = tcp.matrix_world.to_quaternion()
    return (
        tuple(round(v, 7) for v in tcp.matrix_world.translation),
        tuple(round(v, 7) for v in quat),
    )


def ensure_tcp():
    tcp = bpy.data.objects.get(TCP_NAME)
    target_collection = bpy.data.collections.get("robots_new") or bpy.context.scene.collection
    if tcp is None:
        tcp = bpy.data.objects.new(TCP_NAME, None)
        target_collection.objects.link(tcp)
    elif target_collection not in tcp.users_collection:
        for collection in list(tcp.users_collection):
            collection.objects.unlink(tcp)
        target_collection.objects.link(tcp)
    tcp.empty_display_type = "ARROWS"
    tcp.empty_display_size = 0.3
    tcp.show_name = True
    tcp.hide_viewport = False
    tcp.hide_render = False
    tcp.hide_set(False)
    ensure_tcp_marker(tcp, target_collection)
    return tcp


def ensure_tcp_marker(tcp, target_collection):
    marker = bpy.data.objects.get(TCP_MARKER_NAME)
    if marker is None:
        marker = bpy.data.objects.new(TCP_MARKER_NAME, None)
        target_collection.objects.link(marker)
    elif target_collection not in marker.users_collection:
        for collection in list(marker.users_collection):
            collection.objects.unlink(marker)
        target_collection.objects.link(marker)
    marker.parent = tcp
    marker.matrix_parent_inverse.identity()
    marker.location = (0.0, 0.0, 0.0)
    marker.rotation_euler = (0.0, 0.0, 0.0)
    marker.scale = (1.0, 1.0, 1.0)
    marker.empty_display_type = "SPHERE"
    marker.empty_display_size = 0.035
    marker.show_name = False
    marker.hide_viewport = False
    marker.hide_render = False
    marker.hide_set(False)
    return marker


def _start_backend():
    global _PROCESS
    if _PROCESS is not None and _PROCESS.poll() is None:
        return True
    _PROCESS = subprocess.Popen(
        BACKEND_CMD,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    line = _PROCESS.stdout.readline().strip()
    if not line:
        return False
    try:
        payload = json.loads(line)
    except Exception:
        return False
    return payload.get("status") == "ready"


def _stop_backend():
    global _PROCESS
    if _PROCESS is None:
        return
    if _PROCESS.poll() is None:
        try:
            _PROCESS.stdin.write(json.dumps({"command": "quit"}) + "\n")
            _PROCESS.stdin.flush()
        except Exception:
            pass
        try:
            _PROCESS.terminate()
        except Exception:
            pass
    _PROCESS = None


def _backend_solve(payload):
    if not _start_backend():
        return None
    try:
        _PROCESS.stdin.write(json.dumps({"command": "solve", **payload}) + "\n")
        _PROCESS.stdin.flush()
        line = _PROCESS.stdout.readline().strip()
        if not line:
            return None
        result = json.loads(line)
        if result.get("status") != "ok":
            return None
        return result["result"]
    except Exception:
        return None


def sync_tcp_to_hand():
    global _LAST_TCP_STATE
    ee = _ee_matrix_world()
    tcp = ensure_tcp()
    if ee is None or tcp is None:
        return None
    tcp.matrix_world = ee.copy()
    _LAST_TCP_STATE = _tcp_state(tcp)
    return tcp


def solve_once():
    global _HANDLER_GUARD, _LAST_TCP_STATE
    if _HANDLER_GUARD:
        return False
    scene = bpy.context.scene
    if not getattr(scene, "robot_left_pink_ik_enabled", False):
        return False
    tcp = ensure_tcp()
    state = _tcp_state(tcp)
    if state == _LAST_TCP_STATE:
        return False
    q_values = _visible_q()
    ee = _ee_matrix_world()
    if q_values is None or ee is None:
        return False
    payload = {
        "q": q_values,
        "current_ee": {
            "position": list(ee.translation),
            "quaternion_xyzw": _quat_xyzw(ee.to_quaternion()),
        },
        "target": {
            "position": list(tcp.matrix_world.translation),
            "quaternion_xyzw": _quat_xyzw(tcp.matrix_world.to_quaternion()),
        }
    }
    _HANDLER_GUARD = True
    try:
        result = _backend_solve(
            {
                **payload,
                "iterations": 60,
                "dt": 0.02,
                "posture_cost": 1e-4,
                "damping_cost": 1e-4,
            }
        )
        if result is None:
            return False
        _set_visible_q(result["q"])
        bpy.context.view_layer.update()
        _LAST_TCP_STATE = state
    finally:
        _HANDLER_GUARD = False
    return True


@persistent
def robot_left_pink_handler(scene, depsgraph):
    if _HANDLER_GUARD:
        return
    solve_once()


class ROBOTLEFT_OT_enable_pink_ik(Operator):
    bl_idname = "robot_left.enable_pink_ik"
    bl_label = "Enable Left Pink IK"
    bl_description = "Start the Pink backend and enable live IK for robot_left"

    def execute(self, context):
        _stop_backend()
        if not _start_backend():
            self.report({"ERROR"}, "Could not start Pink backend")
            return {"CANCELLED"}
        tcp = sync_tcp_to_hand()
        if tcp is None:
            self.report({"ERROR"}, "Could not initialize TCP")
            return {"CANCELLED"}
        context.scene.robot_left_pink_ik_enabled = True
        self.report({"INFO"}, "Left Pink IK enabled")
        return {"FINISHED"}


class ROBOTLEFT_OT_sync_pink_tcp(Operator):
    bl_idname = "robot_left.sync_pink_tcp"
    bl_label = "Snap TCP To Hand"
    bl_description = "Snap the Pink TCP control to the current robot flange"

    def execute(self, context):
        tcp = sync_tcp_to_hand()
        if tcp is None:
            self.report({"ERROR"}, "Could not snap TCP")
            return {"CANCELLED"}
        self.report({"INFO"}, "TCP snapped to current flange pose")
        return {"FINISHED"}


class VIEW3D_PT_robot_left_pink_ik(Panel):
    bl_label = "Robot Left Pink IK"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Robotics"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        layout.prop(scene, "robot_left_pink_ik_enabled", text="Enabled")
        layout.operator("robot_left.enable_pink_ik")
        layout.operator("robot_left.sync_pink_tcp")
        if bpy.data.objects.get(TCP_NAME) is not None:
            layout.label(text=f"TCP: {TCP_NAME}")


CLASSES = (
    ROBOTLEFT_OT_enable_pink_ik,
    ROBOTLEFT_OT_sync_pink_tcp,
    VIEW3D_PT_robot_left_pink_ik,
)


def ensure_handler():
    handlers = bpy.app.handlers.depsgraph_update_post
    if robot_left_pink_handler not in handlers:
        handlers.append(robot_left_pink_handler)


def remove_handler():
    handlers = bpy.app.handlers.depsgraph_update_post
    if robot_left_pink_handler in handlers:
        handlers.remove(robot_left_pink_handler)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.robot_left_pink_ik_enabled = BoolProperty(
        name="Pink IK",
        default=False,
        description="Drive robot_left using the Pink backend in the blender Conda environment",
    )
    ensure_handler()
    ensure_tcp()


def unregister():
    remove_handler()
    _stop_backend()
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
    if hasattr(bpy.types.Scene, "robot_left_pink_ik_enabled"):
        del bpy.types.Scene.robot_left_pink_ik_enabled


if __name__ == "__main__":
    sys.modules.setdefault("robot_left_pink_ik_addon", sys.modules[__name__])
    register()
