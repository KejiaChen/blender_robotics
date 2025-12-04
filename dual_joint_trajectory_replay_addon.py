bl_info = {
    "name": "Dual Arm Trajectory + Scrub (MoveIt txt)",
    "version": (1, 0, 2),
    "blender": (4, 0, 0),
    "category": "Animation",
    "description": "Load a dual-arm trajectory (t + 7 pos + 7 vel per arm) and scrub with a slider",
}

import bpy, os, math, bisect
from bpy.types import Panel, Operator
from bpy.props import StringProperty, BoolProperty, IntProperty, FloatProperty, EnumProperty
from mathutils import Vector
import bmesh

# -------- Internal state --------
TRAJ_A = None     # {'t': [...], 'q': [[7],...]} for Arm A
TRAJ_B = None     # same for Arm B
ARMREFS = {'A': None, 'B': None}   # caches (obj,bone) lists per arm

# -------- Utilities --------
def get_joint_bone(base_name, index, suffix, bone_name):
    obj_name = f"{base_name}{index}{suffix}"
    obj = bpy.data.objects.get(obj_name)
    if not obj:
        return None, None, f"Object '{obj_name}' not found"
    if obj.type != 'ARMATURE' or not getattr(obj, "pose", None):
        return obj, None, f"'{obj.name}' is not an armature with pose"
    bone = obj.pose.bones.get(bone_name)
    if not bone:
        return obj, None, f"'{obj.name}' has no pose bone '{bone_name}'"
    try:
        if bone.rotation_mode != 'XYZ':
            bone.rotation_mode = 'XYZ'
    except Exception:
        pass
    return obj, bone, None

def ensure_arm_refs(scene, which):
    """Build/cache (obj,bone) list for arm 'A' or 'B' using that arm's suffix."""
    suffix = scene.sa_suffix_a if which == 'A' else scene.sa_suffix_b
    key = (scene.sa_base_name, scene.sa_start_index, scene.sa_joints, suffix, scene.sa_bone_name)
    cached = ARMREFS.get(which)
    if cached and cached.get('key') == key:
        return cached['refs']

    refs, errors = [], []
    for j in range(scene.sa_start_index, scene.sa_start_index + scene.sa_joints):
        obj, bone, err = get_joint_bone(scene.sa_base_name, j+1, suffix, scene.sa_bone_name)
        if err:
            errors.append(err); refs.append((None, None))
        else:
            refs.append((obj, bone))
            obj.animation_data_create()

    if errors:
        print(f"[Dual] Mapping warnings {which}:", "; ".join(errors))
    ARMREFS[which] = {'key': key, 'refs': refs}
    return refs

def _global_time_range():
    tmins, tmaxs = [], []
    if TRAJ_A: tmins.append(TRAJ_A['t'][0]); tmaxs.append(TRAJ_A['t'][-1])
    if TRAJ_B: tmins.append(TRAJ_B['t'][0]); tmaxs.append(TRAJ_B['t'][-1])
    if not tmins: return 0.0, 1.0
    return min(tmins), max(tmaxs)

def parse_single_arm(filepath, delim_mode="AUTO", time_unit="SECONDS", has_header=False, degrees=False):
    if not os.path.isfile(filepath):
        raise RuntimeError(f"File not found: {filepath}")
    times, joints = [], []
    header_skipped = False
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"): 
                continue
            if   delim_mode == "COMMA": parts = s.split(",")
            elif delim_mode == "TAB":   parts = s.split("\t")
            elif delim_mode == "SPACE": parts = s.split()
            else:                       parts = s.split(",") if "," in s else (s.split("\t") if "\t" in s else s.split())
            if has_header and not header_skipped:
                header_skipped = True
                continue
            try:
                vals = [float(x) for x in parts]
            except ValueError:
                raise RuntimeError(f"Non-numeric data in line: {line}")
            if len(vals) < 15:  # t + 7 pos + 7 vel (vel ignored)
                raise RuntimeError(f"Expected 15 numbers (t + 7 pos + 7 vel). Got {len(vals)}")
            t = vals[0] * (0.001 if time_unit == "MILLISECONDS" else 1.0)
            qpos = vals[1:8]
            if degrees:
                qpos = [math.radians(v) for v in qpos]
            times.append(t); joints.append(qpos)
    if not times:
        raise RuntimeError("No valid data rows found.")

    # sort & dedup
    pairs = sorted(zip(times, joints), key=lambda x: x[0])
    times, joints = [], []
    for t,q in pairs:
        if times and abs(t - times[-1]) < 1e-12:
            times[-1] = t; joints[-1] = q
        else:
            times.append(t); joints.append(q)
    return {'t': times, 'q': joints}

def sample_q(traj, t):
    ts, qs = traj['t'], traj['q']
    if t <= ts[0]: return qs[0]
    if t >= ts[-1]: return qs[-1]
    i = bisect.bisect_left(ts, t)
    t0,t1 = ts[i-1], ts[i]; q0,q1 = qs[i-1], qs[i]
    if t1 <= t0 + 1e-12: return q1
    r = (t - t0) / (t1 - t0)
    return [(1-r)*a + r*b for a,b in zip(q0,q1)]

def apply_pose_from_scrub(context):
    scn = context.scene
    t0, t1 = _global_time_range()
    duration = max(t1 - t0, 1e-9)
    t = t0 + scn.sa_scrub * duration
    scn.sa_time = t

    # Arm A
    if TRAJ_A:
        qa = sample_q(TRAJ_A, t)
        refsA = ensure_arm_refs(scn, 'A')
        for j, (obj, bone) in enumerate(refsA[:len(qa)]):
            if obj and bone:
                bone.rotation_euler[1] = qa[j]
    # Arm B
    if TRAJ_B:
        qb = sample_q(TRAJ_B, t)
        refsB = ensure_arm_refs(scn, 'B')
        for j, (obj, bone) in enumerate(refsB[:len(qb)]):
            if obj and bone:
                bone.rotation_euler[1] = qb[j]

def _on_scrub_update(self, context):
    try:
        apply_pose_from_scrub(context)
    except Exception as e:
        print("[Dual] Scrub error:", e)

# -------- TCP utilities (ADD) --------
def _ensure_collection(name):
    coll = bpy.data.collections.get(name)
    if not coll:
        coll = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(coll)
    _make_collection_renderable(coll)   # <-- ADD THIS LINE
    return coll

def _ensure_tcp_marker(name="TCP_Marker_Template"):
    obj = bpy.data.objects.get(name)
    if obj and obj.type == 'MESH':
        return obj
    # Make a tiny low-res UV sphere as a template
    mesh = bpy.data.meshes.new(name + "_Mesh")
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=12, v_segments=8, radius=1.0)
    bm.to_mesh(mesh); bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.hide_set(True); obj.hide_render = True
    obj.display_type = 'WIRE'
    return obj

def _parse_tcp_positions(filepath, delim_mode="AUTO", has_header=False, column_major=False, y_offset=0.0):
    """Each row: timestamp + 16 numbers (flattened 4x4). Returns list of (x,y,z)."""
    if not os.path.isfile(filepath):
        raise RuntimeError(f"File not found: {filepath}")
    pts = []
    header_skipped = False
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if   delim_mode == "COMMA": parts = s.split(",")
            elif delim_mode == "TAB":   parts = s.split("\t")
            elif delim_mode == "SPACE": parts = s.split()
            else:                       parts = s.split(",") if "," in s else (s.split("\t") if "\t" in s else s.split())
            if has_header and not header_skipped:
                header_skipped = True
                continue
            try:
                vals = [float(x) for x in parts]
            except ValueError:
                raise RuntimeError(f"Non-numeric data in line: {line}")
            if len(vals) < 17:
                raise RuntimeError(f"Expected 17 numbers (t + 16 matrix). Got {len(vals)}")
            m = vals[1:17]
            if column_major:
                # m00 m10 m20 m30  m01 m11 m21 m31  m02 m12 m22 m32  m03 m13 m23 m33
                x, y, z = m[12], m[13], m[14]
            else:
                # m00 m01 m02 m03  m10 m11 m12 m13  m20 m21 m22 m23  m30 m31 m32 m33
                x, y, z = m[3], m[7], m[11]
            pts.append((x, y+y_offset, z))
    if not pts:
        raise RuntimeError("No valid TCP rows found.")
    return pts

def _scatter_tcp_points(points, radius=0.01, coll_name="TCP Points", material_name=None):
    template = _ensure_tcp_marker()
    coll = _ensure_collection(coll_name)
    if material_name:
        _assign_material_to_template(template, material_name)  # <-- add this line

    for i, (x, y, z) in enumerate(points):
        inst = template.copy()
        inst.data = template.data
        inst.location = Vector((x, y, z))
        inst["tcp_i"] = i   
        inst.scale = (radius, radius, radius)
        coll.objects.link(inst)
    # Frame view on created collection
    for o in bpy.context.selected_objects:
        o.select_set(False)
    for o in coll.objects:
        o.select_set(True)
    bpy.context.view_layer.objects.active = next(iter(coll.objects), None)
    try:
        bpy.ops.view3d.view_selected(use_all_regions=False)
    except:
        pass
    return len(points)

# -------- TCP clear utilities (ADD) --------
def _delete_collection_and_contents(coll):
    # unlink and delete all objects in the collection
    for obj in list(coll.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    # unlink this collection from any parents
    for parent in list(coll.users_scene) if hasattr(coll, "users_scene") else []:
        parent.collection.children.unlink(coll)
    # If still linked under scene root
    try:
        bpy.context.scene.collection.children.unlink(coll)
    except Exception:
        pass
    # finally remove the collection datablock
    try:
        bpy.data.collections.remove(coll)
    except Exception:
        pass

def _clear_tcp_by_prefix(prefix):
    # Remove ALL TCP collections that match our naming scheme for that arm.
    # We keep the hidden TCP_Marker_Template intact.
    to_delete = [c for c in bpy.data.collections if c.name.startswith(prefix)]
    for coll in to_delete:
        _delete_collection_and_contents(coll)
    return len(to_delete)

# -------- TCP material utility (ADD) --------
def _assign_material_to_template(template_obj, mat_name):
    """Assigns material to the shared mesh so all instances use it."""
    mat = bpy.data.materials.get(mat_name)
    if not mat:
        print(f"[TCP] Material '{mat_name}' not found; using default grey.")
        return False
    mats = template_obj.data.materials
    mats.clear()
    mats.append(mat)
    return True

# ---- visibility helpers (ADD) ----
def _make_collection_renderable(coll):
    # Collection-level toggles
    try:
        coll.hide_viewport = False
        coll.hide_render = False
    except Exception:
        pass

    # View Layer toggles (camera/disable in this view layer)
    def _find_layer_collection(lc, name):
        if lc.collection.name == name:
            return lc
        for c in lc.children:
            f = _find_layer_collection(c, name)
            if f: return f
        return None

    lc = _find_layer_collection(bpy.context.view_layer.layer_collection, coll.name)
    if lc:
        lc.exclude = False
        try:  lc.hide_viewport = False
        except Exception: pass
        try:
            lc.holdout = False
            lc.indirect_only = False
        except Exception:
            pass

# -------- Load / Clear operators --------
class SA_OT_load_a(Operator):
    bl_idname = "sa.load_traj_a"
    bl_label = "Load Arm A"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        global TRAJ_A, ARMREFS
        scn = context.scene
        folder = scn.sa_folder_a or ""
        path = os.path.join(folder, scn.da_common_filename) if scn.da_common_filename else ""
        if not path:
            self.report({'ERROR'}, "Arm A: set folder and filename first"); return {'CANCELLED'}
        try:
            TRAJ_A = parse_single_arm(path, scn.da_delim, scn.sa_time_unit, scn.sa_has_header, scn.sa_degrees)
        except Exception as e:
            self.report({'ERROR'}, f"A: {e}"); return {'CANCELLED'}
        ARMREFS['A'] = None
        dur = TRAJ_A['t'][-1] - TRAJ_A['t'][0]
        self.report({'INFO'}, f"A: {len(TRAJ_A['t'])} samples ({dur:.3f}s)")
        apply_pose_from_scrub(context)
        return {'FINISHED'}

class SA_OT_load_b(Operator):
    bl_idname = "sa.load_traj_b"
    bl_label = "Load Arm B"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        global TRAJ_B, ARMREFS
        scn = context.scene
        folder = scn.sa_folder_b or ""
        path = os.path.join(folder, scn.da_common_filename) if scn.da_common_filename else ""
        if not path:
            self.report({'ERROR'}, "Arm B: set folder and filename first"); return {'CANCELLED'}
        try:
            TRAJ_B = parse_single_arm(path, scn.da_delim, scn.sa_time_unit, scn.sa_has_header, scn.sa_degrees)
        except Exception as e:
            self.report({'ERROR'}, f"B: {e}"); return {'CANCELLED'}
        ARMREFS['B'] = None
        dur = TRAJ_B['t'][-1] - TRAJ_B['t'][0]
        self.report({'INFO'}, f"B: {len(TRAJ_B['t'])} samples ({dur:.3f}s)")
        apply_pose_from_scrub(context)
        return {'FINISHED'}

class SA_OT_clear_a(Operator):
    bl_idname = "sa.clear_traj_a"
    bl_label = "Clear A"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        global TRAJ_A, ARMREFS
        TRAJ_A = None; ARMREFS['A'] = None
        context.scene.sa_time = 0.0
        self.report({'INFO'}, "Cleared Arm A"); return {'FINISHED'}

class SA_OT_clear_b(Operator):
    bl_idname = "sa.clear_traj_b"
    bl_label = "Clear B"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        global TRAJ_B, ARMREFS
        TRAJ_B = None; ARMREFS['B'] = None
        context.scene.sa_time = 0.0
        self.report({'INFO'}, "Cleared Arm B"); return {'FINISHED'}

# -------- TCP visualize operators (ADD) --------
class SA_OT_tcp_visualize_a(Operator):
    bl_idname = "sa.tcp_visualize_a"
    bl_label = "Visualize TCP A"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        scn = context.scene
        folder = scn.sa_folder_a or ""
        fname  = scn.da_tcp_filename or ""
        if not folder or not fname:
            self.report({'ERROR'}, "Set Arm A folder and TCP filename first"); return {'CANCELLED'}
        path = os.path.join(folder, fname)
        try:
            pts = _parse_tcp_positions(
                path,
                delim_mode=scn.da_delim,
                has_header=scn.sa_has_header,
                column_major=scn.tcp_column_major,
                y_offset=0.281
            )
            # Downsample
            if scn.tcp_step > 1:
                pts = pts[::scn.tcp_step]
            n = _scatter_tcp_points(pts, radius=scn.tcp_radius, coll_name=f"TCP A ({fname})", material_name=scn.tcp_material)
        except Exception as e:
            self.report({'ERROR'}, f"TCP A: {e}"); return {'CANCELLED'}
        self.report({'INFO'}, f"TCP A: Plotted {n} points")
        return {'FINISHED'}

class SA_OT_tcp_visualize_b(Operator):
    bl_idname = "sa.tcp_visualize_b"
    bl_label = "Visualize TCP B"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        scn = context.scene
        folder = scn.sa_folder_b or ""
        fname  = scn.da_tcp_filename or ""
        if not folder or not fname:
            self.report({'ERROR'}, "Set Arm B folder and TCP filename first"); return {'CANCELLED'}
        path = os.path.join(folder, fname)
        try:
            pts = _parse_tcp_positions(
                path,
                delim_mode=scn.da_delim,
                has_header=scn.sa_has_header,
                column_major=scn.tcp_column_major,
                y_offset=-0.281
            )
            if scn.tcp_step > 1:
                pts = pts[::scn.tcp_step]
            n = _scatter_tcp_points(pts, radius=scn.tcp_radius, coll_name=f"TCP B ({fname})", material_name=scn.tcp_material)
        except Exception as e:
            self.report({'ERROR'}, f"TCP B: {e}"); return {'CANCELLED'}
        self.report({'INFO'}, f"TCP B: Plotted {n} points")
        return {'FINISHED'}
    
# -------- TCP clear operators (ADD) --------
class SA_OT_tcp_clear_a(Operator):
    bl_idname = "sa.tcp_clear_a"
    bl_label  = "Clear TCP A"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        n = _clear_tcp_by_prefix("TCP A (")
        self.report({'INFO'}, f"TCP A: cleared {n} collection(s)")
        return {'FINISHED'}

class SA_OT_tcp_clear_b(Operator):
    bl_idname = "sa.tcp_clear_b"
    bl_label  = "Clear TCP B"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        n = _clear_tcp_by_prefix("TCP B (")
        self.report({'INFO'}, f"TCP B: cleared {n} collection(s)")
        return {'FINISHED'}


# -------- Panel --------
class VIEW3D_PT_dual_arm_traj(Panel):
    bl_label = "Dual Arm Trajectory"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Robotics"

    def draw(self, context):
        scn = context.scene
        layout = self.layout

        # Mapping
        box = layout.box()
        box.label(text="Mapping", icon='BONE_DATA')
        row = box.row(align=True)
        row.prop(scn, "sa_base_name", text="Base")
        row.prop(scn, "sa_bone_name",  text="Bone")
        row = box.row(align=True)
        row.prop(scn, "sa_start_index", text="Start idx")
        row.prop(scn, "sa_joints",      text="Joints")
        row = box.row(align=True)
        row.prop(scn, "sa_suffix_a", text="Suffix A (e.g. .001)")
        row.prop(scn, "sa_suffix_b", text="Suffix B (e.g. .002)")

        # Dual-arm paths (each row has its own built-in folder icon)
        box = layout.box()
        box.label(text="Dual-arm paths", icon='FILE_FOLDER')
        box.prop(scn, "sa_folder_a", text="Arm A folder")  # DIR_PATH -> opens directory picker
        box.prop(scn, "sa_folder_b", text="Arm B folder")
        box.prop(scn, "da_common_filename", text="Common filename")

        # Format (shared)
        row = box.row(align=True)
        row.prop(scn, "da_delim", text="Delimiter")
        row.prop(scn, "sa_time_unit", text="Time")
        row = box.row(align=True)
        row.prop(scn, "sa_has_header", text="Header")
        row.prop(scn, "sa_degrees", text="Degrees")

        # Load/status
        row = box.row(align=True)
        row.operator("sa.load_traj_a", text="Load A", icon='FILE_REFRESH')
        row.operator("sa.load_traj_b", text="Load B", icon='FILE_REFRESH')

        if TRAJ_A:
            t0, t1 = TRAJ_A['t'][0], TRAJ_A['t'][-1]
            box.label(text=f"A: {len(TRAJ_A['t'])} samples, duration {t1 - t0:.3f}s")
            box.operator("sa.clear_traj_a", text="Clear A", icon='TRASH')
        if TRAJ_B:
            t0, t1 = TRAJ_B['t'][0], TRAJ_B['t'][-1]
            box.label(text=f"B: {len(TRAJ_B['t'])} samples, duration {t1 - t0:.3f}s")
            box.operator("sa.clear_traj_b", text="Clear B", icon='TRASH')

        # Scrub
        layout.separator()
        box = layout.box()
        box.label(text="Scrub", icon='PLAY')
        box.prop(scn, "sa_scrub", text="Scrub (0–100%)")
        row = box.row(align=True)
        row.label(text=f"Time: {scn.sa_time:.3f} s")
        if not TRAJ_A and not TRAJ_B:
            box.label(text="Load a trajectory first.", icon='INFO')

        # --- TCP options (ADD) ---
        box = layout.box()
        box.label(text="TCP Visualization", icon='SPHERE')
        box.prop(scn, "da_tcp_filename", text="TCP common filename")
        row = box.row(align=True)
        row.prop(scn, "tcp_radius", text="Sphere radius")
        row.prop(scn, "tcp_step",   text="Step (every Nth)")
        box.prop(scn, "tcp_column_major", text="Matrix is Column-Major")
        box.prop(scn, "tcp_material", text="Material")

        row = box.row(align=True)
        row.operator("sa.tcp_visualize_a", text="Visualize TCP A", icon='DOT')
        row.operator("sa.tcp_visualize_b", text="Visualize TCP B", icon='DOT')

        # --- ADD: per-arm TCP clear buttons ---
        row = box.row(align=True)
        row.operator("sa.tcp_clear_a", text="Clear TCP A", icon='TRASH')
        row.operator("sa.tcp_clear_b", text="Clear TCP B", icon='TRASH')


# -------- Register --------
classes = (
    SA_OT_load_a, SA_OT_load_b,
    SA_OT_clear_a, SA_OT_clear_b,
    SA_OT_tcp_visualize_a, SA_OT_tcp_visualize_b,
    SA_OT_tcp_clear_a, SA_OT_tcp_clear_b,
    VIEW3D_PT_dual_arm_traj,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    # Mapping
    bpy.types.Scene.sa_base_name   = StringProperty(name="Base",  default="fer_link")
    bpy.types.Scene.sa_bone_name   = StringProperty(name="Bone",  default="Bone")
    bpy.types.Scene.sa_start_index = IntProperty( name="Start index", default=0, min=0, max=999)
    bpy.types.Scene.sa_joints      = IntProperty( name="Joints", default=7, min=1, max=50)
    bpy.types.Scene.sa_suffix_a    = StringProperty(name="Suffix A", default=".001")
    bpy.types.Scene.sa_suffix_b    = StringProperty(name="Suffix B", default=".002")

    # Folders + common filename
    bpy.types.Scene.sa_folder_a        = StringProperty(name="Arm A folder", default="/home/tp2/ws_humble/trajectories_leader/blender_render", subtype='DIR_PATH')
    bpy.types.Scene.sa_folder_b        = StringProperty(name="Arm B folder", default="/home/tp2/ws_humble/trajectories_follower/blender_render", subtype='DIR_PATH')
    bpy.types.Scene.da_common_filename = StringProperty(name="Common filename", default="traj.txt")

    # Shared format
    bpy.types.Scene.da_delim       = EnumProperty(
        name="Delimiter",
        items=[("AUTO","Auto",""),("SPACE","Space",""),("COMMA","Comma",""),("TAB","Tab","")],
        default="AUTO"
    )
    bpy.types.Scene.sa_time_unit   = EnumProperty(
        name="Time unit", items=[("SECONDS","Seconds",""),("MILLISECONDS","Milliseconds","")],
        default="SECONDS"
    )
    bpy.types.Scene.sa_has_header  = BoolProperty(name="Header row", default=False)
    bpy.types.Scene.sa_degrees     = BoolProperty(name="Degrees input", default=False)

    # Scrub
    bpy.types.Scene.sa_scrub = FloatProperty(
        name="Scrub", min=0.0, max=1.0, default=0.0, subtype='FACTOR',
        description="0=start, 1=end across loaded arms",
        update=_on_scrub_update
    )
    bpy.types.Scene.sa_time  = FloatProperty(name="Time (s)", default=0.0, precision=4)

    # TCP shared options (ADD)
    bpy.types.Scene.da_tcp_filename   = StringProperty(name="TCP filename", default="tcp.txt")
    bpy.types.Scene.tcp_radius        = FloatProperty(name="TCP sphere radius", default=0.005, min=0.0001, soft_max=0.2)
    bpy.types.Scene.tcp_step          = IntProperty(name="TCP step", default=5, min=1)
    bpy.types.Scene.tcp_column_major  = BoolProperty(name="Column-major 4x4", default=True)
    bpy.types.Scene.tcp_material = StringProperty(name="TCP Material", default="trajectoryBlue")

def unregister():
    del bpy.types.Scene.sa_base_name
    del bpy.types.Scene.sa_bone_name
    del bpy.types.Scene.sa_start_index
    del bpy.types.Scene.sa_joints
    del bpy.types.Scene.sa_suffix_a
    del bpy.types.Scene.sa_suffix_b

    del bpy.types.Scene.sa_folder_a
    del bpy.types.Scene.sa_folder_b
    del bpy.types.Scene.da_common_filename

    del bpy.types.Scene.da_delim
    del bpy.types.Scene.sa_time_unit
    del bpy.types.Scene.sa_has_header
    del bpy.types.Scene.sa_degrees

    del bpy.types.Scene.sa_scrub
    del bpy.types.Scene.sa_time

    del bpy.types.Scene.da_tcp_filename
    del bpy.types.Scene.tcp_radius
    del bpy.types.Scene.tcp_step
    del bpy.types.Scene.tcp_column_major
    del bpy.types.Scene.tcp_material


    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()
