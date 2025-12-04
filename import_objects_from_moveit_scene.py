bl_info = {
    "name": "MoveIt Scene Loader (Boxes → Cubes, unique names)",
    "version": (1, 0, 1),
    "blender": (4, 0, 0),
    "category": "3D View",
    "author": "You",
    "description": "Import MoveIt .scene/.txt as cubes, auto-rename if names already exist.",
}

import bpy, re
from bpy.types import Panel, Operator, PropertyGroup
from bpy.props import StringProperty, BoolProperty
from mathutils import Quaternion, Vector

# Allow re-run with Alt+P
try:
    unregister()  # type: ignore
except Exception:
    pass

# -------------------- helpers --------------------

def _ensure_collection(name="MoveIt_Import"):
    coll = bpy.data.collections.get(name)
    if not coll:
        coll = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(coll)
    return coll

def _unique_name(base, existing_names):
    """Return base or base.NNN if taken (Blender-style .001, .002, ...)."""
    if base not in existing_names:
        return base
    # Find next free suffix
    i = 1
    while True:
        nm = f"{base}.{i:03d}"
        if nm not in existing_names:
            return nm
        i += 1

def _parse_scene_file(path):
    entries = []

    def nonempty_lines(fp):
        for raw in fp:
            s = raw.strip()
            if s:
                yield s

    def parse_floats(line, n=None):
        parts = re.split(r"[,\s]+", line.strip())
        vals = [float(p) for p in parts]
        if n is not None and len(vals) != n:
            raise ValueError(f"Expected {n} floats, got {len(vals)}\nLine: {line}")
        return vals

    with open(path, "r", encoding="utf-8") as f:
        it = iter(nonempty_lines(f))
        # optional header like "(noname)+"
        try:
            first = next(it)
        except StopIteration:
            return entries
        if first.startswith("* "):
            it = iter([first] + list(it))

        for line in it:
            if not line.startswith("* "):
                continue
            name = line[2:].strip() or "cube"
            pos  = parse_floats(next(it), 3)
            quat = parse_floats(next(it), 4)     # qx qy qz qw (xyzw)
            _    = next(it)                      # "1"
            shape= next(it).strip().lower()      # "box"
            size = parse_floats(next(it), 3)
            _ = next(it)                         # "0 0 0"
            _ = next(it)                         # "0 0 0 1"
            _ = next(it)                         # "0 0 0 0"
            _ = next(it)                         # "0"
            entries.append({
                "name": name, "pos": pos, "quat_xyzw": quat, "shape": shape, "size": size
            })
    return entries

def _spawn_cube_unique(base_name, pos, quat_xyzw, size, normalize_quat=True, shade_flat=True):
    """Create a brand-new cube with a unique object and mesh name."""
    # Reserve a unique object name up front
    existing_obj_names = {o.name for o in bpy.data.objects}
    obj_name = _unique_name(base_name, existing_obj_names)

    # Create a fresh cube (new object + new mesh datablock)
    bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
    obj = bpy.context.active_object
    obj.name = obj_name

    # Ensure the mesh datablock also has a unique name
    existing_mesh_names = {m.name for m in bpy.data.meshes}
    mesh_name = _unique_name(obj_name + "_Mesh", existing_mesh_names)
    obj.data.name = mesh_name

    # Orientation: file is (x,y,z,w) -> Blender wants (w,x,y,z)
    qx, qy, qz, qw = quat_xyzw
    q = Quaternion((qw, qx, qy, qz))
    if normalize_quat:
        # Blender 4.x: no q.length — normalize safely
        try:
            q.normalize()  # in-place; raises ZeroDivisionError if zero
        except Exception:
            # fallback: manual normalization with small epsilon guard
            n = (q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z) ** 0.5
            if n > 1e-12:
                q = Quaternion((q.w/n, q.x/n, q.y/n, q.z/n))
            # else leave as-is (will likely be identity anyway)

    obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = q
    obj.location = Vector(pos)

    # Dimensions: set in local space; reset scale first to avoid compounding
    sx, sy, sz = size
    obj.scale = (1.0, 1.0, 1.0)
    obj.dimensions = (sx, sy, sz)

    if shade_flat:
        try:
            bpy.ops.object.shade_flat()
        except Exception:
            pass

    return obj


# -------------------- properties --------------------

class MoveItSceneProps(PropertyGroup):
    scene_path: StringProperty(
        name="Scene File",
        description="MoveIt .scene/.txt file",
        subtype='FILE_PATH',
        default=""
    )
    make_new_collection: BoolProperty(
        name="New collection per import",
        default=True
    )
    collection_name: StringProperty(
        name="Collection",
        default="MoveIt_Import"
    )
    normalize_quat: BoolProperty(
        name="Normalize quaternions",
        default=True
    )
    shade_flat: BoolProperty(
        name="Shade flat",
        default=True
    )

# -------------------- operators --------------------

class MOVEIT_SCENE_OT_Browse(Operator):
    bl_idname = "moveit_scene.browse"
    bl_label = "Browse…"
    filepath: StringProperty(subtype="FILE_PATH")
    filter_glob: StringProperty(default="*.scene;*.txt", options={'HIDDEN'})
    def execute(self, context):
        context.scene.moveit_scene_props.scene_path = self.filepath
        return {'FINISHED'}
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

class MOVEIT_SCENE_OT_Import(Operator):
    bl_idname = "moveit_scene.import"
    bl_label = "Import .scene"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.moveit_scene_props
        path = bpy.path.abspath(props.scene_path)
        if not path:
            self.report({'ERROR'}, "Select a .scene/.txt file first.")
            return {'CANCELLED'}

        try:
            entries = _parse_scene_file(path)
        except Exception as e:
            self.report({'ERROR'}, f"Read failed: {e}")
            return {'CANCELLED'}
        if not entries:
            self.report({'WARNING'}, "No entries found.")
            return {'CANCELLED'}

        # target collection
        if props.make_new_collection:
            base = bpy.path.display_name_from_filepath(path)
            coll = _ensure_collection(f"MoveIt_Import_{base}")
        else:
            coll = _ensure_collection(props.collection_name)

        created = []
        for e in entries:
            if e.get("shape", "box") != "box":
                continue

            obj = _spawn_cube_unique(
                base_name=e["name"],
                pos=e["pos"],
                quat_xyzw=e["quat_xyzw"],
                size=e["size"],
                normalize_quat=props.normalize_quat,
                shade_flat=props.shade_flat,
            )

            # move to target collection
            for c in list(obj.users_collection):
                c.objects.unlink(obj)
            coll.objects.link(obj)
            created.append(obj)

        # focus viewport on created
        for o in context.selected_objects:
            o.select_set(False)
        for o in created:
            o.select_set(True)
        if created:
            context.view_layer.objects.active = created[0]
            try:
                bpy.ops.view3d.view_selected(use_all_regions=False)
            except Exception:
                pass

        self.report({'INFO'}, f"Imported {len(created)} boxes into '{coll.name}'.")
        return {'FINISHED'}

# -------------------- panel --------------------

class MOVEIT_SCENE_PT_Main(Panel):
    bl_label = "Load & Spawn"
    bl_idname = "MOVEIT_SCENE_PT_Main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "MoveIt Scene"

    def draw(self, context):
        layout = self.layout
        props = context.scene.moveit_scene_props

        box = layout.box()
        row = box.row(align=True)
        row.prop(props, "scene_path", text="")
        row.operator("moveit_scene.browse", text="Browse…", icon='FILE_FOLDER')

        box.separator()
        box.prop(props, "normalize_quat")
        box.prop(props, "shade_flat")
        box.prop(props, "make_new_collection")
        if not props.make_new_collection:
            box.prop(props, "collection_name")

        box.separator()
        box.operator("moveit_scene.import", icon='MESH_CUBE')

# -------------------- registration --------------------

classes = (
    MoveItSceneProps,
    MOVEIT_SCENE_OT_Browse,
    MOVEIT_SCENE_OT_Import,
    MOVEIT_SCENE_PT_Main,
)

def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.moveit_scene_props = bpy.props.PointerProperty(type=MoveItSceneProps)

def unregister():
    del bpy.types.Scene.moveit_scene_props
    for c in reversed(classes):
        bpy.utils.unregister_class(c)

if __name__ == "__main__":
    register()
