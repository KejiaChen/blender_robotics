import bpy, os
from mathutils import Matrix

# Output folder next to your .blend
out_dir = bpy.path.abspath("//meshes/")
os.makedirs(out_dir, exist_ok=True)

# Selected meshes only
meshes = [o for o in bpy.context.selected_objects if o.type == "MESH"]
if not meshes:
    raise RuntimeError("Select one or more MESH objects and run again.")

# Remember selection to restore later
prev_active = bpy.context.view_layer.objects.active
prev_sel = list(bpy.context.selected_objects)

count = 0
for obj in meshes:
    # Make a temp duplicate so we can zero its world transform
    dup = obj.copy()
    dup.data = obj.data  # share data; exporter evaluates without modifying
    bpy.context.collection.objects.link(dup)
    dup.matrix_world = Matrix.Identity(4)

    # Select only the duplicate
    bpy.ops.object.select_all(action='DESELECT')
    dup.select_set(True)
    bpy.context.view_layer.objects.active = dup

    # File path
    safe = bpy.path.clean_name(obj.name)
    path = os.path.join(out_dir, f"{safe}.stl")

    # New Blender 4.1+ STL exporter
    bpy.ops.wm.stl_export(
        filepath=path,
        export_selected_objects=True,  # like "Selection Only"
        ascii_format=False,            # binary STL
        apply_modifiers=True,
        use_scene_unit=True            # keep meters if your scene uses Metric, Unit Scale 1.0
    )
    count += 1
    print("Exported", path)

    # Clean up the temporary object
    bpy.data.objects.remove(dup, do_unlink=True)

# Restore selection/active
bpy.ops.object.select_all(action='DESELECT')
for o in prev_sel:
    o.select_set(True)
bpy.context.view_layer.objects.active = prev_active

print(f"Done. Exported {count} STL file(s) to {out_dir}")
