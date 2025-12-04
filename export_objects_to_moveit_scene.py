# .scene exporter (no axis mapping) — uses Blender world frame as-is.
# Format per your example:
# (noname)+
# * <name>
# <px> <py> <pz>
# <qx> <qy> <qz> <qw>
# 1
# mesh
# <sx> <sy> <sz>
# 0 0 0
# 0 0 0 1
# 0 0 0 0
# 0
# .

import bpy, os

OUTPUT_PATH = bpy.path.abspath("//scene.scene")  # writes next to your .blend
TYPE_LINE = "box"                               # keep as "mesh" (change to "box" if needed)
SANITIZE_NAMES = True                            # replace spaces with underscores

# Collect selected mesh objects
objs = [o for o in bpy.context.selected_objects if o.type == "MESH"]
if not objs:
    raise RuntimeError("No MESH objects selected. Select your objects and run again.")
objs.sort(key=lambda o: o.name.lower())

lines = []
lines.append("(noname)+")  # file header

for o in objs:
    name = o.name.strip()
    if SANITIZE_NAMES:
        name = name.replace(" ", "_")

    # World-space translation and quaternion (Blender frame)
    t = o.matrix_world.to_translation()
    q = o.matrix_world.to_quaternion()  # Blender stores as (w,x,y,z)

    # Dimensions in meters (axis-aligned extents)
    dims = o.dimensions

    # Write block: position (x y z), orientation (x y z w)
    lines.append(f"* {name}")
    lines.append(f"{t.x:.6f} {t.y:.6f} {t.z:.6f}")
    lines.append(f"{q.x:.6f} {q.y:.6f} {q.z:.6f} {q.w:.6f}")
    lines.append("1")
    lines.append(TYPE_LINE)
    lines.append(f"{dims.x:.6f} {dims.y:.6f} {dims.z:.6f}")
    lines.append("0 0 0")
    lines.append("0 0 0 1")
    lines.append("0 0 0 0")
    lines.append("0")

lines.append(".")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"Wrote {OUTPUT_PATH} with {len(objs)} object(s) in BLENDER WORLD frame (no mapping).")
