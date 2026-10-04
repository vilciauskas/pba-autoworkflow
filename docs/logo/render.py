"""Cycles render of the idealised Na-Fe-Fe hexacyanoferrate cell (transparent PNG).

usage: python logo/render.py OUT.png RES SAMPLES POLY(0|1) [VIEW]
"""

import json
import math
import os
import sys

import bpy
from mathutils import Vector

out, res, samples, poly = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] == "1"
view = tuple(float(v) for v in (sys.argv[5].split(",") if len(sys.argv) > 5 else "1.0,-0.62,0.48".split(",")))

D = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "structure.json")))
atoms, bonds = D["atoms"], D["bonds"]
C0 = Vector(D["centre"])

import os

RADII = {"Fe_C": 0.46, "Fe_N": 0.46, "C": 0.27, "N": 0.27, "O_coord": 0.40,
         "O_zeo": 0.40, "H": 0.23, "Na": 0.74}
# role: (RGB, roughness, metallic).  Saturated base colours; metals carry the hue
# in their reflections, so they read as coloured chrome rather than plastic.
PALETTES = {
    "prussian_gold": {
        "Fe_C": ((0.015, 0.09, 0.62), 0.16, 0.90),   # [Fe(CN)6] iron: Prussian blue metal
        "Fe_N": ((1.00, 0.66, 0.12), 0.24, 1.00),    # N-bonded iron: polished gold
        "C":    ((0.30, 0.31, 0.34), 0.28, 0.55),    # gunmetal
        "N":    ((0.05, 0.42, 1.00), 0.12, 0.10),    # azure gloss
        "O":    ((1.00, 0.06, 0.04), 0.12, 0.00),
        "H":    ((1.00, 1.00, 1.00), 0.10, 0.00),
        "Na":   ((0.62, 0.10, 1.00), 0.10, 0.25),    # vivid violet
        "octa": (0.02, 0.18, 0.95),
    },
    "copper_teal": {
        "Fe_C": ((0.00, 0.42, 0.48), 0.15, 0.90),
        "Fe_N": ((0.98, 0.45, 0.22), 0.24, 1.00),    # copper
        "C":    ((0.30, 0.31, 0.34), 0.28, 0.55),
        "N":    ((0.00, 0.75, 0.85), 0.12, 0.10),
        "O":    ((1.00, 0.08, 0.10), 0.12, 0.00),
        "H":    ((1.00, 1.00, 1.00), 0.10, 0.00),
        "Na":   ((1.00, 0.78, 0.05), 0.10, 0.30),    # yellow Na+ for contrast with teal
        "octa": (0.00, 0.55, 0.60),
    },
    "chrome_cobalt": {
        "Fe_C": ((0.05, 0.20, 0.95), 0.12, 1.00),
        "Fe_N": ((0.92, 0.93, 0.96), 0.16, 1.00),    # chrome
        "C":    ((0.30, 0.31, 0.34), 0.28, 0.55),
        "N":    ((0.25, 0.55, 1.00), 0.12, 0.10),
        "O":    ((1.00, 0.20, 0.05), 0.12, 0.00),
        "H":    ((1.00, 1.00, 1.00), 0.10, 0.00),
        "Na":   ((1.00, 0.12, 0.55), 0.10, 0.25),    # magenta Na+
        "octa": (0.05, 0.25, 1.00),
    },
}
PAL = PALETTES[os.environ.get("PALETTE", "prussian_gold")]
STYLE = {r: (RADII[r], *PAL["O" if r.startswith("O") else r]) for r in RADII}
BOND_R = {"covalent": 0.095, "coord": 0.055}

bpy.ops.wm.read_factory_settings(use_empty=True)
scn = bpy.context.scene
scn.render.engine = "CYCLES"
scn.cycles.device = "CPU"
scn.cycles.samples = samples
scn.cycles.use_denoising = True
scn.render.film_transparent = True
scn.render.resolution_x = scn.render.resolution_y = res
scn.render.image_settings.file_format = "PNG"
scn.render.image_settings.color_mode = "RGBA"
scn.view_settings.view_transform = "AgX"
try:                                   # AgX's default look desaturates; Punchy restores colour
    scn.view_settings.look = "AgX - Punchy"
except TypeError:
    scn.view_settings.view_transform = "Standard"


def material(name, rgb, rough, metal, alpha=1.0, transmission=0.0, coat=0.6, spec=0.6):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1.0)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    b.inputs["Coat Weight"].default_value = coat
    b.inputs["Coat Roughness"].default_value = 0.04
    b.inputs["Specular IOR Level"].default_value = spec
    if alpha < 1.0:
        b.inputs["Alpha"].default_value = alpha
    if transmission:
        b.inputs["Transmission Weight"].default_value = transmission
        b.inputs["IOR"].default_value = 1.35
    return m


MATS = {r: material(r, s[1], s[2], s[3]) for r, s in STYLE.items()}

# one template mesh per radius, instanced as linked duplicates (keeps the scene light)
bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=1.0)
sphere = bpy.context.active_object.data
bpy.ops.object.shade_smooth()
bpy.data.objects.remove(bpy.context.active_object)
bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=1.0, depth=1.0)
cyl = bpy.context.active_object.data
bpy.ops.object.shade_smooth()
bpy.data.objects.remove(bpy.context.active_object)

col = bpy.data.collections.new("crystal"); scn.collection.children.link(col)


# one sphere mesh and one cylinder mesh per role, each carrying its material
SPH, CYL = {}, {}
for r, m in MATS.items():
    SPH[r] = sphere.copy(); SPH[r].materials.append(m)
    CYL[r] = cyl.copy(); CYL[r].materials.append(m)


def place(mesh, name, loc, scale, rot=None):
    o = bpy.data.objects.new(name, mesh)
    o.location = loc; o.scale = scale
    if rot is not None:
        o.rotation_mode = "QUATERNION"; o.rotation_quaternion = rot
    col.objects.link(o)
    return o


for k, a in enumerate(atoms):
    r = STYLE[a["role"]][0]
    place(SPH[a["role"]], f"{a['role']}{k}", Vector(a["xyz"]) - C0, (r, r, r))

for i, j, kind in bonds:
    p, q = Vector(atoms[i]["xyz"]) - C0, Vector(atoms[j]["xyz"]) - C0
    rb = BOND_R[kind]
    for (start, end, role) in ((p, (p + q) / 2, atoms[i]["role"]), ((p + q) / 2, q, atoms[j]["role"])):
        d = end - start
        rot = Vector((0, 0, 1)).rotation_difference(d.normalized())
        place(CYL[role], "bond", (start + end) / 2, (rb, rb, d.length), rot)

if poly:  # translucent [Fe(CN)6] octahedra on complete units
    pm = material("octa", PAL["octa"], 0.35, 0.0, alpha=0.42, coat=0.0, spec=0.15)
    cs = [Vector(a["xyz"]) - C0 for a in atoms if a["role"] == "C"]
    for a in atoms:
        if a["role"] != "Fe_C":
            continue
        f = Vector(a["xyz"]) - C0
        v = [c for c in cs if (c - f).length < 2.1]
        if len(v) != 6:
            continue
        me = bpy.data.meshes.new("octa")
        faces = []
        idx = {tuple(round(x, 3) for x in (c - f).normalized()): n for n, c in enumerate(v)}
        dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        for sx in (1, -1):
            for sy in (1, -1):
                for sz in (1, -1):
                    faces.append((idx[(sx, 0, 0)], idx[(0, sy, 0)], idx[(0, 0, sz)]))
        me.from_pydata([tuple(c) for c in v], [], faces)
        me.materials.append(pm)
        o = bpy.data.objects.new("octa", me); col.objects.link(o)

# camera: perspective, aimed at the cell centre along `view`
cam = bpy.data.cameras.new("cam"); cam.lens = 85
camo = bpy.data.objects.new("cam", cam); scn.collection.objects.link(camo); scn.camera = camo
vd = Vector(view).normalized()
camo.location = vd * 62.0
camo.rotation_mode = "QUATERNION"
camo.rotation_quaternion = (-vd).to_track_quat("-Z", "Y")
# fit: the cell (plus overhang) must fill ~88 % of the frame
cam.sensor_fit = "VERTICAL"; cam.sensor_height = 24.0
extent = max(max(abs((Vector(a["xyz"]) - C0).dot(ax)) for a in atoms)
             for ax in (camo.matrix_world.to_3x3() @ Vector((1, 0, 0)),
                        camo.matrix_world.to_3x3() @ Vector((0, 1, 0)))) + 0.8
cam.lens = cam.sensor_height / 2 * 62.0 / (extent / 0.84)

# lighting: large soft key, cool fill, warm rim; dim neutral world
# studio gradient environment: bright overhead, dark floor.  Metals reflect their
# surroundings, so a flat grey world is what made them look dull; the gradient
# gives every sphere a crisp horizon line in its reflection.
world = bpy.data.worlds.new("w"); scn.world = world; world.use_nodes = True
nt = world.node_tree; N = nt.nodes
tc = N.new("ShaderNodeTexCoord"); sep = N.new("ShaderNodeSeparateXYZ")
ramp = N.new("ShaderNodeValToRGB"); bgn = N["Background"]
nt.links.new(tc.outputs["Generated"], sep.inputs[0])
nt.links.new(sep.outputs["Z"], ramp.inputs["Fac"])
nt.links.new(ramp.outputs["Color"], bgn.inputs["Color"])
ramp.color_ramp.elements[0].position = 0.30; ramp.color_ramp.elements[0].color = (0.16, 0.18, 0.22, 1)
ramp.color_ramp.elements[1].position = 0.66; ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1)
bgn.inputs["Strength"].default_value = 0.9


def light(name, loc, energy, size, rgb):
    L = bpy.data.lights.new(name, "AREA"); L.energy = energy; L.size = size; L.color = rgb
    o = bpy.data.objects.new(name, L); o.location = loc
    o.rotation_mode = "QUATERNION"; o.rotation_quaternion = (-Vector(loc)).to_track_quat("-Z", "Y")
    scn.collection.objects.link(o)


up = Vector((0, 0, 1)); side = vd.cross(up).normalized()
light("key", (vd * 30 + side * 22 + up * 26), 9000, 14, (1.0, 0.97, 0.92))
light("fill", (vd * 30 - side * 26 + up * 4), 2600, 18, (0.85, 0.90, 1.0))
light("rim", (-vd * 28 + up * 18), 5000, 10, (1.0, 0.92, 0.85))

scn.render.filepath = out
if os.environ.get("SAVE_BLEND"):          # editable scene: open in Blender, tweak, F12
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(os.environ["SAVE_BLEND"]))
if not os.environ.get("NO_RENDER"):
    bpy.ops.render.render(write_still=True)
print("rendered", out, "objects:", len(col.objects))
