import bpy
import json
from pathlib import Path


ROOT = Path(r"C:\Users\htsy4\Documents\ChatGPT\wolrdgen\artifacts\scenegen_zen_room_test")
GLB = ROOT / "zen_room_combined.glb"
BLEND = ROOT / "zen_room_scenegen.blend"
REPORT = ROOT / "blender_import_report.json"

# Remove the default scene before importing the generated GLB.
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
for datablocks in (bpy.data.meshes, bpy.data.curves, bpy.data.materials, bpy.data.images):
    for block in list(datablocks):
        if block.users == 0:
            datablocks.remove(block)

bpy.ops.import_scene.gltf(filepath=str(GLB), import_pack_images=True)

name_map = {
    "geometry_0": "SceneGen_Lamp",
    "geometry_1": "SceneGen_Bench",
    "geometry_2": "SceneGen_Plant",
    "geometry_3": "SceneGen_Rug",
}

imported = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
for obj in imported:
    original = obj.name
    mapped = next((label for key, label in name_map.items() if key in original), None)
    if mapped:
        obj.name = mapped
        obj.data.name = f"{mapped}_Mesh"

bpy.context.scene.unit_settings.system = "METRIC"
bpy.context.scene.unit_settings.length_unit = "METERS"

# Keep all transforms and embedded textures, then frame the complete generated scene.
bpy.ops.object.select_all(action="DESELECT")
for obj in imported:
    obj.select_set(True)
if imported:
    bpy.context.view_layer.objects.active = imported[0]

for area in bpy.context.screen.areas:
    if area.type == "VIEW_3D":
        area.spaces.active.shading.type = "MATERIAL"
        region = next((r for r in area.regions if r.type == "WINDOW"), None)
        if region:
            with bpy.context.temp_override(area=area, region=region):
                bpy.ops.view3d.view_selected(use_all_regions=False)

bpy.ops.wm.save_as_mainfile(filepath=str(BLEND))

report = {
    "source_glb": str(GLB),
    "blend_file": str(BLEND),
    "mesh_count": len(imported),
    "objects": [],
}
for obj in imported:
    materials = []
    for slot in obj.material_slots:
        mat = slot.material
        images = []
        if mat and mat.use_nodes:
            for node in mat.node_tree.nodes:
                if node.type == "TEX_IMAGE" and node.image:
                    images.append(node.image.name)
        materials.append({"name": mat.name if mat else None, "images": images})
    report["objects"].append(
        {
            "name": obj.name,
            "location": list(obj.location),
            "rotation_euler": list(obj.rotation_euler),
            "scale": list(obj.scale),
            "vertices": len(obj.data.vertices),
            "polygons": len(obj.data.polygons),
            "materials": materials,
        }
    )

REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
print("SCENEGEN_IMPORT_COMPLETE", json.dumps(report))
