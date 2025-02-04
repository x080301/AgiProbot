# import open3d as o3d
import bpy

# ******************* #
# This blender script can be run by python file: generate_syn_dataset_with_blender.py
# It read existing motor model from a .blend file and generate .obj file without label.
# ******************* #

# ******************* #
# delete needless objects
# ******************* #
delete_list = ['Camera', 'Empty', '0000_Plane', '0000_ClampingSystem_fix', '0000_ClampingSystem_movable']
collection = bpy.data.collections['Collection 1']
for object_name in delete_list:
    obj = collection.objects[object_name]
    bpy.data.objects.remove(obj)

# ******************* #
#  join the remaining objects
# ******************* #
has_active_object = False
for obj in collection.objects:
    bpy.data.objects[obj.name].select_set(True)
    if not has_active_object:
        bpy.context.view_layer.objects.active = bpy.data.objects[obj.name]
        has_active_object = True
        joined_object_name = obj.name
    # print(obj.name)
bpy.ops.object.join()

# ******************* #
# remesh the model, so that overlapping boundaries of objects will not appear in the final result.
# ******************* #
object = bpy.data.objects[joined_object_name]

modifier = object.modifiers.new(name="Remesh", type='REMESH')
modifier.mode = 'VOXEL'
modifier.voxel_size = 0.01
bpy.ops.object.modifier_apply(modifier="Remesh")

# ******************* #
# resave as .obj
# ******************* #
bpy.ops.export_scene.obj(filepath='Intermediate_results/intermediate_obj.obj', use_triangles=True)
