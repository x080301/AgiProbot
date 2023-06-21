# import open3d as o3d
# import os
import bpy

# 将blender读取为obj数据形式
#   读取blender文件

# ******************* #
#   删除背景部分
# ******************* #
delete_list = ['Camera', 'Empty', '0000_Plane', '0000_ClampingSystem_fix', '0000_ClampingSystem_movable']
collection = bpy.data.collections['Collection 1']
for object_name in delete_list:
    obj = collection.objects[object_name]
    bpy.data.objects.remove(obj)

# ******************* #
#   合并剩余模型
# ******************* #
has_active_object = False
for obj in collection.objects:
    bpy.data.objects[obj.name].select_set(True)
    if not has_active_object:
        bpy.context.view_layer.objects.active = bpy.data.objects[obj.name]
        has_active_object = True
        joined_object_name = obj.name
    #print(obj.name)
bpy.ops.object.join()

# ******************* #
#   重构网格
# ******************* #
object = bpy.data.objects[joined_object_name]

modifier = object.modifiers.new(name="Remesh", type='REMESH')
modifier.mode = 'VOXEL'
modifier.voxel_size = 0.01
bpy.ops.object.modifier_apply(modifier="Remesh")

# ******************* #
#   保存为.obj
# ******************* #
bpy.ops.export_scene.obj(filepath='C:/Users/Lenovo/Desktop/test.obj', use_triangles=True)

blender_file_dir = 'E:/datasets/agiprobot/agi_large_motor_dataset_syn/'

'''flag = False
for file_dir_batch in os.listdir(blender_file_dir):
    for file_dir in os.listdir(blender_file_dir + file_dir_batch):
        for file_name in os.listdir(blender_file_dir + file_dir_batch + '/' + file_dir):
            if 'Szene.blend' not in file_name:
                continue
            else:
                print(file_name)
                bpy.ops.import_scene.fbx(filepath=blender_file_dir + file_dir_batch + '/' + file_dir+'/'+file_name)


'''
r'''
mesh = o3d.io.read_triangle_mesh(r'C:\Users\Lenovo\Desktop\0Szene.obj')
# pcd = mesh.sample_points_uniformly(number_of_points=10000)
# print(pcd.points)
o3d.visualization.draw_geometries([mesh], window_name=".obj")

'''
