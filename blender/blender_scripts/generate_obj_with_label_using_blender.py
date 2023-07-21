# import open3d as o3d
import bpy


def object_grouping(object_list):
    """
    This function was written based on observation of how these objects are named.

    :param object_list
    :return: object_dict: objects are grouped.
    """

    object_dict = {'Void': [], 'Background': [], 'Gear': [], 'Connector': [], 'Screws': [], 'Solenoid': [],
                   'Electrical Connector': [], 'Main Housing': [], 'Noise': [], 'Inner Gear': []
                   }

    # find Trainanle, gears, all bolts and max cylinder id
    new_list = []
    max_cylinder_id = 0
    for obj_name in object_list:
        if 'Bolt' in obj_name:
            object_dict['Screws'].append(obj_name)
        elif 'Triangle' in obj_name:
            object_dict['Connector'].append(obj_name)
        elif 'gear' in obj_name:
            object_dict['Gear'].append(obj_name)
        elif 'z_function' in obj_name:
            object_dict['Connector'].append(obj_name)
        else:
            new_list.append(obj_name)

        if 'Cylinder' in obj_name:
            cylinder_id = int(obj_name.split('_')[-1])
            if cylinder_id > max_cylinder_id:
                max_cylinder_id = cylinder_id

    object_list = new_list

    # Cylinders
    object_dict['Main Housing'].append('5555_Cylinder_' + str(max_cylinder_id))
    object_list.remove('5555_Cylinder_' + str(max_cylinder_id))
    object_dict['Solenoid'].append('5555_Cylinder_' + str(max_cylinder_id - 1))
    object_list.remove('5555_Cylinder_' + str(max_cylinder_id - 1))
    object_dict['Main Housing'].append('5555_Cylinder_' + str(max_cylinder_id - 2))
    object_list.remove('5555_Cylinder_' + str(max_cylinder_id - 2))
    object_dict['Gear'].append('5555_Cylinder_' + str(max_cylinder_id - 3))
    object_list.remove('5555_Cylinder_' + str(max_cylinder_id - 3))
    object_dict['Screws'].append('5555_Cylinder_' + str(max_cylinder_id - 5))
    object_list.remove('5555_Cylinder_' + str(max_cylinder_id - 5))
    object_dict['Screws'].append('5555_Cylinder_' + str(max_cylinder_id - 6))
    object_list.remove('5555_Cylinder_' + str(max_cylinder_id - 6))
    object_dict['Electrical Connector'].append('5555_Cylinder_' + str(max_cylinder_id - 7))
    object_list.remove('5555_Cylinder_' + str(max_cylinder_id - 7))

    new_list = []
    for obj_name in object_list:
        if 'Cylinder' in obj_name:
            object_dict['Connector'].append(obj_name)
        else:
            new_list.append(obj_name)
    object_list = new_list

    # rest
    object_dict['Electrical Connector'].append('3333_cube_')
    object_list.remove('3333_cube_')
    object_dict['Electrical Connector'].append('3333_cube_.002')
    object_list.remove('3333_cube_.002')
    object_dict['Connector'].append('3333_cube_.001')
    object_list.remove('3333_cube_.001')

    if len(object_list) > 0:
        print('unexpect object:\n')
        print(object_list)
        exit(-1)
    else:
        return object_dict


# ******************* #
# delete needless objects
# ******************* #
delete_list = ['Camera', 'Empty', '0000_Plane', '0000_ClampingSystem_fix', '0000_ClampingSystem_movable']
collection = bpy.data.collections['Collection 1']
for object_name in delete_list:
    obj = collection.objects[object_name]
    bpy.data.objects.remove(obj)

object_name_list = []
for obj in collection.objects:
    object_name_list.append(obj.name)

grouped_objects_dict = object_grouping(object_name_list)

for semantic_part in grouped_objects_dict.keys():
    if len(grouped_objects_dict[semantic_part]) > 0:

        # select objects in one semantic type
        for obj_name in grouped_objects_dict[semantic_part]:
            bpy.data.objects[obj_name].select_set(True)

        # export as an .obj file
        bpy.ops.export_scene.obj(filepath='Intermediate_results/intermediate_' + semantic_part + '_part.obj',
                                 use_triangles=True, use_selection=True)

        # deselect selected objects
        for obj_name in grouped_objects_dict[semantic_part]:
            bpy.data.objects[obj_name].select_set(False)
