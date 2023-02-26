import numpy as np
import open3d as o3d
from para_extracter import ParaExtracter
import copy

import matplotlib.pyplot as plt


def test_load_data():
    extracter = ParaExtracter()
    print('loading')
    point_cloud, num_points = extracter.load_pcd_data('D:/Jupyter/AgiProbot/GUI_agi-master/pcdfile/B1_17_gear.pcd')

    point_cloud = point_cloud[:, 0: 3]

    print('{num} points in point cloud are loaded'.format(num=num_points))

    print('The type of the point cloud data is {pcd_type}'.format(pcd_type=type(point_cloud)))
    print('The shape of the point cloud data is {pcd_shape}'.format(pcd_shape=point_cloud.shape))
    print('visualizing')

    point_cloud_o3d = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utility.Vector3dVector(point_cloud)
    o3d.visualization.draw_geometries([point_cloud_o3d])


def test_prediction():
    # Define the object and provide the necessary information
    extracter = ParaExtracter()
    extracter.load_model()
    extracter.load_pcd_data(
        'D:/Jupyter/AgiProbot/GUI_agi-master/pcdfile/A21_9_gear.pcd')  # ('D:/Jupyter/AgiProbot/GUI_agi-master/pcdfile/A1_13_screws.pcd')

    # Run the model
    extracter.run()

    # get what you need
    classification_prediction = extracter.get_classification_prediction()
    print(classification_prediction)

    segementation_prediction = extracter.get_segmentation_prediction()
    segementation_prediction = segementation_prediction[:, 3].flatten()
    print(np.histogram(segementation_prediction, bins=10, range=(0, 10)))
    plt.hist(segementation_prediction, bins=10, range=(0, 10))
    plt.show()


def test_load_pretrained_model(expected_return=True):
    extracter = ParaExtracter()
    init_model = copy.deepcopy(extracter.model)

    print('loading pretrained model')
    if expected_return is True:
        extracter.load_model()

    loaded_model = copy.deepcopy(extracter.model)

    for para_name in list(init_model.state_dict()):
        if (init_model.state_dict()[para_name] != loaded_model.state_dict()[para_name]).sum() != 0:
            print('New model is loaded')
            return True

    print('Model is not changed.')
    return False


def test_find_screws_position(with_cover=True):
    # Define the object and provide the necessary information
    extracter = ParaExtracter()
    extracter.load_model()
    if with_cover:
        extracter.load_pcd_data('D:/Jupyter/AgiProbot/GUI_agi-master/pcdfile/A1_13_screws.pcd')
    else:
        extracter.load_pcd_data('D:/Jupyter/AgiProbot/GUI_agi-master/pcdfile/A1_12_gear.pcd')

    # Run the model
    extracter.run()

    bolt_positions, _, bolt_num, bolt_piont_clouds = extracter.find_screws()

    print(bolt_positions)
    print(bolt_num)
    print(bolt_piont_clouds)
    if bolt_piont_clouds is not None:
        print(bolt_piont_clouds.shape)


def test_find_screws_direction(with_cover=True):
    # Define the object and provide the necessary information
    extracter = ParaExtracter()
    extracter.load_model()
    if with_cover:
        extracter.load_pcd_data('D:/Jupyter/AgiProbot/GUI_agi-master/pcdfile/A1_13_screws.pcd')
    else:
        extracter.load_pcd_data('D:/Jupyter/AgiProbot/GUI_agi-master/pcdfile/A1_12_gear.pcd')

    # Run the model
    extracter.run()

    bolt_positions, cover_screw_normal, bolt_num, bolt_piont_clouds = extracter.find_screws()

    print('**************************')
    print(type(bolt_positions))
    print(type(cover_screw_normal))
    print(type(bolt_num))
    print(type(bolt_piont_clouds))

    print(bolt_positions)
    print(bolt_num)
    print(bolt_piont_clouds)

    print('\n*************************')
    if bolt_piont_clouds is not None:
        print(bolt_piont_clouds.shape)

        print(cover_screw_normal)

    print('\n*************************')


def test_find_gear_position(with_cover=True):
    # Define the object and provide the necessary information
    extracter = ParaExtracter()
    extracter.load_model()
    if with_cover:
        extracter.load_pcd_data('D:/Jupyter/AgiProbot/GUI_agi-master/pcdfile/A1_13_screws.pcd')
    else:
        extracter.load_pcd_data('D:/Jupyter/AgiProbot/GUI_agi-master/pcdfile/A1_12_gear.pcd')

    # Run the model
    extracter.run()

    gear, gearpositions = extracter.find_gears()

    print('\n*************************')

    print(gear)
    print(gearpositions)
    if with_cover is False:
        print(np.asarray(gear).shape)
        print(np.asarray(gearpositions).shape)

    print('\n*************************')


def test_if_cover_existence(with_cover=True):
    extracter = ParaExtracter()
    extracter.load_model()

    if with_cover:
        extracter.load_pcd_data('D:/Jupyter/AgiProbot/GUI_agi-master/pcdfile/A1_13_screws.pcd')
    else:
        extracter.load_pcd_data('D:/Jupyter/AgiProbot/GUI_agi-master/pcdfile/B1_17_gear.pcd')

    extracter.run()
    print(extracter.if_cover_existence())


def pipline_example():
    # ***************************************
    # pipline example
    # ***************************************
    # import package ParaExtracter
    from para_extracter import ParaExtracter

    # Define the object and provide necessary information
    extracter = ParaExtracter()
    extracter.load_model()
    extracter.load_pcd_data('D:/Jupyter/AgiProbot/GUI_agi-master/pcdfile/A1_13_screws.pcd')

    # Run the model
    extracter.run()

    # get what you need
    segementation_prediction = extracter.get_segmentation_prediction()
    classification_prediction = extracter.get_classification_prediction()

    if extracter.if_cover_existence():
        bolt_positions, cover_screw_normal, bolt_num, bolt_piont_clouds = extracter.find_screws()
    else:
        gear_piont_clouds, gear_positions = extracter.find_gears()

    # further data
    extracter.load_pcd_data('D:/Jupyter/AgiProbot/GUI_agi-master/pcdfile/B1_17_gear.pcd')
    extracter.run()
    segementation_prediction = extracter.get_segmentation_prediction()
    classification_prediction = extracter.get_classification_prediction()
    # ...


if __name__ == '__main__':
    # print(test_load_pretrained_model(expected_return=True))
    # print(test_load_pretrained_model(expected_return=False))
    #
    # test_load_data()
    # test_prediction()
    #
    # test_if_cover_existence(with_cover=True)
    # test_if_cover_existence(with_cover=False)
    #
    # test_find_screws_position(with_cover=True)
    # test_find_screws_position(with_cover=False)
    #
    test_find_screws_direction(with_cover=True)
    test_find_screws_direction(with_cover=False)
    #
    # test_find_gear_position(with_cover=True)
    # test_find_gear_position(with_cover=False)

    # pipline_example()

    pass
