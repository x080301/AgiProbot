import numpy as np
import open3d as o3d
from para_extracter.para_extracter import ParaExtracter
from para_extracter.utilities.utilities import calculate_mIoU
import copy


def visualization_pointcloud_noemal():
    pcd = o3d.io.read_point_cloud('data/cover.pcd')  # ("/home/bi/study/thesis/pyqt/cover.pcd")
    downpcd = pcd.voxel_down_sample(voxel_size=0.002)  # 下采样滤波，体素边长为0.002m
    downpcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))  # 计算法线，只考虑邻域内的20个点
    '''# downpcd.estimate_normals(
    #     search_param=o3d.geometry.KDTreeSearchParamRadius(radius=0.01))  # 计算法线，搜索半径1cm，只考虑邻域内的20个点
    nor = downpcd.normals
    normal = []
    for ele in nor:
        normal.append(ele)
    normal = np.array(normal)
    model = DBSCAN(eps=0.02, min_samples=100)
    yhat = model.fit_predict(normal)  # genalize label based on index
    clusters = np.unique(yhat)
    noise = []
    clusters_new = []
    for i in clusters:
        noise.append(i) if np.sum(i == yhat) < 300 or i == -1 else clusters_new.append(i)
    for clu in clusters_new:
        row_ix = np.where(yhat == clu)
        normal = np.squeeze(np.mean(normal[row_ix, :3], axis=1))
        # positions.append(position)'''

    o3d.visualization.draw_geometries([downpcd], "Open3D normal estimation", width=800, height=600, left=50, top=50,
                                      point_show_normal=True, mesh_show_wireframe=False,
                                      mesh_show_back_face=False)  # 可视化法线


def test_load_data():
    extracter = ParaExtracter()
    print('loading')
    point_cloud, num_points = extracter.load_pcd_data('D:/Jupyter/AgiProbot/GUI_agi-master/pcdfile/A1_13_screws.pcd')

    point_cloud = point_cloud[:, 0: 3]

    print('{num} points in point cloud are loaded'.format(num=num_points))

    print('The type of the point cloud data is {pcd_type}'.format(pcd_type=type(point_cloud)))
    print('The shape of the point cloud data is {pcd_shape}'.format(pcd_shape=point_cloud.shape))
    print('visualizing')

    point_cloud_o3d = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utility.Vector3dVector(point_cloud)
    o3d.visualization.draw_geometries([point_cloud_o3d])


def test_load_pretrained_model(expected_return):
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


def pipline_example():
    # ***************************************
    # pipline example
    # ***************************************

    # Define the object and provide the necessary information
    extracter = ParaExtracter()
    extracter.load_model()
    extracter.load_pcd_data('D:/Jupyter/AgiProbot/GUI_agi-master/pcdfile/A1_13_screws.pcd')

    # Run the model
    extracter.run()

    # get what you need
    segementation_prediction = extracter.get_segmentation_prediction()
    classification_prediction = extracter.get_classification_prediction()

    bolt_positions, bolt_normal, bolt_num, bolt_piont_clouds = extracter.find_bolts()


if __name__ == '__main__':
    test_load_data()
    pass
