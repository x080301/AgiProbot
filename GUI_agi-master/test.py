import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
pcd = o3d.io.read_point_cloud("/home/bi/study/thesis/pyqt/cover.pcd")
downpcd = pcd.voxel_down_sample(voxel_size=0.002)  # 下采样滤波，体素边长为0.002m
downpcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))  # 计算法线，只考虑邻域内的20个点
# downpcd.estimate_normals(
#     search_param=o3d.geometry.KDTreeSearchParamRadius(radius=0.01))  # 计算法线，搜索半径1cm，只考虑邻域内的20个点
nor=downpcd.normals
normal=[]
for ele in nor:
    normal.append(ele)
normal=np.array(normal)
model = DBSCAN(eps=0.02, min_samples=100)
yhat = model.fit_predict(normal)  # genalize label based on index
clusters = np.unique(yhat)
noise = []
clusters_new = []
positions = []
for i in clusters:
    noise.append(i) if np.sum(i == yhat) < 300 or i == -1 else clusters_new.append(i)
flag=0
bolts__=1
for clu in clusters_new :
    row_ix = np.where(yhat == clu)
    normal = np.squeeze(np.mean(normal[row_ix, :3], axis=1))
    #positions.append(position)

o3d.visualization.draw_geometries([downpcd], "Open3D normal estimation", width=800, height=600, left=50, top=50,
                                  point_show_normal=True, mesh_show_wireframe=False,
                                  mesh_show_back_face=False)  # 可视化法线