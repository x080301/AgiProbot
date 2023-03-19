import numpy as np
import open3d as o3d

pcd = o3d.data.DemoICPPointClouds()
src = o3d.io.read_point_cloud('C:/Users/Lenovo/Desktop/MotorClean.plz_plus1.pcd')  # 006_labelled_color.pcd')
tar = o3d.io.read_point_cloud('C:/Users/Lenovo/Desktop/MotorClean.plz_color.pcd')
# src = tar


pipreg = o3d.pipelines.registration

reg = pipreg.registration_icp(src, tar, max_correspondence_distance=0.01, estimation_method=
# pipreg.TransformationEstimationPointToPlane())
# pipreg.TransformationEstimationForGeneralizedICP())
# pipreg.TransformationEstimationForColoredICP())
pipreg.TransformationEstimationPointToPoint())

print(reg.transformation)

print(reg)

import copy

'''srcDraw = copy.deepcopy(src)
tarDraw = copy.deepcopy(tar)
srcDraw.paint_uniform_color([1, 1, 0])
tarDraw.paint_uniform_color([0, 1, 1])
o3d.visualization.draw_geometries([srcDraw, tarDraw])'''

tarDraw = copy.deepcopy(tar)
srcDraw = copy.deepcopy(src)
srcDraw.paint_uniform_color([1, 1, 0])
tarDraw.paint_uniform_color([0, 1, 1])
srcDraw.transform(reg.transformation)
o3d.visualization.draw_geometries([srcDraw, tarDraw])
