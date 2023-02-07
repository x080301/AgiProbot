import os
import numpy as np
import open3d as o3d
import csv
import math
from sklearn.cluster import DBSCAN
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR



def base_to_camera(cam_to_base_transform, xyz, calc_angle=False):
    '''
    now do the base to camera transform
    '''

        # squeeze the first two dimensions
    xyz_transformed2 = xyz.reshape(-1, 3)  # [N=X*Y, 3]

        # homogeneous transformation
    if calc_angle:
        xyz_transformed2 = np.hstack((xyz_transformed2, np.zeros((xyz_transformed2.shape[0], 1))))  # [N, 4]
    else:
        xyz_transformed2 = np.hstack((xyz_transformed2, np.ones((xyz_transformed2.shape[0], 1))))  # [N, 4]

    cam_to_base_transform = np.matrix(cam_to_base_transform)
    base_to_cam_transform = cam_to_base_transform.I
    xyz_transformed2 = np.matmul(base_to_cam_transform, xyz_transformed2.T).T  # [N, 4]

    return xyz_transformed2[:, :-1].reshape(xyz.shape)  # [X, Y, 3]



def camera_to_base(xyz, calc_angle=False):
    '''
    '''
    cam_to_base_transform = [[-1.0721407e-01,-9.4186008e-01 ,3.1844112e-01,-2.3087662e+02],
                              [-9.6728820e-01 ,2.4749031e-02,-2.5246987e-01 ,1.1985071e+03],
                              [ 2.2991017e-01,-3.3509266e-01,-9.1370356e-01 ,7.4048785e+02],
                              [ 0.0000000e+00 ,0.0000000e+00 ,0.0000000e+00 ,1.0000000e+00]]
        # squeeze the first two dimensions
    xyz_transformed2 = xyz.reshape(-1, 3)  # [N=X*Y, 3]

        # homogeneous transformation
    if calc_angle:
        xyz_transformed2 = np.hstack((xyz_transformed2, np.zeros((xyz_transformed2.shape[0], 1))))  # [N, 4]
    else:
        xyz_transformed2 = np.hstack((xyz_transformed2, np.ones((xyz_transformed2.shape[0], 1))))  # [N, 4]


    xyz_transformed2 = np.matmul(cam_to_base_transform, xyz_transformed2.T).T  # [N, 4]

    return xyz_transformed2[:, :-1].reshape(xyz.shape)  # [X, Y, 3]



def get_Corordinate_inCam(cam_pos_x, cam_pos_y, cam_pos_z, alpha, beta, theta, cor):

    alpha = float(alpha)
    beta = float(beta)
    theta = float(theta)
    cor = np.array(cor).T
    cam_pos = np.array([float(cam_pos_x), float(cam_pos_y), float(cam_pos_z)]).T
    cor = cor - cam_pos

    c_mw = np.array([[math.cos(beta)*math.cos(theta), math.cos(beta)*math.sin(theta), -math.sin(beta)],
            [-math.cos(alpha)*math.sin(theta)+math.sin(alpha)*math.sin(beta)*math.cos(theta), math.cos(alpha)*math.cos(theta)+math.sin(alpha)*math.sin(beta)*math.sin(theta), math.sin(alpha)*math.cos(beta)],
            [math.sin(alpha)*math.sin(theta)+math.cos(alpha)*math.sin(beta)*math.cos(theta), -math.sin(alpha)*math.cos(theta)+math.cos(alpha)*math.sin(beta)*math.sin(theta), math.cos(alpha)*math.cos(beta)] ])

    cor_new = c_mw.dot(cor)

    return cor_new



def get_Corordinate_inBlensor_rw(cam_pos_x, cam_pos_y, cam_pos_z, alpha, beta, theta, cor_new):
    cor_inBlensor_Cam = cor_new
    alpha = float(alpha)
    beta = float(beta)
    theta = float(theta)
    cam_pos = np.array([float(cam_pos_x), float(cam_pos_y), float(cam_pos_z)]).T

    c_mw = np.array([[math.cos(beta)*math.cos(theta), math.cos(beta)*math.sin(theta), -math.sin(beta)],
            [-math.cos(alpha)*math.sin(theta)+math.sin(alpha)*math.sin(beta)*math.cos(theta), math.cos(alpha)*math.cos(theta)+math.sin(alpha)*math.sin(beta)*math.sin(theta), math.sin(alpha)*math.cos(beta)],
            [math.sin(alpha)*math.sin(theta)+math.cos(alpha)*math.sin(beta)*math.cos(theta), -math.sin(alpha)*math.cos(theta)+math.cos(alpha)*math.sin(beta)*math.sin(theta), math.cos(alpha)*math.cos(beta)] ])
    c_mw_I = np.linalg.inv(c_mw)
    cor = c_mw_I.dot(cor_inBlensor_Cam) + cam_pos
    return cor



def get_panel(point_1, point_2, point_3):

    x1 = point_1[0]
    y1 = point_1[1]
    z1 = point_1[2]

    x2 = point_2[0]
    y2 = point_2[1]
    z2 = point_2[2] 

    x3 = point_3[0]
    y3 = point_3[1]
    z3 = point_3[2]
    
    a = (y2-y1)*(z3-z1) - (y3-y1)*(z2-z1)
    b = (z2-z1)*(x3-x1) - (z3-z1)*(x2-x1)
    c = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
    d = 0 - (a*x1 + b*y1 + c*z1)

    return (a, b, c, d)



def set_Boundingbox(panel_list, point_cor):

    if panel_list['panel_up'][0]*point_cor[0] + panel_list['panel_up'][1]*point_cor[1] + panel_list['panel_up'][2]*point_cor[2] + panel_list['panel_up'][3] <= 0 :   # panel 1
        if panel_list['panel_bot'][0]*point_cor[0] + panel_list['panel_bot'][1]*point_cor[1] + panel_list['panel_bot'][2]*point_cor[2] + panel_list['panel_bot'][3] >= 0 : # panel 2
            if panel_list['panel_front'][0]*point_cor[0] + panel_list['panel_front'][1]*point_cor[1] + panel_list['panel_front'][2]*point_cor[2] + panel_list['panel_front'][3] <= 0 : # panel 3
                if panel_list['panel_behind'][0]*point_cor[0] + panel_list['panel_behind'][1]*point_cor[1] + panel_list['panel_behind'][2]*point_cor[2] + panel_list['panel_behind'][3] >= 0 : # panel 4
                    if panel_list['panel_right'][0]*point_cor[0] + panel_list['panel_right'][1]*point_cor[1] + panel_list['panel_right'][2]*point_cor[2] + panel_list['panel_right'][3] >= 0 : #panel 5
                        if panel_list['panel_left'][0]*point_cor[0] + panel_list['panel_left'][1]*point_cor[1] + panel_list['panel_left'][2]*point_cor[2] + panel_list['panel_left'][3] >= 0 : # panel 6

                            return True
    return False



def cut_motor(whole_scene):
    Corners = [(35,880,300), (35,1150,300), (-150,1150,300), (-150,880,300), (35,880,50), (35,1150,50), (-150,1150,50), (-150,880,50)]
    cam_to_robot_transform = [[-1.0721407e-01,-9.4186008e-01 ,3.1844112e-01,-2.3087662e+02],
                              [-9.6728820e-01 ,2.4749031e-02,-2.5246987e-01 ,1.1985071e+03],
                              [ 2.2991017e-01,-3.3509266e-01,-9.1370356e-01 ,7.4048785e+02],
                              [ 0.0000000e+00 ,0.0000000e+00 ,0.0000000e+00 ,1.0000000e+00]]
    cor_inCam = []
    for corner in Corners:
        cor_inCam_point = base_to_camera(cam_to_robot_transform, np.array(corner))
        cor_inCam.append(np.squeeze(np.array(cor_inCam_point)))

    panel_1 = get_panel(cor_inCam[0], cor_inCam[1], cor_inCam[2])
    panel_2 = get_panel(cor_inCam[5], cor_inCam[6], cor_inCam[4])
    panel_3 = get_panel(cor_inCam[0], cor_inCam[3], cor_inCam[4])
    panel_4 = get_panel(cor_inCam[1], cor_inCam[2], cor_inCam[5])
    panel_5 = get_panel(cor_inCam[0], cor_inCam[1], cor_inCam[4])
    panel_6 = get_panel(cor_inCam[2], cor_inCam[3], cor_inCam[6])
    panel_list = {'panel_up':panel_1, 'panel_bot':panel_2, 'panel_front':panel_3, 'panel_behind':panel_4, 'panel_right':panel_5, 'panel_left':panel_6}

    patch_motor = []
    residual_scene=[]
    for point in whole_scene:
        point_cor = (point[0], point[1], point[2])
        if set_Boundingbox(panel_list, point_cor):
            patch_motor.append(point)
        else:
            residual_scene.append(point)
    return np.array(patch_motor),np.array(residual_scene)



def Read_PCD(file_path):

    pcd = o3d.io.read_point_cloud(file_path)
    colors = np.asarray(pcd.colors)
    points = np.asarray(pcd.points)
    return np.concatenate([points, colors], axis=-1)



def open3d_save_pcd(pc ,FileName = None):
    sampled = np.asarray(pc)
    PointCloud_koordinate = sampled[:, 0:3]

    #visuell the point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(PointCloud_koordinate)
    point_cloud.colors = o3d.utility.Vector3dVector(np.float64( sampled[:, 3:]))
    o3d.io.write_point_cloud(FileName +'.pcd', point_cloud, write_ascii=True)



def points2pcd(pcd_file_path, points):

    handle = open(pcd_file_path, 'a')
    
    point_num=points.shape[0]

    handle.write('# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1')
    string = '\nWIDTH ' + str(point_num)
    handle.write(string)
    handle.write('\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0')
    string = '\nPOINTS ' + str(point_num)
    handle.write(string)
    handle.write('\nDATA ascii')

   # int rgb = ((int)r << 16 | (int)g << 8 | (int)b); 
    for i in range(point_num):
        r,g,b = points[i,3], points[i,4], points[i,5]
        rgb = int(r)<<16 | int(g)<<8 | int(b)
        string = '\n' + str(points[i, 0]) + ' ' + str(points[i, 1]) + ' ' + str(points[i, 2]) + ' ' + str(rgb)
        handle.write(string)
    handle.close()




    return cor_inZivid_np



def find_bolts(seg_motor, eps, min_points):
    bolts = []
    for point in seg_motor:
        if point[3] == 255. and point[4] == 0. and point[5] == 0. : bolts.append(point[0:3])
    bolts = np.asarray(bolts)
    model = DBSCAN(eps=eps, min_samples=min_points)
    yhat = model.fit_predict(bolts)  # genalize label based on index
    clusters = np.unique(yhat)
    noise = []
    clusters_new = []
    positions = []
    for i in clusters:
        noise.append(i) if np.sum(i == yhat) < 200 or i == -1 else clusters_new.append(i)
    for clu in clusters_new :
        row_ix = np.where(yhat == clu)
        position = np.squeeze(np.mean(bolts[row_ix, :3], axis=1))
        positions.append(position)
    
    return positions, len(clusters_new)



def save_pcd_asIMG(pc ,FileName = None):

    sampled = np.asarray(pc)
    PointCloud_koordinate = sampled[:, 0:3]
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(PointCloud_koordinate)
    point_cloud.colors = o3d.utility.Vector3dVector(sampled[:, 3:])

    vis = o3d.visualization.Visualizer()  
    vis.create_window(visible=False) #works for me with False, on some systems needs to be true

    ctr = vis.get_view_control()

    vis.add_geometry(point_cloud)
    vis.get_render_option().point_size = 1.0

    ctr.rotate(0.0, -350.0)
    ctr.rotate(-500.0, 0.0)
    ctr.rotate(0.0, -500.0)
    ctr.rotate(-150.0, 0.0)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(FileName, do_render=True)
    vis.destroy_window()





