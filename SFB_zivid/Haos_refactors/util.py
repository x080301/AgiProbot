import rospy
import numpy as np
import math
from scipy.spatial.transform import Rotation
import tf2_msgs.msg
import sensor_msgs.point_cloud2 as pc2
import struct
import pandas as pd
from math import sin, cos, atan, pi, sqrt

transform_ee_c_arr = np.array([[0.96333, 0.00567, 0.26827, -67.92999],
                               [-0.01035, 0.99982, 0.01605, -104.24841],
                               [-0.26813, -0.01824, 0.96321, 127.98334],
                               [0.0, 0.0, 0.0, 1.0]])
"""
transform_ee_c_arr = np.array([[0.96322 ,  -0.00087  ,  0.26872 , -68.84004],
                                [  -0.00415   , 0.99983  ,  0.01812, -105.10566],
                                [  -0.26869  , -0.01857 ,   0.96305 , 127.69376],
                                [   0.       ,  0.      ,   0.      ,   1.]])

"""

global_transformation = np.asarray([[0, 1, 0, -0.860],
                                    [-1, 0, 0, -0.140],
                                    [0, 0, 1, 1.1],
                                    [0, 0, 0, 1]])

def read_robot_pose():
    # tf_topic = rospy.get_param('/tf')
    success = False
    while not success:
        pose = rospy.wait_for_message('/tf', tf2_msgs.msg.TFMessage, timeout=10)
        if pose.transforms[0].header.frame_id == "zivid_base":
            translation = pose.transforms[0].transform.translation
            translation = np.asarray([translation.x, translation.y, translation.z])
            rotation = pose.transforms[0].transform.rotation

            rotation = Rotation.from_quat([rotation.x, rotation.y, rotation.z, rotation.w])
            success = True

    return translation, rotation

def euler_from_quaternion(x, y, z, w):
    """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians


def deg_to_rad(list):
    return [e * 2 * pi / 360 for e in list]


def rad_to_deg(list):
    return [e * 180 / pi for e in list]


def rpy2rv(roll, pitch, yaw):
    list = deg_to_rad([roll, pitch, yaw])
    alpha = list[2]
    beta = list[1]
    gamma = list[0]

    ca = math.cos(alpha)
    cb = math.cos(beta)
    cg = math.cos(gamma)
    sa = math.sin(alpha)
    sb = math.sin(beta)
    sg = math.sin(gamma)

    r11 = ca * cb
    r12 = ca * sb * sg - sa * cg
    r13 = ca * sb * cg + sa * sg
    r21 = sa * cb
    r22 = sa * sb * sg + ca * cg
    r23 = sa * sb * cg - ca * sg
    r31 = -sb
    r32 = cb * sg
    r33 = cb * cg

    theta = math.acos((r11 + r22 + r33 - 1) / 2)
    sth = math.sin(theta)
    kx = (r32 - r23) / (2 * sth)
    ky = (r13 - r31) / (2 * sth)
    kz = (r21 - r12) / (2 * sth)

    return [(theta * kx), (theta * ky), (theta * kz)]


def decode_pcd_col(data):

    points = []
    colors = []

    for line in pc2.read_points(data, skip_nans=False): 
        points.append([e for e in line])

        hexx = hex(struct.unpack('<I', struct.pack('<f', line[3]))[0])
        hexxx =int(hexx, 16)
        hexxxx=hex(hexxx)
        r = (hexxx & 0x00FF0000)>> 16
        g = (hexxx & 0x0000FF00)>> 8
        b = (hexxx & 0x000000FF)
        
        rgb = [r/255.,g/255., b/255.]
        colors.append(rgb)

    points = np.asarray(points) * 1000
    xyz_local = points[:,:3]
    return xyz_local, colors


def transform_robot_global(xyz_local, translation, rotation):
    # Convert robots TCP to camera Position

    transform = np.eye(4)
    # transform[:3, :3] = rotation.as_dcm()
    transform[:3, :3] = rotation.as_matrix()
    transform[:3, 3] = translation.T * 1000
    global transform_ee_c_arr
    base_cam = np.matmul(transform, transform_ee_c_arr)

    xyz_global = xyz_local.dot(base_cam[:3, :3].T) + [base_cam[:3, 3][0], base_cam[:3, 3][1], base_cam[:3, 3][2]]

    return xyz_global


def calculate_robot_pose():
    translation, rotation = read_robot_pose()
    transform = np.eye(4)

    transform[:3, :3] = rotation.as_matrix()
    transform[:3, 3] = translation.T * 1000

    return transform


def rgba2rgb(rgba, background=(255, 255, 255)):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros((row, col, 3), dtype='float32')
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]

    a = np.asarray(a, dtype='float32') / 255.0

    R, G, B = background

    rgb[:, :, 0] = r * a + (1.0 - a) * R
    rgb[:, :, 1] = g * a + (1.0 - a) * G
    rgb[:, :, 2] = b * a + (1.0 - a) * B

    return np.asarray(rgb, dtype='uint8')

def decode_segmentation_masks(mask, n_classes):
    colormap = pd.DataFrame(
        {0: [0, 0, 0], 1: [170, 170, 170], 2: [238, 183, 13], 3: [20, 68, 102], 4: [84, 110, 122], 5: [0, 150, 150]})
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for l in range(0, n_classes):
        idx = mask == l
        r[idx] = colormap[l][0]
        g[idx] = colormap[l][1]
        b[idx] = colormap[l][2]
    rgb = np.stack([r, g, b], axis=2)

    return rgb

def get_cam_rotation(pos):
    x, y, z = pos
    alpha, beta, gamma = 0.0, 0.0, 0.0
    if y == 0:
        if z <= 0:
            alpha = 0.0
        else:
            alpha = pi
    elif y <= 0 and z <= 0:
        alpha = pi / 2 - atan(z / y)
    elif y <= 0 and z > 0:
        alpha = pi / 2 - atan(z / y)
    elif y > 0 and z > 0:
        alpha = 3 * pi / 2 - atan(z / y)
    elif y > 0 and z <= 0:
        alpha = 3 * pi / 2 - atan(z / y)

    d = sqrt(y * y + z * z)
    if d != 0:
        beta = - atan(x / d)

    return -alpha, beta, gamma

def adjust_gamma(alpha, beta, gamma, phi):
    rotation = Rotation.from_euler("XYZ", [alpha, beta, gamma], degrees=False)
    x = rotation.as_matrix()[:, 0]
    xx = np.asarray([x[0], x[1], 0])
    cos = np.dot(x, xx) / (np.linalg.norm(x) * np.linalg.norm(xx) + 1e-16)
    degree = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    rad = degree / 180 * pi
    if pi / 4 <= phi <= 3 * pi / 4:
        # x towards (-1, 0, 0)
        if x[0] <= 0 and x[2] >= 0:
            gamma = rad
        elif x[0] <= 0 and x[2] < 0:
            gamma = -1 * rad
        elif x[0] > 0 and x[2] > 0:
            gamma = np.pi - rad
        elif x[0] > 0 and x[2] < 0:
            gamma = rad - np.pi
    elif 5 * pi / 4 <= phi <= 7 * pi / 4:
        # x towards (1, 0, 0)
        if x[0] <= 0 and x[2] >= 0:
            gamma = np.pi - rad
        elif x[0] <= 0 and x[2] < 0:
            gamma = np.pi + rad
        elif x[0] > 0 and x[2] > 0:
            gamma = rad
        elif x[0] > 0 and x[2] < 0:
            gamma = -1 * rad
    elif 3 * pi / 4 < phi < 5 * pi / 4:
        # x towards (0, -1, 0)
        if x[1] <= 0 and x[2] >= 0:
            gamma = rad
        elif x[1] <= 0 and x[2] < 0:
            gamma = -1 * rad
        elif x[1] > 0 and x[2] > 0:
            gamma = np.pi - rad
        elif x[1] > 0 and x[2] < 0:
            gamma = rad - np.pi
    else:
        # x towards (0, 1, 0)
        if x[1] <= 0 and x[2] >= 0:
            gamma = np.pi - rad
        elif x[1] <= 0 and x[2] < 0:
            gamma = np.pi + rad
        elif x[1] > 0 and x[2] > 0:
            gamma = rad
        elif x[1] > 0 and x[2] < 0:
            gamma = -1 * rad
    return alpha, beta, gamma