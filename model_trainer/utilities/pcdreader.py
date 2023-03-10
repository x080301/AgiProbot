import numpy as np
import open3d

rgb_dic = {'Void': [207, 207, 207],
           'Background': [0, 0, 128],
           'Gear': [165, 205, 255],
           'Connector': [0, 255, 0],
           'Screws': [255, 0, 0],
           'Solenoid': [255, 165, 0],
           'Electrical Connector': [255, 255, 0],
           'Main Housing': [0, 100, 0],
           'Noise': [223, 200, 200]}
labels = ['Void', 'Background', 'Gear', 'Connector', 'Screws', 'Solenoid', 'Electrical Connector', 'Main Housing',
          'Noise']


def read_pcd(pcd_file):
    points = []
    colors = []
    with open(pcd_file, 'r') as f:

        head_flag = True
        while True:
            # for i in range(12):
            oneline = f.readline()

            if head_flag:
                if 'DATA ascii' in oneline:
                    head_flag = False
                    continue
                else:
                    continue

            if not oneline:
                break

            x, y, z, label, _ = list(oneline.strip('\n').split(' '))  # '0 0 0 1646617 8 -1\n'

            point_label = labels[int(label)]
            if point_label != 'Noise':
                points.append(np.array([x, y, z]))
                point_color = rgb_dic[point_label]
                colors.append(np.array([a / 255.0 for a in point_color]))

    points = np.array(points)
    colors = np.array(colors)

    return points, colors


if __name__ == "__main__":
    points, colors = read_pcd('C:/Users/Lenovo/Desktop/MotorClean.plz.pcd')
    print(points.shape)
    print(colors.shape)

    # visuell the point cloud
    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(points)
    point_cloud.colors = open3d.utility.Vector3dVector(colors)

    open3d.io.write_point_cloud('C:/Users/Lenovo/Desktop/MotorClean.plz_color.pcd', point_cloud)
    open3d.visualization.draw_geometries([point_cloud])

    '''points = np.empty([0])
    colors = np.empty([0])
    for i in range(pcd.shape[0]):
        points = np.append(points, pcd[i, 0:3])
        colors = np.append(colors, np.asarray(rgb_label[int(pcd[i, 3])]))

    points = points.reshape((-1, 3))
    colors = colors.reshape((-1, 3))

    print(points.shape)
    print(colors.shape)'''

    pass
