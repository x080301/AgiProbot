import numpy as np
import open3d

# labels = ['Void', 'Background', 'Gear', 'Connector', 'Screws', 'Solenoid', 'Electrical Connector', 'Main Housing',
#           'Noise', 'Inner Gear']

# https://www.sioe.cn/yingyong/yanse-rgb-16/
rgb_dic = {'Void': [207, 207, 207],
           'Background': [0, 0, 128],
           'Gear': [102, 140, 255],  # [102, 179, 255],  # [102,140,255],#[102, 204, 255],
           # [153, 221, 255],  # [12, 132, 198],#[204, 255, 255],  # [59, 98, 142],#[165, 205, 255],
           'Connector': [102, 255, 102],  # [0, 255, 0],
           'Screws': [247, 77, 77],
           'Solenoid': [255, 165, 0],
           'Electrical Connector': [255, 255, 0],
           'Main Housing': [0, 100, 0],
           'Noise': [223, 200, 200],
           'Inner Gear': [255, 20, 147]
           # [175, 238, 238]  # [255, 105, 180]  # [255, 182, 193]  # [107, 218, 250]  # [219, 112 ,147]
           # [221, 160, 221]  # [255, 182, 193]  # [199, 21, 133]
           }
#

''' {'Void': [207, 207, 207],
        'Background': [0, 0, 128],
        'Gear': [120, 152, 225],
        'Connector': [118, 218, 145],
        'Screws': [247, 77, 77],
        'Solenoid': [239, 166, 102],
        'Electrical Connector': [153, 135, 206],
        'Main Housing': [99, 178, 238],
        'Noise': [223, 200, 200]}'''
# full_scaned3
'''
{'Void': [207, 207, 207],
           'Background': [0, 0, 128],
           'Gear': [59, 98, 142],
           'Connector': [125, 152, 71],
           'Screws': [247, 77, 77],
           'Solenoid': [201, 121, 55],
           'Electrical Connector': [255, 255, 0],
           'Main Housing': [56, 132, 152],
           'Noise': [223, 200, 200]}
'''
# full_scaned2
'''
{'Void': [207, 207, 207],
           'Background': [0, 0, 128],
           'Gear': [12, 132, 198],
           'Connector': [0, 255, 0],
           'Screws': [247, 77, 77],
           'Solenoid': [255, 165, 16],
           'Electrical Connector': [255, 255, 0],
           'Main Housing': [65, 183, 172],
           'Noise': [223, 200, 200]}
'''
# full_scaned
'''{'Void': [207, 207, 207],
'Background': [0, 0, 128],
'Gear': [165, 205, 255],
'Connector': [0, 255, 0],
'Screws': [255, 0, 0],
'Solenoid': [255, 165, 0],
'Electrical Connector': [255, 255, 0],
'Main Housing': [0, 100, 0],
'Noise': [223, 200, 200]}'''


class PcdReader:

    def read_pcd_ASCII(self, pcd_file):
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

                x, y, z, _, label, _ = list(oneline.strip('\n').split(' '))  # '0 0 0 1646617 8 -1\n'

                point_label = list(rgb_dic.keys())[int(label)]
                if point_label != 'Noise':
                    points.append(np.array([x, y, z]))
                    point_color = rgb_dic[point_label]
                    colors.append(np.array([a / 255.0 for a in point_color]))

        self.points = np.array(points)
        self.colors = np.array(colors)

        print(self.points.shape)
        print(self.colors.shape)

        return self.points, self.colors

    def save_pcd(self, save_dir):
        point_cloud = open3d.geometry.PointCloud()
        point_cloud.points = open3d.utility.Vector3dVector(self.points)
        point_cloud.colors = open3d.utility.Vector3dVector(self.colors)

        open3d.io.write_point_cloud(save_dir, point_cloud, write_ascii=True)
        open3d.visualization.draw_geometries([point_cloud])


if __name__ == "__main__":
    pcd_reader = PcdReader()

    pcd_reader.read_pcd_ASCII('C:/Users/Lenovo/Desktop/large_motor_inside (1).pcd')
    pcd_reader.save_pcd('C:/Users/Lenovo/Desktop/large_motor_inside_labeled.pcd')
