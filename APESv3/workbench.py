def find_sampling_score_bin_boundary():
    from utils.data_analysis import sampling_score_bin_boundary

    sampling_score_bin_boundary(saved_sampling_score_dir='shapenet_sampling_scores.pt', layer_to_visualize=0)
    sampling_score_bin_boundary(saved_sampling_score_dir='shapenet_sampling_scores.pt', layer_to_visualize=1)
    sampling_score_bin_boundary(saved_sampling_score_dir='modelnet_sampling_scores.pt', layer_to_visualize=0)
    sampling_score_bin_boundary(saved_sampling_score_dir='modelnet_sampling_scores.pt', layer_to_visualize=1)


# from utils.data_analysis import draw_pcd
#
# for i in range(5):
#     draw_pcd(idx=i, saved_sampling_score_dir='shapenet_sampling_scores.pt')
#     draw_pcd(idx=i, saved_sampling_score_dir='modelnet_sampling_scores.pt')

def visualization_histogram_in_boundary():
    from utils.data_analysis import visualization_sampling_score_in_bin
    from utils.data_analysis import visualization_sampling_score

    # visualization_sampling_score(saved_sampling_score_dir='shapenet_sampling_scores.pt', layer_to_visualize=0,
    #                              z_normalization_miu=True, show_plt=False,
    #                              save_dir=r'C:/Users/Lenovo/Desktop/SamplingScore2/', idx=None)
    # visualization_sampling_score(saved_sampling_score_dir='shapenet_sampling_scores.pt', layer_to_visualize=1,
    #                              z_normalization_miu=True, show_plt=False,
    #                              save_dir=r'C:/Users/Lenovo/Desktop/SamplingScore2/', idx=None)
    # visualization_sampling_score(saved_sampling_score_dir='modelnet_sampling_scores.pt', layer_to_visualize=0,
    #                              z_normalization_miu=True, show_plt=False,
    #                              save_dir=r'C:/Users/Lenovo/Desktop/SamplingScore2/', idx=None)
    # visualization_sampling_score(saved_sampling_score_dir='modelnet_sampling_scores.pt', layer_to_visualize=1,
    #                              z_normalization_miu=True, show_plt=False,
    #                              save_dir=r'C:/Users/Lenovo/Desktop/SamplingScore2/', idx=None)

    for i in range(6):
        if i == 5:
            i = None
        visualization_sampling_score_in_bin(saved_sampling_score_dir='shapenet_sampling_scores.pt',
                                            layer_to_visualize=0,
                                            show_plt=False,
                                            save_dir=r'C:/Users/Lenovo/Desktop/SamplingScore2/', idx=i,
                                            bin_boundary=[-1.245e-05, -8.971e-06, -5.476e-06, -7.46e-07, 8.45e-06])

        visualization_sampling_score_in_bin(saved_sampling_score_dir='shapenet_sampling_scores.pt',
                                            layer_to_visualize=1,
                                            show_plt=False,
                                            save_dir=r'C:/Users/Lenovo/Desktop/SamplingScore2/', idx=i,
                                            bin_boundary=[-1.68e-05, -1.155e-05, -6.46e-06, 2.405e-07, 1.274e-05])

        visualization_sampling_score_in_bin(saved_sampling_score_dir='modelnet_sampling_scores.pt',
                                            layer_to_visualize=0,
                                            show_plt=False,
                                            save_dir=r'C:/Users/Lenovo/Desktop/SamplingScore2/', idx=i,
                                            bin_boundary=[-8.08e-06, -5.421e-06, -2.851e-06, 3.737e-07, 6.065e-06])

        visualization_sampling_score_in_bin(saved_sampling_score_dir='modelnet_sampling_scores.pt',
                                            layer_to_visualize=1,
                                            show_plt=False,
                                            save_dir=r'C:/Users/Lenovo/Desktop/SamplingScore2/', idx=i,
                                            bin_boundary=[-0.0001078, -7.882e-05, -5.652e-05, -2.619e-05, 5.914e-05])


def visualization_heatmap():
    from utils.visualization import visualization_heatmap_one_shape
    import pickle
    import os
    from tqdm import tqdm

    mapping = {0: 'airplane', 1: 'bathtub', 2: 'bed', 3: 'bench', 4: 'bookshelf', 5: 'bottle', 6: 'bowl', 7: 'car',
               8: 'chair', 9: 'cone', 10: 'cup', 11: 'curtain', 12: 'desk', 13: 'door', 14: 'dresser', 15: 'flower_pot',
               16: 'glass_box', 17: 'guitar', 18: 'keyboard', 19: 'lamp', 20: 'laptop', 21: 'mantel', 22: 'monitor',
               23: 'night_stand',
               24: 'person', 25: 'piano', 26: 'plant', 27: 'radio', 28: 'range_hood', 29: 'sink', 30: 'sofa',
               31: 'stairs',
               32: 'stool', 33: 'table', 34: 'tent', 35: 'toilet', 36: 'tv_stand', 37: 'vase', 38: 'wardrobe',
               39: 'xbox'}

    save_path = f'/home/team1/cwu/FuHaoWorkspace/test_results/2024_02_03_23_07_modelnet_nonuniform_newdownsampling/heat_map'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in tqdm(range(20)):
        with open(
                f'/home/team1/cwu/FuHaoWorkspace/test_results/2024_02_03_23_07_modelnet_nonuniform_newdownsampling/intermediate_result_{i}.pkl',
                'rb') as f:
            data_dict = pickle.load(f)

        sampling_score_batch = data_dict['sampling_score']  # (B, num_layers, H, N)
        sample_batch = data_dict['samples']  # (B,N,3)
        label_batch = data_dict['ground_truth']

        B = sampling_score_batch.shape[0]

        for j in range(B):
            sampling_score = sampling_score_batch[j]  # (num_layers, H, N)
            sample = sample_batch[j]  # (N,3)
            category = mapping[int(label_batch[j])]

            visualization_heatmap_one_shape(i * B + j, sample, category, sampling_score, save_path)


# from utils.data_analysis import estimate_sigma
# estimate_sigma()

# import numpy as np
# bins = np.linspace(0, 8, 11)
# print(bins)

# from utils.data_analysis import chi_square_test
# #
# chi_square_test()


# find_sampling_score_bin_boundary()
# visualization_histogram_in_boundary()
visualization_heatmap()
