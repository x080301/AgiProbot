from utils.visualization import visualization_heatmap


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


def num_points_in_bins():
    import pickle
    from tqdm import tqdm
    for i in tqdm(range(20)):
        with open(
                f'/home/team1/cwu/FuHaoWorkspace/test_results/2024_02_03_23_07_modelnet_nonuniform_newdownsampling/intermediate_result_{i}.pkl',
                'rb') as f:
            data_dict = pickle.load(f)
        num_points_in_bins = data_dict['idx_in_bins']  # B * num_layers * (H,N)
        probability_of_bins = data_dict['probability_of_bins']  # (B, num_layers, num_bins)

        for j in range(16):
            for k in range(2):
                print(
                    f'num_points_in_bins,sample{i * 16 + j},layer{k}:{[item.shape[1] for item in num_points_in_bins[j][k]]}')
                print(f'probability_of_bins,sample{i * 16 + j},layer{k}:{probability_of_bins[j][k]}')


def visualization_downsampled_points():
    from utils.visualization import visualization_downsampled_points

    visualization_downsampled_points()


def visualization_points_in_bins():
    from utils.visualization import visualization_points_in_bins
    visualization_points_in_bins()


def visualization_histogram():
    from utils.visualization import visualization_histogram
    visualization_histogram()


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
# visualization_heatmap()
# num_points_in_bins()
# visualization_downsampled_points()
# visualization_points_in_bins()
visualization_histogram()
