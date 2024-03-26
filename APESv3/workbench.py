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


def visualization_downsample_segmentation():
    from utils.visualization import visualize_segmentation_predictions_downsampled, visualize_segmentation_predictions
    import os

    save_dirs = os.listdir(r'E:/datasets/APES/test_results')

    for save_dir in save_dirs:
        save_dir = f'E:/datasets/APES/test_results/{save_dir}'

        if 'AnTao' in save_dir:
            continue

        if 'Shapenet' in save_dir:
            visualize_segmentation_predictions_downsampled(save_path=save_dir)


def visualization_all():
    from utils.visualization import visualization_heatmap, visualization_downsampled_points, \
        visualization_points_in_bins, visualization_histogram, get_statistic_data_all_samples, \
        visualize_segmentation_predictions, visualize_few_points, visualize_segmentation_predictions_downsampled
    import os

    save_dirs = os.listdir(r'C:\Users\Lenovo\Desktop\test_results')

    for save_dir in save_dirs:
        save_dir = f'C:/Users/Lenovo/Desktop/test_results/{save_dir}'

        view_range = 0.6  # 0.6
        # save_dir = 'C:/Users/Lenovo/Desktop/2024_02_26_19_49_Modelnet_Token_Std_4bin'
        # save_dir = 'C:/Users/Lenovo/Desktop/2024_02_26_19_50_Modelnet_Token_Std_8bin'
        # save_dir = 'C:/Users/Lenovo/Desktop/2024_02_26_20_22_Shapenet_Token_Std'
        # save_dir = 'C:/Users/Lenovo/Desktop/2024_02_21_01_47_Modelnet_Token_Std_2'

        visualization_all = True



        if 'Shapenet' in save_dir:
            visualize_segmentation_predictions(save_path=save_dir)
            visualize_segmentation_predictions_downsampled(save_path=save_dir)

        visualization_histogram(save_path=f'{save_dir}', visualization_all=visualization_all)

        visualization_points_in_bins(save_path=f'{save_dir}', view_range=view_range,
                                     visualization_all=visualization_all)

        visualization_downsampled_points(save_path=f'{save_dir}',
                                         view_range=view_range, visualization_all=visualization_all)

        visualization_heatmap(save_path=f'{save_dir}', view_range=view_range, visualization_all=visualization_all)

        for M in [16, 8, 32, 64, 128]:
            visualize_few_points(M, save_path=save_dir, visualization_all=visualization_all)

        get_statistic_data_all_samples(save_path=save_dir)


def visualize_statistic_data_all_samples():
    from utils.visualization import get_statistic_data_all_samples
    save_dir = 'C:/Users/Lenovo/Desktop/2024_02_27_07_23_Modelnet_Token_Std_12bin'
    get_statistic_data_all_samples(save_path=save_dir)
    visualization_heatmap(save_path=f'{save_dir}', view_range=0.6)


def visualization_points_in_gray():
    from utils.visualization import visualization_points_in_gray

    import os

    save_dirs = os.listdir(r'E:/datasets/APES/test_results')

    for save_dir in save_dirs:
        save_dir = f'E:/datasets/APES/test_results/{save_dir}'

        if '2024_02_21_01_47' not in save_dir:
            continue

        visualization_points_in_gray(save_path=save_dir)


def crop_histogram():
    from utils.visualization import crop_image
    import os
    from tqdm import tqdm

    crop_area = (72, 35, 555, 440)

    for file_name in tqdm(os.listdir('D:/master/semester7/master_arbeit/ECCV/Figures/cls/histogram')):
        if 'cropped' in file_name:
            continue
        input_image_path = f'D:/master/semester7/master_arbeit/ECCV/Figures/cls/histogram/{file_name}'
        output_image_path = f'D:/master/semester7/master_arbeit/ECCV/Figures/cls/histogram/cropped_{file_name}'

        crop_image(input_image_path, output_image_path, crop_area)


def copy_and_rename():
    from utils.visualization import copy_rename

    # copy_rename('E:/datasets/APES/test_results/2024_02_21_01_47_Modelnet_Token_Std/few_points', 'airplane', 27,
    #             'D:/master/semester7/master_arbeit/ECCV/Figures/few/pictures')
    # copy_rename('E:/datasets/APES/test_results/2024_02_21_01_47_Modelnet_Token_Std/few_points', 'chair', 99,
    #                 'D:/master/semester7/master_arbeit/ECCV/Figures/few/pictures')
    copy_rename('E:/datasets/APES/test_results/2024_02_21_01_47_Modelnet_Token_Std/few_points', 'chair', 55,
                'D:/master/semester7/master_arbeit/ECCV/Figures/few/pictures')


def copy_and_rename_and_crop():
    from utils.visualization import copy_and_crop

    # copy_rename('E:/datasets/APES/test_results/2024_02_21_01_47_Modelnet_Token_Std/few_points', 'airplane', 27,
    #             'D:/master/semester7/master_arbeit/ECCV/Figures/few/pictures')
    # copy_rename('E:/datasets/APES/test_results/2024_02_21_01_47_Modelnet_Token_Std/few_points', 'chair', 99,
    #                 'D:/master/semester7/master_arbeit/ECCV/Figures/few/pictures')
    copy_and_crop('chair', 95, (55, 30, 329, 348), mode='apes')
    copy_and_crop('chair', 54, (60, 29, 325, 358), mode='v1')


visualization_all()
# copy_and_rename_and_crop()
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
# visualization_points_in_bins()
# visualization_histogram()
# visualization_all()
# visualize_statistic_data_all_samples()
# visualization_downsample_segmentation()
# visualization_points_in_gray()
# crop_histogram()
