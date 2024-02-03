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


# from utils.data_analysis import estimate_sigma
# estimate_sigma()

# import numpy as np
# bins = np.linspace(0, 8, 11)
# print(bins)

# from utils.data_analysis import chi_square_test
# #
# chi_square_test()


# find_sampling_score_bin_boundary()
visualization_histogram_in_boundary()
