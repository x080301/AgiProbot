# from utils.data_analysis import sampling_score_bin_boundary
#
# sampling_score_bin_boundary(saved_sampling_score_dir='shapenet_sampling_scores.pt', layer_to_visualize=0)
# sampling_score_bin_boundary(saved_sampling_score_dir='shapenet_sampling_scores.pt', layer_to_visualize=1)
# sampling_score_bin_boundary(saved_sampling_score_dir='modelnet_sampling_scores.pt', layer_to_visualize=0)
# sampling_score_bin_boundary(saved_sampling_score_dir='modelnet_sampling_scores.pt', layer_to_visualize=1)

from utils.data_analysis import visualization_sampling_score

visualization_sampling_score(saved_sampling_score_dir='shapenet_sampling_scores.pt', layer_to_visualize=0,
                             z_normalization_miu=True, show_plt=False,
                             save_dir=r'C:/Users/Lenovo/Desktop/SamplingScore2/', idx=None)
visualization_sampling_score(saved_sampling_score_dir='shapenet_sampling_scores.pt', layer_to_visualize=1,
                             z_normalization_miu=True, show_plt=False,
                             save_dir=r'C:/Users/Lenovo/Desktop/SamplingScore2/', idx=None)
visualization_sampling_score(saved_sampling_score_dir='modelnet_sampling_scores.pt', layer_to_visualize=0,
                             z_normalization_miu=True, show_plt=False,
                             save_dir=r'C:/Users/Lenovo/Desktop/SamplingScore2/', idx=None)
visualization_sampling_score(saved_sampling_score_dir='modelnet_sampling_scores.pt', layer_to_visualize=1,
                             z_normalization_miu=True, show_plt=False,
                             save_dir=r'C:/Users/Lenovo/Desktop/SamplingScore2/', idx=None)

# from utils.data_analysis import draw_pcd
#
# for i in range(5):
#     draw_pcd(idx=i, saved_sampling_score_dir='shapenet_sampling_scores.pt')
#     draw_pcd(idx=i, saved_sampling_score_dir='modelnet_sampling_scores.pt')

# from utils.data_analysis import visualization_sampling_score_in_bin

# for i in range(6):
#     if i == 5:
#         i = None
#     visualization_sampling_score_in_bin(saved_sampling_score_dir='shapenet_sampling_scores.pt', layer_to_visualize=0,
#                                         show_plt=False,
#                                         save_dir=r'C:/Users/Lenovo/Desktop/SamplingScore2/', idx=i,
#                                         bin_boundary=[-0.609, -0.477, -0.362, -0.242, -0.095, 0.132, 0.623])
#
#     visualization_sampling_score_in_bin(saved_sampling_score_dir='shapenet_sampling_scores.pt', layer_to_visualize=1,
#                                         show_plt=False,
#                                         save_dir=r'C:/Users/Lenovo/Desktop/SamplingScore2/', idx=i,
#                                         bin_boundary=[-0.679, -0.514, -0.381, -0.241, -0.065, 0.200, 0.723])
#
#     visualization_sampling_score_in_bin(saved_sampling_score_dir='modelnet_sampling_scores.pt', layer_to_visualize=0,
#                                         show_plt=False,
#                                         save_dir=r'C:/Users/Lenovo/Desktop/SamplingScore2/', idx=i,
#                                         bin_boundary=[-0.689, -0.522, -0.380, -0.230, -0.045, 0.219, 0.721])
#
#     visualization_sampling_score_in_bin(saved_sampling_score_dir='modelnet_sampling_scores.pt', layer_to_visualize=1,
#                                         show_plt=False,
#                                         save_dir=r'C:/Users/Lenovo/Desktop/SamplingScore2/', idx=i,
#                                         bin_boundary=[-0.560, -0.492, -0.428, -0.346, -0.220, 0.014, 0.629])


# from utils.data_analysis import estimate_sigma
# estimate_sigma()

# import numpy as np
# bins = np.linspace(0, 8, 11)
# print(bins)

# from utils.data_analysis import chi_square_test
# #
# chi_square_test()
