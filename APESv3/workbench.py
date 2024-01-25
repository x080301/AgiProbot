from utils.data_analysis import visualization_sampling_score_in_bin

for i in range(5):
    visualization_sampling_score_in_bin(saved_sampling_score_dir='shapenet_sampling_scores.pt', layer_to_visualize=0,
                                        show_plt=False,
                                        save_dir=r'C:/Users/Lenovo/Desktop/SamplingScore2/', idx=i,
                                        bin_boundary=[0.33, 0.53, 0.89, 1.26])
    visualization_sampling_score_in_bin(saved_sampling_score_dir='modelnet_sampling_scores.pt', layer_to_visualize=0,
                                        show_plt=False,
                                        save_dir=r'C:/Users/Lenovo/Desktop/SamplingScore2/', idx=i,
                                        bin_boundary=[0.84, 1.15, 1.51, 2.12])
    visualization_sampling_score_in_bin(saved_sampling_score_dir='shapenet_sampling_scores.pt', layer_to_visualize=1,
                                        show_plt=False,
                                        save_dir=r'C:/Users/Lenovo/Desktop/SamplingScore2/', idx=i,
                                        bin_boundary=[0.47, 0.73, 1.03, 1.53])
    visualization_sampling_score_in_bin(saved_sampling_score_dir='modelnet_sampling_scores.pt', layer_to_visualize=1,
                                        show_plt=False,
                                        save_dir=r'C:/Users/Lenovo/Desktop/SamplingScore2/', idx=i,
                                        bin_boundary=[0.11, 0.21, 0.38, 0.84])
# from utils.data_analysis import estimate_sigma
# estimate_sigma()

# import numpy as np
# bins = np.linspace(0, 8, 11)
# print(bins)

# from utils.data_analysis import chi_square_test
# #
# chi_square_test()
