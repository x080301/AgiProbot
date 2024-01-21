import numpy as np
from scipy.optimize import minimize
import scipy
import torch
import matplotlib.pyplot as plt


def maxwell_pdf(v, sigma):
    return np.sqrt(2 / np.pi) * (v ** 2 * np.exp(-v ** 2 / (2 * sigma ** 2))) / sigma ** 3


def negative_log_likelihood(params, data):
    sigma = params[0]
    likelihoods = maxwell_pdf(data, sigma)
    print(-np.mean(np.log(likelihoods)))
    return -np.mean(np.log(likelihoods))


def _estimate_sigma():
    layer_to_visualize = 0

    tensor = torch.load('modelnet_sampling_scores.pt')
    num_batch = tensor.shape[0]
    tensor = tensor[:, layer_to_visualize + 3, :]

    # tensor = torch.reshape(tensor, (-1,))
    # tensor = tensor[tensor > -1.5]
    # print(tensor[tensor == 0])
    # exit(-1)

    if layer_to_visualize != 0:
        tensor = torch.reshape(tensor, (-1,))
        tensor = tensor[tensor > -1.5]
        tensor = torch.reshape(tensor, (num_batch, 1024))

    # tensor = (tensor - torch.mean(tensor, dim=1, keepdim=True)) / torch.std(tensor, dim=1, unbiased=False, keepdim=True)
    tensor = tensor / torch.std(tensor, dim=1, unbiased=False, keepdim=True)

    data = tensor.flatten().cpu().numpy()

    initial_guess = [1]
    result = minimize(negative_log_likelihood, initial_guess, args=(data,))
    estimated_sigma = result.x[0]

    print(f"estimated sigma: {estimated_sigma}")


def estimate_sigma(tensor=torch.load('modelnet_sampling_scores.pt'), layer_to_visualize=0):
    num_batch = tensor.shape[0]
    tensor = tensor[:, layer_to_visualize + 3, :]

    if layer_to_visualize != 0:
        tensor = torch.reshape(tensor, (-1,))
        tensor = tensor[tensor > -1.5]
        tensor = torch.reshape(tensor, (num_batch, 1024))

    tensor = tensor / torch.std(tensor, dim=1, unbiased=False, keepdim=True)

    data = tensor.flatten().cpu().numpy()

    sigma = np.sqrt(np.pi / 8) * np.mean(data)

    print(f'estimated sigma:{sigma}')

    return sigma


def chi_square_test(tensor=torch.load('modelnet_sampling_scores.pt'), normal_ditribution_test=False):
    if normal_ditribution_test:
        data = torch.normal(0, 1, size=(100000,))

        data = torch.nn.functional.softmax(data, dim=0).cpu().numpy()

        estimated_sigma = np.sqrt(np.pi / 8) * np.mean(data)

        bins = np.linspace(0, estimated_sigma * 10, 11)
    else:

        layer_to_visualize = 0

        estimated_sigma = estimate_sigma(tensor, layer_to_visualize=layer_to_visualize)

        num_batch = tensor.shape[0]
        tensor = tensor[:, layer_to_visualize + 3, :]

        if layer_to_visualize != 0:
            tensor = torch.reshape(tensor, (-1,))
            tensor = tensor[tensor > -1.5]
            tensor = torch.reshape(tensor, (num_batch, 1024))

        tensor = tensor / torch.std(tensor, dim=1, unbiased=False, keepdim=True)
        data = tensor.flatten().cpu().numpy()

        bins = np.linspace(0, 8, 201)
    observed_counts, _ = np.histogram(data, bins)
    observed_counts = observed_counts / np.sum(observed_counts)  # / len(data)

    cdf_values = scipy.stats.maxwell.cdf(bins, scale=estimated_sigma)
    expected_counts = np.diff(cdf_values)  # * len(data)

    # cdf_values = scipy.stats.maxwell.cdf(bins, scale=estimated_sigma * 2)
    # observed_counts = np.diff(cdf_values)

    # print(observed_counts)
    # print(expected_counts)

    bar_width = 0.35
    index = np.arange(len(observed_counts))
    plt.bar(index, observed_counts, bar_width, color='blue', label='observed_distribution')
    plt.bar(index + bar_width, expected_counts, bar_width, color='red', label='expected_Maxwell-Boltzmann_distribution')
    plt.legend()
    plt.show()

    chi_square, p_value = scipy.stats.chisquare(observed_counts, expected_counts)  # expected_counts)

    print(f'p_value:{p_value}')
