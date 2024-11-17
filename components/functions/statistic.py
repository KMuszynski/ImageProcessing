import numpy as np


def calculate_mean(image_array):
    return np.mean(image_array)


def calculate_variance(image_array):
    return np.var(image_array)


def calculate_standard_deviation(image_array):
    return np.std(image_array)


def calculate_variation_coefficient_i(image_array):
    mean = calculate_mean(image_array)
    stdev = calculate_standard_deviation(image_array)
    return stdev / mean if mean != 0 else 0


def calculate_asymmetry_coefficient(image_array):
    mean = calculate_mean(image_array)
    stdev = calculate_standard_deviation(image_array)
    skewness = np.mean(((image_array - mean) / stdev) ** 3)
    return skewness


def calculate_flattening_coefficient(image_array):
    mean = calculate_mean(image_array)
    stdev = calculate_standard_deviation(image_array)
    kurtosis = np.mean(((image_array - mean) / stdev) ** 4) - 3
    return kurtosis


def calculate_variation_coefficient_ii(image_array):
    mean = calculate_mean(image_array)
    variance = calculate_variance(image_array)
    return variance / mean if mean != 0 else 0


def calculate_entropy(image_array):
    """information source entropy."""
    histogram, _ = np.histogram(image_array, bins=256, range=(0, 256), density=True)
    histogram = histogram[histogram > 0]  # Remove zero entries to avoid log issues
    entropy = -np.sum(histogram * np.log2(histogram))
    return entropy
