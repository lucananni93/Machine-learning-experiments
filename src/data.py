import numpy as np


def orange_blue_dataset(n_samples: int,
                        blue_mean=None, blue_cov=None, blue_mean_num=10,
                        orange_mean=None, orange_cov=None, orange_mean_num=10) -> tuple:
    """Generates a binary dataset from multivariate normal distributions of means

    :param n_samples: total number of samples
    :param blue_mean: int or array of mean values for first class
    :param blue_cov: int or covariance matrix for first class
    :param blue_mean_num: number of distributions generating first class
    :param orange_mean: int or array of mean values for second class
    :param orange_cov: int or covariance matrix for second class
    :param orange_mean_num: number of distributions generating second class
    :return: X, Y, parameters
    """
    ORANGE = 1
    BLUE = 0

    blue_mean = [1, 0] if blue_mean is None else blue_mean
    blue_cov = np.eye(2) if blue_cov is None else blue_cov

    orange_mean = [1, 0] if orange_mean is None else orange_mean
    orange_cov = np.eye(2) if orange_cov is None else orange_cov

    means_blue = np.random.multivariate_normal(blue_mean, blue_cov, blue_mean_num)
    means_orange = np.random.multivariate_normal(orange_mean, orange_cov, orange_mean_num)

    n_blue = int(n_samples / 2)
    n_orange = n_samples - n_blue

    blue_samples = np.zeros((n_blue, len(blue_mean)))
    orange_samples = np.zeros((n_orange, len(orange_mean)))

    for i in range(n_blue):
        mean_i = means_blue[np.random.choice(np.arange(means_blue.shape[0])), :]
        blue_samples[i, :] = np.random.multivariate_normal(mean_i, np.eye(2) / 5, 1)

    for i in range(n_orange):
        mean_i = means_orange[np.random.choice(np.arange(means_orange.shape[0])), :]
        orange_samples[i, :] = np.random.multivariate_normal(mean_i, np.eye(2) / 5, 1)

    X = np.vstack((blue_samples, orange_samples))
    Y = np.append(np.repeat(BLUE, n_blue), np.repeat(ORANGE, n_orange)).reshape(-1, 1)

    parameters = {
        'blue_mean_generator': blue_mean,
        'blue_cov_generator': blue_cov,
        'orange_mean_generator': orange_mean,
        'orange_cov_generator': orange_cov,
        'blue_means': means_blue,
        'orange_means': means_orange,
        'observations_cov': np.eye(2) / 5
    }
    return X, Y, parameters


__all__ = [
    'orange_blue_dataset'
]
