import functools
import operator

import numpy as np


# def range_cost(x, theta, y_min, y_max):
#     pass


def compute_cost(x, theta, y):
    pred = dot_product(x, theta)
    return sum(sum((y - pred) ** 2))


def dot_product(x, theta):
    product = x * theta
    return np.array([functools.reduce(operator.add, lis) for lis in product])


def gradient_descent(
        x, y, theta, alpha=0.0001, beta1=-99,
        beta2=-99, epsilon=10 ** -8, iteration=-99):
    vdwdb, sdwdb = [0, 0]
    nutrients_count = x.shape[2]
    input_sample_size = x.shape[0]
    temp_theta = np.zeros(shape=theta.shape)
    for i in range(0, theta.shape[0]):
        temp_theta[i] = [
                            (alpha / input_sample_size) * sum(sum(((
                                    (dot_product(x, theta) - y)
                                    * x[:, i:i + 1].reshape(input_sample_size,
                                                            nutrients_count)
                            ))))
                        ] * nutrients_count
    # Adams optimisation.
    # Need correction
    if beta1 != -99 and beta2 != -99:
        vdwdb = beta1 * vdwdb + (1 - beta1) * temp_theta
        vdwdb_corrected = vdwdb / (1 - beta1 ** iteration)
        sdwdb = beta2 * sdwdb + (1 - beta2) * (sdwdb * sdwdb)
        sdwdb_corrected = sdwdb / (1 - beta2 ** iteration)
        temp_theta = vdwdb_corrected / (np.sqrt(sdwdb_corrected) + epsilon)

    for i in range(0, theta.shape[0]):
        theta[i] = theta[i] - temp_theta[i]
    for i in range(0, theta.shape[0]):
        theta[i] = [max(x, 0) for x in theta[i]]
