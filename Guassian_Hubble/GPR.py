import numpy as np
from numba import njit

# Squared exponential kernel
def kernel(argument_1, argument_2, sigma_f=1, length=1):
    return (
        sigma_f**2 *
        np.exp(-((argument_1 - argument_2)**2) /
        (2*length**2))
    )

def kernel_zhang(argument_1, argument_2, sigma_f=1, length=1):
    return (
        sigma_f**2 *
        np.exp(-3*np.abs(argument_1 - argument_2) /
        (length)) *
        (
            1 +
            (3 * np.abs(argument_1 - argument_2)) / length +
            (27 * (argument_1 - argument_2)**2) / (7 * length**2) +
            (18 * np.abs(argument_1 - argument_2)**3) / (7 * length**3) +
            (27 * (argument_1 - argument_2)**4) / (35 * length**4)
        )
    )

def cov_matrix(x1, x2, sigma_f=1, length=1):
    return np.array([[kernel(a, b, sigma_f, length) for a in x1] for b in x2])

def gpr(data_x, data_y, predict_x, noise=0, sigma_f=1, length=1):
    k = cov_matrix(data_x, data_x, sigma_f, length) + (3e-7+noise) * np.identity(len(data_x))

    k_inv = np.linalg.inv(
        k
    )
    k_s = cov_matrix(data_x, predict_x, sigma_f, length)

    k_ss = cov_matrix(predict_x, predict_x, sigma_f, length)

    # Mean
    mean_predict = k_s @ k_inv @ data_y

    # Covariance
    cov_predict = k_ss - (
        k_s@k_inv@k_s.T
    )

    # Adding value larger than machine epsilon to ensure positive semi definite
    cov_predict = cov_predict + 3e-7*np.ones(np.shape(cov_predict)[0])

    # Variance
    var_predict = np.diag(cov_predict)

    return mean_predict, cov_predict, var_predict

def cov_matrix_zhang(x1, x2, sigma_f=1, length=1):
    return np.array([[kernel_zhang(a, b, sigma_f, length) for a in x1] for b in x2])

def gpr_zhang(data_x, data_y, predict_x, noise=0, sigma_f=1, length=1):
    k = cov_matrix_zhang(data_x, data_x, sigma_f, length) + (3e-7+noise) * np.identity(len(data_x))

    k_inv = np.linalg.inv(
        k
    )
    k_s = cov_matrix_zhang(data_x, predict_x, sigma_f, length)

    k_ss = cov_matrix_zhang(predict_x, predict_x, sigma_f, length)

    # Mean
    mean_predict = k_s @ k_inv @ data_y

    # Covariance
    cov_predict = k_ss - (
        k_s@k_inv@k_s.T
    )

    # Adding value larger than machine epsilon to ensure positive semi definite
    cov_predict = cov_predict + 3e-7*np.ones(np.shape(cov_predict)[0])

    # Variance
    var_predict = np.diag(cov_predict)

    return mean_predict, cov_predict, var_predict
