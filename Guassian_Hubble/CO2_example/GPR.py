import numpy as np
from numba import njit

# Squared exponential kernel


@njit('f8(f8, f8, f8[:])', nogil=True)
def kernel(x1, x2, hp):

    k1 = hp[0]**2 * np.exp(
        -(x1 - x2)**2 / (2*hp[1]**2)
    )

    k2 = hp[2]**2 * np.exp(
        (-(x1 - x2)**2 / (2*hp[3]**2)) -
        (2*np.sin(np.pi * np.abs(x1 - x2))**2 / (hp[4]**2))
    )

    k3 = hp[5]**2 * (
        1 +
        (x1 - x2)**2 / (2*hp[7]*hp[6]**2)
    )**(-hp[7])

    k4 = hp[8]**2 * np.exp(
        -(x1 - x2)**2 / (2*hp[9]**2)
    )

    return k1 + k2 + k3 + k4


@njit('f8[:,:](f8[:], f8[:], f8[:])', nogil=True)
def cov_matrix(x1, x2, hp):
    cm = np.zeros((len(x2), len(x1)))
    for b in range(len(x2)):
        for a in range(len(x1)):
            cm[b][a] = kernel(x1[a], x2[b], hp)

    return cm


def gpr(data_x, data_y, predict_x, hp):
    k = cov_matrix(data_x, data_x, hp) + (3e-7+hp[10]) * np.eye(len(data_x))

    k_inv = np.linalg.inv(
        k
    )
    k_s = cov_matrix(data_x, predict_x, hp)

    k_ss = cov_matrix(predict_x, predict_x, hp)

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
