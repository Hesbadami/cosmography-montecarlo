import numpy as np
from scipy.stats import truncnorm
from tqdm import tqdm
from GPR import cov_matrix

# Import Data
data = np.genfromtxt('co2.txt')
t = data[:, 2]
co2 = data[:, 3]


# Initial Truncnorm Function
def N(mean, sigma, lower, upper):
    return truncnorm(
        (lower - mean)/sigma,
        (upper - mean)/sigma,
        mean,
        sigma
    )


# Initial values for hyperparameters
def get_hp(ini):
    return [
        # kernel 1:
        N(ini[0], 5, 0, 100),  # σ_f_trend -> 66
        N(ini[1], 1, 10, 90),  # l_trend -> 67

        # kernel 2:
        N(ini[2], 0.2, 0, 4),  # σ_f_periodic -> 2.4
        N(ini[3], 5, 10, 200),  # l_periodic_decay -> 90
        N(ini[4], 0.5, 0, 2),  # l_periodic -> 1.3

        # kernel 3:
        N(ini[5], 0.05, 0, 1),  # σ_f_quad -> 0.66
        N(ini[6], 0.5, 0, 5),  # l_quad -> 1.2
        N(ini[7], 0.1, 0, 5),  # α_quad -> 0.78

        # kernel 4:
        N(ini[8], 0.05, 0, 1),  # σ_f_noise -> 0.18
        N(ini[9], 0.1, 0, 3),  # l_noise -> 1.6
        N(ini[10], 0.05, 0, 1),  # σ_n_noise -> 0.19
    ]


# Marginal likelihood
def mlog(hp):
    x = t
    y = co2
    print('hi')
    k = cov_matrix(x, x, hp) + (3e-7+hp[10]) * np.eye(len(x))
    print('bye')
    k_inv = np.linalg.inv(k)
    n = len(x)
    return (
        -(1/2) * y.T @ k_inv @ y -
        1/2 * np.log(np.abs(k)) -
        n/2 * np.log(2*np.pi)
    )


ini = np.array([2, 50, 0.5, 170, 1, 0, 2, 2.5, 0.05, 0.05, 0.05])

print(mlog(ini))

# for j in tqdm(range(50000)):
#     ini = get_hp(ini)
