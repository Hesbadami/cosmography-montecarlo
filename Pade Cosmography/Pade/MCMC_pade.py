import numpy as np
from numba import njit
from scipy.stats import truncnorm

data = np.genfromtxt('mock_pantheon.txt')
z1 = data[:, 0]
muu = data[:, 1]
error = data[:, 2]

Nx = 100  # accuracy of integral


@njit("f8(f8, f8[:])", nogil=True)
def E(z, param):
    H_1 = 1 + param[0]
    H_2 = -pow(param[0], 2) + param[1]
    H_3 = 3*pow(param[0], 2)*(1+param[0]) - param[1]*(3+4*param[0]) - param[2]
    H_4 = -3*pow(param[0], 2)*(4+8*param[0]+5*pow(param[0], 2)) + \
        param[1] * (12+32*param[0]+25*pow(param[0], 2)-4*param[1]) + \
        param[2] * (8+7*param[0]) + param[3]

    Q_1 = (-6*H_1*H_4 + 12*H_2*H_3)/(24*H_1*H_3 - 36*H_2**2)
    Q_2 = (3*H_2*H_4 - 4*H_3**2)/(24*H_1*H_3 - 36*H_2**2)
    P_0 = 1
    P_1 = H_1 + Q_1
    P_2 = H_2/2 + Q_1*H_1 + Q_2

    EE = (P_0 + P_1*z + P_2*z**2)/(1 + Q_1*z + Q_2*z**2)
    return 1/EE


@njit("f8(f8, f8[:])", nogil=True)
def mu(z, param):
    zc = np.linspace(0, z, Nx)
    dz = zc[1] - zc[0]
    Es = np.array([E(z, param) for z in zc])
    E_int = np.cumsum(Es) * dz
    muu = 5*np.log10((1+z)*E_int[-1])
    return muu


@njit("f8(f8[:])", nogil=True)
def xi2(param):
    A = 0
    B = 0
    C = 0
    for i in range(len(z1)):
        A += pow(mu(z1[i], param)-muu[i], 2)/pow(error[i], 2)
        B += (mu(z1[i], param)-muu[i])/pow(error[i], 2)
        C += 1/pow(error[i], 2)
        x2 = A - pow(B, 2)/C
    return x2


def N(mean, sigma, lower, upper):
    NN = truncnorm((lower - mean) / sigma, (upper - mean) / sigma, mean, sigma)
    return NN


sample_old0 = -0.8
sigma0 = 0.01
sample_old1 = 1.9
sigma1 = 0.03
sample_old2 = -2
sigma2 = 0.05
sample_old3 = 4
sigma3 = 0.08
X2_old = 5000

f = open("pantheon_pade22.txt", "w")
for j in range(50000):
    N0 = N(sample_old0, sigma0, -1, 0).rvs()
    N1 = N(sample_old1, sigma1, 0, 3).rvs()
    N2 = N(sample_old2, sigma2, -2, 3).rvs()
    N3 = N(sample_old3, sigma3, -2, 6).rvs()
    param = np.array([N0, N1, N2, N3])
    X2_new = xi2(param)
    u = np.random.rand()
    if u < min(1, np.exp(-(X2_new - X2_old)/2)):
        sample_old0 = N0
        sample_old1 = N1
        sample_old2 = N2
        sample_old3 = N3
        X2_old = X2_new
        f.write(
            str(1.0) + "  " + str(0.0) + "  " + str(sample_old0) + "  "
            + str(sample_old1) + "  " + str(sample_old2) + "  " +
            str(sample_old3) + "  " + str(X2_old) + "\n"
        )
    else:
        f.write(
            str(1.0) + "  " + str(0.0) + "  " + str(sample_old0) + "  "
            + str(sample_old1) + "  " + str(sample_old2) + "  " +
            str(sample_old3) + "  " + str(X2_old) + "\n"
        )

    b = (j+1) % 100
    if b == 0:
        print(j+1)
