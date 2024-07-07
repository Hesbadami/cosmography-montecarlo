import numpy as np
from numba import njit

def H(z, params):
    return z

def dH(z, params):
    return z

def Xi2(z, params, y_train):
    sum(
        (H(z[i], params) - y_train[i])**2 / error[i]
    ) / 2*m

def dXi2(z, params, y_train):
    sum(
        (H(z[i], params) - y_train[i])*dH_domega(z[i], params) / error[i]
    ) / m

def m(beta1, m_old, H):
    beta1 * m_old + (1 - beta1) * H

def v(beta2, v_old, H):
    beta2 * v_old + (1 - beta2) * H**2

def Adam(params, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-07, batch_size=32, epochs=1):
    params = [0, 0]
    m = [0., 0.]
    v = [0., 0.]

    for i in range(2):
        m[i] = beta1* m[i] + (1 - beta1) * 
