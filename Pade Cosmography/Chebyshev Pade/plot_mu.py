import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

data1= np.genfromtxt('mock_pantheon.txt')
z1= data1[:, 0]
mu1= data1[:, 1]
e1= data1[:, 2]

def E_LCDM(z, param):
    EE= np.sqrt(param[0]*pow(1+z,3) + (1-param[0]))
    return 1/EE

def E_Taylor4(z, param):
    y= z/(1+z)
    k1= 1 + param[0]
    k2= 2 - pow(param[0],2) + 2*param[0] + param[1]
    k3= 6 + 3*pow(param[0],3) - 3*pow(param[0],2) + 6*param[0] - 4*param[0]*param[1] + 3*param[1] - param[2]
    k4= - 15*pow(param[0],4) + 12*pow(param[0],3) + 25*param[1]*pow(param[0],2) + 7*param[0]*param[2] - 4*pow(param[1],2) - 16*param[0]*param[1] - 12*pow(param[0],2) + param[3] - 4*param[2] + 12*param[1] + 24*param[0] + 24
    EE= 1 + k1*y + (1/2)*k2*pow(y,2) + (1/6)*k3*pow(y,3) + (1/24)*k4*pow(y,4)
    return 1/EE

def E_Pade22(z, param):
    H1= 1 + param[0]
    H2= -pow(param[0],2) + param[1]
    H3= 3*pow(param[0],2)*(1+param[0]) - param[1]*(3+4*param[0]) - param[2]
    H4= -3*pow(param[0],2)*(4+8*param[0]+5*pow(param[0],2)) + param[1]*(12+32*param[0]+25*pow(param[0],2)-4*param[1]) + param[2]*(8+7*param[0]) + param[3]

    Q1= (-6*H1*H4 + 12*H2*H3)/(24*H1*H3 - 36*H2**2)
    Q2= (3*H2*H4 - 4*H3**2)/(24*H1*H3 - 36*H2**2)
    P0= 1
    P1= H1 + Q1
    P2= H2/2 + Q1*H1 + Q2

    EE= (P0 + P1*z + P2*z**2)/(1 + Q1*z + Q2*z**2)
    return 1/EE

def E_cheb(z, param):
    H_1 = 1 + param[0]
    H_2 = -pow(param[0], 2) + param[1]
    H_3 = 3*pow(param[0], 2)*(1+param[0]) - param[1]*(3+4*param[0]) - param[2]
    H_4 = -3*pow(param[0], 2)*(4+8*param[0]+5*pow(param[0], 2)) + \
        param[1] * (12+32*param[0]+25*pow(param[0], 2)-4*param[1]) + \
        param[2] * (8+7*param[0]) + param[3]

    c_0 = H_2/4 + H_4/64 + 1
    c_1 = H_1 + H_3/8
    c_2 = H_2/4 + H_4/48
    c_3 = H_3/24
    c_4 = H_4/192

    T_0 = 1
    T_1 = z
    T_2 = 2*z**2 - 1
    T_3 = 4*z**3 - 3*z
    T_4 = 8*z**4 - 8*z**2 + 1

    EE = c_0 + c_1*T_1 + c_2*T_2 + c_3*T_3 + c_4*T_4
    return 1/EE

def E_cheb_Pade22(z, param):
    H_1 = 1 + param[0]
    H_2 = -pow(param[0], 2) + param[1]
    H_3 = 3*pow(param[0], 2)*(1+param[0]) - param[1]*(3+4*param[0]) - param[2]
    H_4 = -3*pow(param[0], 2)*(4+8*param[0]+5*pow(param[0], 2)) + \
        param[1] * (12+32*param[0]+25*pow(param[0], 2)-4*param[1]) + \
        param[2] * (8+7*param[0]) + param[3]

    c_0 = H_2/4 + H_4/64 + 1
    c_1 = H_1 + H_3/8
    c_2 = H_2/4 + H_4/48
    c_3 = H_3/24
    c_4 = H_4/192

    T_1 = z
    T_2 = 2*z**2 - 1

    B_1 = (-2*c_1*c_4 + 2*c_2*c_3 - 2*c_3*c_4) / \
        (c_1*c_3 - c_2**2 + 10*c_2*c_4 - 5*c_3**2 - 24*c_4**2)
    B_2 = (2*c_2*c_4 - 2*c_3**2 - 8*c_4**2) / \
        (c_1*c_3 - c_2**2 + 10*c_2*c_4 - 5*c_3**2 - 24*c_4**2)
    A_0 = B_1*c_1 / 2 - 3*B_1*c_3/2 - B_2*c_2 + 4*B_2*c_4 + c_0 - 3*c_4
    A_1 = B_1*c_1/2 - 3*B_1*c_3/2 + B_2*c_0 - 2*B_2*c_2 + 5*B_2*c_4 + \
        c_2 - 4*c_4
    A_2 = B_1*c_0 - B_1*c_2 + B_1*c_4 - B_2*c_1 + 3*B_2*c_3 + c_1 - 3*c_3

    EE = (A_0 + A_1*T_1 + A_2*T_2) / (1 + B_1*T_1 + B_2*T_2)
    return 1/EE

def mu(z, param, E):
    mu0= 42.384 - 5*np.log10(0.7)
    muu= 5*np.log10((1+z)*quad(E, 0, z, args=param)[0]) + mu0
    return muu

z= np.linspace(0.01, 2.5)
param1= [0.3]
param2 = [-0.55, 1, -0.35, 3.11]

A = [mu(zz, param1,  E_LCDM) for zz in z]
B = [mu(zz, param2,  E_Taylor4) for zz in z]
C = [mu(zz, param2,  E_Pade22) for zz in z]
D = [mu(zz, param2,  E_cheb) for zz in z]
F = [mu(zz, param2,  E_cheb_Pade22) for zz in z]

plt.plot(z, A, color='b', linewidth=0.9, zorder= 3, label='$\Lambda$CDM')
plt.plot(z, B, color='r', linewidth=0.9, zorder= 3, label='Taylor 4')
plt.plot(z, C, color='g', linewidth=0.9, zorder= 3, label='Pade22')
plt.plot(z, D, color='m', linewidth=0.9, zorder= 3, label='Chebyshev4')
plt.plot(z, F, color='k', linewidth=0.9, zorder= 3, label='Chebyshev Pade22')
kwargs = dict(color='y', capsize=1, elinewidth=0.7, linewidth=1.5, ms=4)
plt.errorbar(z1, mu1, yerr= e1, fmt='y.', alpha=0.7, **kwargs, label='Observational data')
plt.xlabel('z')
plt.ylabel('$\mu$(z)')
plt.legend(loc= 4, prop={'size':11})
plt.savefig('mu.pdf')
plt.show()
