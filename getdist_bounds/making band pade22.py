import numpy as np
from numba import njit
#-----------------------------------------------------------------------------
x= np.genfromtxt('pade22.txt')
q= x[:,2]
j= x[:,3]
s= x[:,4]
l= x[:,5]


# cond = (qq>(-0.552-0.14)) * (qq < (-0.552+0.13)) \
#             * (jj>(1.31-0.90)) * (jj<(1.31+0.92)) \
#             * (ss>(1.3-2.1)) * (ss<(1.3+1.8)) \
#             * (ll>(2.9-3.9)) * (ll<(2.9+3.3))
#
# q = qq[cond]
# j = jj[cond]
# s = ss[cond]
# l = ll[cond]

#------------------------------------------------------------------------------
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
    mu0= 42.384 - 5*np.log10(0.7)
    muu = 5*np.log10((1+z)*E_int[-1]) + mu0
    return muu

f= open ("pantheon_pade22_band_2sigma.txt", "w")
z= np.linspace(0.01, 2.5, 100)

for i in range(len(z)):
    A=[]
    for k in range(len(q)):
        param= np.array([q[k], j[k], s[k], l[k]])
        A.append(mu(z[i], param))
    f.write(str(z[i]) + "  " + str(np.mean(A)+np.std(A)) + "  " + str(np.mean(A)-np.std(A)) + "\n")
f.close()

