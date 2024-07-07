import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes,mark_inset
#-----------------------------------------------------------------------------
def E_model(z, param):
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

def X_model(z, param):
    I= quad(E_model, 0, z, args=param)[0]
    return I

def mu_model(z, param):
    mu0= 42.384 - 5*np.log10 (0.7)
    muu= 5*np.log10((1+z)*X_model(z, param)) + mu0
    return muu

def E_LCDM(z, param):
    EE= np.sqrt(param[0]*pow(1+z,3) + (1-param[0]))
    return 1/EE

def X_LCDM(z, param):
    I= quad(E_LCDM, 0, z, args=param)[0]
    return I

def mu_LCDM(z, param):
    mu0= 42.384 - 5*np.log10 (0.7)
    muu= 5*np.log10((1+z)*X_LCDM(z, param)) + mu0
    return muu
#-----------------------------------------------------------------------------
x1= np.genfromtxt('mudata.txt')
z1= x1[:,0]
mu1_max= x1[:,1]
mu1_min= x1[:,2]

SN_pantheon=np.genfromtxt('mock_pantheon.txt')
Z1=SN_pantheon[:,0]
muo1=SN_pantheon[:,1]
error1=SN_pantheon[:,2]
#-------------------------------------------------------------------------------
z= np.linspace(0.01, 2.5, 601)
param1= [-0.552, 1.31, 1.3, 2.9] #cosmographic
param2= [0.3] #LCDM

A= [] #cosmography
B= [] #LCDM
for i in range(len(z)):
    A.append(mu_model(z[i], param1))
    B.append(mu_LCDM(z[i], param2))

C= [] #delta LCDM
D= [] # cosmo - LCDM
for i in range(len(A)):
    C.append(B[i]-B[i])
    D.append(((A[i]-B[i])/B[i])*100)

E1= []
E2= []
for i in range(len(x1)):
    E1.append(((mu1_max[i]-mu_LCDM(z1[i], param2))/mu_LCDM(z1[i], param2))*100)
    E2.append(((mu1_min[i]-mu_LCDM(z1[i], param2))/mu_LCDM(z1[i], param2))*100)

GG = gridspec.GridSpec(6, 1)
ax= plt.subplot(GG[0:5,0])
plt.subplots_adjust(hspace=0)
plt.plot(z, A, 'b', label='Pade (2,2) Cosmography', linewidth=0.5, zorder= 3)
plt.plot(z, B, 'r', label='$\Lambda$CDM model', linewidth=0.5, zorder= 3)
plt.fill_between(z1, mu1_max, mu1_min, facecolor='blue', alpha=0.3, zorder= 3)
kwargs = dict(color='g', capsize=1, elinewidth=1, linewidth=2, ms=4)
plt.errorbar(Z1, muo1, yerr=error1, fmt='.', alpha=0.7, **kwargs, label='mock data Pantheon')
plt.xticks(alpha=0)
plt.ylabel ("$\mu$(z)")
plt.xscale('log')
plt.legend(loc= 'lower right')
plt.yticks([34, 36, 38, 40, 42, 44, 46])
#plt.title('E')
axins = zoomed_inset_axes(ax, 2, loc=2)
axins.plot(z, A, 'b', linewidth=0.5, zorder= 3)
axins.plot(z, B, 'r', linewidth=0.5, zorder= 3)
axins.fill_between(z1, mu1_max, mu1_min, facecolor='blue', alpha=0.3, zorder= 3)
axins.errorbar(Z1, muo1, yerr=error1, fmt='.', alpha=0.7, **kwargs, label='mock data Pantheon')
axins.set_xlim(1, 2.6)
axins.set_ylim(44, 48)
plt.xticks([1.2, 1.5, 1.8, 2.1, 2.4])
plt.yticks(visible=False)
mark_inset(ax,axins,loc1=1,loc2=3, ec="k", lw=0.5)

plt.subplot(GG[-1,0])
plt.subplots_adjust(hspace=0)
#plt.plot(z, C, 'k:', label='$\Lambda$CDM')
plt.plot(z, D, 'b', label='Cosmography method', linewidth=1)
plt.fill_between(z1, E1, E2, facecolor='blue', alpha=0.3)
plt.xlabel ("log z")
plt.ylabel ("Residuals(%)")
plt.axhline(y= 0.25, color='grey', linestyle=':', linewidth=1)
plt.ylim(-0.5, 1.2)
plt.yticks([0.25])
plt.xscale('log')
plt.savefig('residual_pade22_new.pdf')
plt.show()
print(D[-1])




