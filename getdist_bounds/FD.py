import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

data = np.loadtxt('4d.txt')

data = data[:, :2]

q = data[:, 0]  # np.genfromtxt('xdata.txt')
j = data[:, 1]  # np.genfromtxt('ydata.txt')

qmin, jmin = data.min(axis=0)
qmax, jmax = data.max(axis=0)

Q, J = np.mgrid[qmin:qmax:100j, jmin:jmax:100j]

positions = np.vstack([Q.ravel(), J.ravel()])
values = data.T

kernel = stats.gaussian_kde(values)

Z = np.reshape(kernel(positions).T, Q.shape)
norm = Z.sum()

levels = [0.99*norm, 0.95*norm, 0.68*norm]

fig, ax = plt.subplots()
# ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
#           extent=[qmin, qmax, jmin, jmax], aspect='auto')

# ax.contourf(Q, J, Z, levels)

# ax.set_xlim([qmin, qmax])
# ax.set_ylim([jmin, jmax])

# plt.show()
