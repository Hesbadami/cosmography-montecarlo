import numpy as np
from GPR import gpr_zhang as gpr
import matplotlib.pyplot as plt
plt.style.use('bmh')


# Import data
data = np.genfromtxt('hubble.dat')
z = data[:, 0]
H = data[:, 1]
e = data[:, 2]

x = z
y = H
noise = np.var(e)

x_predict = np.arange(0, 2.5, 0.1)

model = gpr(x, y, x_predict, noise, np.mean(y), 1)


def plot_GPR(data_x, data_y, data_error, model, predict_x):
    mean = model[0]
    std = np.sqrt(model[2])

    alpha = (0.2, 0.1, 0.05)
    for i in range(1, 4):
        x_lines = predict_x
        y_lower = mean - i * std
        y_upper = mean + i * std
        label = f"{i}sigma confidence region"
        plt.fill_between(
            x_lines, y_lower, y_upper, color='r', alpha=alpha[i-1], label=label
        )

    plt.plot(
        predict_x, mean, color='blue', linewidth=2, label='Predicted function'
    )
    plt.errorbar(
        data_x, data_y, yerr=data_error, fmt='.', label='Observed points',
        capsize=2, ecolor='r', elinewidth=0.9, ms=8
    )
    plt.legend()
    plt.show()


plot_GPR(x, y, e, model, x_predict)
