import numpy as np
from GPR import gpr
import matplotlib.pyplot as plt
plt.style.use('bmh')


# Import data
data = np.genfromtxt('co2.txt')
t = data[:, 2]
co2 = data[:, 3]


x = t
y = co2

hp = np.array([66, 67, 2.4, 90, 1.3, 0.66, 1.2, 0.78, 0.18, 1.6, 0.19])

x_predict = np.arange(1958.2027, 2030, 0.084)

model = gpr(x, y, x_predict, hp)


def plot_GPR(data_x, data_y, model, predict_x):
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
        predict_x, mean, color='blue', linewidth=1, label='Predicted function'
    )
    plt.scatter(
        data_x, data_y, s=1, label='Observed points',
    )
    plt.legend()
    plt.show()


plot_GPR(x, y, model, x_predict)
