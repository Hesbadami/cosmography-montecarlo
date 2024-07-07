import matplotlib.pyplot as plt
from getdist import plots
from getdist import loadMCSamples
plt.style.use(['science', 'grid'])

sample = loadMCSamples(
    '/home/sesh/Code/getdist_bounds/pade22'
)

#sample.updateSettings({'contours': [0.68, 0.95, 0.99]})

g = plots.getSinglePlotter(
    width_inch=7,
    chain_dir='/home/sesh/Code/getdist_bounds',
    analysis_settings={
        'ignore_rows': 0
    },
)


g.settings.num_plot_contours = 3
g.plot_2d(
    'pade22',
    ['x0', 'x1'],
    filled=True
    # line_args=[{'ls':'-', 'color':'darkred'}],
)

plt.show()
