

import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D





x_vals = numpy.linspace(0,10,1000)

y_vals = numpy.linspace(0,10,1000)

xx, yy = numpy.meshgrid(x_vals, y_vals)

z_vals = numpy.sin(xx) + numpy.cos(yy)


fig = plt.figure(figsize=(16,10))
axes = fig.add_axes([.1,.1,.8,.8], projection='3d')
axes.plot_surface(xx, yy, z_vals, cmap='coolwarm')
plt.show()