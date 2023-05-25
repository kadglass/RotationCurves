


import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x_vals = numpy.linspace(0,10,1000)

y_vals = numpy.linspace(0,10,1000)

xx, yy = numpy.meshgrid(x_vals, y_vals)

z_vals = numpy.sin(xx) + numpy.cos(yy)

gradients = numpy.gradient(z_vals)

y_grads = gradients[0] #axis 0 is rows, which is y dimension

x_grads = gradients[1] #axis 1 is columns, which is x direction


dim = 2

fig = plt.figure(figsize=(16,10))

if dim == 3:

    axes_data = fig.add_axes([.025,.1,.275,.8], projection='3d')
    
    axes_x_grad = fig.add_axes([.35,.1,.275,.8], projection='3d')
    
    axes_y_grad = fig.add_axes([.675,.1,.275,.8], projection='3d')
    
    axes_data.plot_surface(xx, yy, z_vals, cmap='coolwarm')
    
    axes_x_grad.plot_surface(xx, yy, x_grads, cmap='coolwarm')
    
    axes_y_grad.plot_surface(xx, yy, y_grads, cmap='coolwarm')
    
elif dim == 2:
    
    axes_data = fig.add_axes([.025,.1,.275,.8])
    
    axes_x_grad = fig.add_axes([.35,.1,.275,.8])
    
    axes_y_grad = fig.add_axes([.675,.1,.275,.8])
    
    axes_data.contour(xx, yy, z_vals, 20, cmap='coolwarm')
    
    axes_x_grad.contour(xx, yy, x_grads, 20, cmap='coolwarm')
    
    axes_y_grad.contour(xx, yy, y_grads, 20, cmap='coolwarm')

axes_data.set_title("Input Data")
axes_x_grad.set_title("X gradients")
axes_y_grad.set_title("Y gradients")

plt.show()


