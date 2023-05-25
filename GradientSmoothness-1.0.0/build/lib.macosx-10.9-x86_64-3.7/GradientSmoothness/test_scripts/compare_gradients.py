
#import sys
#sys.path.insert(1, "/home/oneills2/.eclipse-workspace/GradientSmoothness")

from calculate_smoothness import calculate_smoothness

import numpy
from scipy.spatial.distance import cosine as cosine_dist
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import time

x_vals = numpy.linspace(0,10,1000)

y_vals = numpy.linspace(0,10,1000)

xx, yy = numpy.meshgrid(x_vals, y_vals)

z_vals = numpy.sin(2.0*xx) + numpy.cos(yy)

gradients = numpy.gradient(z_vals)

y_grads = gradients[0] #axis 0 is rows, which is y dimension

x_grads = gradients[1] #axis 1 is columns, which is x direction




n_rows, n_cols = z_vals.shape

out_dists = numpy.empty((n_rows, n_cols), dtype=numpy.float64)


start_time = time.time()
score = calculate_smoothness(x_grads.astype(numpy.float32), y_grads.astype(numpy.float32), numpy.zeros(x_grads.shape, dtype=numpy.uint8), x_grads.shape[0], x_grads.shape[1])
cython_time = time.time() - start_time

print("Score: ", score, "Cython time: ", cython_time)


start_time = time.time()

for row_idx in range(n_rows):
    
    if row_idx%100 == 0:
        print(row_idx)
    
    for col_idx in range(n_cols):
        
        
        
        curr_grad = (x_grads[row_idx, col_idx], y_grads[row_idx, col_idx])
        
        if col_idx > 0:
            left_grad = (x_grads[row_idx, col_idx - 1], y_grads[row_idx, col_idx - 1])
            left_dist = cosine_dist(curr_grad, left_grad)
        else:
            left_dist = 0.0
            
            
        if col_idx < (n_cols - 1):
            right_grad = (x_grads[row_idx, col_idx + 1], y_grads[row_idx, col_idx + 1])
            right_dist = cosine_dist(curr_grad, right_grad)
        else:
            right_dist = 0.0
            
            
        if row_idx > 0:
            top_grad = (x_grads[row_idx - 1, col_idx], y_grads[row_idx - 1, col_idx])
            top_dist = cosine_dist(curr_grad, top_grad)
        else:
            top_dist = 0.0
            
            
        if row_idx < (n_rows - 1):
            bot_grad = (x_grads[row_idx + 1, col_idx], y_grads[row_idx + 1, col_idx])
            bot_dist = cosine_dist(curr_grad, bot_grad)
        else:
            bot_dist = 0.0
            
        
        
        
        '''
        print("-----")
        print(row_idx, col_idx)
        print("Curr gradient: ", curr_grad)
        print("top: ", top_grad, top_dist)
        print("bot: ", bot_grad, bot_dist)
        print("left: ", left_grad, left_dist)
        print("right: ", right_grad, right_dist)
        
        #exit()
        '''
        
        curr_dist = top_dist + bot_dist + left_dist + right_dist
        
        out_dists[row_idx, col_idx] = curr_dist
        
python_time = time.time() - start_time
        
        
print("Smoothness score: ", numpy.sum(out_dists), "Python time: ", python_time)




fig = plt.figure(figsize=(16,10))
axes_data = fig.add_axes([.1,.1,.4,.8], projection='3d')
axes_dist = fig.add_axes([.5,.1,.4,.8], projection='3d')
axes_data.plot_surface(xx, yy, z_vals, cmap='coolwarm')
axes_dist.plot_surface(xx, yy, out_dists, cmap='coolwarm')
plt.show()
        
        
        