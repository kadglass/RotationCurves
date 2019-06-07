






from ..calculate_smoothness import calculate_smoothness

import numpy

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D




def CalculateAxesSpacing(num_rows, 
                          num_cols, 
                          top_margin=.1, 
                          bottom_margin=.1, 
                          left_margin=.1, 
                          right_margin=.1, 
                          subplot_percentage=.8,
                          subplot_halfgap_vertical=None,
                          subplot_halfgap_horizontal=None
                          ):
    '''
    
    '''
    available_vertical = 1.0 - top_margin - bottom_margin
    
    vertical_per_plot = available_vertical/float(num_rows)
    
    subplot_vertical = subplot_percentage*vertical_per_plot
    
    if subplot_halfgap_vertical is None:
        subplot_halfgap_vertical = (1.0 - subplot_percentage)*vertical_per_plot/2.0
    
    top = 1.0 - top_margin
    
    available_horizontal = 1.0 - left_margin - right_margin
    
    horizontal_per_plot = available_horizontal/float(num_cols)
    
    subplot_horizontal = subplot_percentage*horizontal_per_plot
    
    if subplot_halfgap_horizontal is None:
        subplot_halfgap_horizontal = (1.0 - subplot_percentage)*horizontal_per_plot/2.0
    
    left = left_margin + subplot_halfgap_horizontal
    
    #rect = left, bottom, width, height
    axes_list = []
    
    for col_num in range(num_cols):
    
        left = left_margin + col_num*horizontal_per_plot + subplot_halfgap_horizontal
    
        for row_num in range(num_rows):
        
            bottom = top - row_num*vertical_per_plot - subplot_halfgap_vertical - subplot_vertical
            
            width = subplot_horizontal
            
            height = subplot_vertical
            
            axes_list.append([left, bottom, width, height])
        
    return axes_list













def test():
    ################################################################################
    # Set up a data grid
    ################################################################################
    
    n_rows = 1000
    
    n_cols = 1000
    
    x_vals = numpy.linspace(0,10,n_cols)
    
    y_vals = numpy.linspace(0,10,n_rows)
    
    xx, yy = numpy.meshgrid(x_vals, y_vals)
    
    
    ################################################################################
    # Dataset 1
    ################################################################################
    
    def evaluate_surface(f, xx, yy, out=None):
        
        z_vals = f(xx, yy)
    
        gradients = numpy.gradient(z_vals)
        
        y_grads = gradients[0] #axis 0 is rows, which is y dimension
        
        x_grads = gradients[1] #axis 1 is columns, which is x direction
        
        score = calculate_smoothness(x_grads.astype(numpy.float32), 
                                     y_grads.astype(numpy.float32), 
                                     numpy.zeros(x_grads.shape, dtype=numpy.uint8), 
                                     n_rows, 
                                     n_cols,
                                     1e-13,
                                     out)
        
        return score
    
    
    ################################################################################
    # Dataset 2
    ################################################################################
    
    def surf1(xx, yy):
        
        return numpy.sin(xx) + numpy.cos(yy)
    
    def surf2(xx, yy):
        
        return numpy.sin(2.0*xx) + numpy.cos(yy)
    
    def surf3(xx, yy):
        
        return numpy.zeros(xx.shape, dtype=numpy.float32)
    
    def surf4(xx, yy):
        
        return numpy.random.normal(size=xx.shape)
    
    ################################################################################
    # Evaluate surfaces
    ################################################################################
    
    surfaces = [surf1, surf2, surf3, surf4]
    
    num_surfaces = len(surfaces)
    
    out_data = numpy.zeros(xx.shape, dtype=numpy.float32)
    
    scores = []
    
    dists = []
    
    for curr_surf in surfaces:
        
        out_data.fill(0.0)
        
        curr_score = evaluate_surface(curr_surf, xx, yy, out_data)
        
        scores.append(curr_score)
        
        dists.append(out_data.copy())
        
    #print(scores)
    
    fig = plt.figure(figsize=(16,10))
    
    axes_locs = CalculateAxesSpacing(num_surfaces, 
                                     2, 
                                     top_margin=.1, 
                                     bottom_margin=.1, 
                                     left_margin=.1, 
                                     right_margin=.1, 
                                     )
    
    axes_list = []
    for axes_loc in axes_locs:
        
        curr_axes = fig.add_axes(axes_loc, projection='3d')
        
        axes_list.append(curr_axes)
        
        
    for surf_idx in range(num_surfaces):
        
        left_axes = axes_list[surf_idx]
        right_axes = axes_list[surf_idx+num_surfaces]
        
        left_axes.plot_surface(xx, yy, surfaces[surf_idx](xx,yy), cmap='coolwarm')
        
        right_axes.plot_surface(xx, yy, dists[surf_idx], cmap='coolwarm')
        
        right_axes.set_title("Score: " + str(scores[surf_idx]))
        
    plt.show()
    
    
    
    
    
if __name__ == "__main__":
    
    test()


