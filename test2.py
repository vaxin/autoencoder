import numpy as np
# Import these to display the result
import matplotlib.pyplot as plt
import matplotlib.cm as cm		
 
vals = np.linspace(-np.pi,np.pi,100)	
# vals.shape : (100,)
xgrid, ygrid = np.meshgrid(vals,vals)		
# xgrid.shape : (100,100)		
# ygrid.shape : (100,100)
 
# Simple Gaussian : mean = 0, std = 1, amplitude = 1 
the_gaussian = np.exp(-(xgrid/2)**2-(ygrid/2)**2)
 
# Simple sine wave grating : orientation = 0, phase = 0, amplitude = 1, frequency = 10/(2*pi) 
the_sine = np.sin(xgrid * 10)	
 
# Elementwise multiplication of Gaussian and sine wave grating   
#the_gabor = the_gaussian * the_sine	 
 
# Plot every pixel value, using a gray colormap    
plt.imshow(the_sine,cm.gray) 		
 
# Display the result    
plt.show()				
