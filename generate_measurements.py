# this script generates synthetic measurements for lower dimensional parameter inputs theta 

# imports
import numpy as np
from forward_model import Forward_Model as fm

# plots FEM solution and measurement operator applied to solution for different gridsizes.

# meshsize 
N = 256
# initialize forward model 
fm = fm(1/N)


theta_16 = np.ones(16) # 'true' parameter that generated the measurement in 16 dim case
theta_16[5] = 0.1
theta_16[10] = 10

theta_4 = np.ones(4) # 'true' parameter that generated the measurement in 4 dimensional case
theta_4[0] = 0.1
theta_4[3] = 10

z_16, u_16 = fm(theta_16, return_coefficients=True)
z_4, u_4 = fm(theta_4, return_coefficients=True)

# store values
np.save('z_hat_16', z_16)
np.save('z_hat_4', z_4)

