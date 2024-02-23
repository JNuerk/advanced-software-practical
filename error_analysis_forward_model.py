# Test the forward model against input/output pairs. 
# The reference solutions come from a C++ implementation

# imports 
import numpy as np
from forward_model import Forward_Model


if __name__ == '__main__':

    print('Please enter:')
    N = int(input('FEM meshwidth (enter denominator i.e. for h=1/16 just enter 16):'))
    h = 1/N
    my_model = Forward_Model(h)
    print(f'Starting model test with meshwidth h=1/{int(1/h)}:')
    for i in range(10):
        print('Verifiying against data set', i)
        
        # Read the input vector
        f_input = open ("testing/input.{}.txt".format(i), 'r')
        theta = np.fromfile(f_input, count=64, sep=" ")
        
        this_z = my_model(theta)
        f_output_z = open ("testing/output.{}.z.txt".format(i), 'r')
        reference_z = np.fromfile(f_output_z, count=13**2, sep=" ")
        print ("  || z-z_ref ||  : ", np.linalg.norm(this_z - reference_z))

        
    


