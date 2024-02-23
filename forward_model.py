# imports
import numpy as np
from scipy.sparse.linalg import spsolve
import scipy.io

class Forward_Model():
    """Defines the forward model.
    
    Given an 2x2, 4x4 or 8x8 (dimensionality of benchmark) stiffness matrix, the forward model computes the solution z 
    of the poisson equation on a uniform grid on the domain [0,1]^2
    and evaluates it at measurement points on a 13x13 grid.
    
    Args:   h (float): meshwidth of the FE-grid, allowed values are of the form 1/(2^n) for a natural number n.
            inputs_positive (Bool): defaul is True which assumes that only positive inputs are passed into the model.
                                    If False exp(input) is passed to the model.
    Attribute: The following attributes of the class are precomputed since they only depend on the meshwidth h:
                M (ndarray): measurement matrix/operator
                A_loc (ndarray): matrix for local contributions
                b (ndarray): RHS for linear FEM solver
                boundaries (list): locate boundary labels
    """
    def __init__(self, h, inputs_positive=True):
        
        self.h = h
        self.inputs_pos = inputs_positive
        # check for allowed values of h 
        if not (np.log2(1/h)%1 == 0):
            raise ValueError('Meshwidth h must be of the form 1/(2^n) for a natural number n.')
        
        if self.inputs_pos:
            print('Input values to forward model are assumed to be positive.')
        else:
            print('Negative inputs to forward model are allowed; exp(input) is passed to the model.')
        
        # Construct measurement matrix, M, for measurements
        xs = np.arange(1./14,13./14,1./14);    # measurement points

        M = np.zeros((13,13,(int(1/self.h)+1)**2))
        for k in range((int(1/self.h)+1)**2) :
            c = self.inv_ij_to_dof_index(k)
            for i in range(13) :
                for j in range(13) :
                    M[i,j,k] = self.phi(xs[i]-self.h*c[0], xs[j]-self.h*c[1])
        M = M.reshape((13**2, (int(1/h)+1)**2))
        M = scipy.sparse.csr_matrix(M)
        self.M = M
        
        # constract local overlap matrix
        self.A_loc = np.array([[2./3,  -1./6,  -1./3,  -1./6],
                               [-1./6,  2./3,  -1./6,  -1./3],
                               [-1./3, -1./6,   2./3,  -1./6],
                               [-1./6, -1./3,  -1./6,   2./3]])
        
        # construct boundary conditions
        self.boundaries = ([self.ij_to_dof_index(i,0) for i in range(int(1/self.h)+1)] +
                           [self.ij_to_dof_index(i,int(1/self.h)) for i in range(int(1/self.h)+1)] +
                           [self.ij_to_dof_index(0,j+1) for j in range(int(1/self.h)-1)] +
                           [self.ij_to_dof_index(int(1/self.h),j+1) for j in range(int(1/self.h)-1)])
        self.b = np.ones((int(1/self.h)+1)**2)*10*self.h**2
        self.b[self.boundaries] = 0

    # Define characteristic function of unit square
    def heaviside(self, x) :
        if x<0 :
            return 0
        else :
            return 1
        
    def S(self, x,y) :
        return self.heaviside(x)*self.heaviside(y) * (1-self.heaviside(x-self.h))*(1-self.heaviside(y-self.h))

    # Define tent function on the domain [0,2h]x[0,2h]
    def phi(self,x,y) :
        return ((x+self.h)*(y+self.h)*self.S(x+self.h,y+self.h) + (self.h-x)*(self.h-y)*self.S(x,y) 
                + (x+self.h)*(self.h-y)*self.S(x+self.h,y) + (self.h-x)*(y+self.h)*self.S(x,y+self.h))/self.h**2
    
    # Define conversion function for dof's from 2D to scalar label, and
    # its inverse
    def ij_to_dof_index(self,i,j) :
        return (int(1/self.h)+1)*j+i

    def inv_ij_to_dof_index(self,k) :
        return [k-(int(1/self.h)+1)*int(k/(int(1/self.h)+1)),int(k/(int(1/self.h)+1))]


        
    def __call__(self, theta, return_coefficients=False):
        """Given stiffness parameter, returns the FEM solution of the poisson equation.

        Arguments
        ---------

        theta : list or ndarray of length 64, 16 or 4.
                Parameter values of stiffness field.
        return_coefficients : bool, optional
                Default False, if True the values of the FEM solution evaluated at the 
                measurement points and the coefficients of the FEM solution are returned.

        Returns
        -------
        ndarray or tuple
                Values of FEM-solution evaluated at measurement points.
        """
        # Initialize matrix A for FEM linear solve, AU = b
        A = np.zeros(((int(1/self.h)+1)**2,(int(1/self.h)+1)**2))
        
        # check dimensionality of input theta
        if np.sqrt(len(theta)) != 8 and np.sqrt(len(theta)) != 4 and np.sqrt(len(theta)) != 2:
            raise ValueError('Stiffness field theta must be of lenght: 4, 16 or 64 ')

        # check if input is positive or not.
        if not self.inputs_pos:
            theta = np.exp(theta)

        # Build A by summing over contribution from each cell
        sqrt_dim_theta = int(np.sqrt(len(theta))) # length of stiffness field in x direction (same for y direction).
        t = int(1/(self.h*sqrt_dim_theta)) # multiple of FE grid width in parameter space grid
        for i in range(int(1/self.h)) :
            for j in range (int(1/self.h)) :
                # Find local coefficient in 8x8 grid
                theta_loc = theta[int(i/t)+int(j/t)*sqrt_dim_theta]

                # Update A by including contribution from cell (i,j)
                dof = [self.ij_to_dof_index(i,j),
                       self.ij_to_dof_index(i,j+1),
                       self.ij_to_dof_index(i+1,j+1),
                       self.ij_to_dof_index(i+1,j)]
                A[np.ix_(dof,dof)] += theta_loc * self.A_loc

        # Enforce boundary condition: Zero out rows and columns, then
        # put a one back into the diagonal entries.
        A[self.boundaries,:] = 0
        A[:,self.boundaries] = 0
        A[self.boundaries,self.boundaries] = 1

        # Solve linear equation for coefficients, U, and then
        # get the Z vector by multiplying by the measurement matrix
        u = spsolve(scipy.sparse.csr_matrix(A), self.b)
        
        z = self.M * u
        
        if return_coefficients:
            return z, u # return both FEM solution values at measurement points and FEM coefficients.
        else:
            return z # only return FEM solution evaluated at measurement points.

