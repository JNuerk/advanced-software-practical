# imports
import time
import numpy as np
import tinyDA as tda
from tinyDA import Proposal
import pickle

from forward_model import Forward_Model


class Prior():
    """
    Base class for the prior distribution.
    
    Args: 
            dim (int): dimension of the distribution
            sig_pr (float): standard deviation.
    """
    def __init__(self,dim, sig_pr):
        self.dim = dim
        self.sig_pr = sig_pr
        
    def rvs(self):
        """ Get a sample from the dirstribution.
        Here this does not correspond to a sample because the formula is not
        known but the function is only needed if no initial parameter is given and
        for reading the dimension
        """
        return np.ones(self.dim)
    
    # the pdf and log pdf are both unnormalised
    def pdf(self,x):
        if x.shape[0]!=self.dim:
            raise ValueError('Dimension incorrect')
        return np.prod(np.exp(-1/(2*self.sig_pr**2)*np.log(x)))
    
    def logpdf(self,x):
        return -np.linalg.norm(np.log(x))**2/(2*self.sig_pr**2)
        
        
class Likelihood():
    """
    Likelihood function.
    
    Args:
            z (np.array): array of measured data.
            sig (float): standard deviation.
    """
    def __init__(self, z, sig):
        self.z = z
        self.sig = sig
    
    def logpdf(self,x):
        misfit = x - self.z
        return -np.dot(misfit,misfit)/(2*self.sig**2)
        
    
    
class BenchmarkProposal(Proposal):
    
    def __init__(self,dim, sig_prop):
        self.sig_prop = sig_prop
        self.dim = dim
    def setup_proposal(self, **kwargs):
        pass
    
    def adapt(self, **kwargs):
        pass
    
    def make_proposal(self, link):
        # draw a sample based on current state 
        xi = np.random.normal(scale=self.sig_prop,size=self.dim)
        return link.parameters*np.exp(xi)
        
    def get_acceptance(self, proposal_link, previous_link):
        if np.isnan(proposal_link.posterior):
            return 0
        else:
            # get the acceptance probability.
            return np.exp(proposal_link.posterior-previous_link.posterior)*np.prod(proposal_link.parameters/previous_link.parameters)
        
        
if __name__ == "__main__":
    
    #########################################################################
    ####### do all precomputations necessary for forward solver #############
    #########################################################################
    print('Doing precomputations for FEM solver...')
    # Read the mesh width
    n = int(input('Select denominator for meshwidth, e.g. for h=1/16 just enter 16:'))
    h = 1/n
    
    print('Initializing model and posterior...')
    #initialize the forward model
    my_model = Forward_Model(h)
    
    # define prior & Likelihood and parameters like in benchmark paper
    sig_pr = 2
    sig = 0.05
    sig_prop = 0.0725
    initial_parameters = np.ones(64)
    # load measurements
    z_measurement = np.loadtxt('z_hat.txt')
    my_prior = Prior(dim=initial_parameters.size, sig_pr=sig_pr)
    my_loglike = Likelihood(z_measurement, sig)
    my_posterior = tda.Posterior(my_prior, my_loglike, my_model)
    # define proposal
    my_proposal = BenchmarkProposal(dim=initial_parameters.size, sig_prop=sig_prop)

    # read inputs for makrov chains
    print('Please enter:')
    chain_length = int(input('Length of each Markov chain:'))
    number_of_chains = int(input(('Number of independent Markov chains:')))
    burnin = int(input('Burnin:'))
    # run the sampler and time it
    tic = time.time()
    my_chains = tda.sample(my_posterior, my_proposal, iterations=chain_length, 
                           n_chains=number_of_chains, initial_parameters = initial_parameters)
    toc = time.time()
    print(f'Time to compute the chains:{(toc-tic)/60} minutes' )    

    # write time data in seconds to txt file
    with open('MCMC_chains\TIME_MCMC_chains_length_%d_numChains_%d_h_1_%d.txt'%(chain_length,number_of_chains,n), 'w') as f:
        f.write('Time to compute the chains in seconds:%d'%(toc-tic))
    
    # convert to inference data and save it to disc
    idata = tda.to_inference_data(my_chains, burnin=burnin)
    array_dict = tda.get_samples(my_chains,burnin=burnin)
    with open('MCMC_chains\MCMC_chains_dict_length_%d_numChains_%d_h_1_%d.pkl'%(chain_length,number_of_chains,n), 'wb') as f:
        pickle.dump(array_dict, f)
    idata.to_json('MCMC_chains\MCMC_chains_length_%d_numChains_%d_h_1_%d.json'%(chain_length,number_of_chains,n))
    
    

