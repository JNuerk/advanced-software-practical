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
    
    # Initialize the forward models
    print('Initializing model and posterior...')
    #initialize the forward model
    my_model_l0 = Forward_Model(h=1/8)
    my_model_l1 = Forward_Model(h=1/16)
    my_model_l2 = Forward_Model(h=1/32)
    
    # define prior & Likelihood and parameters like in benchmark paper
    sig_pr = 2
    sig = 0.05
    sig_prop = 0.0725
    initial_parameters = np.ones(64)
    # load measurements
    z_measurement = np.loadtxt('z_hat.txt')
    my_prior = Prior(dim=initial_parameters.size, sig_pr=sig_pr)
    my_loglike = Likelihood(z_measurement, sig)
    # define hirarchy of posteriors
    my_posterior_l0 = tda.Posterior(my_prior, my_loglike, my_model_l0)
    my_posterior_l1 = tda.Posterior(my_prior, my_loglike, my_model_l1)
    my_posterior_l2 = tda.Posterior(my_prior, my_loglike, my_model_l2)

    # read inputs for makrov chains
    print('Please enter:')
    l = input('Enter model levels (at least two), comma separated, starting with coarsest level, allowed values are 0,1,2:')
    levels = list(map(int, l.split(',')))
    levels_str = '_'.join(map(str, levels)) # for writing in file name
    possible_posteriors = [my_posterior_l0, my_posterior_l1,  my_posterior_l2] # list of all available posteriors
    my_posteriors = [possible_posteriors[i] for i in levels] # posteriors that are actually used in sampling

    # define proposal
    my_proposal = BenchmarkProposal(dim=initial_parameters.size, sig_prop=sig_prop)

    # read inputs for makrov chains
    print('Please enter:')
    chain_length = int(input('Length of each Markov chain:'))
    number_of_chains = int(input(('Number of independent Markov chains:')))
    burnin = int(input('Burnin:'))

    # input subsampling rate or list of rates for more levels
    if len(my_posteriors) == 2:
        sub_rate = int(input('Subsampling rate:'))
        sub_rate_str = str(sub_rate)
    else:
        s = input('Enter subampling rates, comma separated, starting with rate for coarsest level:')
        sub_rate = list(map(int, s.split(',')))
        sub_rate_str = '_'.join(map(str, sub_rate))

    # run the sampler and time it
    tic = time.time()
    my_chains = tda.sample(my_posteriors, my_proposal, iterations=chain_length, 
                           n_chains=number_of_chains, initial_parameters = initial_parameters, 
                           subsampling_rate=sub_rate,
                           force_sequential=True)
    toc = time.time()
    print(f'Time to compute the chains:{(toc-tic)/60} minutes' )    
    # write time data in seconds to txt file
    with open('MLDA_chains\TIME_MLDA_chains_length_%d_numChains_%d_subrates_%s_levels_%s.txt'%(chain_length,number_of_chains,sub_rate_str,levels_str), 'w') as f:
        f.write('Time to compute the chains in seconds:%d'%(toc-tic))

    # convert to inference data and dictionary with numpy array
    if my_chains['sampler'] == "DA":
        idata = tda.to_inference_data(my_chains, level='fine', burnin=burnin)
        array_dict = tda.get_samples(my_chains, level='fine', burnin=burnin)
    elif my_chains['sampler'] == 'MLDA':
        idata = tda.to_inference_data(my_chains, level=2, burnin=burnin)
        array_dict = tda.get_samples(my_chains,level=2, burnin=burnin)
    else:
        print('There went something wrong while storing the chains, check levels in your model and selected level in tinyDA for extracting values.')

    # store data to disc
    idata.to_json('MLDA_chains\MLDA_chains_length_%d_numChains_%d_subrates_%s_levels_%s.json'%(chain_length,number_of_chains,sub_rate_str,levels_str))

    with open('MLDA_chains\MLDA_chains_dict_length_%d_numChains_%d_subrates_%s_levels_%s.pkl'%(chain_length,number_of_chains,sub_rate_str, levels_str), 'wb') as f:
        pickle.dump(array_dict, f)
    
    

