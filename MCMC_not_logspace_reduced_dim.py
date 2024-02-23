# imports
import time
import numpy as np
import tinyDA as tda
import pickle
import scipy.stats as stats

from forward_model import Forward_Model

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
    my_model = Forward_Model(h, inputs_positive=False) # we allow for arbitrary inputs since we pass exp(inputs) to the model

    
    # define prior & Likelihood and parameters like in benchmark paper
    sig_pr = 2
    sig = 0.05
    sig_prop = 0.0725
    # dimensionality of input parameters theta 
    input_dim = 16
    initial_parameters = np.zeros(input_dim)  # since x = log(theta)
    # load measurements
    z_measurement = np.load('z_hat_%s.npy'%input_dim) 

    # define the prior distribution
    my_prior = stats.multivariate_normal(mean=np.zeros(input_dim), cov=sig_pr**2)

    # define the loglikelihood function
    cov_likelihood = sig**2*np.eye(z_measurement.size)
    my_loglike = tda.GaussianLogLike(z_measurement, cov_likelihood)
    my_posterior = tda.Posterior(my_prior, my_loglike, my_model)

    # define the proposal distribution
    cov_proposal = sig_prop**2*np.eye(input_dim)
    my_proposal = tda.AdaptiveMetropolis(cov_proposal, adaptive=True) 

    # read inputs for makrov chains
    print('Please enter:')
    chain_length = int(input('Length of each Markov chain:'))
    number_of_chains = int(input(('Number of independent Markov chains:')))
    comment_str = input("Comment for specifying current run:")
    burnin = int(input('Burnin:'))
    # run the sampler and time it
    tic = time.time()
    my_chains = tda.sample(my_posterior, my_proposal, iterations=chain_length, 
                           n_chains=number_of_chains, initial_parameters = initial_parameters)
    toc = time.time()
    print(f'Time to compute the chains:{(toc-tic)/60} minutes' )    

    # write time data in seconds to txt file
    with open('MCMC_chains_not_logspace\TIME_MCMC_chains_length_%d_numChains_%d_h_1_%d_reduced_dim_%s.txt'%(chain_length,number_of_chains,n, comment_str), 'w') as f:
        f.write('Time to compute the chains in seconds:%d'%(toc-tic))
    
    # convert to inference data and save it to disc
    idata = tda.to_inference_data(my_chains, burnin=burnin)
    array_dict = tda.get_samples(my_chains,burnin=burnin)
    with open('MCMC_chains_not_logspace\MCMC_chains_dict_length_%d_numChains_%d_h_1_%d_reduced_dim_%s.pkl'%(chain_length,number_of_chains,n,comment_str), 'wb') as f:
        pickle.dump(array_dict, f)
    idata.to_json('MCMC_chains_not_logspace\MCMC_chains_length_%d_numChains_%d_h_1_%d_reduced_dim_%s.json'%(chain_length,number_of_chains,n, comment_str))
    
    