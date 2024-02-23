# imports
import time
import numpy as np
import tinyDA as tda
import pickle
import scipy.stats as stats


from forward_model import Forward_Model

if __name__ == "__main__":
    # Initialize the forward models
    print("Initializing model and posterior...")
    # initialize the forward model
    my_model_l0 = Forward_Model(h=1 / 8, inputs_positive=False)  # also negative inputs are allowed therefore exp(input) is passed to the model
    my_model_l1 = Forward_Model(h=1 / 16, inputs_positive=False)
    my_model_l2 = Forward_Model(h=1 / 32, inputs_positive=False)

    # define prior & Likelihood and parameters like in benchmark paper
    sig_pr = 2

    # increase the Likelihoods variance on coarser level (specific values arise from FE-error comparison)
    sig_l2 = 0.05
    sig_l1 = 0.05
    sig_l0 = 0.05
    # proposal variance
    sig_prop = 0.0725
    initial_parameters = np.load('MAP_posterior_level_2_not_logspace.npy') #we take a precomputed MAP estimated as the starting sample
    # load measurements
    z_measurement = np.loadtxt("z_hat.txt")

    # define the prior distribution
    my_prior = stats.multivariate_normal(mean=np.zeros(64), cov=sig_pr**2)

    # define the loglikelihood functions for the levels
    cov_likelihood_l2 = sig_l2**2 * np.eye(z_measurement.size)
    cov_likelihood_l1 = sig_l1**2 * np.eye(z_measurement.size)
    cov_likelihood_l0 = sig_l0**2 * np.eye(z_measurement.size)

    # change to adaptive Likelihood to see if acceptance rate goes up.
    my_loglike_l2 = tda.GaussianLogLike(z_measurement, cov_likelihood_l2)
    my_loglike_l1 = tda.GaussianLogLike(z_measurement, cov_likelihood_l1)
    my_loglike_l0 = tda.GaussianLogLike(z_measurement, cov_likelihood_l0)

    # define hirarchy of posteriors
    my_posterior_l0 = tda.Posterior(my_prior, my_loglike_l0, my_model_l0)
    my_posterior_l1 = tda.Posterior(my_prior, my_loglike_l1, my_model_l1)
    my_posterior_l2 = tda.Posterior(my_prior, my_loglike_l2, my_model_l2)

    # read inputs for makrov chains
    print("Please enter:")
    l = input(
        "Enter model levels (at least two), comma separated, starting with coarsest level (0 <-> h=1/8, 1 <-> h=1/16, 2 <-> h=1/32), allowed values are 0,1,2:"
    )
    levels = list(map(int, l.split(",")))
    levels_str = "_".join(map(str, levels))  # for writing in file name
    possible_posteriors = [
        my_posterior_l0,
        my_posterior_l1,
        my_posterior_l2,
    ]  # list of all available posteriors
    my_posteriors = [
        possible_posteriors[i] for i in levels
    ]  # posteriors that are actually used in sampling

    comment_str = input("Comment for specifying current run:")
    # define the proposal distribution
    cov_proposal = sig_prop**2*np.eye(64)
    my_proposal =  tda.AdaptiveMetropolis(cov_proposal, adaptive=True)
    # read inputs for makrov chains
    print("Please enter:")
    chain_length = int(input("Length of each Markov chain:"))
    number_of_chains = int(input(("Number of independent Markov chains:")))
    burnin = int(input("Burnin:"))
    
    # input subsampling rate or list of rates for more levels
    if len(my_posteriors) == 2:
        sub_rate = int(input("Subsampling rate:"))
        sub_rate_str = str(sub_rate)
    else:
        s = input(
            "Enter subampling rates, comma separated, starting with rate for coarsest level:"
        )
        sub_rate = list(map(int, s.split(",")))
        sub_rate_str = "_".join(map(str, sub_rate))

    # run the sampler and time it
    tic = time.time()
    my_chains = tda.sample(
        my_posteriors,
        my_proposal,
        iterations=chain_length,
        n_chains=number_of_chains,
        initial_parameters=initial_parameters,
        subsampling_rate=sub_rate)
    toc = time.time()
    print(f"Time to compute the chains:{(toc-tic)/60} minutes")
    # write time data in seconds to txt file
    with open(
        "MLDA_chains_not_logspace\TIME_MLDA_chains_length_%d_numChains_%d_subrates_%s_levels_%s_%s.txt"
        % (chain_length, number_of_chains, sub_rate_str, levels_str, comment_str),
        "w",
    ) as f:
        f.write("Time to compute the chains in seconds:%d" % (toc - tic))

    # convert to inference data and dictionary with numpy array
    if my_chains["sampler"] == "DA":
        idata = tda.to_inference_data(my_chains, level="fine", burnin=burnin)
        array_dict = tda.get_samples(my_chains, level="fine", burnin=burnin)
    elif my_chains["sampler"] == "MLDA":
        idata = tda.to_inference_data(my_chains, level=2, burnin=burnin)
        array_dict = tda.get_samples(my_chains, level=2, burnin=burnin)
    else:
        print(
            "There went something wrong while storing the chains, check levels in your model and selected level in tinyDA for extracting values."
        )

    # store data to disc
    idata.to_json(
        "MLDA_chains_not_logspace\MLDA_chains_length_%d_numChains_%d_subrates_%s_levels_%s_%s.json"
        % (chain_length, number_of_chains, sub_rate_str, levels_str, comment_str)
    )

    with open(
        "MLDA_chains_not_logspace\MLDA_chains_dict_length_%d_numChains_%d_subrates_%s_levels_%s_%s.pkl"
        % (chain_length, number_of_chains, sub_rate_str, levels_str, comment_str),
        "wb",
    ) as f:
        pickle.dump(array_dict, f)
