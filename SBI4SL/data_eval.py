from sbi.utils.user_input_checks_utils import float32
import torch
import torch.nn as nn
import pickle
import sbi
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore
from SBI4SL.calibration import *

def posterior_analysis(posteriors): # samples shape (test_size, n_samples), output of sample_from_posterior in calibration file
    ''' 
    analyzes the each posterior distribution in posteriors
    returns array with median and sigmas
    '''
    # Inicializando listas
    median_data = []
    sig1p_data = []
    sig2p_data = []
    sig3p_data = []
    sig1m_data = []
    sig2m_data = []
    sig3m_data = []
    
    # Loop sobre images
    for posterior_distribution in posteriors: # for each group of posterior samples in the samples array
        median = np.median(posterior_distribution)
        median_data.append(median)
        
        sig1p = np.quantile(posterior_distribution, 0.5+0.34)
        sig1p_data.append(sig1p)
        
        sig1m = np.quantile(posterior_distribution, 0.5-0.34)
        sig1m_data.append(sig1m)
        
        sig2p = np.quantile(posterior_distribution, 0.5+0.34+0.13)
        sig2p_data.append(sig2p)
        
        sig2m = np.quantile(posterior_distribution, 0.5-0.34-0.13)
        sig2m_data.append(sig2m)
        
        sig3p = np.quantile(posterior_distribution, 0.5+0.34+0.13+0.02)
        sig3p_data.append(sig3p)
        
        sig3m = np.quantile(posterior_distribution, 0.5-0.34-0.13-0.02)
        sig3m_data.append(sig3m)
        
    result = np.stack([median_data, sig1p_data, sig1m_data, sig2p_data, sig2m_data, sig3p_data, sig3m_data], axis=-1)
    
    return result # shape (test_size, 7), 7 because median plus 6 sigmas

def percentile_of_true(posteriors, true_vals): 
    # usa a saida de sample_from_posterior, mesmo que param_analysis 
    # true_vals eh array de valores true, deve estar na mesma ordem que params
    perc_for_true_value = []
    
    for index, samples in enumerate(posteriors):
        perc_for_true_value.append(percentileofscore(samples, true_vals[index])) # a porcentagem do posterior necessária para encontrar o valor real
    
    return np.array(perc_for_true_value)


def make_dataframe(analysis_result, true_params, percentile_of_true, param_name):
    full_array = np.concatenate([true_params, analysis_result, percentile_of_true], axis=1) # create new array containing true value in first column
    col_names = []
    col_names.append(param_name+'_true')
    col_names.append(param_name+'_pred')
    col_names.append(param_name+'_sig1p')
    col_names.append(param_name+'_sig1m')
    col_names.append(param_name+'_sig2p')
    col_names.append(param_name+'_sig2m')
    col_names.append(param_name+'_sig3p')
    col_names.append(param_name+'_sig3m')
    col_names.append(param_name+'_perc_of_true')

    results_df = pd.DataFrame(full_array, columns = col_names)
    return results_df

# defining utility denormalization function
def undo_normalization(arr, min, max):
    #print(min.type)
    #print(max.type)
    #print(arr.type)
    # Função oposta da normalização
    range = max - min
    new_arr = arr * range
    new_arr += min
    return new_arr
 
def get_physrange_results(norm_results_df, physrange_params, param_name, data_type):
    
    # norm results_df is the output of make_dataframe
    # physrange params is the non normalized dataframe of parameters
    denorm_results = pd.DataFrame()
    
    if data_type.endswith("real"): # add id and true error to real data
        denorm_results["objid"] = physrange_params["objid"]
        denorm_results[param_name+"_true_err"] = physrange_params[param_name+"_err"]
    
    for name, values in norm_results_df.items(): # iterate over results, denormalize and append as column to denorm dataframe
        if not name.endswith("perc_of_true"): # if param isnt percentile of true score, denormalize and append to new df
            print(f"{name} caught in if")
            denorm_results[name] = undo_normalization(values, physrange_params[param_name].min(), physrange_params[param_name].max())
    
        else: # if param is percentile score, append as is
            print(f"{name} caught in else")
            denorm_results[name] = values
    
    return denorm_results


# ------------------------ main function ------------------------

def data_eval(posterior_samples, # array with posterior samples for a given set
              params, # true params  (Must be a list for a single parameter!!!!)
              parameter_name,
              physrange_params, # true values non normalized for denormalization (Must be single parameter!!!!)
              data_type, # type od data (real or simulation)
              n_samples=5000, # number of posterior samples
    ): # sad face

    '''
    wrapper for inference procedure of test data using posterior samples object
    '''

    print("running posterior analyses")
    analysed_posteriors = posterior_analysis(posterior_samples) # run analysis function on parameters

    print("getting percentiles of true value")
    perc_of_true = percentile_of_true(posterior_samples, params) # get list of percentiles of true value in each posterior
    
    print("Analysis loop:", params.shape, analysed_posteriors.shape, perc_of_true.shape)

    norm_results = make_dataframe(analysed_posteriors, params, perc_of_true, parameter_name)
    
    denorm_results = get_physrange_results(norm_results, physrange_params, parameter_name, data_type)
    
    return norm_results, denorm_results