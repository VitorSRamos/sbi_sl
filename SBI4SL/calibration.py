import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import percentileofscore
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from SBI4SL.data_eval import percentile_of_true
from time import time

def is_in_interval(val, int_min, int_max):
    if int_min < val < int_max:
        return True
    else:
        return False

def sample_from_posterior(images, 
                          posterior, 
                          n_samples=5000, 
                          max_attempts=100,
                          calibration_factor=None
                         ):
    ''' 
    util function to get posterior samples
    returns uncalibrated posterior if calibration_factor is None
    else, returns both uncalibrated and calibrated posterior as well
    '''
    samples = []
    print("sampling from posterior")
    
    for image_index, image in enumerate(images):
        # sampling sometimes fails for no apparent reason. causes issues in other parts of the code
        
        got_valid_posterior=False # flag to catch problems after max attempts
        
        for i in range(max_attempts): # maximum number of attempts for each image
            try:
                post_samples = posterior.set_default_x(image).sample((n_samples,), show_progress_bars = False).cpu() # shape (n_samples, n_params)
                # if line above didnt cause an error, we continue
                samples.append(post_samples[:,0]) # since we analyze a single parameter, append sub array with shape (n_samples)
                got_valid_posterior=True # update flag
                break # stop trying
            
            except Exception as error:
                print(f"Erro de sampling for image {image_index} {error}")
        
        if not got_valid_posterior: # in case sampling fails all attempts
            raise Exception("SamplingError: Max attempts exceeded")
            
    print("sampling done")
    
    uncalibrated_posteriors = np.stack(samples, axis=0)
    assert uncalibrated_posteriors.shape[0] == images.shape[0]
    
    #samples_corrected = np.moveaxis(samples, 2, 0)
    #print(samples.shape)

    if calibration_factor:
        calibrated_posteriors = rescale_posteriors(uncalibrated_posteriors, calibration_factor)
        return uncalibrated_posteriors, calibrated_posteriors
    else:
        return uncalibrated_posteriors

def rescale_posteriors(posterior_samples, factor):
    
    # rescales the posterior distribution uncertainty by a given factor
    # sigma -> factor x sigma

    rescaled_posteriors = []
    
    for distribution in posterior_samples:
        rescaled_distribution = []
        median = np.quantile(distribution, 0.5)
        
        for point in distribution:
            sigma = point - median
            rescaled_point = median + (factor*sigma)
            rescaled_distribution.append(rescaled_point)
        
        rescaled_posteriors.append(rescaled_distribution)
    
    return np.stack(rescaled_posteriors)

def get_reliability_curve(posteriors, true_vals):

    percentiles_of_true = percentile_of_true(posteriors, true_vals)
    
    # get posterior coverage array
    confidence_intervals = np.linspace(0.001, 50, 20) # intervalos (pra cima e para baixo, partindo de 50) em que a cobertura sera avaliada
    
    posterior_coverage = [] 
    for interval in confidence_intervals: 
        # Para cada volume da lista, conta quantos elementos em percentiles_of_true estão cobertos 
        post_coverage = np.count_nonzero([is_in_interval(item, 50-interval, 50+interval) for item in percentiles_of_true]) 
        posterior_coverage.append(post_coverage)
    
    norm_pc = [item/percentiles_of_true.shape[0] for item in posterior_coverage] # divide as contagens pelo comprimento da lista p obter porcentagem
    norm_pv = [item*2/100 for item in confidence_intervals] # divide as porcentagens por 100, multiplica por 2 porque interval e p cima e p baixo

    return np.array([norm_pv, norm_pc])

def beta(x, a, b, c):
    # beta function to fit reliability plot
    # described in kull et al. 
    # https://proceedings.mlr.press/v54/kull17a.html
    return 1 / ( 1 + 1/(np.exp(c)*(x**a/(1-x)**b)) )

def fit_curve_to_beta(reliability_curve):
    params, _ = optimize.curve_fit(beta, reliability_curve[0], reliability_curve[1], maxfev=5000)
    return params

def sum_sq_residuals(beta_params):
    # returns sum of squared residuals between beta function with parameters as input and diagonal
    x = np.linspace(0.001,1,50)
    fitted_curve = beta(x,beta_params[0], beta_params[1], beta_params[2])
    squared_residuals = (fitted_curve - x)**2
    
    return np.sum(squared_residuals)

def compare_given_factor(factor, uncalibrated_posterior_samples, true_vals):
    
    # function that returns difference between beta curve for posterior calibrated with given factor and diagonal
    # to use in scipy.golden to find ideal factor
    
    rescaled_posteriors = rescale_posteriors(uncalibrated_posterior_samples, factor) # reescale posterior with factor
    reliability_curve = get_reliability_curve(rescaled_posteriors, true_vals) # get posterior reliability curve
    beta_parameters = fit_curve_to_beta(reliability_curve) # fit reliability curve to beta function
    difference_to_identity = sum_sq_residuals(beta_parameters) # compare fitted beta function with diagonal
    
    return difference_to_identity

def find_calibration_factor(posterior, # trained posterior object
                            calibration_images, # images in calibration set
                            calibration_params, # true values in calibration set
                            n_samples=5000,
                            max_attempts=100,
                           ):
    
    # generate posterior samples for calibration set
    uncalibrated_posterior_samples = sample_from_posterior(calibration_images, posterior, n_samples=n_samples, max_attempts=max_attempts)

    # find ideal calibration factor for set
    t_i = time()
    calibration_factor = optimize.golden(compare_given_factor, args=(uncalibrated_posterior_samples, calibration_params))
    t_f = time() # stopping timer
    print(f'\nTotal calibration time: {(t_f-t_i)/60} minutes\n')    

    return calibration_factor