# ---------------- imports ---------------- 
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime

from SBI4SL.model_train import *
from SBI4SL.utils import *
from SBI4SL.architectures import *
from SBI4SL.utils_1.checks import *
from SBI4SL.make_plots import *
from SBI4SL.data_eval import *


#  ---------------- open files (all must be numpy array or pandas dataframe) ---------------- 
sim_imgs=np.load("/Parent/added_noise_test/data_prep/data/prep_sim_imgs.npy")
sim_params=pd.read_csv("/Parent/added_noise_test/data_prep/data/norm_sim_params.csv")
sim_physrange=pd.read_csv("/Parent/added_noise_test/data_prep/data/sim_params.csv")
'''talvez o ideal seja definir param_list no começo e colocar no usecols de todos os dataframes'''

real_imgs=np.load("/Parent/added_noise_test/data_prep/data/prep_real_imgs.npy")
real_params=pd.read_csv("/Parent/added_noise_test/data_prep/data/norm_real_params.csv")
real_physrange=pd.read_csv("/Parent/added_noise_test/data_prep/data/real_params.csv")

param_list = ["thetaE"] # list of parameters to run inference on. Must be columns on params dataframes


#  ---------------- data checks ---------------- 
#assert images_normalized(sim_imgs, real_imgs)
#assert params_normalized(parameters=parameters, sim_params, real_params)
'''print checks ok to pipeline report'''


#  ---------------- define architecture list ---------------- 
input_size=sim_imgs[0].shape

hyperparams_dict = {"n_out_embedding" : [8], # number of outputs in the embedding net
                    "hidden features" : [24, 32, 48], 
                    "n_transforms" : [4, 6],
                    "density_estimator": ["nsf"]
                    }

architectures = make_architecture_list(inception_feature_extractor_87, hyperparams_dict, input_size)


#  ---------------- create results directory ---------------- 
results_path = make_pipeline_output_dir(output_name="quick_Test_Results")
pipeline_report = open(results_path+"/"+"pipeline_report.txt", "x")
print(f"Starting Inference Pipeline on {datetime.now()} UTC", file=pipeline_report)
''' print data shapessssssss '''


#  ---------------- seeds, gpu, stdout ---------------- 
random_seed=42
torch.manual_seed(42)
np.random.seed(42)

gpu_index=2
set_gpu_index(gpu_index)
print(f"Device: {torch.cuda.get_device_name()}, index {torch.cuda.current_device()}", file=pipeline_report)

# anotando stdout pro console
console_stdout = sys.stdout
#console_stderr = sys.stderr

#  ---------------- data preparation ---------------- 
train_sim_imgs, test_sim_imgs, train_sim_params, test_sim_params = train_test_split(sim_imgs, sim_params.to_numpy(), test_size=0.2, random_state=random_seed)
calibration_sim_imgs, test_sim_imgs, calibration_sim_params, test_sim_params = train_test_split(test_sim_imgs, test_sim_params, test_size=0.5, random_state=random_seed)

print("Train imgs, params:", train_sim_imgs.shape, train_sim_params.shape, file=pipeline_report)
print("Test imgs, params:", test_sim_imgs.shape, test_sim_params.shape, file=pipeline_report)
print("Calibration imgs, params:", calibration_sim_imgs.shape, calibration_sim_params.shape, file=pipeline_report)

#  ---------------- main loop  ---------------- 
for model_index, model in enumerate(architectures): # loop over all architectures
    for param_index, param_name in enumerate(param_list): # loop over all parameters
        
        model_path = results_path +"/"+ f"Model_{model_index}" # model directory name
        param_path = model_path +"/"+ str(param_name) # parameter directory (inside model directory)
        
        os.makedirs(param_path) # creates model and param_name directory
        model_report = open(model_path +"/"+ "Model_Report.txt", "x") # report for model in model directory (not param_name)
        
        sys.stdout =  model_report # all print statements print to model_report
        #sys.stderr =  model_report # all print statements print to model_report

        print(f"Starting main loop for Model {model_index} at {datetime.now()}")
                
        # clearing gpu memory 
        torch.cuda.empty_cache()

        # printing model params to model output file
        print(f"\nembedding: {model[0]} \nhidden_features: {model[1]} \nn_transforms: {model[2]} \ndensity_estimator: {model[3]}\n")
        
        try:
            # ---------------- training model ---------------- 
            '''em algum lugar tem que salvar o plot da arquitetura'''
            posterior, training_history = model_train(train_sim_imgs, 
                                                      train_sim_params, 
                                                      architecture=model,
                                                      patience=200, 
                                                      max_epochs=3000, 
                                                      batch_size=512, 
                                                      learning_rate=0.005, 
                                                      val_frac=0.1,
                                                      )
            # Saving training history
            print("saving training history") ;   
            with open(param_path+"/training_history.pkl", "wb") as handle:
                pickle.dump(training_history, handle)    
            # Saving posterior
            print("saving posterior file")
            with open(param_path+"/saved_posterior.pkl", "wb") as handle:
                pickle.dump(posterior, handle)
            
            # ---------------- Posterior Calibration ----------------
            calibration_factor = find_calibration_factor(posterior,
                                                            calibration_sim_imgs, 
                                                            calibration_sim_params,
                                                            n_samples=10000,
                                                            max_attempts=100,
                                                           )            
            
            
            print(f"Calibration Factor: {calibration_factor}")
            
            # generating posterior objects for test set (sim and real)
            uncalib_sim_posterior_samples, calib_sim_posterior_samples = sample_from_posterior(test_sim_imgs, posterior, n_samples=10000, calibration_factor=calibration_factor)
            uncalib_real_posterior_samples, calib_real_posterior_samples = sample_from_posterior(real_imgs, posterior, n_samples=10000, calibration_factor=calibration_factor)
            
            # ---------------- data eval sim uncalibrated ----------------
            uncalib_sim_norm_results, uncalib_sim_denorm_results = data_eval(uncalib_sim_posterior_samples,
                                                                                    test_sim_params,
                                                                                    param_name,
                                                                                    sim_physrange,
                                                                                    n_samples=10000,
                                                                                    data_type="uncalib_sim",
                                                                                    )
            
            print("Saving uncalibrated simulated posterior samples")
            np.save(param_path+'/uncalib_sim_posterior_samples.npy', uncalib_sim_posterior_samples)
            
            print("Saving uncalibrated simulated normalized results")
            uncalib_sim_norm_results.to_csv(param_path+'/uncalib_sim_results_norm.csv', index=False)
            
            print("Saving uncalibrated simulated Physical Range results")
            uncalib_sim_denorm_results.to_csv(param_path+'/uncalib_sim_results_physical.csv', index=False)
            
            # ---------------- data eval sim calibrated ----------------
            calib_sim_norm_results, calib_sim_denorm_results = data_eval(calib_sim_posterior_samples,
                                                                                    test_sim_params,
                                                                                    param_name,
                                                                                    sim_physrange,
                                                                                    n_samples=10000,
                                                                                    data_type="calib_sim",
                                                                                    )
            
            print("Saving calibrated simulated posterior samples")
            np.save(param_path+'/calib_sim_posterior_samples.npy', calib_sim_posterior_samples)
            
            print("Saving calibrated simulated normalized results")
            calib_sim_norm_results.to_csv(param_path+'/calib_sim_results_norm.csv', index=False)
            
            print("Saving calibrated simulated Physical Range results")
            calib_sim_denorm_results.to_csv(param_path+'/calib_sim_results_physical.csv', index=False)
            
            # ---------------- data eval real uncalibrated ----------------
            uncalib_real_norm_results, uncalib_real_denorm_results = data_eval(uncalib_real_posterior_samples,
                                                                                    real_params.to_numpy(),
                                                                                    param_name,
                                                                                    real_physrange,
                                                                                    n_samples=10000,
                                                                                    data_type="uncalib_real",
                                                                                    )
            print("Saving uncalibrated real posterior samples")
            np.save(param_path+'/uncalib_real_posterior_samples.npy', uncalib_real_posterior_samples)
            
            print("Saving uncalibrated real normalized results")
            uncalib_real_norm_results.to_csv(param_path+'/uncalib_real_results_norm.csv', index=False)
            
            print("Saving uncalibrated real Physical Range results")
            uncalib_real_denorm_results.to_csv(param_path+'/uncalib_real_results_physical.csv', index=False)
    
            # ---------------- data eval real uncalibrated ----------------
            calib_real_norm_results, calib_real_denorm_results = data_eval(calib_real_posterior_samples,
                                                                                    real_params.to_numpy(),
                                                                                    param_name,
                                                                                    real_physrange,
                                                                                    n_samples=10000,
                                                                                    data_type="calib_real",
                                                                                    )
            print("Saving calibrated real posterior samples")
            np.save(param_path+'/calib_real_posterior_samples.npy', calib_real_posterior_samples)
            
            print("Saving calibrated real normalized results")
            calib_real_norm_results.to_csv(param_path+'/calib_real_results_norm.csv', index=False)
            
            print("Saving calibrated real Physical Range results")
            calib_real_denorm_results.to_csv(param_path+'/calib_real_results_physical.csv', index=False)
            
            # ---------------- make plots ----------------
            print("Making history Plots")
            plot_history(training_history, f"Model_{model_index}", param_name, save_as=param_path+"/training_history.jpg")

            print("Making 1x1 Plots")
            plot_1x1(uncalib_sim_denorm_results, f"Model_{model_index}", param_name, save_as=param_path+"/uncalib_1x1.jpg")
            plot_1x1(calib_sim_denorm_results, f"Model_{model_index}", param_name, save_as=param_path+"/calib_1x1.jpg")

            print("Making scatter Plots")
            plot_real_scatter(uncalib_real_denorm_results, f"Model_{model_index}", param_name, save_as=param_path+"/uncalib_real_scatter.jpg")
            plot_real_scatter(calib_real_denorm_results, f"Model_{model_index}", param_name, save_as=param_path+"/calib_real_scatter.jpg")

            print("Making reliability Plots")
            plot_pcp(uncalib_sim_posterior_samples, calib_sim_posterior_samples, test_sim_params, f"Model_{model_index}", param_name, save_as=param_path+"/sim_pcp.jpg", data_type="sim")
            plot_pcp(uncalib_real_posterior_samples, calib_real_posterior_samples, real_params.to_numpy(), f"Model_{model_index}", param_name, save_as=param_path+"/real_pcp.jpg", data_type="real")
            
            # ---------------- closing loop ----------------
            
            
            print(f"Succesful end for Model {model_index} at {datetime.now()}")
            print(f"Model {model_index}: OK", file=pipeline_report)
            
            #sys.stderr = console_stderr
            sys.stdout = console_stdout
            model_report.close()
            
        except Exception as error:
            print(f"Error for Model {model_index}: {error}. Ending at {datetime.now()}")
            print(f"Model {model_index}: Error", file=pipeline_report)
        
print(f"Pipeline finished. Current Time: {datetime.now()} UTC", file=pipeline_report)
pipeline_report.close()