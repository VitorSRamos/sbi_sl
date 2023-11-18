import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
from SBI4SL.calibration import * # get_reliability_curve

# setting plot parameters
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 20})

' -------------------------------------------------------------------------- '
def plot_history(training_history, # history object
                 model, # model index for plot title
                 param_name, # parameter name for plot title
                 save_as, # path to save file
                ):
    fig, ax = plt.subplots()
    ax.set_title(F'Training History - {model}, {param_name}')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('-LogProb')
    plt.plot([-item for item in training_history['training_log_probs']], label = 'Training')
    plt.plot([-item for item in training_history['validation_log_probs']], label = 'Validation')
    plt.legend()
    plt.grid()
    print("saving training plot")
    plt.savefig(save_as, format='jpg')

' -------------------------------------------------------------------------- '

def plot_1x1(results_df, # data to plot
            model, # model index for plot title
            param_name, # parameter name for plot title
            save_as, # path to save file
           ):
    
    # Binando
    bin_param = results_df.groupby(pd.cut(results_df[param_name+'_true'], 30)).agg('mean') # agrupo valores pela média a partir de bins (cut) no valor true

    # Plot
    fig, ax = plt.subplots(2, 2, figsize=(8,8), height_ratios=[4, 1], width_ratios=[4, 1])
    #plt.suptitle(param_name+' (range original {:.3f} - {:.3f})'.format(limits_df[param_name][0], limits_df[param_name][1]))
    plt.suptitle(F'1x1 binned - {model}, {param_name}')
    
    # plot 1x1
    ax[0, 0].fill_between(bin_param[param_name+'_true'], bin_param[param_name+'_sig3m'], bin_param[param_name+'_sig3p'], alpha=0.3, color='C2')
    ax[0, 0].fill_between(bin_param[param_name+'_true'], bin_param[param_name+'_sig2m'], bin_param[param_name+'_sig2p'], alpha=0.6, color='C2')
    ax[0, 0].fill_between(bin_param[param_name+'_true'], bin_param[param_name+'_sig1m'], bin_param[param_name+'_sig1p'], alpha=0.9, color='C0')
    
    ax[0, 0].set_ylabel('Predicted')
    ax[0, 0].plot(bin_param[param_name+'_true'], bin_param[param_name+'_pred'], color='C8')
    
    x_min, x_max  = ax[0,0].get_xlim() # guardo limites gerados automaticamente pelo plot dos pontos
    y_min, y_max = ax[0,0].get_ylim()
    ax_min = min(x_min, y_min) # acho o menor entre os dois para garantir que a reta 1x1 tbm seja diagonal no plot
    ax_max = max(x_max, y_max)
    ax[0, 0].set_xlim(ax_min, ax_max) # forço esses limites no plot
    ax[0, 0].set_ylim(ax_min, ax_max)
    ax[0, 0].plot([-1000.0, 1000.0], [-1000.0, 1000.0], '--', color='k') # ploto reta com range absurdo
    ax[0, 0].grid(alpha=0.3)
    
    # histograma true
    ax[1, 0].hist(results_df[param_name+'_true'], bins=30) #
    ax[1, 0].set_xlabel('True')
    ax[1, 0].set_xlim(ax_min, ax_max) # uso limtes do plot 1x1
    #ax[1, 0].set_ylim(ax_min, ax_max)
    #ax[1, 0].set_xlim(0, 1)
    ax[1, 0].grid(alpha=0.3)

    # histograma pred
    ax[0, 1].hist(results_df[param_name+'_pred'], bins=30, orientation='horizontal') #
    ax[0, 1].set_xlim(ax[0, 1].get_xlim()[::-1]) #invert the order of x-axis values
    ax[0, 1].yaxis.tick_right() #move ticks to the right
    #ax[0, 1].set_xlim(ax_min, ax_max) 
    ax[0, 1].set_ylim(ax_min, ax_max) # uso limites do plot 1x1
    #ax[0, 1].set_ylim(0, 1)
    ax[0, 1].grid(alpha=0.3)

    # subplot vazio
    ax[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(save_as, format='jpeg')

' -------------------------------------------------------------------------- '

def plot_real_scatter(real_results_df, # data to plot
                      model, # model index for plot title
                      param_name, # parameter name for plot title
                      save_as, # path to save file
                     ):
    
    # make scatter plot
    # calculate uncertainty in inference
    real_results_df[param_name+"_pred_yerr_m"] = real_results_df[param_name+"_pred"] - real_results_df[param_name+"_sig1m"]
    real_results_df[param_name+"_pred_yerr_p"] = real_results_df[param_name+"_sig1p"] - real_results_df[param_name+"_pred"]

    fig, ax = plt.subplots(1, 1, figsize=(8,8))
    plt.suptitle(F'1x1 scatter - {model}, {param_name}')
    
    #plt.suptitle("Einstein Radius (arcsec)")
    
    # plot 1x1    
    ax.set_ylabel('Predicted')
    ax.set_xlabel("True")


    ax.errorbar(real_results_df[param_name+"_true"], real_results_df[param_name+"_pred"], 
                     xerr=real_results_df[param_name+"_true_err"],
                     yerr=np.array([real_results_df[param_name+"_pred_yerr_m"],real_results_df[param_name+"_pred_yerr_p"]]), 
                     fmt="none", 
                     c="C8",
                     alpha=0.75,
                     zorder=0,
                    )
    ax.scatter(real_results_df[param_name+"_true"], real_results_df[param_name+"_pred"], alpha=0.9, s=15, zorder=1), #label = f"r2: {r2:.2f}; mean precision = {precision:.2f}",)
    
    
    x_min, x_max  = ax.get_xlim() # guardo limites gerados automaticamente pelo plot dos pontos
    y_min, y_max = ax.get_ylim()
    ax_min = min(x_min, y_min) # acho o menor entre os dois para garantir que a reta 1x1 tbm seja diagonal no plot
    ax_max = max(x_max, y_max)
    ax.set_xlim(ax_min, ax_max) # forço esses limites no plot
    ax.set_ylim(ax_min, ax_max)
    ax.plot([-1000.0, 1000.0], [-1000.0, 1000.0], '--', color='k') # ploto reta com range absurdo
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_as, format='jpeg', dpi="figure")

' -------------------------------------------------------------------------- '
def plot_pcp(uncalibrated_posteriors,
             calibrated_posteriors,
             true_vals,
             model,
             param_name,
             save_as,
             data_type,
             ):
    
    # making both curves
    uncalib_curve = get_reliability_curve(uncalibrated_posteriors, true_vals)
    calib_curve = get_reliability_curve(calibrated_posteriors, true_vals)

    # plot
    fig, ax = plt.subplots(1,1, figsize=(8,8))
    plt.suptitle(F'Reliability_plot - {data_type}, {model}, {param_name}')
    
    ax.set_ylabel('Empirical Coverage')
    ax.set_xlabel('Confidence Interval')
    

    ax.plot(uncalib_curve[0], uncalib_curve[1], label="Uncalibrated")
    ax.plot(calib_curve[0], calib_curve[1], label="Calibrated")
    plt.legend()
    ax.grid()
    ax.plot([0, 1], [0, 1], '--', color='k')
    plt.savefig(save_as, format='jpeg')








'''


def plot_pcp(uncalibrated_posteriors,
             calibrated_posteriors
            model,
            param_name,
            save_as,
            data_type,
            ):
    
    # get posterior coverage    
    confidence_intervals = np.linspace(0, 50, 20) # intervalos (pra cima e para baixo, partindo de 50) em que a cobertura sera avaliada
    
    test_percentages = results[param_name+"_perc_of_true"]
    posterior_coverage = [] 
    for interval in confidence_intervals: 
        # Para cada volume da lista, conta quantos elementos em test_percentages estão cobertos 
        post_coverage = np.count_nonzero([is_in_interval(item, 50-interval, 50+interval) for item in test_percentages]) 
        posterior_coverage.append(post_coverage)
    
    
    norm_pc = [item/test_percentages.shape[0] for item in posterior_coverage] # divide as contagens pelo comprimento da lista p obter porcentagem
    norm_pv = [item*2/100 for item in confidence_intervals] # divide as porcentagens por 100, multiplica por 2 porque interval e p cima e p baixo


    # plot
    fig, ax = plt.subplots(1,1, figsize=(8,8))
    plt.suptitle(F'PCP - {data_type}, {model}, {param_name}')
    
    ax.set_ylabel('Empirical Coverage')
    ax.set_xlabel('Confidence Interval')
    ax.plot(norm_pv, norm_pc)
    ax.plot([0, 1], [0, 1], '--', color='k')
    plt.savefig(save_as, format='jpeg')
'''