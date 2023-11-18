import torch
import torch.nn as nn
import sbi
from sbi.utils.user_input_checks_utils import float32
from sbi.inference import SNPE # simulate_for_sbi, prepare_for_sbi
from sbi import utils as utils
from time import time

def model_train(images, 
                params,
                architecture, # list of (embedding_net, hidden_features, n_transforms, density_estimator). output of make_architecture_list
                patience=200, 
                max_epochs=3000, 
                batch_size=512, 
                learning_rate=0.005, 
                val_frac=0.1,
               ): # sad face

    ''' 
    wrapper for the model training routine 
    Returns posterior and training history
    '''

    # opening architecture
    embedding, hidden_features, n_transforms, density_estimator= architecture


    # loading tensors to gpu    
    gpu_index = torch.cuda.current_device()
    print(f"Loading tensors to GPU {torch.cuda.current_device()}. Current memory usage: {torch.cuda.memory_allocated()} bytes")
    params = torch.tensor(params, dtype=float32, device="cuda")
    images = torch.tensor(images, dtype=float32, device="cuda")
    print(f"Load Successful. Memory allocated: {torch.cuda.memory_allocated()} bytes")


    # instantiating embedding net
    #embedding_net = embedding
    #input_size=images[].shape
    #embedding_summary = summary(embedding_net, input_size, device="cpu")
    

    # simulation-based inference procedures
    print("\nBeggining SBI procedures:")
    print('Instantiating density estimator')
    neural_posterior = utils.posterior_nn(model=density_estimator,
                                          embedding_net=embedding,
                                          hidden_features=hidden_features,
                                          num_transforms=n_transforms,
                                          )
    
    print('Defining inference method')
    inference = SNPE(density_estimator=neural_posterior,
                     device="cuda",
                     )
    
    print("Asserting data devices")
    torch.device(f"cuda:{gpu_index}")
    assert images.device == torch.device(f"cuda:{gpu_index}") and params.device == torch.device(f"cuda:{gpu_index}"), f"DEVICE ERROR: images: {images.device}, params: {params.device}; Expected {torch.device(f'cuda:{gpu_index}')}"
    
    print('Loading simulations')
    inference.append_simulations(params, images)
    
    print("Starting training")
    
    t_i = time() # starting timer
    density_estimator = inference.train(stop_after_epochs=patience,
                                        max_num_epochs=max_epochs,
                                        training_batch_size=batch_size,
                                        learning_rate=learning_rate,
                                        validation_fraction=val_frac,
                                        )
    t_f = time() # stopping timer
    
    print(f'\nTotal training time: {(t_f-t_i)/60} minutes\n')
    
    # writing embedding net to file
    #with open(output_path+"embedding_net.txt", "w") as text_file:
    #    text_file.write(str(embedding_summary))
    
    
    # Creating training history object
    training_history = inference._summary
    
    # Creating Posterior object
    posterior = inference.build_posterior(density_estimator)

    return posterior, training_history

    '''
    # Saving training history
    print("saving training history")    
    with open(output_path+"/training_history.pkl", "wb") as handle:
        pickle.dump(training_history, handle)

    # Saving posterior
    print("saving posterior file")
    with open(output_path+"/saved_posterior.pkl", "wb") as handle:
        pickle.dump(posterior, handle)
    '''

    
    
    