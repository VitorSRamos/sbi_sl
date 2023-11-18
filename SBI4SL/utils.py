import os
import itertools
def make_pipeline_output_dir(output_name="Results"):
    
    # try to find directory name not taken
    # starts with output_name, adds (i) to end in case it exists
    # create directory when it doesnt exist
    
    proposal = output_name
    dir_version = 0
    
    while os.path.isdir(proposal): # if exists, add 1 to (i)
        dir_version +=1
        proposal = output_name + f" ({dir_version})"

    # after proposal is accepted
    os.makedirs(proposal)
    return proposal

def make_architecture_list(embedding_net, hyperparams_dict, input_size):
    # takes in embedding_net dict with keys n_out_embedding, hidden features, n_transforms, density_estimator
    # returns list with shape (embedding, hidden_features, n_transforms, density_estimator)
    arch_params = list(itertools.product(*hyperparams_dict.values())) # gets all combinations of values in dict
    arch_list = [[embedding_net(input_size, item[0]), item[1], item[2], item[3]] for item in arch_params] # makes the list of architectures/hyperparams
    return arch_list