import torch

def set_gpu_index(gpu_index):
    assert torch.cuda.is_available(), "GPU availability assertion failed."
    torch.cuda.set_device(gpu_index) # set index
    device = torch.device(f"cuda:{gpu_index}")
    return device

'''
# making sure path exists
assert os.path.isdir(output_path)


def clear_gpu_memory():
    # clearing gpu memory 
    torch.cuda.empty_cache()


def set_seeds(seed):
    # fixing seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)


def load_data(params, images):
    # loading data
    params = pd.read_csv(param_path)
    images = np.load(image_path)
    print(f'Images shape: {images.shape}')
    print(f'params shape: {params.shape}')


# filtering parameters
#filtered_params = params[['thetaE', 'zl', 'zs', 'sig_v']]
filtered_params = params[param_list]


# checking image normalization:
if np.min(images) == 0 and np.max(images) == 255:
    print("normalizing images from 0-255 to 0-1")
    images = images/255

assert np.min(images) == 0.0  and np.max(images) == 1.0, f"images are not normalized. min: {np.min(images)}, max: {np.max(images)}"

# cheking parameter normalization
assert np.min(params.to_numpy()) == 0.0 and np.max(params.to_numpy()) == 1.0, f"parameters are not normalized. min: {np.min(params.to_numpy())}, max: {np.max(params.to_numpy())}"
'''