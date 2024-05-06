import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from collections import defaultdict

def get_go_emotions_loaders(batch_size: int=32, seed: int=0):    

    # Load GoEmotions dataset
    dataset_dict = load_dataset("go_emotions", "simplified")

    loaders = defaultdict(list)
    for split in tqdm(['train', 'validation', 'test'], desc='INFO - Loading datasets'):
        # Get shuffling order
        np.random.seed(seed)
        _order = np.random.permutation(range(dataset_dict[split].num_rows))
        
        # Get number of batches 
        num_batches = (dataset_dict[split].num_rows // batch_size) + 1
        # Process batches
        for batch_idx in range(num_batches):
            elements = _order[batch_size*batch_idx:batch_size*(1+batch_idx)]
            loaders[split].append(dataset_dict[split][elements])
    
    return loaders