
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoImageProcessor, BeitImageProcessor

# Create a new IterableDataset with the transformations
class TransformedDataset(torch.utils.data.IterableDataset):
    def __init__(self, base_dataset, processor):
        self.base_dataset = base_dataset
        self.processor = processor

    def __iter__(self):
        for example in self.base_dataset:
            # Apply processor
            yield (self.processor(example["image"].convert('RGB'))['pixel_values'][0], example["label"])

def get_imagenet_loaders(processor_name: str, batch_size: int=32, seed: int=0):    

    loaders = {}
    for split in tqdm(['train', 'validation', 'test'], desc='INFO - Loading datasets'):
        # Load imagenet dataset in streaming mode
        dataset = load_dataset('imagenet-1k', split=split, streaming=True, trust_remote_code=True).shuffle(seed=seed)
        
        # Get processor
        if 'beit' in processor_name:
           processor = BeitImageProcessor.from_pretrained(processor_name + '-patch16-224')
        elif 'deit' in processor_name:
           processor = AutoImageProcessor.from_pretrained(processor_name + '-patch16-224')
        else:
            processor = AutoImageProcessor.from_pretrained(processor_name)

        # Create a new IterableDataset with the transformations 
        transformed_dataset = TransformedDataset(dataset, processor)
        loaders[split] = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=False)

    return loaders