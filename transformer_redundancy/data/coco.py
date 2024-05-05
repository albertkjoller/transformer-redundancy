import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoCaptions
from pathlib import Path

class COCORev(torch.utils.data.Dataset):

    def __init__(self, base_dataset, processor):
        self.base_dataset = base_dataset
        self.processor = processor

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        return idx

    def __get_batch_repr__(self, idxs):
        images, texts = [], []
        for idx in idxs:
            _image, _texts = self.base_dataset[idx.item()]
            images.append(_image.convert('RGB'))
            texts.append(_texts[0])
        
        return self.processor(text=texts, images=images, return_tensors="pt", padding=True)


def get_coco_loaders(DATA_PATH: Path, processor, batch_size=128, dset_tags: str = ['val2017']):
    loaders = {}
    for dset_tag in dset_tags:
        # Load COCO dataset from pre-saved files
        dataset = CocoCaptions(root=DATA_PATH / f'COCO/{dset_tag}', annFile=DATA_PATH / f'COCO/annotations/captions_{dset_tag}.json')
        # Create DataLoader
        loaders[dset_tag] = DataLoader(COCORev(dataset, processor), batch_size=batch_size, shuffle=False)

    return loaders