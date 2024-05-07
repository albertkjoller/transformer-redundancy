from .go_emotions import get_go_emotions_loaders
from .coco import get_coco_loaders
from .imagenet import get_imagenet_loaders
from .speech_commands import get_speech_commands_loaders

def get_loaders(dataset_name: str, batch_size: int=32, seed: int=0, **kwargs):
    if dataset_name == 'imagenet-1k':
        return get_imagenet_loaders(processor_name=kwargs['processor_name'], batch_size=batch_size, seed=seed)
    elif dataset_name == 'go_emotions':
        return get_go_emotions_loaders(batch_size=batch_size, seed=seed)
    elif dataset_name == 'coco':
        return get_coco_loaders(DATA_PATH=kwargs['coco_path'], processor=kwargs['processor_name'], batch_size=batch_size, seed=seed, dset_tags=['val2017'])
    elif dataset_name == 'speech_commands':
        return get_speech_commands_loaders(model_name=kwargs['model_name'], batch_size=batch_size, seed=seed)