from transformers import Wav2Vec2FeatureExtractor, AutoModelForAudioClassification, AutoFeatureExtractor
from datasets import load_dataset
from torch.utils.data import DataLoader

def load_processor(name):
    if "wav2vec2" in name and 'large' in name:
        return Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large")
    elif "wav2vec2" in name:
        return Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    elif 'wavlm' in name.lower() and 'large' in name:
        return AutoFeatureExtractor.from_pretrained("microsoft/wavlm-large")
    elif 'wavlm' in name.lower():
        return AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base")
    
    else:
        raise NotImplementedError(f"Processor for {name} not implemented...")

def process_dataset(example, processor):
    audio = example["audio"]
    processed = processor(audio["array"], return_tensors="pt", padding='max_length', max_length=16000, sampling_rate=16000)
    processed["input_values"] = processed["input_values"].squeeze()
    return processed

def get_speech_commands_loaders(model_name: str, batch_size: int, num_proc: int = 1, seed: int = 0, splits=["validation", "test"]):

    # Load processor
    processor = load_processor(model_name)
    # Get dataset and filter
    dataset = load_dataset("speech_commands", "v0.02")
    dataset = dataset.filter(lambda example: example["label"] != 35)
    
    loaders = {}
    for split in splits:
        # Shuffle the dataset split
        _dset = dataset[split].shuffle(seed=seed)

        if 'speaker' not in model_name:
            # Process dataset
            _dset = _dset.map(lambda x: process_dataset(x, processor), remove_columns=['file', 'audio', 'is_unknown', 'speaker_id', 'utterance_id'], num_proc=num_proc)
            _dset.set_format(type="torch", columns=["input_values", "label"])
        else:
            raise NotImplementedError(f"Speaker dataloader not implemented...")
        
        # Create DataLoader
        loaders[split] = DataLoader(_dset, batch_size=batch_size, shuffle=False)

    return loaders