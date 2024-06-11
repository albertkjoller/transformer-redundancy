from transformers import Wav2Vec2FeatureExtractor, AutoModelForAudioClassification, AutoFeatureExtractor
from datasets import load_dataset
from torch.utils.data import DataLoader
import numpy as np

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

def prepare_dataset_speaker(dataset, processor, num_proc: int = 1):
    dataset = dataset.map(lambda x: process_dataset(x, processor), remove_columns=['file', 'audio', 'is_unknown', 'label', 'utterance_id'], num_proc=num_proc)

    # filter out speaker_ids that occur less than 50 times
    speaker_ids = dataset['speaker_id']
    unique, counts = np.unique(speaker_ids, return_counts=True)
    speaker_ids = unique[counts >= 50]
    dataset = dataset.filter(lambda example: example["speaker_id"] in speaker_ids, num_proc=num_proc)
    
    # Map speaker IDs to classes
    speaker_ids = dataset['speaker_id']
    unique_speakers = np.unique(speaker_ids)
    speaker_id_to_class = {speaker_id: i for i, speaker_id in enumerate(unique_speakers)}
    dataset = dataset.map(lambda example: {'input_values': example['input_values'], 'label': speaker_id_to_class[example['speaker_id']]}, num_proc=num_proc)
    dataset.set_format(type="torch", columns=["input_values", "label"])

    # split into train, validation, and test set
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    return dataset["train"], dataset["test"]

def get_speech_commands_loaders(model_name: str, batch_size: int, num_proc: int = 1, seed: int = 0, splits=["validation", "test"], shuffle: bool = False):

    # Load processor
    processor = load_processor(model_name)
    # Get dataset and filter
    dataset = load_dataset("speech_commands", "v0.02")
    dataset = dataset.filter(lambda example: example["label"] != 35)
    
    loaders = {}
    if 'speaker' not in model_name:
        for split in splits:
            # Shuffle the dataset split
            _dset = dataset[split].shuffle(seed=seed)

            # Process dataset
            _dset = _dset.map(lambda x: process_dataset(x, processor), remove_columns=['file', 'audio', 'is_unknown', 'speaker_id', 'utterance_id'], num_proc=num_proc)
            _dset.set_format(type="torch", columns=["input_values", "label"])
            
            # Create DataLoader
            loaders[split] = DataLoader(_dset, batch_size=batch_size, shuffle=shuffle)

    else:

        # Shuffle the dataset split
        _dset = dataset["train"].shuffle(seed=seed)
        train_dataset, eval_dataset = prepare_dataset_speaker(_dset, processor, num_proc=num_proc)
        loaders["train"] = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        loaders["validation"] = DataLoader(eval_dataset, batch_size=batch_size, shuffle=shuffle)

    return loaders