from transformer_redundancy.train_surrogate import get_domain, get_analyzer, FeatureReproducingModel, TransformerBasedMimicker, OneLayerFeatureReproducingModel, OneLayerTransformerBasedMimicker
from transformer_redundancy.data import get_loaders
from transformer_redundancy.methods.utils import extract_features

from collections import OrderedDict, defaultdict
import torch
from torch import nn

class AssembledModel(nn.Module):

    def __init__(self, feature_extractor, feature_projection, remaining_network):
        super(AssembledModel, self).__init__()

        self.feature_extractor = feature_extractor
        self.feature_projection = feature_projection
        self.remaining_network = remaining_network
        self.name = ''

    def forward(self, input_values):
        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        hidden_states, extract_features = self.feature_projection(extract_features)
        _, last_hidden_states = self.remaining_network.mimicker(hidden_states)

        pooled_projection = self.remaining_network.projector(last_hidden_states).mean(dim=1)
        return self.remaining_network.classifier(pooled_projection)


def load_model(model_path, analyzer, embedding_dim, n_layers, hidden_dim, model_type):
    if 'mimicker' in model_path.name:
        # Load feature extraction network parts
        feature_extractor = analyzer.model.wav2vec2.feature_extractor if 'wav2vec2' in model_path.as_posix() else analyzer.model.wavlm.feature_extractor
        feature_projector = analyzer.model.wav2vec2.feature_projection if 'wav2vec2' in model_path.as_posix() else analyzer.model.wavlm.feature_projection

        # Load pre-trained weights
        state_dict = torch.load(model_path)

        # Setup mimicking architecture
        if model_type == 'linear' and n_layers == 2:
            mimicker = FeatureReproducingModel(in_dim=embedding_dim, embedding_dim=embedding_dim, hidden_dim=hidden_dim)
        elif model_type == 'linear' and n_layers == 1:
            mimicker = OneLayerFeatureReproducingModel(in_dim=embedding_dim, embedding_dim=embedding_dim, hidden_dim=hidden_dim)
        elif model_type == 'transformer' and n_layers == 2:
            mimicker = TransformerBasedMimicker(embedding_dim=embedding_dim, hidden_dim=hidden_dim)
        else:
            mimicker = OneLayerTransformerBasedMimicker(embedding_dim=embedding_dim, hidden_dim=hidden_dim)

        # Set up classification network architecture
        _elements = [("mimicker", mimicker)]
        _elements += [("projector", analyzer.model.projector), ("classifier", analyzer.model.classifier)]
        remaining_network = nn.Sequential(OrderedDict(_elements))
        # Load model weights
        remaining_network.load_state_dict(state_dict)

        # Assemble model
        model = AssembledModel(feature_extractor, feature_projector, remaining_network)
        model.name = model_path.name
        return model, remaining_network

    else:
        analyzer.model.name = model_path.name
        return analyzer.model, nn.Sequential(OrderedDict([('encoder', analyzer.model.wav2vec2.encoder if 'wav2vec' in model_path.as_posix() else analyzer.model.wavlm.encoder), ('projector', analyzer.model.projector), ('classifier', analyzer.model.classifier)]))
    

if __name__ == '__main__':

    import os
    from pathlib import Path

    from dotenv import load_dotenv
    from huggingface_hub import login
    from tqdm import tqdm
    import numpy as np

    import argparse
    parser = argparse.ArgumentParser(description='Run Visual Transformer results.')
    ### Experiment parameters ###
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save-path', type=str, default='../experiments', help='Directory to save the bsub files.')
    ### Data parameters ###
    parser.add_argument('--dataset-name', type=str, choices=['speech_commands', 'imagenet-1k'])
    parser.add_argument('--processor-name', type=str)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--num-proc', type=int, default=1)
    ### Model parameters ###
    parser.add_argument('--model-name', type=str)
    parser.add_argument('--model-folder', type=str)
    parser.add_argument('--mimicker-model-folder', type=str)
    parser.add_argument('--hidden-dims', nargs="+", type=int, required=True)
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'])
    # Parse arguments
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    login(os.getenv('HF_TOKEN'))

    # Clear pytorch cache
    torch.cuda.empty_cache()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Get data domain and loaders
    domain = get_domain(args)
    loaders, label2cat = get_loaders(**args.__dict__)
    analyzer = get_analyzer(args)

    EXP_PATH = Path(args.mimicker_model_folder) / args.model_name
    embedding_dim = 1024 if 'large' in args.model_name else 768

    # Extract model paths
    meta_info = {}
    for training_procedure in ['mimicker', 'non_mimicker']:
        for model_type in ['linear']:
            for hidden_dim in args.hidden_dims:
                for n_layers in [1, 2]:
                    model_path = EXP_PATH / f"{training_procedure}_{model_type}_{n_layers}layer_hidden_dim={hidden_dim}.pt"
                    
                    # Store meta info
                    meta_info[model_path] = {'embedding_dim': embedding_dim, 'n_layers': n_layers, 'hidden_dim': hidden_dim, 'model_type': model_type}

    # Add original model
    meta_info[Path(args.model_folder) / args.model_name] = {'embedding_dim': None, 'n_layers': None, 'hidden_dim': None, 'model_type': None}

    # Setup results storage
    num_params = {}
    num_params_full = {}
    all_preds = defaultdict(list)
    all_probs = defaultdict(list)
    all_labels = []
    all_timings = defaultdict(list)

    store_labels = True
    for model_idx, model_path in enumerate(meta_info.keys()):
        try:
            model, remaining_network = load_model(model_path, analyzer, **meta_info[model_path])
            model.to(args.device)
            model.eval()

            # Count number of parameters
            num_params[model_path.name] = sum(p.numel() for p in remaining_network.parameters())
            num_params_full[model_path.name] = sum(p.numel() for p in model.parameters())

            # Test model
            with torch.no_grad():
                # Set up timing
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

                for i in range(100): # GPU warm-up
                    _ = model(torch.randn(1, 16000).to(args.device))

                for batch in tqdm(loaders['test'], desc=f"Testing {model_path.name}"): # num_batches is repetitions
                    # Get data from batch
                    inputs = batch['input_values'].to(args.device)
                    labels = batch['label']

                    # Forward pass with time tracking
                    starter.record()
                    logits = model(inputs) if 'mimicker' in model_path.name else model(inputs).logits
                    ender.record()

                    # Synchronize and get time
                    torch.cuda.synchronize() # WAIT FOR GPU SYNC
                    curr_time = starter.elapsed_time(ender)
                    all_timings[model_path.name].append(curr_time / len(labels)) # Average time per sample

                    # Get predictions and probabilities
                    probs, preds = nn.functional.softmax(logits, dim=1).max(dim=1)
                    all_probs[model_path.name].extend(probs.to(torch.float16).cpu().numpy())
                    all_preds[model_path.name].extend(preds.cpu().numpy())
                    if store_labels:
                        all_labels.extend(labels.cpu().numpy())

            store_labels = False

        except FileNotFoundError:
            if training_procedure != 'non_mimicker' and n_layers != 2:
                print(f"File not found - {model_path}")

    # Save results
    results = {
        'num_params': num_params,
        'num_params_full': num_params_full,
        'all_preds': dict(all_preds),
        'all_probs': dict(all_probs),
        'all_labels': all_labels,
        'all_timings': dict(all_timings),
    }
    torch.save(results, EXP_PATH / 'results.pt')