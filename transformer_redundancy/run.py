import torch
from transformer_redundancy.data import *
from transformer_redundancy.models.vision import *
from transformer_redundancy.models.text import *
from transformer_redundancy.models.audio import *
from transformer_redundancy.methods.utils import extract_features, __compute_jacobian__
from transformer_redundancy.methods.cka import compute_cka_from_tensors

def get_domain(args):
    if args.dataset_name in ['imagenet-1k']:
        return 'vision'
    elif args.dataset_name in ['go_emotions']:
        return 'text'
    elif args.dataset_name in ['coco']:
        return 'multimodal'
    elif args.dataset_name in ['speech_commands']:
        return 'audio'
    else:
        raise ValueError(f"Dataset {args.dataset_name} not recognized.")

def get_analyzer(args):
    if 'deit' in args.model_name:
        return DeiTForLayerwiseAnalysis(args.model_name, device=args.device)
    elif 'beit' in args.model_name:
        return BeitForLayerwiseAnalysis(args.model_name, device=args.device)
    elif 'dinov2' in args.model_name:
        return DinoV2ForLayerwiseAnalysis(args.model_name, device=args.device)
    elif 'roberta' in args.model_name:
        return RoBERTaForLayerwiseAnalysis(args.model_name, device=args.device)
    elif 'clip' in args.model_name:
        raise NotImplementedError("CLIP models not supported.")
        # return CLIPForLayerwiseAnalysis(args.model_name, device=args.device)
    elif 'wav2vec2' in args.model_name:
        return Wav2VecForLayerwiseAnalysis(args.model_name, model_folder=args.model_folder, pruned=args.pruned, device=args.device)
    elif 'wavlm' in args.model_name.lower():
        return WavLMForLayerwiseAnalysis(args.model_name, model_folder=args.model_folder, pruned=args.pruned, device=args.device)

if __name__ == '__main__':
    
    import os
    import argparse
    from dotenv import load_dotenv
    from huggingface_hub import login
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description='Run Visual Transformer experiments.')
    ### Experiment parameters ###
    parser.add_argument('mode', type=str, nargs='+', choices=['cosine-similarity', 'cka-similarity', 'jacobian-similarity'])
    parser.add_argument('--within-block', action='store_true')
    parser.add_argument('--jacobian-between-layers', action='store_true')
    parser.add_argument('--jacobian-chunk-size', type=int, default=1)
    parser.add_argument('--c-per-iter', type=int, default=20)
    parser.add_argument('--max-iter', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save-path', type=str, default='../experiments', help='Directory to save the bsub files.')
    ### Data parameters ###
    parser.add_argument('--dataset-name', type=str, choices=['coco', 'imagenet-1k', 'go_emotions', 'speech_commands'])
    parser.add_argument('--processor-name', type=str)
    parser.add_argument('--batch-size', type=int, default=128)
    ### Model parameters ###
    parser.add_argument('--model-name', type=str)
    parser.add_argument('--model-folder', type=str, default=None)
    parser.add_argument('--pruned', action='store_true')
    parser.add_argument('--distilled', action='store_true')
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

    # Create save path
    os.makedirs(args.save_path, exist_ok=True)
    
    # Get data domain and loaders
    domain = get_domain(args)
    loaders = get_loaders(**args.__dict__)
    analyzer = get_analyzer(args)

    # Setup storage system
    results = {
        'all-cosine-similarities': [],
        'all-cka-similarities': [],
        'all-jacobian-batch-similarities': [],
        'all-jacobian-layer-similarities': [],
    }

    current_iteration = 0
    pbar = tqdm(total=args.max_iter)
    with torch.no_grad():
        for batch in loaders["test"]:
            if args.dataset_name == 'imagenet-1k':
                inputs = {'pixel_values': batch[0].to(args.device)}
            elif args.dataset_name == 'go_emotions':
                inputs = batch["text"]
            elif args.dataset_name == 'speech_commands':
                inputs = batch["input_values"].to(args.device)
            # elif args.dataset_name == 'coco':
            #     inputs = batch[0].to(args.device)

            if 'cosine-similarity' in args.mode or 'cka-similarity' in args.mode:
                pbar.set_description(f"Iteration {current_iteration}/{args.max_iter}: Computing feature similarities...") # set pbar description

                # Extract intermediate features (within and between transformer encoder blocks)
                features, n_feature_layers = extract_features(inputs, analyzer)

                # Compute similarities between intermediate features and input
                cos_similarities = torch.zeros((n_feature_layers, n_feature_layers))
                cka_similarities = torch.zeros((n_feature_layers, n_feature_layers))
                for layer_i in range(n_feature_layers):
                    for layer_j in range(layer_i+1, n_feature_layers):
                        if 'cosine-similarity' in args.mode:
                            cos_similarities[layer_i, layer_j] = torch.nn.functional.cosine_similarity(features[f'layer{layer_i}'].flatten(1), features[f'layer{layer_j}'].flatten(1)).mean()
                        if 'cka-similarity' in args.mode:
                            cka_similarities[layer_i, layer_j] = compute_cka_from_tensors(features[f'layer{layer_i}'], features[f'layer{layer_j}'], kernel_func='linear')
                
                if 'cosine-similarity' in args.mode:
                    cos_similarities += cos_similarities.T + torch.eye(n_feature_layers) # Symmetrize
                    results['all-cosine-similarities'].append(cos_similarities)
                if 'cka-similarity' in args.mode:
                    cka_similarities += cka_similarities.T + torch.eye(n_feature_layers) # Symmetrize
                    results['all-cka-similarities'].append(cka_similarities)

            if 'jacobian-similarity' in args.mode:
                jacobians, jacobian_similarities_batch = {}, {}
                for k in range(analyzer.num_layers):
                    base_desc = f"Iteration {current_iteration}/{args.max_iter}: Transformer layer {k+1}/{analyzer.num_layers} ==> " # set pbar description

                    # Get intermediate representations, operations and intermediates (+ features)
                    operations, intermediates, info = analyzer.__element_wise_breakdown__(inputs, from_layer=k, within_block=args.within_block)
                    # Compute Jacobian for each data point in the batch
                    J = __compute_jacobian__(analyzer, intermediates=intermediates, operations=operations, pbar=pbar, current_iteration=current_iteration, k=k, info=info, base_desc=base_desc, **args.__dict__)
                    
                    # Store Jacobians for computing similarities between layers
                    if args.jacobian_between_layers:
                        jacobians[k] = J

                    # Compute similarities between Jacobians across the batch
                    _batch_sim_output = torch.ones((args.batch_size, args.batch_size))
                    for i in range(args.batch_size):
                        pbar.set_description(base_desc + f"Similarity of {i+1}/{args.batch_size}") # update pbar info

                        # Compute pairwise cosine similarity between Jacobians of different data points
                        for j in range(i+1, args.batch_size):
                            _batch_sim_output[i,j] = _batch_sim_output[j,i] = torch.nn.functional.cosine_similarity(J[i].to(args.device).flatten(), J[j].to(args.device).flatten(), dim=0).item()
                    
                    # Store batch similarities 
                    jacobian_similarities_batch[f'layer{k}'] = _batch_sim_output
                results['all-jacobian-batch-similarities'].append(jacobian_similarities_batch)

                if args.jacobian_between_layers:
                    layer_similarities = torch.ones((args.batch_size, analyzer.num_layers, analyzer.num_layers))
                    for _data_idx in range(args.batch_size):
                        # Set description
                        pbar.set_description(f"Similarity of layer {i+1}/{analyzer.num_layers} (data point {_data_idx+1}/{args.batch_size})")
                        for i in range(jacobians.keys().__len__()):
                            # Compute pairwise cosine similarity between Jacobians of different data points
                            for j in range(i+1, jacobians.keys().__len__()):
                                layer_similarities[_data_idx, i, j] = layer_similarities[_data_idx, j, i] = torch.nn.functional.cosine_similarity(jacobians[i][_data_idx], jacobians[j][_data_idx], dim=0)

                    layer_similarities = layer_similarities.mean(dim=0)
                    results['all-jacobian-layer-similarities'].append(layer_similarities)

            # Check if maximum iterations are reached
            if current_iteration >= args.max_iter:
                break
            current_iteration += 1

    os.makedirs(f"{args.save_path}/{domain}", exist_ok=True)
    torch.save(results, f'{args.save_path}/{domain}/{args.model_name.split("/")[-1]}_within={args.within_block}_{args.mode}_results.pth')