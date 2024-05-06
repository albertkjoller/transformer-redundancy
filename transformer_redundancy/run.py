import torch
from transformer_redundancy.data import *
from transformer_redundancy.models.vision import *
from transformer_redundancy.models.text import *
from transformer_redundancy.methods.utils import extract_features
from transformer_redundancy.methods.cka import compute_cka_from_tensors

def get_domain(args):
    if args.dataset_name in ['imagenet-1k']:
        domain = 'vision'
    elif args.dataset_name in ['go_emotions']:
        domain = 'text'
    elif args.dataset_name in ['coco']:
        domain = 'multimodal'
    # elif args.datset_name in ['audio']:
    #     domain = 'audio'
    else:
        raise ValueError(f"Dataset {args.dataset_name} not recognized.")
    return domain

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
    elif 'wav2vec' in args.model_name:
        raise NotImplementedError("Wav2Vec models not supported.")
        # return Wav2VecForLayerwiseAnalysis(args.model_name, device=args.device)    

if __name__ == '__main__':
    
    import os
    import argparse
    from dotenv import load_dotenv
    from huggingface_hub import login
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description='Run Visual Transformer experiments.')
    ### Experiment parameters ###
    parser.add_argument('mode', type=str, nargs='+', choices=['cka-similarity', 'jacobian-similarity'])
    parser.add_argument('--within-block', action='store_true')
    parser.add_argument('--jacobian-between-layers', action='store_true')
    parser.add_argument('--max-iter', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save-path', type=str, default='../experiments', help='Directory to save the bsub files.')
    ### Data parameters ###
    parser.add_argument('--dataset-name', type=str, choices=['coco', 'imagenet-1k', 'go_emotions'])
    parser.add_argument('--processor-name', type=str)
    parser.add_argument('--batch-size', type=int, default=128)
    ### Model parameters ###
    parser.add_argument('--model-name', type=str)
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
    loaders = get_loaders(**args)
    analyzer = get_analyzer(args)
    
    # Setup storage system
    results = {
        'all-cka-similarities': [],
        'all-jacobians': [],
    }

    current_iteration = 0
    pbar = tqdm(total=args.max_iter)
    with torch.no_grad():
        for batch in loaders["test"]:
            # TODO: format inputs and add to gpu
            # inputs = f(batch)

            if 'cka-similarity' in args.mode:
                # Extract intermediate features (within and between transformer encoder blocks)
                features, n_feature_layers = extract_features(inputs, analyzer)

                # Compute similarities between intermediate features and input
                similarities = torch.zeros((n_feature_layers, n_feature_layers))
                for layer_i in range(n_feature_layers):
                    for layer_j in range(layer_i+1, n_feature_layers):
                    similarities[layer_i, layer_j] = compute_cka_from_tensors(features[f'layer{layer_i}'], features[f'layer{layer_j}'], kernel_func='linear')
                similarities += similarities.T + torch.eye(n_feature_layers)
                results['all-cka-similarities'].append(similarities)

            if 'jacobian-similarity' in args.mode:
                jacobians = {}
                jacobian_similarities_batch = []
                for k in range(analyzer.num_layers):
                    # Get intermediate representations, operations and intermediates (+ features)
                    outputs, operations, intermediates = analyzer.__element_wise_breakdown__(inputs, from_layer=k, within_block=args.within_block)
                        
                    # Get Jacobians
                    # TODO: THIS WORKS FOR TEXT !
                    J = []
                    for z in intermediates:
                        J.append( torch.vmap(torch.func.jacrev(lambda x: analyzer.__get_output_from__(x.unsqueeze(0).unsqueeze(0), operations)))(z).cpu().squeeze([0,1]).mean(dim=1) )
                    J = torch.stack(J)
                    
                    # Store Jacobians for computing similarities between layers
                    if args.jacobian_between_layers:
                        jacobians[k] = J

                    # Compute similarities between Jacobians across the batch
                    _batch_sim_output = torch.ones((args.batch_size, args.batch_size))
                    for i in range(args.batch_size):
                        # Set description
                        pbar.set_description(f'Iteration {current_iteration}/{args.max_iter}: Transformer layer {k+1}/{analyzer.num_layers} ==> Similarity of {i+1}/{args.batch_size}')
                        # Compute pairwise cosine similarity between Jacobians of different data points
                        for j in range(i+1, args.batch_size):
                            _batch_sim_output[i,j] = _batch_sim_output[j,i] = torch.nn.functional.cosine_similarity(J[i].to(args.device).flatten(), J[j].to(args.device).flatten(), dim=0).item()
                    
                    # Store batch similarities 
                    jacobian_similarities_batch.append(_batch_sim_output)



            if args.jacobian_between_layers:
                for _data_idx in range(len(X)):
                    pbar.set_description(f'Iteration {current_iter}/{max_iter} ==> Processed {num_data_processed}/{len(X)} points in batch')
                    layer_similarities[num_data_processed] = {layer_i: {layer_j: torch.nn.functional.cosine_similarity(jacobians[layer_i][_data_idx], jacobians[layer_j][_data_idx], dim=1) for layer_j in range(layer_i, num_layers)} for layer_i in range(num_layers)}
                    num_data_processed += 1


            # Check if maximum iterations are reached
            if current_iteration >= args.max_iter:
                break
            current_iteration += 1

    torch.save(results, os.path.join(args.save_path, f'{domain}_{args.model_name}_{args.mode}_results.pth'))



    ### IMAGES ###
    # X = torch.randn(1, 3, 224, 224).to('cuda')
    # inputs = {'pixel_values': X}

    # loaders = get_imagenet_loaders('facebook/deit-tiny', batch_size=2, seed=0)
    # analyzer = BeitForLayerwiseAnalysis('microsoft/beit-base', device='cuda')
    # analyzer = DeiTForLayerwiseAnalysis('facebook/deit-tiny-distilled', device='cuda')
    # analyzer = DeiTForLayerwiseAnalysis('facebook/deit-base', device='cuda')
    # analyzer = DinoV2ForLayerwiseAnalysis('facebook/dinov2-small-imagenet1k-1-layer', device='cuda')

    ### TEXT ###
    # loaders = get_go_emotions_loaders(batch_size=10, seed=0)
    # analyzer = RoBERTaForLayerwiseAnalysis('SamLowe/roberta-base-go_emotions', device='cuda')

    # inputs, labels, ids = next(iter(loaders["test"])).values()
    