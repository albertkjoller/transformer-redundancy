import torch
from transformer_redundancy.data import *
from transformer_redundancy.models.vision import *
from transformer_redundancy.models.text import *
from transformer_redundancy.models.audio import *
from transformer_redundancy.methods.utils import extract_features, __compute_jacobian__, prune_model_by_heuristic, prune_model_backward
from transformer_redundancy.methods.cka import compute_cka_from_tensors
from transformer_redundancy.methods.procrustes import procrustes_similarity
from transformer_redundancy.methods.mutual_knn import mutual_knn

def get_domain(args):
    if args.dataset_name in ['imagenet-1k']:
        return 'vision'
    elif args.dataset_name in ['go_emotions']:
        return 'text'
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
    elif 'wav2vec2' in args.model_name:
        return Wav2VecForLayerwiseAnalysis(args.model_name, model_folder=args.model_folder, device=args.device)
    elif 'wavlm' in args.model_name.lower():
        return WavLMForLayerwiseAnalysis(args.model_name, model_folder=args.model_folder, device=args.device)

if __name__ == '__main__':
    
    import os
    from collections import defaultdict
    import argparse
    from dotenv import load_dotenv
    from huggingface_hub import login
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description='Run Visual Transformer experiments.')
    ### Experiment parameters ###
    parser.add_argument('mode', type=str, nargs='+', choices=['pruning-performance', 'cosine-similarity', 'cka-similarity', 'jacobian-similarity', 'procrustes-similarity', 'mutual-knn'])
    parser.add_argument('--within-block', action='store_true')
    parser.add_argument('--jacobian-between-layers', action='store_true')
    parser.add_argument('--jacobian-chunk-size', type=int, default=1)
    parser.add_argument('--jacobian-single-target', action='store_true')
    parser.add_argument('--prune-by', nargs='+', choices=['backward', 'forward', 'forward-backward', 'block-influence', 'knn-block-influence', 'multi-block-influence', 'jacobian-rank'], default=[])
    parser.add_argument('--prune-amount-range', nargs='+', type=int, help='Number of layers to prune.', default=[])
    parser.add_argument('--custom-pruning', nargs='+', type=int, help='Order.', default=[])
    parser.add_argument('--layers-core-first', nargs='+', type=int, help='Index.', default=[])
    parser.add_argument('--c-per-iter', type=int, default=20)
    parser.add_argument('--max-iter', type=int, default=50)
    parser.add_argument('--start-iter', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save-path', type=str, default='../experiments', help='Directory to save the bsub files.')
    ### Data parameters ###
    parser.add_argument('--dataset-name', type=str, choices=['coco', 'imagenet-1k', 'go_emotions', 'speech_commands'])
    parser.add_argument('--coco-path', type=str, default=None)
    parser.add_argument('--processor-name', type=str)
    parser.add_argument('--batch-size', type=int, default=128)
    ### Model parameters ###
    parser.add_argument('--model-name', type=str)
    parser.add_argument('--model-folder', type=str, default=None)
    parser.add_argument('--distilled', action='store_true')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'])
    # Parse arguments
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()
    login(os.getenv('HF_TOKEN'))
    # os.environ["HF_HOME"] = os.getenv('HF_HOME')

    # Clear pytorch cache
    torch.cuda.empty_cache()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Create save path
    os.makedirs(args.save_path, exist_ok=True)

    # Get data domain and loaders
    domain = get_domain(args)
    loaders, label2cat = get_loaders(**args.__dict__)
    analyzer = get_analyzer(args)
    save_filename = f'{args.save_path}/{domain}/{args.model_name.split("/")[-1]}/within={args.within_block}_{args.mode}_start-iter={args.start_iter}'
    print(save_filename)

    # Setup storage system
    results = {
        'accuracy': {k: defaultdict(list) for k in args.prune_by + ['custom', 'core-first-block-influence']},
        'all-cosine-similarities': [],
        'all-cka-similarities': [],
        'all-mutual_knn-similarities': {'within': [], 'between': []},
        'all-procrustes-similarities': [],
        'all-block-influences': [],
        'all-core-last-block-influences': [],
        'all-knn-block-influences': [],
        'all-multi-block-influences': [],
        'all-jacobian-batch-similarities': [],
        'all-jacobian-layer-similarities': [],
    }

    ### FEATURE EXTRACTION AND SIMILARITY COMPUTATION ###
    current_iteration = 0
    args.max_iter = args.max_iter + args.start_iter
    pbar = tqdm(total=args.max_iter)
    with torch.no_grad():
        for batch in loaders["validation"]:
            if current_iteration < args.start_iter:
                current_iteration += 1
                pbar.update(1)
            
            else:
                if args.dataset_name == 'imagenet-1k':
                    inputs = {'pixel_values': batch[0].to(args.device)}
                    labels = batch[1].to('cpu') if args.jacobian_single_target else None
                elif args.dataset_name == 'go_emotions':
                    inputs = batch["text"]
                    labels = torch.tensor([_label[0] for _label in batch["labels"]]) if args.jacobian_single_target else None
                elif args.dataset_name == 'speech_commands':
                    inputs = batch["input_values"].to(args.device)
                    labels = batch["label"] if args.jacobian_single_target else None
                elif args.dataset_name == 'coco':
                    inputs = {k: v.to(args.device) for k, v in loaders["test"].dataset.__get_batch_repr__(batch).items()}
                    labels = None
                    
                if any([_m in ['cosine-similarity', 'cka-similarity', 'procrustes-similarity', 'mutual-knn'] for _m in args.mode]) or 'block-influence' in args.prune_by:
                    pbar.set_description(f"Iteration {current_iteration}/{args.max_iter}: Computing feature similarities...") # set pbar description

                    # Extract intermediate features (within and between transformer encoder blocks)
                    features, n_feature_layers = extract_features(inputs, analyzer, register_intermediate=args.within_block)

                    # Compute block inference scores
                    if 'block-influence' in args.prune_by:
                        assert not args.within_block, "Block inference scores can only be computed between blocks."
                        features_between_blocks = {i: features[f"layer{k}"] for i, k in enumerate(range(n_feature_layers))}

                        bi_scores = torch.zeros((args.batch_size, analyzer.num_layers - 1))
                        for layer_i in range(analyzer.num_layers - 1):
                            # Compute elements of block inference scores                        
                            dot_prod = torch.einsum('id,id->i', features_between_blocks[layer_i].flatten(1), features_between_blocks[layer_i+1].flatten(1))
                            norm_before = torch.linalg.norm(features_between_blocks[layer_i].flatten(1), ord=2, dim=1)
                            norm_after = torch.linalg.norm(features_between_blocks[layer_i+1].flatten(1), ord=2, dim=1)

                            # Compute block inference scores
                            bi_scores[:, layer_i] = 1 - dot_prod / (norm_before * norm_after)
                        
                        # Store block inference scores
                        results['all-block-influences'].append(bi_scores)

                    if 'knn-block-influence' in args.prune_by:
                        assert not args.within_block, "Block inference scores can only be computed between blocks."
                        features_between_blocks = {i: features[f"layer{k}"] for i, k in enumerate(range(n_feature_layers))}

                        knn_bi = torch.stack([torch.tensor([1 - mutual_knn(features_between_blocks[_layer].flatten(1), features_between_blocks[_layer+1].flatten(1), topk=_k) for _layer in range(analyzer.num_layers - 1)]) for _k in [2, 4, 8, 16, 32, 64]])
                        results['all-knn-block-influences'].append(knn_bi)

                    if 'multi-block-influence' in args.prune_by:
                        assert not args.within_block, "Block inference scores can only be computed between blocks."
                        features_between_blocks = {i: features[f"layer{k}"] for i, k in enumerate(range(n_feature_layers))}

                        knn_mbi = torch.stack([torch.stack([torch.tensor([1 - mutual_knn(features_between_blocks[layer_i].flatten(1), features_between_blocks[layer_j].flatten(1), topk=_k) for layer_i in range(analyzer.num_layers)]) for layer_j in range(analyzer.num_layers)]) for _k in [2, 4, 8, 16, 32, 64]])
                        knn_mbi = knn_mbi.mean(dim=1)
                        results['all-multi-block-influences'].append(knn_mbi)

                    if 'procrustes-similarity' in args.mode:
                        # Extract features for withing and between encoder blocks
                        within_features = torch.stack([features[f'layer{i}'].flatten(1) for i in range(n_feature_layers)[::2]]).permute(1,0,2)
                        # within_features = torch.stack([features[f'layer{i}'].mean(dim=1).flatten(1) for i in range(n_feature_layers)[::2]]).permute(1,0,2)
                        between_features = torch.stack([features[f'layer{i}'].flatten(1) for i in range(n_feature_layers)[1::2]]).permute(1,0,2)
                        # between_features = torch.stack([features[f'layer{i}'].mean(dim=1).flatten(1) for i in range(n_feature_layers)[1::2]]).permute(1,0,2)

                        proc_sims_within = [procrustes_similarity(within_features[:, k, :], within_features[:, k+1, :])[0] for k in range(analyzer.num_layers - 1)]
                        proc_sims_between = [procrustes_similarity(between_features[:, k, :], between_features[:, k+1, :])[0] for k in range(analyzer.num_layers - 1)]

                        results['all-procrustes-similarities'].append({'within': proc_sims_within, 'between': proc_sims_between})
                                                                            
                    if 'cosine-similarity' in args.mode:
                        # Extract features for withing and between encoder blocks
                        within_features = torch.stack([features[f'layer{i}'].flatten(1) for i in range(n_feature_layers)[::2]]).permute(1,0,2)
                        between_features = torch.stack([features[f'layer{i}'].flatten(1) for i in range(n_feature_layers)[1::2]]).permute(1,0,2)

                        # Compute cosine similarities between intermediate features and input
                        cos_similarities_within = torch.zeros((n_feature_layers // 2, n_feature_layers // 2))
                        cos_similarities_between = torch.zeros((n_feature_layers // 2, n_feature_layers // 2))
                        # Iterate through batch members
                        for _idx in range(args.batch_size):
                            # Even features
                            normalized_within_features = torch.nn.functional.normalize(within_features[_idx], p=2, dim=1)
                            cos_similarities_within += torch.mm(normalized_within_features, normalized_within_features.T)
                            # Odd features                    
                            normalized_odd_features = torch.nn.functional.normalize(between_features[_idx], p=2, dim=1)
                            cos_similarities_between += torch.mm(normalized_odd_features, normalized_odd_features.T)
                        
                        # Take average of batch and store results
                        cos_similarities_within /= args.batch_size
                        cos_similarities_between /= args.batch_size
                        results['all-cosine-similarities'].append({'within': cos_similarities_within, 'between': cos_similarities_between})

                    if 'cka-similarity' in args.mode:
                        # Compute CKA similarities between intermediate features and input
                        cka_similarities = torch.zeros((n_feature_layers, n_feature_layers))
                        for layer_i in range(n_feature_layers):
                            for layer_j in range(layer_i+1, n_feature_layers):
                                cka_similarities[layer_i, layer_j] = compute_cka_from_tensors(features[f'layer{layer_i}'], features[f'layer{layer_j}'], kernel_func='linear')
                        
                        cka_similarities += cka_similarities.T + torch.eye(n_feature_layers) # Symmetrize
                        results['all-cka-similarities'].append(cka_similarities)

                    if 'mutual-knn' in args.mode:

                        within_features = torch.stack([features[f'layer{i}'].flatten(1) for i in range(n_feature_layers)[::2]]).permute(1,0,2)
                        between_features = torch.stack([features[f'layer{i}'].flatten(1) for i in range(n_feature_layers)[1::2]]).permute(1,0,2)

                        top_ks = [2**i for i in range(10) if 2**i <= args.batch_size]
                        for _feats_type, _feats in [('within', within_features), ('between', between_features)]:
                            mutual_knn_sims = {_k: torch.zeros((analyzer.num_layers, analyzer.num_layers)) for _k in top_ks}
                            for _top_k in top_ks:
                                pbar.set_description(f"Iteration {current_iteration}/{args.max_iter}: Mutual kNN ({_feats_type} - k={_top_k})") # set pbar description
                                for _i in range(analyzer.num_layers):
                                    for _j in range(_i, analyzer.num_layers):
                                        mutual_knn_sims[_top_k][_i, _j] = mutual_knn_sims[_top_k][_j, _i] = mutual_knn(_feats[:, _i, :], _feats[:, _j, :], topk=_top_k)
                            results['all-mutual_knn-similarities'][_feats_type].append(mutual_knn_sims)

                if 'jacobian-similarity' in args.mode:
                    jacobians, jacobian_similarities_batch = {}, {}
                    for k in range(analyzer.num_layers):
                        base_desc = f"Iteration {current_iteration}/{args.max_iter}: Transformer layer {k+1}/{analyzer.num_layers} ==> " # set pbar description

                        # Get intermediate representations, operations and intermediates (+ features)
                        operations, intermediates, info = analyzer.__element_wise_breakdown__(inputs, from_layer=k, within_block=args.within_block)
                        # Compute Jacobian for each data point in the batch
                        J = __compute_jacobian__(analyzer, intermediates=intermediates, operations=operations, target_labels=labels, pbar=pbar, current_iteration=current_iteration, k=k, info=info, base_desc=base_desc, **args.__dict__)
                        
                        # Store Jacobians for computing similarities between layers
                        if args.jacobian_between_layers:
                            jacobians[k] = J

                        # Compute and store similarities between Jacobians across the batch
                        J_norm = torch.nn.functional.normalize(J, p=2, dim=1)
                        jacobian_similarities_batch[f'layer{k}'] = torch.mm(J_norm, J_norm.T)

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
                if current_iteration >= args.max_iter - 1:
                    break
                current_iteration += 1
                pbar.update(1)




    ### PRUNING PERFORMANCE ###
    current_iteration = 0
    pbar = tqdm(total=args.max_iter)
    if 'pruning-performance' in args.mode:
        if args.prune_amount_range == []:
            args.prune_amount_range = [0, analyzer.num_layers]

        assert args.prune_by != [] or args.custom_pruning != [], "Prune by must be specified to compute performance."
        save_filename += f'_pruned-by={args.prune_by}'

        with torch.no_grad():
            for batch in loaders["validation"]:
                if current_iteration < args.start_iter:
                    current_iteration += 1
                else:
                    if args.dataset_name == 'imagenet-1k':
                        inputs = batch[0].to(args.device)
                        labels = batch[1].to('cpu')
                    elif args.dataset_name == 'go_emotions':
                        inputs = batch["text"]
                        labels = torch.tensor([_labels[0] for _labels in batch["labels"]]).flatten() # disregard multi-label format
                    elif args.dataset_name == 'speech_commands':
                        inputs = batch["input_values"].to(args.device)
                        labels = batch["label"]
                    # elif args.dataset_name == 'coco':

                    if 'backward' in args.prune_by:
                        assert args.prune_amount_range != [], "Prune amount must be specified."
                        pbar.set_description(f"Iteration {current_iteration}/{args.max_iter}: Computing backward-pruned accuracies...") # set pbar description

                        for _prune_amount in range(args.prune_amount_range[0], args.prune_amount_range[1]):
                            # Currently only works for audio-models
                            if args.model_name == 'SamLowe/roberta-base-go_emotions':
                                _model, _ = analyzer.load_model()
                                _model.model = prune_model_backward(_model.model, analyzer.num_layers - _prune_amount) # remove last layers first
                            else:
                                _model, _ = analyzer.load_model()
                                _model = prune_model_backward(_model, analyzer.num_layers - _prune_amount) # remove last layers first

                            # Compute accuracy
                            preds = torch.tensor([label2cat[pred[0]["label"]] for pred in _model(inputs)]).cpu() if args.dataset_name == 'go_emotions' else torch.argmax(_model(inputs).logits, dim=1).cpu()
                            results['accuracy']['backward'][_prune_amount].append((preds == labels).sum().item() / args.batch_size)

                    if 'forward' in args.prune_by:
                        pbar.set_description(f"Iteration {current_iteration}/{args.max_iter}: Computing forward-pruned accuracies...") # set pbar description

                        # Compute prune order
                        _prune_order = torch.arange(1, analyzer.num_layers)
                        for _prune_amount in range(args.prune_amount_range[0], min(len(_prune_order), args.prune_amount_range[1])):
                            # Prune model by forward
                            if args.model_name == 'SamLowe/roberta-base-go_emotions':
                                _model, _ = analyzer.load_model()
                                _model.model = prune_model_by_heuristic(_model.model, layers_to_prune=sorted(_prune_order[:_prune_amount].numpy(), reverse=True)) # remove later layers first according to block inference scores
                            else:
                                _model, _ = analyzer.load_model()
                                _model = prune_model_by_heuristic(_model, layers_to_prune=sorted(_prune_order[:_prune_amount].numpy(), reverse=True)) # remove later layers first according to block inference scores

                            # Compute accuracy
                            preds = torch.tensor([label2cat[pred[0]["label"]] for pred in _model(inputs)]) if args.dataset_name == 'go_emotions' else torch.argmax(_model(inputs).logits, dim=1).cpu()
                            results['accuracy']['forward'][_prune_amount].append((preds == labels).sum().item() / args.batch_size)

                    if args.custom_pruning != []:
                        pbar.set_description(f"Iteration {current_iteration}/{args.max_iter}: Computing customly pruned accuracies...") # set pbar description

                        _prune_order = torch.tensor(args.custom_pruning)
                        for _prune_amount in range(args.prune_amount_range[0], min(len(_prune_order), args.prune_amount_range[1])):
                            # Prune model by forward
                            if args.model_name == 'SamLowe/roberta-base-go_emotions':
                                _model, _ = analyzer.load_model()
                                _model.model = prune_model_by_heuristic(_model.model, layers_to_prune=sorted(_prune_order[:_prune_amount].numpy(), reverse=True)) # remove later layers first according to block inference scores
                            else:
                                _model, _ = analyzer.load_model()
                                _model = prune_model_by_heuristic(_model, layers_to_prune=sorted(_prune_order[:_prune_amount].numpy(), reverse=True)) # remove later layers first according to block inference scores

                            # Compute accuracy
                            preds = torch.tensor([label2cat[pred[0]["label"]] for pred in _model(inputs)]) if args.dataset_name == 'go_emotions' else torch.argmax(_model(inputs).logits, dim=1).cpu()
                            results['accuracy']['custom'][_prune_amount].append((preds == labels).sum().item() / args.batch_size)

                    if 'forward-backward' in args.prune_by:
                        pbar.set_description(f"Iteration {current_iteration}/{args.max_iter}: Computing forward/backward-pruned accuracies...") # set pbar description

                        # Compute prune order                        
                        _prune_order = torch.tensor([k//2 if i%2 == 0 else analyzer.num_layers - k//2 for (i, k) in enumerate(range(2, analyzer.num_layers+1))])
                        for _prune_amount in range(args.prune_amount_range[0], min(len(_prune_order), args.prune_amount_range[1])):
                            # Prune model by forward
                            if args.model_name == 'SamLowe/roberta-base-go_emotions':
                                _model, _ = analyzer.load_model()
                                _model.model = prune_model_by_heuristic(_model.model, layers_to_prune=sorted(_prune_order[:_prune_amount].numpy(), reverse=True)) # remove later layers first according to block inference scores
                            else:
                                _model, _ = analyzer.load_model()
                                _model = prune_model_by_heuristic(_model, layers_to_prune=sorted(_prune_order[:_prune_amount].numpy(), reverse=True)) # remove later layers first according to block inference scores

                            # Compute accuracy
                            preds = torch.tensor([label2cat[pred[0]["label"]] for pred in _model(inputs)]) if args.dataset_name == 'go_emotions' else torch.argmax(_model(inputs).logits, dim=1).cpu()
                            results['accuracy']['forward-backward'][_prune_amount].append((preds == labels).sum().item() / args.batch_size)

                    if 'block-influence' in args.prune_by:
                        pbar.set_description(f"Iteration {current_iteration}/{args.max_iter}: Computing BI-pruned accuracies...") # set pbar description

                        # Compute prune order
                        _prune_order = torch.argsort(torch.mean(torch.vstack(results['all-block-influences']), dim=0), descending=False) + 1 # Skip first layer
                        for _prune_amount in range(args.prune_amount_range[0], min(len(_prune_order), args.prune_amount_range[1])):
                            # Prune model by block inference scores
                            if args.model_name == 'SamLowe/roberta-base-go_emotions':
                                _model, _ = analyzer.load_model()
                                _model.model = prune_model_by_heuristic(_model.model, layers_to_prune=sorted(_prune_order[:_prune_amount], reverse=True)) # remove later layers first according to block inference scores
                            else:
                                _model, _ = analyzer.load_model()
                                _model = prune_model_by_heuristic(_model, layers_to_prune=sorted(_prune_order[:_prune_amount], reverse=True)) # remove later layers first according to block inference scores

                            # Compute accuracy
                            preds = torch.tensor([label2cat[pred[0]["label"]] for pred in _model(inputs)]) if args.dataset_name == 'go_emotions' else torch.argmax(_model(inputs).logits, dim=1).cpu()
                            results['accuracy']['block-influence'][_prune_amount].append((preds == labels).sum().item() / args.batch_size)

                        if args.layers_core_first != []:
                            pbar.set_description(f"Iteration {current_iteration}/{args.max_iter}: Computing core-first BI-pruned accuracies...") # set pbar description

                            # Remove core layers first
                            _prune_order = list(_prune_order.numpy())
                            for _l in args.layers_core_first:
                                _prune_order.remove(_l)
                            # Define new pruning order
                            _prune_order = args.layers_core_first + _prune_order
                            for _prune_amount in range(args.prune_amount_range[0], min(len(_prune_order), args.prune_amount_range[1])):
                                # Prune model by block inference scores
                                if args.model_name == 'SamLowe/roberta-base-go_emotions':
                                    _model, _ = analyzer.load_model()
                                    _model.model = prune_model_by_heuristic(_model.model, layers_to_prune=sorted(_prune_order[:_prune_amount], reverse=True)) # remove later layers first according to block inference scores
                                else:
                                    _model, _ = analyzer.load_model()
                                    _model = prune_model_by_heuristic(_model, layers_to_prune=sorted(_prune_order[:_prune_amount], reverse=True)) # remove later layers first according to block inference scores

                                # Compute accuracy
                                preds = torch.tensor([label2cat[pred[0]["label"]] for pred in _model(inputs)]) if args.dataset_name == 'go_emotions' else torch.argmax(_model(inputs).logits, dim=1).cpu()
                                results['accuracy']['core-first-block-influence'][_prune_amount].append((preds == labels).sum().item() / args.batch_size)

                    if 'knn-block-influence' in args.prune_by:
                        pbar.set_description(f"Iteration {current_iteration}/{args.max_iter}: Computing kNN-BI-pruned accuracies...") # set pbar description

                        # Compute prune order
                        _prune_order = torch.argsort(torch.mean(torch.vstack(results['all-knn-block-influences']), dim=0), descending=False) + 1 # Skip first layer
                        for _prune_amount in range(args.prune_amount_range[0], min(len(_prune_order), args.prune_amount_range[1])):
                            # Prune model by block inference scores
                            if args.model_name == 'SamLowe/roberta-base-go_emotions':
                                _model, _ = analyzer.load_model()
                                _model.model = prune_model_by_heuristic(_model.model, layers_to_prune=sorted(_prune_order[:_prune_amount], reverse=True)) # remove later layers first according to block inference scores
                            else:
                                _model, _ = analyzer.load_model()
                                _model = prune_model_by_heuristic(_model, layers_to_prune=sorted(_prune_order[:_prune_amount], reverse=True)) # remove later layers first according to block inference scores

                            # Compute accuracy
                            preds = torch.tensor([label2cat[pred[0]["label"]] for pred in _model(inputs)]) if args.dataset_name == 'go_emotions' else torch.argmax(_model(inputs).logits, dim=1).cpu()
                            results['accuracy']['knn-block-influence'][_prune_amount].append((preds == labels).sum().item() / args.batch_size)

                    if 'multi-block-influence' in args.prune_by:
                        pbar.set_description(f"Iteration {current_iteration}/{args.max_iter}: Computing kNN-mBI-pruned accuracies...") # set pbar description

                        # Compute prune order
                        _prune_order = torch.argsort(torch.mean(torch.vstack(results['all-multi-block-influences']), dim=0), descending=False) + 1 # Skip first layer
                        for _prune_amount in range(args.prune_amount_range[0], min(len(_prune_order), args.prune_amount_range[1])):
                            # Prune model by block inference scores
                            if args.model_name == 'SamLowe/roberta-base-go_emotions':
                                _model, _ = analyzer.load_model()
                                _model.model = prune_model_by_heuristic(_model.model, layers_to_prune=sorted(_prune_order[:_prune_amount], reverse=True)) # remove later layers first according to block inference scores
                            else:
                                _model, _ = analyzer.load_model()
                                _model = prune_model_by_heuristic(_model, layers_to_prune=sorted(_prune_order[:_prune_amount], reverse=True)) # remove later layers first according to block inference scores

                            # Compute accuracy
                            preds = torch.tensor([label2cat[pred[0]["label"]] for pred in _model(inputs)]) if args.dataset_name == 'go_emotions' else torch.argmax(_model(inputs).logits, dim=1).cpu()
                            results['accuracy']['multi-block-influence'][_prune_amount].append((preds == labels).sum().item() / args.batch_size)

                    if 'jacobian-rank' in args.prune_by:
                        assert 'jacobian-similarity' in args.mode, "Jacobian similarity scores can only be computed with Jacobian similarities."
                        pbar.set_description(f"Iteration {current_iteration}/{args.max_iter}: Computing Jacobian-pruned accuracies...") # set pbar description

                        # Compute relative eigenvalues (maximum)
                        rel_eigvals = torch.zeros((len(results['all-jacobian-batch-similarities']), analyzer.num_layers))
                        for i, _batch_sim_output in enumerate(results['all-jacobian-batch-similarities']):
                            for j, (k, _batch_sim_output) in enumerate(_batch_sim_output.items()):
                                eigvals, _ = torch.linalg.eigh(_batch_sim_output)
                                rel_eigvals[i, j] = eigvals.max() / sum(eigvals)

                        # Compute prune order                       
                        _prune_order = torch.argsort(torch.mean(rel_eigvals, dim=0), descending=True)
                        _prune_order = _prune_order[_prune_order != 0]
                        for _prune_amount in range(args.prune_amount_range[0], args.prune_amount_range[1]):
                            # Prune model by Jacobian rank
                            if args.model_name == 'SamLowe/roberta-base-go_emotions':
                                _model, _ = analyzer.load_model()
                                _model.model = prune_model_by_heuristic(_model.model, layers_to_prune=sorted(_prune_order[:_prune_amount], reverse=True)) # remove later layers first according to block inference scores
                            else:
                                _model, _ = analyzer.load_model()
                                _model = prune_model_by_heuristic(_model, layers_to_prune=sorted(_prune_order[:_prune_amount], reverse=True)) # remove later layers first according to block inference scores

                            # Compute accuracy
                            preds = torch.tensor([label2cat[pred[0]["label"]] for pred in _model(inputs)]) if args.dataset_name == 'go_emotions' else torch.argmax(_model(inputs).logits, dim=1).cpu()
                            results['accuracy']['jacobian-rank'][_prune_amount].append((preds == labels).sum().item() / args.batch_size)

                    # Check if maximum iterations are reached
                    if current_iteration >= args.max_iter - 1:
                        break
                    current_iteration += 1
                    pbar.update(1)

    os.makedirs(f"{args.save_path}/{domain}/{args.model_name.split('/')[-1]}", exist_ok=True)
    torch.save(results, save_filename + f'_results.pth')