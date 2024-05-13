import torch
from ..models.layerwise import LayerWiseAnalysis

def extract_features(inputs, analyzer: LayerWiseAnalysis, register_intermediate: bool = True):
    # Get features
    analyzer.__register_hooks__(register_intermediate=register_intermediate)
    # Get intermediate representations, operations and intermediates (+ features)
    _, _, _ = analyzer.__element_wise_breakdown__(inputs, from_layer=0, within_block=False)
    
    # Extract features
    features = analyzer.features
    analyzer.features = {}
    return features, features.keys().__len__()

def __compute_jacobian__(analyzer, intermediates: dict, operations: dict, **kwargs):

    if kwargs['dataset_name'] == 'imagenet-1k':
        num_classes = 1000
        mini_batches = num_classes // kwargs['c_per_iter']
        
        # Prepare structure
        Js = torch.zeros((kwargs['batch_size'], intermediates.flatten(1).shape[1] * num_classes))
        for _idx, _z in enumerate(intermediates):    
            _Js = []
            for i in range(mini_batches):
                kwargs['pbar'].set_description(kwargs['base_desc'] + f"Processing {_idx+1}/{kwargs['batch_size']} batch idx - " + f"(class {kwargs['c_per_iter']*(i+1)}/{num_classes})") # update pbar info
                
                # Define function (batched on classes due to memory constraints)
                _func = lambda x: analyzer.__get_output_from__(x.unsqueeze(0), operations)[:, i*kwargs['c_per_iter']:(i+1)*kwargs['c_per_iter']]
                # Compute Jacobian per input for selected classes
                J = torch.vmap(torch.func.jacrev(_func), chunk_size=kwargs['jacobian_chunk_size'])(_z.unsqueeze(0))
                _Js.append(J.cpu().squeeze([0,1]).flatten(1))
                
                # Free cache
                torch.cuda.empty_cache()

            # Add to storage
            Js[_idx] = torch.stack(_Js).flatten().unsqueeze(0)
    
    else:
        if kwargs['dataset_name'] == 'go_emotions':
            num_classes = 28
            feature_size = intermediates[0].shape[-1]
        elif kwargs['dataset_name'] == 'speech_commands':
            num_classes = 35
            feature_size = intermediates.flatten(1).shape[1]
        
        Js = torch.zeros((kwargs['batch_size'], feature_size * num_classes))
        for _idx, _z in enumerate(intermediates):    
            kwargs['pbar'].set_description(kwargs['base_desc'] + f"Processing {_idx+1}/{kwargs['batch_size']} batch idx") # update pbar info

            # Get attention outputs if available (within_block mode only)
            attention_output = None if kwargs['info'].get('attention_outputs') in [None, []] else kwargs['info'].get('attention_outputs')[_idx] 
            position_bias = None if kwargs['info'].get('position_bias') in [None, []] else kwargs['info'].get('position_bias')[_idx] 

            if kwargs['dataset_name'] == 'go_emotions':
                # Define function (batched on classes due to memory constraints)
                _func = lambda x: analyzer.__get_output_from__(x.unsqueeze(0).unsqueeze(0), operations, attention_outputs=attention_output)[:, :num_classes]
                # Compute Jacobian per input for selected classes
                J = torch.vmap(torch.func.jacrev(_func), chunk_size=kwargs['jacobian_chunk_size'])(_z).cpu().squeeze([0,1])

            elif kwargs['dataset_name'] == 'speech_commands':
                # Define function (batched on classes due to memory constraints)
                _func = lambda x: analyzer.__get_output_from__(x.unsqueeze(0), operations, attention_outputs=attention_output, **{'position_bias': position_bias})
                # Compute Jacobian per input for selected classes
                J = torch.vmap(torch.func.jacrev(_func), chunk_size=kwargs['jacobian_chunk_size'])(_z.unsqueeze(0)).cpu().squeeze([0,1])
            
            # Add to storage
            Js[_idx] = J.mean(dim=1).flatten().unsqueeze(0) if kwargs['dataset_name'] == 'go_emotions' else J.flatten().unsqueeze(0)
            # Free cache
            torch.cuda.empty_cache()
    
    return Js

def prune_model_by_heuristic(model, layers_to_prune: list):
    # Get the list of encoder layers
    encoder_layers = model.base_model.encoder.layers
    # Prune layers
    for layer_idx in layers_to_prune:
        del encoder_layers[layer_idx]
    # Update the model's encoder layers
    model.base_model.encoder.layers = encoder_layers
    return model