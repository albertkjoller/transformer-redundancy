from typing import Optional

import torch
from copy import deepcopy
from ..layerwise import LayerWiseAnalysis
from transformers import AutoModelForAudioClassification

class Wav2VecForLayerwiseAnalysis(LayerWiseAnalysis):
      
    def __init__(self, model_name: str, model_folder: Optional[str] = None, device='cuda'):
        super().__init__()

        self.model_name = model_name
        self.model_folder = model_folder
        self.device = device
        self.model, self.num_layers = self.load_model()
        self.model.eval()
        self._hooks_registed = False

    def load_model(self):
        if self.model_folder not in [None, "None"]:
            # Load finetuned Wav2Vec model
            model_path = f"{self.model_folder}/{self.model_name}-finetuned" 
        else: # load pre-trained
            model_path = self.model_name # 'facebook/wav2vec2-base'
        
        # Load model
        model = AutoModelForAudioClassification.from_pretrained(model_path).to(self.device)        
        num_layers = model.wav2vec2.encoder.layers.__len__()
        return model, num_layers
            
    def get_features(self, name: str, is_intermediate: bool = False):
        def hook(module, input, output):
            if is_intermediate:
                self.features[name] = output[0]
            else:
                self.features[name] = output
        return hook
    
    def __register_hooks__(self, register_intermediate: bool = False, layers: list = None, register_init_embedding: bool = False, **kwargs):
        self.features = {}
        self._hooks_registed = True

        # Register forward hooks
        layer_name = 0
        handles = []
        self._register_intermediate = register_intermediate
        
        if layers is None:
            layers = range(self.num_layers)
        
        if register_init_embedding:
            handles.append(self.model.wav2vec2.feature_projection.register_forward_hook(self.get_features("feature_projection")))

        for layer_idx in layers:
            if register_intermediate:
                handles.append(self.model.wav2vec2.encoder.layers[layer_idx].feed_forward.register_forward_hook(self.get_features(f"layer{layer_name}")))
                handles.append(self.model.wav2vec2.encoder.layers[layer_idx].register_forward_hook(self.get_features(f"layer{layer_name + 1}")))
                layer_name += 2
            else:
                handles.append(self.model.wav2vec2.encoder.layers[layer_idx].register_forward_hook(self.get_features(f"layer{layer_idx}")))
        return handles
    
    def __element_wise_breakdown__(self, inputs, from_layer: int, within_block: bool = False):
        # Setup for storing operations from layer
        i = 0
        operations = {}

        # Get input embeddings
        z = self.model.wav2vec2.feature_extractor(inputs).transpose(1, 2)
        z, extracted_features = self.model.wav2vec2.feature_projection(z)
        
        z = z + self.model.wav2vec2.encoder.pos_conv_embed(z)
        z = self.model.wav2vec2.encoder.layer_norm(z)
        z = self.model.wav2vec2.encoder.dropout(z)

        # Pass embeddings through encoder network
        for layer_idx, _layer in enumerate(self.model.wav2vec2.encoder.layers):
            if layer_idx == from_layer and within_block:
                raise NotImplementedError("Within block mode is currently not implemented for Wav2vec...")

            else: # between encoder blocks
                # Get intermediate representation for verification purposes 
                if layer_idx == from_layer:
                    intermediates = z

                z = _layer(z)[0]
                if layer_idx >= from_layer:
                    # Add operation to operations
                    operations[i] = _layer
                    i += 1

        # Pass through remaining layers
        z = self.model.projector(z)
        z = self.model.classifier(z.mean(dim=1)) # pool and classify

        operations[i] = self.model.projector
        operations[i+1] = lambda x: self.model.classifier(x.mean(dim=1))

        _orig_outputs = self.model(inputs).logits
        assert torch.allclose(_orig_outputs, self.__get_output_from__(intermediates, operations)), "Encoder block decomposition is incorrect..."
        assert torch.allclose(_orig_outputs, z), "Full model decomposition is incorrect..."

        if self._hooks_registed:
            # Move features to CPU
            for i, (k, v) in enumerate(self.features.items()):
                self.features[k] = v[0].to('cpu') if not self._register_intermediate or int(k[5:]) % 2 == 1 else v.detach().to('cpu')

        return operations, intermediates, {}
    
    def __get_output_from__(self, intermediates: torch.Tensor, operations: dict, **kwargs):
        __z = intermediates
        # iterate through operations of the last part of the network 
        for i, _op in operations.items():
            if _op.__class__.__name__.lower() == 'wav2vec2encoderlayer' or i == 0: 
                __z = _op(__z)[0]
            else:
                __z = _op(__z)
        return __z
    
    def __average_layers__(self, model, community_filepath: str):
        # Load community
        all_communities = torch.load(community_filepath)

        with torch.no_grad():
            average_layers = []
            for _community in all_communities['wav2vec2-words']:
                for c_idx, layer_idx in enumerate(_community):
                    layer = model.wav2vec2.encoder.layers[layer_idx] # get layer
                    if c_idx == 0: # initialize new layer dict if first layer in community
                        new_layer_dict = dict(layer.named_parameters())
                    else: # add parameters to new layer dict
                        for n, p in layer.named_parameters():
                            new_layer_dict[n] += p
                
                # Average parameters
                for k, v in new_layer_dict.items():
                    new_layer_dict[k] /= len(_community)
                
                # Load weights onto layer structure
                new_layer = deepcopy(layer)
                new_layer.load_state_dict(new_layer_dict)
                average_layers.append(new_layer)

            # Replace layers with average layers
            model.wav2vec2.encoder.layers = torch.nn.ModuleList(average_layers)