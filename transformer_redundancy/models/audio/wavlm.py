from typing import Optional

import torch
from ..layerwise import LayerWiseAnalysis
from transformers import AutoModelForAudioClassification

class WavLMForLayerwiseAnalysis(LayerWiseAnalysis):
      
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
            # Load finetuned wavLM model
            model_path = f"{self.model_folder}/{self.model_name}-finetuned" 
        else: # load pre-trained
            model_path = self.model_name

        # Load model
        model = AutoModelForAudioClassification.from_pretrained(model_path).to(self.device)
        
        # Check if model uses stable layers
        self.stable_layers = 'stable' in model.wavlm.encoder.layers[0].__class__.__name__.lower()
        num_layers = model.wavlm.encoder.layers.__len__()
        return model, num_layers
            
    def get_features(self, name: str, is_intermediate: bool = False):
        def hook(module, input, output):
            if is_intermediate:
                self.features[name] = output[0]
            else:
                self.features[name] = output
        return hook
    
    def __register_hooks__(self, register_intermediate: bool = False):
        self.features = {}
        self._hooks_registed = True

        # Register forward hooks
        layer_name = 0
        handles = []
        self._register_intermediate = register_intermediate
        for layer_idx in range(self.num_layers):
            if register_intermediate:
                handles.append(self.model.wavlm.encoder.layers[layer_idx].feed_forward.register_forward_hook(self.get_features(f"layer{layer_name}")))
                handles.append(self.model.wavlm.encoder.layers[layer_idx].register_forward_hook(self.get_features(f"layer{layer_name + 1}")))
                layer_name += 2
            else:
                handles.append(self.model.wavlm.encoder.layers[layer_idx].register_forward_hook(self.get_features(f"layer{layer_idx}")))    
        return handles
    
    def __element_wise_breakdown__(self, inputs, from_layer: int, within_block: bool = False):
        # Setup for storing operations from layer
        i = 0
        operations = {}

        # Get input embeddings
        z = self.model.wavlm.feature_extractor(inputs).transpose(1, 2)
        z, extracted_features = self.model.wavlm.feature_projection(z)
        
        z = z + self.model.wavlm.encoder.pos_conv_embed(z)
        if not self.stable_layers:
            z = self.model.wavlm.encoder.layer_norm(z)
        z = self.model.wavlm.encoder.dropout(z)

        _pos_bias = None
        # Pass embeddings through encoder network
        for layer_idx, _layer in enumerate(self.model.wavlm.encoder.layers):
            if layer_idx == from_layer and within_block:
                raise NotImplementedError("Within block mode is currently not implemented for Wav2vec...")

            else: # between encoder blocks

                # Get intermediate representation for verification purposes 
                if layer_idx == from_layer:
                    intermediates = z

                z, _pos_bias = _layer(z,  position_bias=_pos_bias)[:2]
                if layer_idx == from_layer:
                    position_bias = _pos_bias

                if layer_idx >= from_layer:
                    # Add operation to operations
                    operations[i] = _layer
                    i += 1

        if self.stable_layers:
            z = self.model.wavlm.encoder.layer_norm(z)
            operations[i] = self.model.wavlm.encoder.layer_norm
            i += 1  

        # Pass through remaining layers
        z = self.model.projector(z)
        z = self.model.classifier(z.mean(dim=1)) # pool and classify
        
        operations[i] = self.model.projector
        operations[i+1] = lambda x: self.model.classifier(x.mean(dim=1))

        _orig_outputs = self.model(inputs).logits
        assert torch.allclose(_orig_outputs, self.__get_output_from__(intermediates, operations, position_bias=_pos_bias), atol=1e-3, rtol=1e-5), "Encoder block decomposition is incorrect..."
        assert torch.allclose(_orig_outputs, z, atol=1e-3, rtol=1e-5), "Full model decomposition is incorrect..."

        if self._hooks_registed:
            # Move features to CPU
            for i, (k, v) in enumerate(self.features.items()):
                self.features[k] = v[0].to('cpu') if not self._register_intermediate or int(k[5:]) % 2 == 1 else v.detach().to('cpu')

        return operations, intermediates, {'position_bias': position_bias}
    
    def __get_output_from__(self, intermediates: torch.Tensor, operations: dict, **kwargs):
        __z = intermediates
        _pos_bias = kwargs['position_bias']

        # iterate through operations of the last part of the network 
        for i, _op in operations.items():
            if _op.__class__.__name__.lower() in ['wavlmencoderlayer', 'wavlmencoderlayerstablelayernorm'] or i == 0: 
                __z, _pos_bias = _op(__z, position_bias=_pos_bias)
            else:
                __z = _op(__z)
        return __z