import torch
from ..layerwise import LayerWiseAnalysis
from transformers import DeiTForImageClassificationWithTeacher, ViTForImageClassification
from copy import copy

class DeiTForLayerwiseAnalysis(LayerWiseAnalysis):
      
    def __init__(self, model_name: str, device='cuda', distilled: bool = False):
        super().__init__()

        self.model_name = model_name
        self.device = device
        self.model, self.num_layers = self.load_model()
        self.model.eval()
        self._hooks_registed = False

    def load_model(self):
        # Load pre-trained DeiT model
        if 'distilled' in self.model_name:
            model = DeiTForImageClassificationWithTeacher.from_pretrained(self.model_name + '-patch16-224').to(self.device) 
            num_layers = len(model.deit.encoder.layer)
        else:
            model = ViTForImageClassification.from_pretrained(self.model_name + '-patch16-224').to(self.device)
            num_layers = len(model.vit.encoder.layer)
            
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
        self._register_intermediate = register_intermediate
        for layer_idx in range(self.num_layers):
            if 'distilled' in self.model_name:
                if register_intermediate:
                    self.model.deit.encoder.layer[layer_idx].intermediate.register_forward_hook(self.get_features(f"layer{layer_name}"))
                    self.model.deit.encoder.layer[layer_idx].register_forward_hook(self.get_features(f"layer{layer_name + 1}"))
                    layer_name += 2
                else:
                    self.model.deit.encoder.layer[layer_idx].register_forward_hook(self.get_features(f"layer{layer_idx}"))
            else:
                if register_intermediate:
                    self.model.vit.encoder.layer[layer_idx].intermediate.register_forward_hook(self.get_features(f"layer{layer_name}"))
                    self.model.vit.encoder.layer[layer_idx].register_forward_hook(self.get_features(f"layer{layer_name + 1}"))
                    layer_name += 2
                else:
                    self.model.vit.encoder.layer[layer_idx].register_forward_hook(self.get_features(f"layer{layer_idx}"))    
    
    def __element_wise_breakdown__(self, inputs, from_layer: int, within_block: bool = False):
        # Setup for storing operations from layer
        i = 0
        operations = {}

        _embedder = self.model.deit.embeddings if 'distilled' in self.model_name else self.model.vit.embeddings
        _encoder = self.model.deit.encoder if 'distilled' in self.model_name else self.model.vit.encoder
        _layernorm = self.model.deit.layernorm if 'distilled' in self.model_name else self.model.vit.layernorm
        _classifier = self.model.cls_classifier if 'distilled' in self.model_name else self.model.classifier 

        # Get input embeddings
        z = _embedder(inputs['pixel_values'])
        
        # Pass embeddings through encoder network
        for layer_idx, _layer in enumerate(_encoder.layer):
            if layer_idx == from_layer and within_block:
                raise NotImplementedError("Within block mode is currently broken for DeiT model...")
                
                # Modified from: https://github.com/huggingface/transformers/blob/main/src/transformers/models/deit/modeling_deit.py (line 289-328)                
                    
                self_attention_outputs = _layer.attention(_layer.layernorm_before(z))
                attention_output = self_attention_outputs[0]
                outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

                # first residual connection
                hidden_states = attention_output + z

                # in DeiT, layernorm is also applied after self-attention
                layer_output = _layer.layernorm_after(hidden_states)
                layer_output = _layer.intermediate(layer_output)

                # Get intermediate representation for verification purposes
                layer_output = _layer.output(layer_output, hidden_states) - hidden_states # workaround for dropout in ViTOutput class
                intermediates = layer_output 

                # second residual connection is done here
                layer_output = layer_output + hidden_states
                z = ((layer_output,) + outputs)[0]
                
                # Add operation to operations
                operations[i] = lambda x: (x + hidden_states,) + outputs
                i += 1

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
        z = _layernorm(z)
        z = _classifier(z[:, 0, :])

        operations[i] = _layernorm
        operations[i+1] = lambda x: _classifier(x[:, 0, :])

        _orig_outputs = self.model(**inputs).cls_logits if 'distilled' in self.model_name else self.model(**inputs).logits
        assert torch.allclose(_orig_outputs, self.__get_output_from__(intermediates, operations)), "Encoder block decomposition is incorrect..."
        assert torch.allclose(_orig_outputs, z), "Full model decomposition is incorrect..."

        if self._hooks_registed:
            # Move features to CPU
            for i, (k, v) in enumerate(self.features.items()):
                self.features[k] = v[0].to('cpu') if not self._register_intermediate or int(k[5:]) % 2 == 1 else v.detach().to('cpu')

        return operations, intermediates, {}
    
    def __get_output_from__(self, intermediates: torch.Tensor, operations: dict):
        __z = intermediates
        # iterate through operations of the last part of the network 
        for i, _op in operations.items():
            if _op.__class__.__name__.lower() in ['vitlayer', 'deitlayer'] or i == 0:
                __z = _op(__z)[0]
            else:
                __z = _op(__z)
        return __z