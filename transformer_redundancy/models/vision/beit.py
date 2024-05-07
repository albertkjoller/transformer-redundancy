import torch
from ..layerwise import LayerWiseAnalysis
from transformers import BeitForImageClassification

class BeitForLayerwiseAnalysis(LayerWiseAnalysis):
      
    def __init__(self, model_name: str, device='cuda'):
        super().__init__()

        self.model_name = model_name
        self.device = device
        self.model, self.num_layers = self.load_model()
        self.model.eval()
        self._hooks_registed = False

    def load_model(self):
        # Load pre-trained BEiT model
        model = BeitForImageClassification.from_pretrained(self.model_name + '-patch16-224').to(self.device)
        # Define number of layers
        num_layers = len(model.beit.encoder.layer)
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
            if register_intermediate:
                self.model.beit.encoder.layer[layer_idx].intermediate.register_forward_hook(self.get_features(f"layer{layer_name}"))
                self.model.beit.encoder.layer[layer_idx].register_forward_hook(self.get_features(f"layer{layer_name + 1}"))
                layer_name += 2
            else:
                self.model.beit.encoder.layer[layer_idx].register_forward_hook(self.get_features(f"layer{layer_idx}"))
    
    def __element_wise_breakdown__(self, inputs, from_layer: int, within_block: bool = False):
        # Setup for storing operations from layer
        i = 0
        operations = {}

        # Get input embeddings
        z = self.model.beit.embeddings(inputs['pixel_values'])[0]
        # Pass embeddings through encoder network
        for layer_idx, _layer in enumerate(self.model.beit.encoder.layer):
            if layer_idx == from_layer and within_block:
                raise NotImplementedError("Within block mode is currently broken for BEiT model...")

                # Modified from: https://github.com/huggingface/transformers/blob/main/src/transformers/models/beit/modeling_beit.py (line 415-445)                
                self_attention_outputs = _layer.attention(_layer.layernorm_before(z))
                attention_output = self_attention_outputs[0]
                outputs = self_attention_outputs[1:] # add self attentions if we output attention weights

                # apply lambda_1 if present
                if _layer.lambda_1 is not None:
                    attention_output = _layer.lambda_1 * attention_output

                # first residual connection
                hidden_states = _layer.drop_path(attention_output) + z

                # in BEiT, layernorm is also applied after self-attention
                layer_output = _layer.layernorm_after(hidden_states)
                layer_output = _layer.intermediate(layer_output)
                layer_output = _layer.output(layer_output)

                if _layer.lambda_2 is not None:
                    layer_output = _layer.lambda_2 * layer_output

                # Get intermediate representation for verification purposes
                intermediates = layer_output[0]
                
                # second residual connection
                layer_output = _layer.drop_path(layer_output) + hidden_states
                z = ((layer_output,) + outputs)[0]

                # Add operation to operations
                operations[i] = lambda x: (_layer.drop_path(x) + hidden_states, ) + outputs
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
        z = self.model.beit.layernorm(z)
        z = self.model.beit.pooler(z)
        z = self.model.classifier(z)

        operations[i] = self.model.beit.layernorm
        operations[i+1] = self.model.beit.pooler
        operations[i+2] = self.model.classifier

        _orig_outputs = self.model(**inputs).logits
        assert torch.allclose(_orig_outputs, self.__get_output_from__(intermediates, operations)), "Encoder block decomposition is incorrect..."
        assert torch.allclose(_orig_outputs, z), "Full model decomposition is incorrect..."

        if self._hooks_registed:
            # Move features to CPU
            for i, (k, v) in enumerate(self.features.items()):
                self.features[k] = v[0].to('cpu') if not self._register_intermediate or int(k[5:]) % 2 == 1 else v.detach().to('cpu')

        return operations, intermediates, {}
    
    def __get_output_from__(self, intermediates: dict, operations: dict):
        __z = intermediates
        # iterate through operations of the last part of the network 
        for i, _op in operations.items():
            if _op.__class__.__name__.lower() == 'beitlayer' or i == 0: # in case of the first layer when running with within_block = True in __element_wise_breakdown__
                __z = _op(__z)[0]
            else:
                __z = _op(__z)
        return __z