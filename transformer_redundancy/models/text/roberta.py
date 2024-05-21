import torch
from ..layerwise import LayerWiseAnalysis
from transformers import pipeline
from collections import defaultdict

class RoBERTaForLayerwiseAnalysis(LayerWiseAnalysis):
      
    def __init__(self, model_name: str, device='cuda'):
        super().__init__()

        self.model_name = model_name
        self.device = device
        self.pipeline, self.num_layers = self.load_model()
        self.pipeline.model.eval()
        self._hooks_registed = False

    def load_model(self):
        # Load pre-trained RoBERTa model
        _pipeline = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None, device=self.device)
        num_layers = _pipeline.model.roberta.encoder.layer.__len__()
        return _pipeline, num_layers

    def get_features(self, name: str, is_intermediate: bool = False):
        def hook(model, input, output):
            if is_intermediate:
                self.features[name].append(output.detach().cpu().mean(dim=1))
            else:
                self.features[name].append(output[0].detach().cpu().mean(dim=1))
        return hook
    
    def __register_hooks__(self, register_intermediate: bool = False):
        self.features = defaultdict(list)
        self._hooks_registed = True

        # Register forward hooks
        layer_name = 0
        handles = []
        self._register_intermediate = register_intermediate
        for layer_idx in range(self.num_layers):
            if register_intermediate:
                handles.append(self.pipeline.model.roberta.encoder.layer[layer_idx].intermediate.register_forward_hook(self.get_features(f"layer{layer_name}", is_intermediate=True)))
                handles.append(self.pipeline.model.roberta.encoder.layer[layer_idx].register_forward_hook(self.get_features(f"layer{layer_name + 1}")))
                layer_name += 2
            else:
                handles.append(self.pipeline.model.roberta.encoder.layer[layer_idx].register_forward_hook(self.get_features(f"layer{layer_idx}")))

        return handles
    
    def __element_wise_breakdown__(self, inputs, from_layer: int, within_block: bool = False):
        # Setup for storing operations from layer
        i = 0
        operations = {}
        
        zs, _zs, _as = [], [], []
        for input_id, _input in enumerate(inputs):
            # Preprocess sentence
            _input = self.pipeline.preprocess({'text': _input})['input_ids'].to(self.device)

            # Get input embeddings
            z = self.pipeline.model.roberta.embeddings(_input)
            
            # Pass embeddings through encoder network
            for layer_idx, _layer in enumerate(self.pipeline.model.roberta.encoder.layer):
                if layer_idx == from_layer and within_block:
                    raise NotImplementedError("Within block mode is currently not implemented for RoBERTa...")

                    # Modified from: https://github.com/huggingface/transformers/blob/main/src/transformers/models/roberta/modeling_roberta.py (line 389-469)                

                    self_attention_outputs = _layer.attention(z)
                    attention_output = self_attention_outputs[0]
                    outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

                    intermediate_output = _layer.intermediate(attention_output)
                    
                    # Get intermediate representation for verification purposes
                    _zs.append(intermediate_output)
                    _as.append(attention_output)

                    layer_output = _layer.output(intermediate_output, attention_output)
                    z = ((layer_output,) + outputs)[0]
                                        
                    # Add operation to operations
                    if input_id == 0:
                        # TODO: fix __get_output_from__ - first operation does not give the exact output wanted due to something with dropout? 
                        operations[i] = lambda x, a: (_layer.output(x, a),) + outputs 
                        i += 1

                else: # between encoder blocks
                    # Get intermediate representation for verification purposes 
                    if layer_idx == from_layer:
                        _zs.append(z)

                    z = _layer(z)[0]
                    if layer_idx >= from_layer and input_id == 0:
                        # Add operation to operations
                        operations[i] = _layer
                        i += 1

            # Pass through remaining layers
            z = self.pipeline.model.classifier(z).sigmoid()
            zs.append(z)

            if input_id == 0:
                operations[i] = lambda x: self.pipeline.model.classifier(x).sigmoid()

        # Stack outputs
        zs = torch.stack(zs).squeeze(1)

        if self._hooks_registed:
            # Move features to CPU
            _features = {}
            for i, (k, v) in enumerate(self.features.items()):
                _features[k] = torch.stack(v).squeeze(1).to('cpu')
        
        # Verify decomposition against most likely label
        _orig_outputs = [self.pipeline.model(self.pipeline.preprocess({'text': _input})['input_ids'].to(self.device)).logits for _input in inputs]
        _orig_outputs = torch.stack(_orig_outputs).squeeze(1).sigmoid().max(dim=1)[0]
        assert torch.allclose(_orig_outputs, self.__get_output_from__(_zs, operations).max(dim=1)[0]), "Encoder block decomposition is incorrect..." 
        assert torch.allclose(_orig_outputs, zs.max(dim=1)[0]), "Full model decomposition is incorrect..."
        
        if self._hooks_registed:
            # Overwrite features
            self.features = _features

        extra = {'attention_outputs': _as}
        return operations, _zs, extra
    
    def __get_output_from__(self, intermediates: torch.Tensor, operations: dict, attention_outputs: list = None):
        self.features = defaultdict(list) # Reset features

        outputs = []
        for j in range(len(intermediates)):
            __z = intermediates[j]
            # iterate through operations of the last part of the network 
            for i, _op in operations.items():
                if i == 0 and attention_outputs is not None:
                    __z = _op(__z, attention_outputs[j])[0]
                elif _op.__class__.__name__.lower() == 'robertalayer' or i == 0: 
                    __z = _op(__z)[0]
                else:
                    __z = _op(__z)
            outputs.append(__z)
        return torch.stack(outputs).squeeze(dim=1)