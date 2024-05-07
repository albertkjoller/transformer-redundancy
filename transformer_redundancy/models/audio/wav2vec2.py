
import torch
from ..layerwise import LayerWiseAnalysis
from transformers import AutoModelForAudioClassification

def prune_model(model, layer):
    # Get the list of encoder layers
    encoder_layers = model.base_model.encoder.layers
    # Remove layers between the given layer number and the classification head
    del encoder_layers[layer+1:]
    # Update the model's encoder layers
    model.base_model.encoder.layers = encoder_layers
    return model

class Wav2VecForLayerwiseAnalysis(LayerWiseAnalysis):
      
    def __init__(self, model_name: str, model_folder: str, pruned: bool = False, device='cuda'):
        super().__init__()

        self.model_name = model_name
        self.model_folder = model_folder
        self.pruned = pruned
        self.device = device
        self.model, self.num_layers = self.load_model()
        self.model.eval()
        self._hooks_registed = False

    def load_model(self):
        # Load pre-trained Wav2Vec model
        model_path = f"{self.model_folder}/{self.model_name}-finetuned" 
        model_path += '-pruned/' if self.pruned else '/'
        model = AutoModelForAudioClassification.from_pretrained(model_path).to(self.device)
        if self.pruned:
            model = prune_model(model, 8)
        num_layers = model.wav2vec2.encoder.layers.__len__()
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
                self.model.wav2vec2.encoder.layers[layer_idx].feed_forward.register_forward_hook(self.get_features(f"layer{layer_name}"))
                self.model.wav2vec2.encoder.layers[layer_idx].register_forward_hook(self.get_features(f"layer{layer_name + 1}"))
                layer_name += 2
            else:
                self.model.wav2vec2.encoder.layers[layer_idx].register_forward_hook(self.get_features(f"layer{layer_idx}"))

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