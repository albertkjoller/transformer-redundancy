import torch

class LayerWiseAnalysis:
      
    def __init__(self):
        pass
    
    def load_model(self):
        raise NotImplementedError("Method 'load_model' must be implemented in derived classes.")

    def __register_hooks__(self, get_features: callable, register_intermediate: bool = False, **kwargs):
        raise NotImplementedError("Method '__register_hooks__' must be implemented in derived classes.")
    
    def __get_intermediate__(self, inputs, __elements__, from_layer: int, model_name: str):
        raise NotImplementedError("Method '__get_intermediate__' must be implemented in derived classes.")

    def __output__(self, intermediate_rep, operations):
        raise NotImplementedError("Method '__output__' must be implemented in derived classes.")

    def __verify_breakdown__(self, _z, __z, operations):
        raise NotImplementedError("Method '__verify_breakdown__' must be implemented in derived classes.")