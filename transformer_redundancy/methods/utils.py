from ..models.layerwise import LayerWiseAnalysis

def extract_features(inputs, analyzer: LayerWiseAnalysis, register_intermediate: bool = True):
    # Get features
    analyzer.__register_hooks__(register_intermediate=register_intermediate)
    # Get intermediate representations, operations and intermediates (+ features)
    _, _, _ = analyzer.__element_wise_breakdown__(inputs, from_layer=0, within_block=False)
    
    # Extract features
    features = analyzer.features
    return features, features.keys().__len__()
