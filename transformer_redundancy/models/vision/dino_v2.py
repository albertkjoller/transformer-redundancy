
    
    elif 'dinov2' in model_type:
        # Load pre-trained DINOv2 model
        model_type = model_type + '-imagenet1k-1-layer'
        model = AutoModelForImageClassification.from_pretrained(model_type).to(device)
        # Define number of layers
        num_layers = len(model.dinov2.encoder.layer)
        __elements__ = {
            '__embedding__': model.dinov2.embeddings,
            '__encoder__': model.dinov2.encoder,
            '__layernorm__': model.dinov2.layernorm,
            '__concatenate__': lambda x: torch.cat([x[:, 0], x[:, 1:].mean(dim=1)], dim=1),
            '__classifier__': model.classifier
        }

    elif 'dinov2' in model_type:
        # Load pre-trained DINOv2 model
        model = AutoModelForImageClassification.from_pretrained(model_type).to(device)
        # Define number of layers
        num_layers = len(model.dinov2.encoder.layer)
        # Register forward hooks
        for layer_idx in range(num_layers):
            # model.dinov2.embeddings.register_forward_hook(get_features("embeddings"))
            model.dinov2.encoder.layer[layer_idx].register_forward_hook(get_features(f"layer{layer_idx}"))

