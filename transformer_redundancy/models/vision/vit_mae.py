
    elif 'vit-mae' in model_type:
        # Load pre-trained ViT-MAE model
        model = ViTMAEModel.from_pretrained(model_type).to(device)
        # Define number of layers
        num_layers = len(model.encoder.layer)
        # Register forward hooks
        for layer_idx in range(num_layers):
            model.encoder.layer[layer_idx].register_forward_hook(get_features(f"layer{layer_idx}"))
