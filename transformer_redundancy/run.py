if __name__ == '__main__':
    
    # import os
    # import argparse
    # from dotenv import load_dotenv, dotenv_values

    # parser = argparse.ArgumentParser(description='Run Visual Transformer experiments.')
    # parser.add_argument('mode', type=str, choices=['cka-similarity', 'input-similarity-jacobian', 'latent-similarity-jacobian'])
    # parser.add_argument('--model-name', type=str)
    # parser.add_argument('--processor-name', type=str)
    # parser.add_argument('--dataset-name', type=str, choices=['zh-plus/tiny-imagenet', 'imagenet-1k'])
    # parser.add_argument('--device', type=str, choices=['cpu', 'cuda'])
    # parser.add_argument('--batch-size', type=int, default=128)
    # parser.add_argument('--max-iter', type=int, default=100)
    # parser.add_argument('--distilled', action='store_true')
    # parser.add_argument('--save-path', type=str, default='/work3/s194253/experiments/ViT', help='Directory to save the bsub files.')
    # args = parser.parse_args()

    # load_dotenv()
    # login(os.getenv('HF_TOKEN'))

    # analyzer = BeitForLayerwiseAnalysis('microsoft/beit-base', device='cuda')    
    # with torch.no_grad():
    #     if args.get_features:
    #         # Get features
    #         analyzer.__register_hooks__(register_intermediate=True)

    #     # Get intermediate representations, operations and intermediates (+ features)
    #     outputs, operations, intermediates = analyzer.__element_wise_breakdown__({'pixel_values': X}, from_layer=7, within_block=True)


    import os
    import torch
    from dotenv import load_dotenv
    from huggingface_hub import login
    from transformer_redundancy.models.vision.beit import BeitForLayerwiseAnalysis
    load_dotenv()
    login(os.getenv('HF_TOKEN'))


    X = torch.randn(1, 3, 224, 224).to('cuda')
    analyzer = BeitForLayerwiseAnalysis('microsoft/beit-base', device='cuda')

    with torch.no_grad():
        # Get features
        analyzer.__register_hooks__(register_intermediate=True)
        # Get intermediate representations, operations and intermediates (+ features)
        outputs, operations, intermediates = analyzer.__element_wise_breakdown__({'pixel_values': X}, from_layer=7, within_block=True)