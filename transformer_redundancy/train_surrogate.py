import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from transformer_redundancy.data import *
from transformer_redundancy.models.vision import *
from transformer_redundancy.models.text import *
from transformer_redundancy.models.audio import *
from transformer_redundancy.methods.utils import extract_features

def get_domain(args):
    if args.dataset_name in ['imagenet-1k']:
        return 'vision'
    elif args.dataset_name in ['go_emotions']:
        return 'text'
    elif args.dataset_name in ['speech_commands']:
        return 'audio'
    else:
        raise ValueError(f"Dataset {args.dataset_name} not recognized.")

def get_analyzer(args):
    if 'deit' in args.model_name:
        return DeiTForLayerwiseAnalysis(args.model_name, device=args.device)
    elif 'beit' in args.model_name:
        return BeitForLayerwiseAnalysis(args.model_name, device=args.device)
    elif 'dinov2' in args.model_name:
        return DinoV2ForLayerwiseAnalysis(args.model_name, device=args.device)
    elif 'roberta' in args.model_name:
        return RoBERTaForLayerwiseAnalysis(args.model_name, device=args.device)
    elif 'wav2vec2' in args.model_name:
        return Wav2VecForLayerwiseAnalysis(args.model_name, model_folder=args.model_folder, device=args.device)
    elif 'wavlm' in args.model_name.lower():
        return WavLMForLayerwiseAnalysis(args.model_name, model_folder=args.model_folder, device=args.device)


class FeatureReproducingModel(nn.Module):

    def __init__(self, in_dim: int, embedding_dim: int, hidden_dim: int = 512):
        super(FeatureReproducingModel, self).__init__()
        
        self.intermediate_representation_encoder = nn.Linear(in_dim, hidden_dim)
        self.intermediate_linear_probe = nn.Linear(hidden_dim, embedding_dim)
        self.final_representation_encoder = nn.Linear(embedding_dim, hidden_dim)
        self.final_linear_probe = nn.Linear(hidden_dim, embedding_dim)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        intermediate_hidden_states = self.intermediate_linear_probe(self.relu(self.layer_norm(self.intermediate_representation_encoder(x))))
        last_hidden_states = self.final_linear_probe(self.relu(self.layer_norm(self.final_representation_encoder(intermediate_hidden_states))))
        return intermediate_hidden_states, last_hidden_states
    

class TransformerBasedMimicker(nn.Module):

    def __init__(self, embedding_dim: int, hidden_dim: int = 768, **kwargs):
        super(TransformerBasedMimicker, self).__init__()

        self.intermediate_transformer_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=8, dim_feedforward=hidden_dim, activation='relu'
        )
        self.last_transformer_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=8, dim_feedforward=hidden_dim, activation='relu'
        )
        
    def forward(self, x):
        intermediate_hidden_states = self.intermediate_transformer_layer(x)
        last_hidden_states = self.last_transformer_layer(intermediate_hidden_states)
        return intermediate_hidden_states, last_hidden_states


if __name__ == '__main__':
    
    import os
    from pathlib import Path
    import argparse
    from collections import OrderedDict

    from dotenv import load_dotenv
    from huggingface_hub import login
    from tqdm import tqdm
    import numpy as np

    parser = argparse.ArgumentParser(description='Run Visual Transformer experiments.')
    ### Experiment parameters ###
    parser.add_argument('--surrogate-type', type=str, default='linear', choices=['linear', 'transformer'])
    parser.add_argument('--val-every', type=int, default=50)
    parser.add_argument('--num-val-batches', type=int, default=None)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--intermediate-layer', type=int)
    parser.add_argument('--last-layer', type=int)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save-path', type=str, default='../experiments', help='Directory to save the bsub files.')
    parser.add_argument('--train-classifier', action='store_true')
    parser.add_argument('--from-pretrained', type=str, default=None)
    parser.add_argument('--avoid-freeze', action='store_true')
    ### Data parameters ###
    parser.add_argument('--dataset-name', type=str, choices=['speech_commands'])
    parser.add_argument('--processor-name', type=str)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--num-proc', type=int, default=1)
    ### Model parameters ###
    parser.add_argument('--model-name', type=str)
    parser.add_argument('--model-folder', type=str, default=None)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'])
    # Parse arguments
    args = parser.parse_args()
    
    if args.num_val_batches is not None:
        assert args.shuffle, "Validation batches must be shuffled to avoid bias..."

    # Load environment variables
    load_dotenv()
    login(os.getenv('HF_TOKEN'))
    # os.environ["HF_HOME"] = os.getenv('HF_HOME')

    # Clear pytorch cache
    torch.cuda.empty_cache()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Create save path
    save_path = os.path.join(args.save_path, f'{args.dataset_name}/{args.model_name}')
    if args.train_classifier:
        model_version = args.from_pretrained.split("/")[-1].split(".pt")[0].split("mimicker_")[1] + f"_finetuned_lr={args.lr}"
    else:
        model_version = f'hidden_dim={args.hidden_dim}_lr={args.lr}_bs={args.batch_size}_layers=[{args.intermediate_layer}, {args.last_layer}]_{args.surrogate_type}'

    if args.avoid_freeze:
        assert args.from_pretrained is not None, "Model must be loaded from a pretrained model..."
        model_version += '_unfrozen'

    os.makedirs(save_path, exist_ok=True)

    # Get data domain and loaders
    domain = get_domain(args)
    args.splits = ['train', 'validation']
    loaders, label2cat = get_loaders(**args.__dict__)
    analyzer = get_analyzer(args)

    # Get input dimension
    if domain == 'audio':
        embedding_shape = (49, 768)
        # embedding_dim = 49 * 768
        embedding_dim = 768

    # Initialize model
    if args.surrogate_type == 'linear':
        model = FeatureReproducingModel(in_dim=embedding_dim, embedding_dim=embedding_dim, hidden_dim=args.hidden_dim)
    else:
        model = TransformerBasedMimicker(embedding_dim=embedding_dim, hidden_dim=args.hidden_dim)
    
    if args.from_pretrained is not None:
        assert args.train_classifier, "Model must be trained with a classifier..."
        model.load_state_dict(torch.load(args.from_pretrained))

        if not args.avoid_freeze:
            # Freeze mimicker model
            for param in model.parameters():
                param.requires_grad = False
            
    if args.train_classifier:
        model = nn.Sequential(OrderedDict([
            ("mimicker", model),
            ("projector", analyzer.model.projector),
            ("classifier", analyzer.model.classifier),
        ]))

    model.to(args.device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss() if not args.train_classifier else nn.NLLLoss()
    
    epoch = 0
    num_steps = loaders["train"].__len__() * args.epochs 
    best_val_loss = np.inf
    
    writer = SummaryWriter(log_dir=os.path.join(save_path, f'logs/{model_version}'))
    with tqdm(range(num_steps)) as pbar:
        for step in pbar:
            
            if step % args.val_every == 0: # validation
                model.eval()
                with torch.no_grad():
                    # Initialize validation variables 
                    val_losses, preds, GT_preds, all_labels = {'intermediate': [], 'last': [], 'total': []}, [], [], []
                    
                    for batch_idx, batch in enumerate(loaders["validation"]):
                        if args.num_val_batches is not None and batch_idx == args.num_val_batches - 1: # stop validation after num_val_batches
                            break

                        pbar.set_description(f"INFO - Epoch {epoch+1}/{args.epochs} - Batch {batch_idx+1}/{args.num_val_batches} - Computing validation performance...")
                        if args.dataset_name == 'speech_commands':
                            inputs = batch["input_values"].to(args.device)
                            labels = batch["label"]

                        # Extract features
                        features, n_feature_layers = extract_features(
                            inputs, analyzer, 
                            layers=[args.intermediate_layer, args.last_layer],
                            register_intermediate=False, 
                            register_init_embedding=True
                        )
                        init_embeddings = features['feature_projection'].to(args.device) #.flatten(1).to(args.device)
                    
                        
                        if not args.train_classifier:
                            # Reproduce intermediate features
                            reproduced_intermediate_features, reproduced_last_hidden_states = model(init_embeddings)

                            # Compute loss and backpropagate
                            intermediate_loss = criterion(features[f'layer{args.intermediate_layer}'].to(args.device), reproduced_intermediate_features)
                            last_loss = criterion(features[f'layer{args.last_layer}'].to(args.device), reproduced_last_hidden_states)
                            val_loss = intermediate_loss + last_loss
                        
                            val_losses['intermediate'].append(intermediate_loss.item())
                            val_losses['last'].append(last_loss.item())
                            val_losses['total'].append(val_loss.item())

                            # Do prediction using original classifier
                            if 'wav2vec' in analyzer.__class__.__name__.lower():                    
                                projected = analyzer.model.projector(reproduced_last_hidden_states).mean(dim=1)
                                preds.append( analyzer.model.classifier(projected).argmax(1).cpu() )
                                GT_preds.append( analyzer.model(inputs).logits.argmax(1).cpu() )
                                all_labels.append( labels )
                            else:
                                raise NotImplementedError(f"Validation accuracy not implemented for {analyzer.__class__.__name__}...")
        
                        else:
                            # Reproduce intermediate features
                            reproduced_intermediate_features, reproduced_last_hidden_states = model.mimicker(init_embeddings)
                            # Classify
                            projected = model.projector(reproduced_last_hidden_states).mean(dim=1)
                            z = torch.log_softmax(model.classifier(projected), dim=1)
                            val_loss = criterion(z, labels.to(args.device))
                            val_losses['total'].append(val_loss.item())

                            _, _preds = z.topk(k=1)
                            preds.append(_preds.cpu().reshape(labels.shape))
                            all_labels.append(labels)

                    # Compute accuracy using classifier layer from original model
                    preds, all_labels = torch.cat(preds), torch.cat(all_labels)
                    val_acc = (preds == all_labels).float().mean()

                    # Store results
                    writer.add_scalar('Loss/Validation (total)', np.mean(val_losses['total']), step)
                    writer.add_scalar('Loss/Validation (intermediate)', np.mean(val_losses['intermediate']), step)
                    writer.add_scalar('Loss/Validation (last)', np.mean(val_losses['last']), step)
                    writer.add_scalar('Accuracy/Validation', val_acc.item(), step)

                    if not args.train_classifier:
                        GT_preds = torch.cat(GT_preds)
                        val_GT_acc = (GT_preds == all_labels).float().mean()
                        writer.add_scalar('Accuracy/Validation (GT)', val_GT_acc.item(), step)

                    if best_val_loss > np.mean(val_losses['total']):
                        best_val_loss = np.mean(val_losses['total'])
                        torch.save(model.state_dict(), os.path.join(save_path, f'mimicker_{model_version}.pt'))
                        print(f"New best model saved...")

            # Get training batch
            model.train()
            batch = next(iter(loaders["train"]))
            if args.dataset_name == 'speech_commands':
                inputs = batch["input_values"].to(args.device)
                labels = batch["label"]

            # Zero gradients
            optimizer.zero_grad()

            # Extract features
            with torch.no_grad():
                features, n_feature_layers = extract_features(
                    inputs, analyzer, 
                    layers=[args.intermediate_layer, args.last_layer],
                    register_intermediate=False, 
                    register_init_embedding=True
                )
            # init_embeddings = features['feature_projection'].flatten(1).to(args.device).requires_grad_()
            init_embeddings = features['feature_projection'].to(args.device).requires_grad_()


            if not args.train_classifier:
                # Reproduce intermediate features
                reproduced_intermediate_features, reproduced_last_hidden_states = model(init_embeddings)
                # Compute loss
                intermediate_loss = criterion(features[f'layer{args.intermediate_layer}'].to(args.device), reproduced_intermediate_features)
                last_loss = criterion(features[f'layer{args.last_layer}'].to(args.device), reproduced_last_hidden_states)
                loss = intermediate_loss + last_loss

            else:
                # Reproduce intermediate features
                reproduced_intermediate_features, reproduced_last_hidden_states = model.mimicker(init_embeddings)
                # Compute loss
                projected = model.projector(reproduced_last_hidden_states).mean(dim=1)
                z = torch.log_softmax(model.classifier(projected), dim=1)
                loss = criterion(z, labels.to(args.device))
            
            # Backpropagate
            loss.backward()
            # Optimize
            optimizer.step()

            with torch.no_grad():
                if not args.train_classifier:
                    # Do prediction using original classifier
                    if 'wav2vec' in analyzer.__class__.__name__.lower():                    
                        projected = analyzer.model.projector(reproduced_last_hidden_states).mean(dim=1)
                        preds = analyzer.model.classifier(projected).argmax(1).cpu()
                        GT_preds = analyzer.model(inputs).logits.argmax(1).cpu()
                    else:
                        raise NotImplementedError(f"Validation accuracy not implemented for {analyzer.__class__.__name__}...")            
                
                    # Compute accuracy using classifier layer from original model
                    acc, GT_acc = (preds == labels).float().mean(), (GT_preds == labels).float().mean()

                else:
                    _, preds = z.topk(k=1)
                    _, preds = z.topk(k=1)
                    preds = preds.cpu().reshape(labels.shape)
                    # Compute accuracy using classifier layer from original model
                    acc = (preds == labels).float().mean()

            # Store results
            writer.add_scalar('Loss/Training (total)', loss.item(), step)
            writer.add_scalar('Accuracy/Training', acc.item(), step)
            if not args.train_classifier:
                writer.add_scalar('Loss/Training (intermediate)', intermediate_loss.item(), step)
                writer.add_scalar('Loss/Training (last)', last_loss.item(), step)
                writer.add_scalar('Accuracy/Training (GT)', GT_acc.item(), step)

            # Update progress bar
            pbar.set_description(f"INFO - Epoch {epoch+1}/{args.epochs} - Train. loss: {loss.item():.3f} - Val. loss: {val_loss.item():.3f} - Train. acc: {acc.item():.3f} - Val. acc: {val_acc.item():.3f}")

            if (step+1) % len(loaders["train"]) == 0:
                epoch += 1