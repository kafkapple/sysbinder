import os
import math
import argparse
import io
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam

from torch.nn.utils import clip_grad_norm_
from torch.nn import DataParallel as DP

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision.utils as vutils

from datetime import datetime

from sysbinder import SysBinderImageAutoEncoder, SysBinderTextAutoEncoder
from data import GlobDataset, EmotionTextDataset
from utils import (linear_warmup, cosine_anneal, log_text_visualizations, 
                  visualize_class_comparison, analyze_emotion_relationships,
                  visualize_emotion_relationships)

from transformers import AutoTokenizer, AutoModel

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

from logger_wrapper import LoggerWrapper

from sentence_transformers import SentenceTransformer

# 1. TextProcessor class definition
class TextProcessor(nn.Module):
    def __init__(self, embedding_dim, hidden_size):
        super().__init__()
        # Adjust intermediate layer size based on embedding dimension
        intermediate_size = max(1024, embedding_dim * 2)  # 2x embedding dim or 1024, whichever is larger
        
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, intermediate_size),
            nn.LayerNorm(intermediate_size),
            nn.ReLU(),
            nn.Linear(intermediate_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
    
    def forward(self, text_embedding):
        return self.projection(text_embedding)

# 2. TextEmbeddingModel class definition
class TextEmbeddingModel:
    def __init__(self, model_name='bert-base-uncased'):
        self.model_name = model_name
        # Define embedding dimensions for each model
        self.embedding_dims = {
            'bert-base-uncased': 768,
            'bge-m3': 768,
            'bge-large': 1024,
            'bge-small': 384
        }
        
        if 'bge' in model_name:
            self.tokenizer = None
            self.model = SentenceTransformer(model_name).cuda()
            self.embedding_dim = self.embedding_dims[model_name]
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).cuda()
            self.embedding_dim = self.embedding_dims[model_name]
    
    def encode(self, texts):
        if 'bge' in self.model_name:
            # Encoding for BGE models
            embeddings = self.model.encode(texts, convert_to_tensor=True)
            return embeddings
        else:
            # Encoding for BERT-based models
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.cuda() for k, v in inputs.items()}
            outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1)

# 3. ArgumentParser definition and argument parsing
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--image_channels', type=int, default=3)

parser.add_argument('--checkpoint_path', default='checkpoint.pt.tar')
parser.add_argument('--data_path', default='data/isear/isear.csv') #clevr-easy/train/*.png') 
parser.add_argument('--log_path', default='logs/')

parser.add_argument('--lr_dvae', type=float, default=3e-4)
parser.add_argument('--lr_enc', type=float, default=1e-4)
parser.add_argument('--lr_dec', type=float, default=3e-4)
parser.add_argument('--lr_warmup_steps', type=int, default=30000)
parser.add_argument('--lr_half_life', type=int, default=250000)
parser.add_argument('--clip', type=float, default=0.05)
parser.add_argument('--epochs', type=int, default=3)

parser.add_argument('--num_iterations', type=int, default=3)
parser.add_argument('--num_slots', type=int, default=4)
parser.add_argument('--num_blocks', type=int, default=8)
parser.add_argument('--cnn_hidden_size', type=int, default=512)
parser.add_argument('--slot_size', type=int, default=2048)
parser.add_argument('--mlp_hidden_size', type=int, default=192)
parser.add_argument('--num_prototypes', type=int, default=64)

parser.add_argument('--vocab_size', type=int, default=4096)
parser.add_argument('--num_decoder_layers', type=int, default=8)
parser.add_argument('--num_decoder_heads', type=int, default=4)
parser.add_argument('--d_model', type=int, default=768)
parser.add_argument('--dropout', type=int, default=0.1)

parser.add_argument('--tau_start', type=float, default=1.0)
parser.add_argument('--tau_final', type=float, default=0.1)
parser.add_argument('--tau_steps', type=int, default=30000)

parser.add_argument('--use_dp', default=True, action='store_true')
parser.add_argument('--data_type', type=str, choices=['clevr', 'isear'], default='isear',
                    help='Data type selection: clevr or isear')

parser.add_argument('--use_text_dvae', action='store_true', default=True,
                    help='Use dVAE for text embeddings')
parser.add_argument('--text_vocab_size', type=int, default=1024,
                    help='Vocabulary size for text dVAE')

parser.add_argument('--debug', action='store_true', default=False,
                   help='Run in debug mode (use only 32 samples)')
parser.add_argument('--max_samples', type=int, default=32,
                    help='Maximum number of samples for visualization')

parser.add_argument('--wandb_project', type=str, default='sysbinder',
                    help='Weights & Biases project name')
parser.add_argument('--wandb_name', type=str, default='isear_test',
                    help='Weights & Biases experiment name')

parser.add_argument('--debug_print', action='store_true', default=False,
                    help='Enable debug prints for attention weights and shapes')

parser.add_argument('--embedding_model', type=str, default='bert-base-uncased',
                    choices=['bert-base-uncased', 'bge-m3', 'bge-large', 'bge-small'],
                    help='Text embedding model selection')

args = parser.parse_args()

# 4. TextEmbeddingModel instance creation
text_embedding_model = TextEmbeddingModel(args.embedding_model)

torch.manual_seed(args.seed)

arg_str_list = ['{}={}'.format(k, v) for k, v in vars(args).items()]
arg_str = '__'.join(arg_str_list)
log_dir = os.path.join(args.log_path, datetime.today().isoformat())

# wandb config setting
wandb_config = {
    'model_type': args.data_type,
    'batch_size': args.batch_size,
    'num_slots': args.num_slots,
    'slot_size': args.slot_size,
    'learning_rate': args.lr_dvae,
    'epochs': args.epochs,
    # Add any other necessary settings
}

# LoggerWrapper initialization
writer = LoggerWrapper(
    log_dir=log_dir,
    project_name="sysbinder",
    experiment_name=f"{args.data_type}_{datetime.today().isoformat()}",
    config=wandb_config
)

def visualize(image, recon_dvae, recon_tf, attns, N=8):

    # tile
    tiles = torch.cat((
        image[:N, None, :, :, :],
        recon_dvae[:N, None, :, :, :],
        recon_tf[:N, None, :, :, :],
        attns[:N, :, :, :, :]
    ), dim=1).flatten(end_dim=1)

    # grid
    grid = vutils.make_grid(tiles, nrow=(1 + 1 + 1 + args.num_slots), pad_value=0.8)

    return grid

# Visualization function
def visualize_embeddings(embeddings, labels=None, epoch=0, log_dir='logs', max_samples=32):
    # Font setting
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # Visualization for entire dataset
    plt.figure(figsize=(15, 8))
    
    # Emotion class definition
    emotion_classes = ["joy", "fear", "anger", "sadness", "disgust", "shame", "guilt"]
    
    # Preprocess labels
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    
    # Use only the first column if labels are 2D array
    if len(labels.shape) > 1:
        labels = labels[:, 0]
    
    # Convert labels to integer if they are float
    labels = np.round(labels).astype(np.int64)
    
    # Balance sample count for each class
    balanced_indices = []
    samples_per_class = max(1, max_samples // len(emotion_classes))
    
    for class_idx in range(len(emotion_classes)):
        class_indices = np.where(labels == class_idx)[0]
        if len(class_indices) > 0:
            selected_indices = np.random.choice(
                class_indices, 
                size=min(samples_per_class, len(class_indices)), 
                replace=False
            )
            balanced_indices.extend(selected_indices)
    
    # Filter data using selected indices
    embeddings = embeddings[balanced_indices]
    labels = labels[balanced_indices]
    
    # Create DataFrame
    df = pd.DataFrame({
        'Dimension 1': embeddings[:, 0],
        'Dimension 2': embeddings[:, 1],
        'Emotion': [emotion_classes[label] for label in labels]
    })
    
    # Main scatter plot
    plt.subplot(1, 2, 1)
    scatter = sns.scatterplot(
        data=df,
        x='Dimension 1',
        y='Dimension 2',
        hue='Emotion',
        style='Emotion',  # Use different marker styles for each emotion
        palette='deep',
        alpha=0.7,
        s=100
    )
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Emotion Class')
    plt.title(f"Text Embeddings by Emotion (Epoch {epoch})")
    
    # Visualization of emotion distribution
    plt.subplot(1, 2, 2)
    emotion_counts = df['Emotion'].value_counts()
    sns.barplot(x=emotion_counts.index, y=emotion_counts.values, palette='deep')
    plt.title("Emotion Distribution")
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save as image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    buf.seek(0)
    
    image = Image.open(buf)
    image = image.convert('RGB')
    image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    
    plt.savefig(os.path.join(log_dir, f'embeddings_by_emotion_epoch_{epoch}.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    return image_tensor

# Dataset loading
if args.data_type == 'clevr':
    train_dataset = GlobDataset(root=args.data_path, phase='train', img_size=args.image_size)
    val_dataset = GlobDataset(root=args.data_path, phase='val', img_size=args.image_size)
else:
    # Load ISEAR dataset - get tokenizer from text_embedding_model
    train_dataset = EmotionTextDataset(
        csv_path=args.data_path,
        tokenizer=text_embedding_model.tokenizer if hasattr(text_embedding_model, 'tokenizer') else None,
        phase='train'
    )
    val_dataset = EmotionTextDataset(
        csv_path=args.data_path,
        tokenizer=text_embedding_model.tokenizer if hasattr(text_embedding_model, 'tokenizer') else None,
        phase='val'
    )

# Add sample limitation for debug mode
if args.debug:
    args.batch_size = 8  # Use smaller batch size in debug mode
    debug_samples = 32  # Number of debug samples
    
    # Use subset of data in debug mode
    train_dataset = torch.utils.data.Subset(
        train_dataset, 
        indices=range(min(debug_samples, len(train_dataset)))
    )
    val_dataset = torch.utils.data.Subset(
        val_dataset,
        indices=range(min(debug_samples // 2, len(val_dataset)))  # Half for validation
    )

train_sampler = None
val_sampler = None

loader_kwargs = {
    'batch_size': args.batch_size,
    'shuffle': True,
    'num_workers': args.num_workers if not args.debug else 0,  # Reduce workers in debug mode
    'pin_memory': True,
    'drop_last': False,
}

train_loader = DataLoader(train_dataset, sampler=train_sampler, **loader_kwargs)
val_loader = DataLoader(val_dataset, sampler=val_sampler, **loader_kwargs)

# Print dataset information
print(f"\n=== Dataset Information ===")
print(f"Training data size: {len(train_dataset):,}")
print(f"Validation data size: {len(val_dataset):,}")
print(f"Batch size: {args.batch_size}")
print(f"Total batches: {len(train_loader):,} (train) / {len(val_loader):,} (val)")
print(f"Debug mode: {'ON' if args.debug else 'OFF'}")
print(f"Debug print: {'ON' if args.debug_print else 'OFF'}")
print("========================\n")

train_epoch_size = len(train_loader)
val_epoch_size = len(val_loader)

# Protect log_interval from becoming 0
log_interval = max(1, train_epoch_size // 5)

# Initialize appropriate model based on data type
if args.data_type == 'clevr':
    model = SysBinderImageAutoEncoder(args)
else:
    model = SysBinderTextAutoEncoder(args)

if os.path.isfile(args.checkpoint_path):
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    start_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']
    best_epoch = checkpoint['best_epoch']
    model.load_state_dict(checkpoint['model'])
else:
    checkpoint = None
    start_epoch = 0
    best_val_loss = math.inf
    best_epoch = 0

model = model.cuda()
if args.use_dp:
    model = DP(model)

optimizer = Adam([
    {'params': (x[1] for x in model.named_parameters() if 'dvae' in x[0]), 'lr': args.lr_dvae},
    {'params': (x[1] for x in model.named_parameters() if 'image_encoder' in x[0]), 'lr': 0.0},
    {'params': (x[1] for x in model.named_parameters() if 'image_decoder' in x[0]), 'lr': 0.0},
])
if checkpoint is not None:
    optimizer.load_state_dict(checkpoint['optimizer'])

# Text processor initialization
if args.data_type == 'isear':
    text_processor = TextProcessor(
        embedding_dim=text_embedding_model.embedding_dim,  # Now correctly accessible
        hidden_size=args.slot_size
    )

def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"GPU Memory - Allocated: {allocated:.1f}MB / Reserved: {reserved:.1f}MB")

# Add variables to store last batch embeddings and emotions
last_batch_embeddings = None
last_batch_emotions = None  # Changed from text to emotions

# Add variables for class-wise visualization
emotion_classes = ["joy", "fear", "anger", "sadness", "disgust", "shame", "guilt"]
last_visualized_emotions = set()  # Track last visualized emotions

# Add variables for collecting data for class-wise visualization
class_visualization_data = {
    'texts': [],
    'slots': [],
    'attns': [],
    'labels': []
}

# Training loop
for epoch in range(start_epoch, args.epochs):
    model.train()
    if args.data_type == 'isear':
        text_processor.train()
    
    for batch_idx, batch in enumerate(train_loader):
        if args.data_type == 'clevr':
            images = batch.cuda()
        elif args.data_type == 'isear':
            # Process text data
            text_data = batch['text']
            emotion_labels = batch['emotion']
            text_embeddings = text_embedding_model.encode(text_data)  # Use encode method
            
            # Store last batch
            last_batch_embeddings = text_embeddings.detach()
            last_batch_emotions = emotion_labels
            
            images = text_embeddings

        global_step = epoch * train_epoch_size + batch_idx

        tau = cosine_anneal(
            global_step,
            args.tau_start,
            args.tau_final,
            0,
            args.tau_steps)

        lr_warmup_factor_enc = linear_warmup(
            global_step,
            0.,
            1.0,
            0.,
            args.lr_warmup_steps)

        lr_warmup_factor_dec = linear_warmup(
            global_step,
            0.,
            1.0,
            0,
            args.lr_warmup_steps)

        lr_decay_factor = math.exp(global_step / args.lr_half_life * math.log(0.5))

        optimizer.param_groups[0]['lr'] = args.lr_dvae
        optimizer.param_groups[1]['lr'] = lr_decay_factor * lr_warmup_factor_enc * args.lr_enc
        optimizer.param_groups[2]['lr'] = lr_decay_factor * lr_warmup_factor_dec * args.lr_dec

        optimizer.zero_grad()
        
        (recon_dvae, cross_entropy, mse, attns) = model(images, tau)

        if args.use_dp:
            mse = mse.mean()
            cross_entropy = cross_entropy.mean()

        loss = mse + cross_entropy
        
        loss.backward()

        clip_grad_norm_(model.parameters(), args.clip, 'inf')

        optimizer.step()
        
        with torch.no_grad():
            if batch_idx % log_interval == 0:
                print(f'\n[Epoch {epoch+1}/{args.epochs}] [{batch_idx:,}/{train_epoch_size:,}]')
                print(f'Loss: {loss.item():.4f} (MSE: {mse.item():.4f}, CE: {cross_entropy.item():.4f})')
                print(f'Learning rate: {optimizer.param_groups[0]["lr"]:.2e}')
                
                writer.add_scalar('TRAIN/loss', loss.item(), global_step)
                writer.add_scalar('TRAIN/cross_entropy', cross_entropy.item(), global_step)
                writer.add_scalar('TRAIN/mse', mse.item(), global_step)

                writer.add_scalar('TRAIN/tau', tau, global_step)
                writer.add_scalar('TRAIN/lr_dvae', optimizer.param_groups[0]['lr'], global_step)
                writer.add_scalar('TRAIN/lr_enc', optimizer.param_groups[1]['lr'], global_step)
                writer.add_scalar('TRAIN/lr_dec', optimizer.param_groups[2]['lr'], global_step)

                print_gpu_memory()

            # If text data, add visualization
            if args.data_type == 'isear':
                # Process all samples in batch
                for sample_idx in range(len(text_data)):
                    # Get emotion label
                    emotion_idx = torch.argmax(emotion_labels[sample_idx]).item()
                    emotion_label = emotion_classes[emotion_idx]
                    
                    # Visualize only if emotion class hasn't been visualized or in debug mode
                    if emotion_label not in last_visualized_emotions or args.debug:
                        slots_data = (model.module if args.use_dp else model).text_encoder.sysbinder(
                            text_embeddings[sample_idx].unsqueeze(0).unsqueeze(0)
                        )[0]
                        
                        # Collect data for class-wise comparison
                        if emotion_label not in [label for label in class_visualization_data['labels']]:
                            class_visualization_data['texts'].append(text_data[sample_idx])
                            class_visualization_data['slots'].append(slots_data)
                            class_visualization_data['attns'].append(attns[sample_idx])
                            class_visualization_data['labels'].append(emotion_label)
                        
                        # Execute debug print conditionally
                        if args.debug_print:
                            print(f"Processing emotion: {emotion_label}")
                            print(f"Text: {text_data[sample_idx]}")
                            print(f"Attention weights shape: {attns[sample_idx].shape}")
                            print(f"Slots shape: {slots_data.shape}")
                        
                        log_text_visualizations(
                            writer=writer,
                            epoch=epoch * len(train_loader) + batch_idx,
                            text=text_data[sample_idx],
                            text_embedding=text_embeddings[sample_idx],
                            slots=slots_data,
                            attns=attns[sample_idx].unsqueeze(0),
                            reconstructed_embedding=recon_dvae[sample_idx],
                            emotion_label=emotion_label,
                            num_blocks=args.num_blocks,
                            tag_prefix='train',
                            debug=args.debug_print
                        )
                        
                        last_visualized_emotions.add(emotion_label)
                    
                    # If all emotion classes have been visualized, create class-wise comparison visualization
                    if len(last_visualized_emotions) == len(emotion_classes):
                        # Class-wise comparison visualization
                        comparison_fig = visualize_class_comparison(
                            texts=class_visualization_data['texts'],
                            slots_list=class_visualization_data['slots'],
                            attns_list=class_visualization_data['attns'],
                            emotion_labels=class_visualization_data['labels'],
                            num_blocks=args.num_blocks
                        )
                        writer.add_figure('train/class_comparison', comparison_fig, epoch)
                        
                        # Emotion class relationship analysis
                        attn_sim, block_sim, emotion_dist = analyze_emotion_relationships(
                            slots_list=class_visualization_data['slots'],
                            attns_list=class_visualization_data['attns'],
                            emotion_labels=class_visualization_data['labels'],
                            num_blocks=args.num_blocks
                        )
                        
                        # Relationship visualization
                        relationship_fig = visualize_emotion_relationships(
                            attn_sim, block_sim, emotion_dist,
                            class_visualization_data['labels']
                        )
                        writer.add_figure('train/emotion_relationships', relationship_fig, epoch)
                        
                        # Record quantitative metrics
                        for i, emotion1 in enumerate(class_visualization_data['labels']):
                            for j, emotion2 in enumerate(class_visualization_data['labels']):
                                if i < j:  # Avoid duplicates
                                    writer.add_scalar(
                                        f'relationships/attention_similarity/{emotion1}_vs_{emotion2}',
                                        attn_sim[i,j], epoch
                                    )
                                    writer.add_scalar(
                                        f'relationships/block_similarity/{emotion1}_vs_{emotion2}',
                                        block_sim[i,j], epoch
                                    )
                                    writer.add_scalar(
                                        f'relationships/emotion_distance/{emotion1}_vs_{emotion2}',
                                        emotion_dist[i,j], epoch
                                    )
                        
                        # Data initialization
                        last_visualized_emotions.clear()
                        class_visualization_data = {
                            'texts': [],
                            'slots': [],
                            'attns': [],
                            'labels': []
                        }
            else:
                # If image data, keep existing visualization
                writer.add_image('train/original', vutils.make_grid(images[:8]), epoch * len(train_loader) + batch_idx)
                writer.add_image('train/recon', vutils.make_grid(recon_dvae[:8]), epoch * len(train_loader) + batch_idx)

    with torch.no_grad():
        
        if args.data_type == 'clevr':
            recon_tf = (model.module if args.use_dp else model).reconstruct_autoregressive(images[:8])
            grid = visualize(images, recon_dvae, recon_tf, attns, N=8)
            writer.add_image('TRAIN_recons/epoch={:03}'.format(epoch+1), grid)
        elif args.data_type == 'isear' and last_batch_embeddings is not None:
            # Move to CPU and convert to numpy
            embeddings_np = last_batch_embeddings.cpu().numpy()
            max_samples = min(args.max_samples, len(embeddings_np))  # Limit to max 16 samples
            
            embedding_tensor = visualize_embeddings(
                embeddings_np,
                labels=last_batch_emotions,  # Changed from text to emotions
                epoch=epoch,
                log_dir=log_dir,
                max_samples=max_samples
            )
            writer.add_image('TEXT/embeddings_epoch={:03}'.format(epoch+1), embedding_tensor)
            writer.add_scalar('TEXT/loss', loss.item(), epoch)

    with torch.no_grad():
        model.eval()
        
        val_cross_entropy = 0.
        val_mse = 0.
        val_count = 0  # Add batch count

        for batch, image in enumerate(val_loader):
            if args.data_type == 'clevr':
                image = image.cuda()
                (recon_dvae, cross_entropy, mse, attns) = model(image, tau)
            elif args.data_type == 'isear':
                # Process text data
                text_data = image['text']  # Validation actually uses batch
                text_embeddings = text_embedding_model.encode(text_data)
                (recon_dvae, cross_entropy, mse, attns) = model(text_embeddings, tau)

            if args.use_dp:
                mse = mse.mean()
                cross_entropy = cross_entropy.mean()

            val_cross_entropy += cross_entropy.item()
            val_mse += mse.item()
            val_count += 1

        # Avoid division by 0
        if val_count > 0:
            val_cross_entropy /= val_count
            val_mse /= val_count
        
        val_loss = val_mse + val_cross_entropy

        writer.add_scalar('VAL/loss', val_loss, epoch+1)
        writer.add_scalar('VAL/cross_entropy', val_cross_entropy, epoch + 1)
        writer.add_scalar('VAL/mse', val_mse, epoch+1)

        print('====> Epoch: {:3} \t Loss = {:F}'.format(epoch+1, val_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1

            torch.save(model.module.state_dict() if args.use_dp else model.state_dict(), os.path.join(log_dir, 'best_model.pt'))

            if 50 <= epoch:
                recon_tf = (model.module if args.use_dp else model).reconstruct_autoregressive(image[:8])
                grid = visualize(image, recon_dvae, recon_tf, attns, N=8)
                writer.add_image('VAL_recons/epoch={:03}'.format(epoch + 1), grid)

        writer.add_scalar('VAL/best_loss', best_val_loss, epoch+1)

        checkpoint = {
            'epoch': epoch + 1,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'model': model.module.state_dict() if args.use_dp else model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        torch.save(checkpoint, os.path.join(log_dir, 'checkpoint.pt.tar'))

        print('====> Best Loss = {:F} @ Epoch {}'.format(best_val_loss, best_epoch))

writer.close()
