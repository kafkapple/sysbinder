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
                  visualize_class_comparison)

from transformers import AutoTokenizer, AutoModel

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

from logger_wrapper import LoggerWrapper

from sentence_transformers import SentenceTransformer

# 1. TextProcessor 클래스 정의
class TextProcessor(nn.Module):
    def __init__(self, embedding_dim, hidden_size):
        super().__init__()
        # 임베딩 차원에 따라 중간 레이어 크기 조정
        intermediate_size = max(1024, embedding_dim * 2)  # 임베딩 차원의 2배 또는 1024 중 큰 값
        
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, intermediate_size),
            nn.LayerNorm(intermediate_size),
            nn.ReLU(),
            nn.Linear(intermediate_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
    
    def forward(self, text_embedding):
        return self.projection(text_embedding)

# 2. TextEmbeddingModel 클래스 정의
class TextEmbeddingModel:
    def __init__(self, model_name='bert-base-uncased'):
        self.model_name = model_name
        # 모델별 임베딩 차원 정의
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
            # BGE 모델용 인코딩
            embeddings = self.model.encode(texts, convert_to_tensor=True)
            return embeddings
        else:
            # BERT 계열 모델용 인코딩
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.cuda() for k, v in inputs.items()}
            outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1)

# 3. ArgumentParser 정의 및 인자 파싱
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=8)
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
parser.add_argument('--num_slots', type=int, default=8)
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
                    help='데이터 종류 선택: clevr 또는 isear')

parser.add_argument('--use_text_dvae', action='store_true', default=True,
                    help='텍스트 임베딩에 dVAE 사용 여부')
parser.add_argument('--text_vocab_size', type=int, default=1024,
                    help='텍스트 dVAE의 vocabulary size')

parser.add_argument('--debug', action='store_true', default=False,
                   help='디버그 모드 실행 (데이터 32개만 사용)')
parser.add_argument('--max_samples', type=int, default=32,
                    help='시각화에 사용할 최대 샘플 수')

parser.add_argument('--wandb_project', type=str, default='sysbinder',
                    help='Weights & Biases project name')
parser.add_argument('--wandb_name', type=str, default='isear_test',
                    help='Weights & Biases experiment name')

parser.add_argument('--debug_print', action='store_true', default=False,
                    help='Enable debug prints for attention weights and shapes')

parser.add_argument('--embedding_model', type=str, default='bert-base-uncased',
                    choices=['bert-base-uncased', 'bge-m3', 'bge-large', 'bge-small'],
                    help='텍스트 임베딩 모델 선택')

args = parser.parse_args()

# 4. TextEmbeddingModel 인스턴스 생성
text_embedding_model = TextEmbeddingModel(args.embedding_model)

torch.manual_seed(args.seed)

arg_str_list = ['{}={}'.format(k, v) for k, v in vars(args).items()]
arg_str = '__'.join(arg_str_list)
log_dir = os.path.join(args.log_path, datetime.today().isoformat())

# wandb config 설정
wandb_config = {
    'model_type': args.data_type,
    'batch_size': args.batch_size,
    'num_slots': args.num_slots,
    'slot_size': args.slot_size,
    'learning_rate': args.lr_dvae,
    'epochs': args.epochs,
    # 필요한 다른 설정들 추가
}

# LoggerWrapper 초기화
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

# 시각화 함수
def visualize_embeddings(embeddings, labels=None, epoch=0, log_dir='logs', max_samples=32):
    # 폰트 설정
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # 전체 데이터에 대한 시각화
    plt.figure(figsize=(15, 8))
    
    # 감정 클래스 정의
    emotion_classes = ["joy", "fear", "anger", "sadness", "disgust", "shame", "guilt"]
    
    # 레이블 전처리
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    
    # labels가 2D 배열인 경우 첫 번째 열만 사용
    if len(labels.shape) > 1:
        labels = labels[:, 0]
    
    # 레이블이 float인 경우 정수로 변환
    labels = np.round(labels).astype(np.int64)
    
    # 각 클래스별 샘플 수 균형 맞추기
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
    
    # 선택된 인덱스로 데이터 필터링
    embeddings = embeddings[balanced_indices]
    labels = labels[balanced_indices]
    
    # DataFrame 생성
    df = pd.DataFrame({
        'Dimension 1': embeddings[:, 0],
        'Dimension 2': embeddings[:, 1],
        'Emotion': [emotion_classes[label] for label in labels]
    })
    
    # 메인 산점도
    plt.subplot(1, 2, 1)
    scatter = sns.scatterplot(
        data=df,
        x='Dimension 1',
        y='Dimension 2',
        hue='Emotion',
        style='Emotion',  # 각 감정별로 다른 마커 스타일 사용
        palette='deep',
        alpha=0.7,
        s=100
    )
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Emotion Class')
    plt.title(f"Text Embeddings by Emotion (Epoch {epoch})")
    
    # 감정별 분포 시각화
    plt.subplot(1, 2, 2)
    emotion_counts = df['Emotion'].value_counts()
    sns.barplot(x=emotion_counts.index, y=emotion_counts.values, palette='deep')
    plt.title("Emotion Distribution")
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # 이미지로 저장
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

# 데이터셋 로드
if args.data_type == 'clevr':
    train_dataset = GlobDataset(root=args.data_path, phase='train', img_size=args.image_size)
    val_dataset = GlobDataset(root=args.data_path, phase='val', img_size=args.image_size)
else:
    # ISEAR 데이터셋 로드 - tokenizer를 text_embedding_model에서 가져옴
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

# 데이터 로드 부분에 디버그용 샘플 제한 추가
if args.debug:
    args.batch_size = 8  # 디버그 모드에서는 작은 배치 사이즈 사용
    debug_samples = 32  # 디버그용 샘플 수
    
    # debug 모드일 때는 일부 데이터만 사용
    train_dataset = torch.utils.data.Subset(
        train_dataset, 
        indices=range(min(debug_samples, len(train_dataset)))
    )
    val_dataset = torch.utils.data.Subset(
        val_dataset,
        indices=range(min(debug_samples // 2, len(val_dataset)))  # 검증은 절반
    )

train_sampler = None
val_sampler = None

loader_kwargs = {
    'batch_size': args.batch_size,
    'shuffle': True,
    'num_workers': args.num_workers if not args.debug else 0,  # 디버그 모드에서는 worker 수 줄임
    'pin_memory': True,
    'drop_last': False,  # drop_last를 False로 변경
}

train_loader = DataLoader(train_dataset, sampler=train_sampler, **loader_kwargs)
val_loader = DataLoader(val_dataset, sampler=val_sampler, **loader_kwargs)

train_epoch_size = len(train_loader)
val_epoch_size = len(val_loader)

# log_interval이 0이 되지 않도록 보호
log_interval = max(1, train_epoch_size // 5)

# 데이터 타입에 따라 적절한 모델 초기화
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

# 텍스트 프로세서 초기화
if args.data_type == 'isear':
    text_processor = TextProcessor(
        embedding_dim=text_embedding_model.embedding_dim,  # 이제 올바르게 접근 가능
        hidden_size=args.slot_size
    )

def print_gpu_memory():
    if torch.cuda.is_available():
        print(f'GPU 메모리 사용량: {torch.cuda.memory_allocated() / 1024**2:.2f}MB')

# 마지막 배치의 텍스트 임베딩과 감정 레이블을 저장할 변수 추가
last_batch_embeddings = None
last_batch_emotions = None  # text 대신 emotions로 변경

# 클래스별 시각화를 위한 변수 추가
emotion_classes = ["joy", "fear", "anger", "sadness", "disgust", "shame", "guilt"]
last_visualized_emotions = set()  # 마지막으로 시각화된 감정들 추적

# 클래스별 시각화를 위한 데이터 수집 변수 추가
class_visualization_data = {
    'texts': [],
    'slots': [],
    'attns': [],
    'labels': []
}

for epoch in range(start_epoch, args.epochs):
    model.train()
    if args.data_type == 'isear':
        text_processor.train()
    
    for batch_idx, batch in enumerate(train_loader):
        if args.data_type == 'clevr':
            images = batch.cuda()
        elif args.data_type == 'isear':
            # 텍스트 데이터 처리
            text_data = batch['text']
            emotion_labels = batch['emotion']
            text_embeddings = text_embedding_model.encode(text_data)  # encode 메서드 사용
            
            # 마지막 배치 저장
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
                print('Train Epoch: {:3} [{:5}/{:5}] \t Loss: {:F} \t MSE: {:F}'.format(
                      epoch+1, batch_idx, train_epoch_size, loss.item(), mse.item()))
                
                writer.add_scalar('TRAIN/loss', loss.item(), global_step)
                writer.add_scalar('TRAIN/cross_entropy', cross_entropy.item(), global_step)
                writer.add_scalar('TRAIN/mse', mse.item(), global_step)

                writer.add_scalar('TRAIN/tau', tau, global_step)
                writer.add_scalar('TRAIN/lr_dvae', optimizer.param_groups[0]['lr'], global_step)
                writer.add_scalar('TRAIN/lr_enc', optimizer.param_groups[1]['lr'], global_step)
                writer.add_scalar('TRAIN/lr_dec', optimizer.param_groups[2]['lr'], global_step)

                print_gpu_memory()

            # 텍스트 데이터의 경우 시각화 추가
            if args.data_type == 'isear':
                # 배치 내의 모든 샘플에 대해 처리
                for sample_idx in range(len(text_data)):
                    # 감정 레이블 가져오기
                    emotion_idx = torch.argmax(emotion_labels[sample_idx]).item()
                    emotion_label = emotion_classes[emotion_idx]
                    
                    # 아직 시각화되지 않은 감정 클래스이거나 디버그 모드인 경우에만 시각화
                    if emotion_label not in last_visualized_emotions or args.debug:
                        slots_data = (model.module if args.use_dp else model).text_encoder.sysbinder(
                            text_embeddings[sample_idx].unsqueeze(0).unsqueeze(0)
                        )[0]
                        
                        # 클래스별 비교를 위한 데이터 수집
                        if emotion_label not in [label for label in class_visualization_data['labels']]:
                            class_visualization_data['texts'].append(text_data[sample_idx])
                            class_visualization_data['slots'].append(slots_data)
                            class_visualization_data['attns'].append(attns[sample_idx])
                            class_visualization_data['labels'].append(emotion_label)
                        
                        # debug print를 조건부로 실행
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
                    
                    # 모든 감정 클래스가 시각화되었다면 클래스별 비교 시각화 생성
                    if len(last_visualized_emotions) == len(emotion_classes):
                        comparison_fig = visualize_class_comparison(
                            texts=class_visualization_data['texts'],
                            slots_list=class_visualization_data['slots'],
                            attns_list=class_visualization_data['attns'],
                            emotion_labels=class_visualization_data['labels'],
                            num_blocks=args.num_blocks
                        )
                        writer.add_figure('train/class_comparison', comparison_fig, epoch)
                        
                        # 데이터 초기화
                        last_visualized_emotions.clear()
                        class_visualization_data = {
                            'texts': [],
                            'slots': [],
                            'attns': [],
                            'labels': []
                        }
            else:
                # 이미지 데이터의 경우 기존 시각화 유지
                writer.add_image('train/original', vutils.make_grid(images[:8]), epoch * len(train_loader) + batch_idx)
                writer.add_image('train/recon', vutils.make_grid(recon_dvae[:8]), epoch * len(train_loader) + batch_idx)

    with torch.no_grad():
        
        if args.data_type == 'clevr':
            recon_tf = (model.module if args.use_dp else model).reconstruct_autoregressive(images[:8])
            grid = visualize(images, recon_dvae, recon_tf, attns, N=8)
            writer.add_image('TRAIN_recons/epoch={:03}'.format(epoch+1), grid)
        elif args.data_type == 'isear' and last_batch_embeddings is not None:
            # CPU로 이동 후 numpy로 변환
            embeddings_np = last_batch_embeddings.cpu().numpy()
            max_samples = min(args.max_samples, len(embeddings_np))  # 최대 16개 샘플로 제한
            
            embedding_tensor = visualize_embeddings(
                embeddings_np,
                labels=last_batch_emotions,  # text 대신 emotions 사용
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
        val_count = 0  # 배치 카운트 추가

        for batch, image in enumerate(val_loader):
            if args.data_type == 'clevr':
                image = image.cuda()
                (recon_dvae, cross_entropy, mse, attns) = model(image, tau)
            elif args.data_type == 'isear':
                # 텍스트 데이터 처리
                text_data = image['text']  # validation에서는 image가 실제로는 batch
                text_embeddings = text_embedding_model.encode(text_data)
                (recon_dvae, cross_entropy, mse, attns) = model(text_embeddings, tau)

            if args.use_dp:
                mse = mse.mean()
                cross_entropy = cross_entropy.mean()

            val_cross_entropy += cross_entropy.item()
            val_mse += mse.item()
            val_count += 1

        # 0으로 나누는 것 방지
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
