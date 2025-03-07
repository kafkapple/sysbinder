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
from utils import linear_warmup, cosine_anneal, log_text_visualizations

from transformers import AutoTokenizer, AutoModel

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# 텍스트 임베딩 모델 초기화
text_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
text_embedding_model = AutoModel.from_pretrained('bert-base-uncased').cuda()

# 텍스트 임베딩 함수
def embed_text(text):
    inputs = text_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    # 입력을 GPU로
    inputs = {k: v.cuda() for k, v in inputs.items()}
    outputs = text_embedding_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

# 텍스트 프로세서 클래스
class TextProcessor(nn.Module):
    def __init__(self, embedding_dim, hidden_size):
        super().__init__()
        # BERT 임베딩을 모델 차원으로 변환하는 프로젝션 레이어들
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, 1024),  # 768 -> 1024
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, hidden_size),    # 1024 -> slot_size (2048)
            nn.LayerNorm(hidden_size)
        )
        
    def forward(self, text_embedding):
        return self.projection(text_embedding)

# 시각화 함수
def visualize_embeddings(embeddings, labels=None, epoch=0, log_dir='logs', max_samples=16):
    # 샘플 수 제한
    if len(embeddings) > max_samples:
        indices = np.random.choice(len(embeddings), max_samples, replace=False)
        embeddings = embeddings[indices]
        if labels is not None:
            labels = [labels[i] for i in indices]
    
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 8))
    
    # 레이블이 있으면 색상으로 구분
    if labels is not None:
        sns.scatterplot(
            x=embeddings[:, 0], 
            y=embeddings[:, 1], 
            hue=labels,  
            palette="deep"
        )
    else:
        sns.scatterplot(x=embeddings[:, 0], y=embeddings[:, 1])
    
    plt.title(f"텍스트 임베딩 시각화 (에포크 {epoch})")
    
    # 이미지로 저장
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # PIL Image를 PyTorch 텐서로 변환
    image = Image.open(buf)
    image = image.convert('RGB')
    image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    
    # 로컬 파일로도 저장
    image.save(os.path.join(log_dir, f'embeddings_epoch_{epoch}.png'))
    plt.close()
    
    return image_tensor

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=40)
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
parser.add_argument('--epochs', type=int, default=1)

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
                    help='데이터 종류 선택: clevr 또는 isear')

parser.add_argument('--use_text_dvae', action='store_true', default=True,
                    help='텍스트 임베딩에 dVAE 사용 여부')
parser.add_argument('--text_vocab_size', type=int, default=1024,
                    help='텍스트 dVAE의 vocabulary size')

parser.add_argument('--debug', action='store_true',
                   help='디버그 모드 실행 (데이터 32개만 사용)')
parser.add_argument('--max_samples', type=int, default=16,
                    help='시각화에 사용할 최대 샘플 수')

args = parser.parse_args()

torch.manual_seed(args.seed)

arg_str_list = ['{}={}'.format(k, v) for k, v in vars(args).items()]
arg_str = '__'.join(arg_str_list)
log_dir = os.path.join(args.log_path, datetime.today().isoformat())
writer = SummaryWriter(log_dir)
writer.add_text('hparams', arg_str)


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


# 데이터셋 로드
if args.data_type == 'clevr':
    train_dataset = GlobDataset(root=args.data_path, phase='train', img_size=args.image_size)
    val_dataset = GlobDataset(root=args.data_path, phase='val', img_size=args.image_size)
else:
    # ISEAR 데이터셋 로드
    train_dataset = EmotionTextDataset(
        csv_path=args.data_path,
        tokenizer=text_tokenizer,
        phase='train'
    )
    val_dataset = EmotionTextDataset(
        csv_path=args.data_path,
        tokenizer=text_tokenizer,
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

# 텍스트 데이터셋 로드
# text_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

# 텍스트 프로세서 초기화
if args.data_type == 'isear':
    text_processor = TextProcessor(embedding_dim=768, hidden_size=args.slot_size)

def print_gpu_memory():
    if torch.cuda.is_available():
        print(f'GPU 메모리 사용량: {torch.cuda.memory_allocated() / 1024**2:.2f}MB')

# 마지막 배치의 텍스트 임베딩을 저장할 변수 추가
last_batch_embeddings = None
last_batch_text = None

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
            text_inputs = text_tokenizer(
                text_data, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            )
            text_inputs = {k: v.cuda() for k, v in text_inputs.items()}
            with torch.no_grad():
                text_embeddings = text_embedding_model(**text_inputs).last_hidden_state.mean(dim=1)
            
            # 마지막 배치 저장
            last_batch_embeddings = text_embeddings.detach()
            last_batch_text = text_data
            
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
                # 배치에서 첫 번째 샘플만 시각화
                log_text_visualizations(
                    writer=writer,
                    epoch=epoch * len(train_loader) + batch_idx,
                    text=text_data[0],  # 첫 번째 텍스트
                    text_embedding=text_embeddings[0],  # 첫 번째 임베딩
                    slots=(model.module if args.use_dp else model).text_encoder.sysbinder(text_embeddings[0].unsqueeze(0).unsqueeze(0))[0],  # 첫 번째 샘플의 슬롯
                    attns=attns[0].unsqueeze(0),  # 첫 번째 샘플의 attention
                    reconstructed_embedding=recon_dvae[0],  # 첫 번째 샘플의 재구성
                    tag_prefix='train'
                )
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
                labels=last_batch_text,
                epoch=epoch,
                log_dir=log_dir,
                max_samples=max_samples  # 최대 샘플 수 전달
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
                text_inputs = text_tokenizer(
                    text_data, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True
                )
                text_inputs = {k: v.cuda() for k, v in text_inputs.items()}
                with torch.no_grad():
                    text_embeddings = text_embedding_model(**text_inputs).last_hidden_state.mean(dim=1)
                
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
