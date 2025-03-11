import os
import math
import argparse
from datetime import datetime

import torch
from torch.optim import Adam
from torch.nn import DataParallel as DP

from sysbinder import SysBinderImageAutoEncoder, SysBinderTextAutoEncoder
from data import GlobDataset, EmotionTextDataset
from text_model import TextEmbeddingModel

from ..data.data_loader import DataLoaderManager
from .clevr_trainer import CLEVRTrainer
from .isear_trainer import ISEARTrainer
from ..visualization.visualizer import Visualizer
from logger_wrapper import LoggerWrapper

def main():
    # ArgumentParser 설정
    parser = argparse.ArgumentParser()
    
    # 기본 설정
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--image_channels', type=int, default=3)
    
    # 경로 설정
    parser.add_argument('--checkpoint_path', default='checkpoint.pt.tar')
    parser.add_argument('--data_path', default='data/isear/isear.csv')
    parser.add_argument('--log_path', default='logs/')
    
    # 학습 파라미터
    parser.add_argument('--lr_dvae', type=float, default=3e-4)
    parser.add_argument('--lr_enc', type=float, default=1e-4)
    parser.add_argument('--lr_dec', type=float, default=3e-4)
    parser.add_argument('--lr_warmup_steps', type=int, default=30000)
    parser.add_argument('--lr_half_life', type=int, default=250000)
    parser.add_argument('--clip', type=float, default=0.05)
    parser.add_argument('--epochs', type=int, default=2)
    
    # 모델 구조 파라미터
    parser.add_argument('--num_iterations', type=int, default=3)
    parser.add_argument('--num_slots', type=int, default=4)
    parser.add_argument('--num_blocks', type=int, default=8)
    parser.add_argument('--slot_size', type=int, default=1024)
    parser.add_argument('--num_prototypes', type=int, default=32)
    parser.add_argument('--mlp_hidden_size', type=int, default=2048)
    
    # 데이터 타입 설정
    parser.add_argument('--data_type', type=str, choices=['clevr', 'isear'], default='clevr')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--debug_print', action='store_true', default=False)
    
    args = parser.parse_args()
    
    # 환경 설정
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.manual_seed(args.seed)
    
    # 로그 디렉토리 설정
    log_dir = os.path.join(args.log_path, datetime.today().isoformat())
    
    # 로거 초기화
    writer = LoggerWrapper(
        log_dir=log_dir,
        project_name="sysbinder",
        experiment_name=f"{args.data_type}_{datetime.today().isoformat()}",
        config={
            'model_type': args.data_type,
            'batch_size': args.batch_size,
            'num_slots': args.num_slots,
            'slot_size': args.slot_size,
            'learning_rate': args.lr_dvae,
            'epochs': args.epochs,
        }
    )
    
    # 텍스트 임베딩 모델 초기화
    text_embedding_model = TextEmbeddingModel(args.embedding_model)
    
    # 데이터셋 로딩
    if args.data_type == 'clevr':
        train_dataset = GlobDataset(root=args.data_path, phase='train', img_size=args.image_size)
        val_dataset = GlobDataset(root=args.data_path, phase='val', img_size=args.image_size)
    else:
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
    
    # 데이터 로더 초기화
    data_loader_manager = DataLoaderManager(args)
    train_loader, val_loader = data_loader_manager.create_dataloaders(train_dataset, val_dataset)
    
    # 모델 초기화
    if args.data_type == 'clevr':
        model = SysBinderImageAutoEncoder(args)
    else:
        model = SysBinderTextAutoEncoder(args)
    
    # 체크포인트 로딩
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
    
    # GPU 설정
    model = model.cuda()
    if args.use_dp:
        model = DP(model)
    
    # 옵티마이저 초기화
    optimizer = Adam([
        {'params': (x[1] for x in model.named_parameters() if 'dvae' in x[0]), 'lr': args.lr_dvae},
        {'params': (x[1] for x in model.named_parameters() if 'image_encoder' in x[0]), 'lr': 0.0},
        {'params': (x[1] for x in model.named_parameters() if 'image_decoder' in x[0]), 'lr': 0.0},
    ])
    
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    # 시각화 관리자 초기화
    visualizer = Visualizer(writer=writer, config=args)
    
    # 트레이너 초기화 및 학습 실행
    trainer_cls = CLEVRTrainer if args.data_type == 'clevr' else ISEARTrainer
    trainer = trainer_cls(
        model=model,
        optimizer=optimizer,
        writer=writer,
        config=args,
        visualizer=visualizer,
        text_embedding_model=text_embedding_model
    )
    
    # 학습 실행
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        start_epoch=start_epoch,
        best_val_loss=best_val_loss,
        best_epoch=best_epoch,
        checkpoint_dir=log_dir
    )
    
    writer.close()

if __name__ == '__main__':
    main() 