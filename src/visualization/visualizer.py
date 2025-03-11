import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from ..utils.training_utils import safe_mean, safe_std

class Visualizer:
    """시각화 관리자 클래스"""
    def __init__(self, writer, config):
        self.writer = writer
        self.config = config
        self.emotion_classes = ["joy", "fear", "anger", "sadness", "disgust", "shame", "guilt"]
    
    def log_training_metrics(self, metrics, global_step):
        """학습 메트릭 로깅"""
        for key, value in metrics.items():
            self.writer.add_scalar(f'TRAIN/{key}', value, global_step)
    
    def log_validation_metrics(self, metrics, epoch):
        """검증 메트릭 로깅"""
        for key, value in metrics.items():
            self.writer.add_scalar(f'VAL/{key}', value, epoch)
    
    def visualize_emotion_metrics(self, emotion_metrics, phase='train'):
        """감정별 메트릭 시각화"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. 감정별 평균 유사도
        emotions = []
        similarities = []
        errors = []
        
        for emotion, metrics in emotion_metrics.items():
            if metrics['similarities']:
                emotions.append(emotion)
                sim_mean = safe_mean(metrics['similarities'])
                sim_std = safe_std(metrics['similarities'])
                similarities.append(sim_mean)
                errors.append(sim_std)
        
        if emotions:
            ax1.bar(emotions, similarities, yerr=errors)
            ax1.set_title(f'{phase.upper()} Reconstruction Quality by Emotion')
            ax1.set_xlabel('Emotion')
            ax1.set_ylabel('Average Cosine Similarity')
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # 2. 감정별 샘플 수
        counts = [metrics['counts'] for metrics in emotion_metrics.values()]
        if any(counts):
            ax2.bar(self.emotion_classes, counts)
            ax2.set_title(f'{phase.upper()} Sample Count by Emotion')
            ax2.set_xlabel('Emotion')
            ax2.set_ylabel('Count')
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        return fig
    
    def visualize_attention(self, attention_maps, title):
        """어텐션 맵 시각화"""
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(attention_maps.cpu().numpy(),
                   cmap='viridis',
                   ax=ax,
                   annot=True,
                   fmt='.2f')
        ax.set_title(title)
        return fig
    
    def visualize_reconstruction_quality(self, train_quality, val_quality):
        """재구성 품질 비교 시각화"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if train_quality and val_quality:
            ax.boxplot([train_quality, val_quality],
                      labels=['Train', 'Val'])
            ax.set_title('Reconstruction Quality Distribution')
            ax.set_ylabel('Quality Score')
        
        return fig
    
    def log_image_grid(self, images, name, global_step):
        """이미지 그리드 로깅"""
        if isinstance(images, torch.Tensor):
            self.writer.add_image(name, images, global_step)
    
    def log_figure(self, figure, name, global_step):
        """Matplotlib 피규어 로깅"""
        self.writer.add_figure(name, figure, global_step)
        plt.close(figure) 