from .base_trainer import BaseTrainer
from ..utils.training_utils import process_batch_visualization
import torch.nn.functional as F

class ISEARTrainer(BaseTrainer):
    """ISEAR 데이터셋을 위한 특화된 트레이너"""
    
    def _prepare_batch(self, batch, text_embedding_model):
        """ISEAR 배치 데이터 준비"""
        text_data = batch['text']
        text_embeddings = text_embedding_model.encode(text_data)
        return text_embeddings.cuda()
    
    def _collect_visualization_data(self, batch, inputs, text_embedding_model, phase):
        """ISEAR 시각화 데이터 수집"""
        return process_batch_visualization(
            batch,
            self.model,
            text_embedding_model,
            phase,
            self._calculate_tau()
        )
    
    def _log_training_step(self, loss, batch_idx, epoch_size):
        """ISEAR 학습 단계 로깅"""
        super()._log_training_step(loss, batch_idx, epoch_size)
        
        # ISEAR 특화 메트릭 로깅
        if hasattr(self.model, 'get_emotion_metrics'):
            emotion_metrics = self.model.get_emotion_metrics()
            for emotion, metrics in emotion_metrics.items():
                self.writer.add_scalar(f'TRAIN/emotion_{emotion}_accuracy',
                                     metrics['accuracy'],
                                     self.global_step)
                self.writer.add_scalar(f'TRAIN/emotion_{emotion}_similarity',
                                     metrics['similarity'],
                                     self.global_step)
    
    def _log_validation_results(self, val_loss, val_viz_data):
        """ISEAR 검증 결과 로깅"""
        self.writer.add_scalar('VAL/loss', val_loss, self.current_epoch + 1)
        
        if val_viz_data is not None:
            emotion_metrics = val_viz_data['emotion_metrics']
            
            # 감정별 메트릭 로깅
            for emotion, metrics in emotion_metrics.items():
                if metrics['similarities']:
                    avg_sim = sum(metrics['similarities']) / len(metrics['similarities'])
                    self.writer.add_scalar(f'VAL/emotion_{emotion}_similarity',
                                         avg_sim,
                                         self.current_epoch)
                    self.writer.add_scalar(f'VAL/emotion_{emotion}_count',
                                         metrics['counts'],
                                         self.current_epoch)
            
            # 전체 재구성 품질
            if 'reconstructions' in val_viz_data:
                recon_quality = F.mse_loss(
                    val_viz_data['embeddings'],
                    val_viz_data['reconstructions']
                ).item()
                self.writer.add_scalar('VAL/reconstruction_quality',
                                     recon_quality,
                                     self.current_epoch)
    
    def _calculate_diversity_loss(self, outputs):
        """감정 다양성 손실 계산"""
        if hasattr(self.model, 'calculate_emotion_diversity'):
            diversity_loss = self.model.calculate_emotion_diversity(outputs)
            self.writer.add_scalar('TRAIN/emotion_diversity',
                                 diversity_loss.item(),
                                 self.global_step)
            return diversity_loss
        return 0.0 