from .base_trainer import BaseTrainer
from ..utils.training_utils import process_image_batch_visualization

class CLEVRTrainer(BaseTrainer):
    """CLEVR 데이터셋을 위한 특화된 트레이너"""
    
    def _prepare_batch(self, batch, text_embedding_model=None):
        """CLEVR 배치 데이터 준비"""
        return batch.cuda()
    
    def _collect_visualization_data(self, batch, inputs, text_embedding_model, phase):
        """CLEVR 시각화 데이터 수집"""
        return process_image_batch_visualization(
            inputs, 
            self.model, 
            phase, 
            self._calculate_tau()
        )
    
    def _log_training_step(self, loss, batch_idx, epoch_size):
        """CLEVR 학습 단계 로깅"""
        super()._log_training_step(loss, batch_idx, epoch_size)
        
        # CLEVR 특화 메트릭 로깅
        if hasattr(self.model, 'get_reconstruction_quality'):
            recon_quality = self.model.get_reconstruction_quality()
            self.writer.add_scalar('TRAIN/reconstruction_quality', 
                                 recon_quality, 
                                 self.global_step)
    
    def _log_validation_results(self, val_loss, val_viz_data):
        """CLEVR 검증 결과 로깅"""
        self.writer.add_scalar('VAL/loss', val_loss, self.current_epoch + 1)
        
        if val_viz_data is not None:
            # 재구성 품질 분포
            self.writer.add_histogram('VAL/reconstruction_quality',
                                    val_viz_data['metrics']['reconstruction_quality'],
                                    self.current_epoch)
            
            # 공간 어텐션 맵
            if 'spatial_attention' in val_viz_data['metrics']:
                self.writer.add_image('VAL/spatial_attention',
                                    val_viz_data['metrics']['spatial_attention'].mean(0),
                                    self.current_epoch) 