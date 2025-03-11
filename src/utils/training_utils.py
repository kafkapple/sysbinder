import torch
import torch.nn.functional as F
import math

def linear_warmup(step, start, end, warmup_start_step, warmup_end_step):
    """선형 웜업 스케줄러"""
    if warmup_start_step >= warmup_end_step:
        return end
    if step < warmup_start_step:
        return start
    if step > warmup_end_step:
        return end
    
    alpha = float(step - warmup_start_step) / float(warmup_end_step - warmup_start_step)
    return start + (end - start) * alpha

def cosine_anneal(step, start, end, start_step, end_step):
    """코사인 어닐링 스케줄러"""
    if start_step >= end_step:
        return end
    if step < start_step:
        return start
    if step > end_step:
        return end
    
    alpha = float(step - start_step) / float(end_step - start_step)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * alpha))
    return end + (start - end) * cosine_decay

def calculate_loss(outputs, inputs, config):
    """손실 함수 계산"""
    recon_dvae, cross_entropy, mse, _ = outputs
    
    if config.use_dp:
        if config.data_type == 'isear':
            return calculate_text_loss(inputs, recon_dvae, cross_entropy, mse)
        else:
            return calculate_image_loss(mse, cross_entropy)
    
    return mse + cross_entropy

def calculate_text_loss(inputs, recon_dvae, cross_entropy, mse):
    """텍스트 데이터에 대한 손실 계산"""
    # 입력과 재구성 텐서 정규화
    inputs_norm = F.normalize(inputs.view(inputs.size(0), -1), p=2, dim=1)
    recon_norm = F.normalize(recon_dvae.view(recon_dvae.size(0), -1), p=2, dim=1)
    
    # 코사인 유사도 손실
    cos_sim = F.cosine_similarity(inputs_norm, recon_norm, dim=1)
    cos_loss = 1 - cos_sim.mean()
    
    # 스케일된 MSE 손실
    feature_scales = inputs.abs().mean(dim=0, keepdim=True)
    scaled_mse = ((inputs - recon_dvae).pow(2) / (feature_scales + 1e-6)).mean()
    
    # 최종 손실 계산
    mse = (scaled_mse + cos_loss) * 10.0
    cross_entropy = cross_entropy.mean()
    
    return mse + cross_entropy

def calculate_image_loss(mse, cross_entropy):
    """이미지 데이터에 대한 손실 계산"""
    return mse.mean() + cross_entropy.mean()

def save_checkpoint(trainer, model, optimizer, log_dir):
    """체크포인트 저장"""
    checkpoint = {
        'epoch': trainer.current_epoch + 1,
        'best_val_loss': trainer.best_val_loss,
        'best_epoch': trainer.best_epoch,
        'model': model.module.state_dict() if trainer.config.use_dp else model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    
    torch.save(checkpoint, f'{log_dir}/checkpoint.pt.tar')
    
    # 최고 성능 모델 저장
    if trainer.current_epoch + 1 == trainer.best_epoch:
        torch.save(
            model.module.state_dict() if trainer.config.use_dp else model.state_dict(),
            f'{log_dir}/best_model.pt'
        ) 