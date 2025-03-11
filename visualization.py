import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
from PIL import Image
import io
import torchvision.utils as vutils
from torch.nn import DataParallel as DP
import networkx as nx
from sklearn.decomposition import PCA

emotion_classes = ["joy", "fear", "anger", "sadness", "disgust", "shame", "guilt"]

def safe_normalize(x, eps=1e-8):
    """안전한 정규화를 위한 헬퍼 함수"""
    if torch.is_tensor(x):
        x = x.cpu().numpy()
    if isinstance(x, np.ndarray):
        x_min = np.nanmin(x)
        x_max = np.nanmax(x)
        if x_max - x_min > eps:
            return (x - x_min) / (x_max - x_min)
        return np.zeros_like(x)
    return x

def safe_mean(x):
    """NaN을 제외한 평균 계산"""
    if not x:
        return 0.0
    if torch.is_tensor(x):
        x = x.cpu().numpy()
    if isinstance(x, np.ndarray):
        return np.nanmean(x)
    if isinstance(x, list):
        x = [v for v in x if not (np.isnan(v) if isinstance(v, (float, int)) else False)]
        return np.mean(x) if x else 0.0
    return x

def safe_std(x):
    """NaN을 제외한 표준편차 계산"""
    if not x:
        return 0.0
    if torch.is_tensor(x):
        x = x.cpu().numpy()
    if isinstance(x, np.ndarray):
        return np.nanstd(x)
    if isinstance(x, list):
        x = [v for v in x if not (np.isnan(v) if isinstance(v, (float, int)) else False)]
        return np.std(x)
    return 0.0

def calculate_block_interactions(block_features):
    num_blocks = block_features.size(0)
    interactions = torch.zeros(num_blocks, num_blocks)
    
    # 직접적인 상호작용
    direct_sim = torch.matmul(block_features, block_features.transpose(-2, -1))
    
    # 간접적인 상호작용 (다른 블록을 통한)
    indirect_sim = torch.zeros_like(direct_sim)
    for i in range(num_blocks):
        for j in range(num_blocks):
            if i != j:
                # 다른 블록들을 통한 경로들의 합
                intermediate_paths = torch.sum(direct_sim[i,:] * direct_sim[:,j])
                indirect_sim[i,j] = intermediate_paths
    
    # 최종 상호작용 = 직접 + 간접
    interactions = direct_sim + 0.5 * indirect_sim
    return interactions

def visualize_block_relationships(text, slots_data, attns_data, model, args):
    """블록 간 관계 시각화 (통합된 버전)"""
    # 1. 블록 간 상호작용 계산
    d_block = args.slot_size // args.num_blocks
    slots_blocks = slots_data.reshape(-1, args.num_blocks, d_block)
    
    # 직접적인 상호작용
    block_features = F.normalize(slots_blocks, dim=-1)
    interactions = calculate_block_interactions(block_features)
    
    # 2. 시각화
    fig = plt.figure(figsize=(20, 15))
    gs = plt.GridSpec(3, 2, figure=fig)
    
    # 2.1 블록 간 상호작용 히트맵
    ax_interactions = fig.add_subplot(gs[0, :])
    sns.heatmap(interactions.cpu().numpy(),
                cmap='viridis',
                xticklabels=[f'Block {i+1}' for i in range(args.num_blocks)],
                yticklabels=[f'Block {i+1}' for i in range(args.num_blocks)],
                annot=True,
                fmt='.2f',
                ax=ax_interactions)
    ax_interactions.set_title('Block Interaction Strength')
    
    # 2.2 블록 그래프 시각화
    ax_graph = fig.add_subplot(gs[1:, 0])
    G = nx.Graph()
    
    for i in range(args.num_blocks):
        G.add_node(i)
    
    threshold = interactions.mean() + interactions.std()
    for i in range(args.num_blocks):
        for j in range(i+1, args.num_blocks):
            if interactions[i,j] > threshold:
                G.add_edge(i, j, weight=interactions[i,j].item())
    
    pos = nx.spring_layout(G)
    nx.draw(G, pos, 
            with_labels=True,
            node_color='lightblue',
            node_size=1000,
            font_size=10,
            font_weight='bold',
            ax=ax_graph)
    
    # 2.3 토큰/이미지-블록 관계 시각화
    ax_block = fig.add_subplot(gs[1:, 1])
    if model is not None:
        mem_attn, _, _ = get_prototype_memory_attention(model, slots_data, args)
        block_attn = mem_attn.mean(dim=(0,1))
        
        k = max(1, args.num_blocks // 2)
        top_k_values, _ = torch.topk(block_attn, k)
        
        sns.heatmap(block_attn.unsqueeze(0).cpu().numpy(),
                    cmap='viridis',
                    xticklabels=[f'Block {i+1}' for i in range(args.num_blocks)],
                    yticklabels=['Attention'],
                    annot=True,
                    fmt='.2f',
                    ax=ax_block)
        ax_block.set_title('Block Attention Strength')
    
    plt.tight_layout()
    return fig

def visualize(image, recon_dvae, recon_tf, attns, args, N=8):
    """이미지 재구성 시각화"""
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

def create_visualization(text, slot_importance, block_activity, token_attention, text_embedding_model, args, emotion_label=None, is_average=False):
    try:
        fig = plt.figure(figsize=(20, 15))
        gs = plt.GridSpec(4, 2, figure=fig, height_ratios=[1, 2, 2, 2], width_ratios=[1, 4])
        
        # 1. 헤더 (감정 레이블)
        ax_header = fig.add_subplot(gs[0, :])
        if is_average:
            header_text = f"Emotion: {emotion_label}"  # 평균의 경우 텍스트 표시하지 않음
        else:
            header_text = f"Emotion: {emotion_label}\nText: {text}"
            
        ax_header.text(0.05, 0.5, header_text, wrap=True, fontsize=12)
        ax_header.axis('off')
        
        # 2. 슬롯 중요도 (히트맵)
        ax_slot_attn = fig.add_subplot(gs[1:3, 0])
        sns.heatmap(slot_importance.reshape(-1, 1).cpu().numpy(),
                    cmap='viridis',
                    yticklabels=range(1, args.num_slots + 1),
                    xticklabels=['Importance'],
                    vmin=0,
                    annot=True,
                    fmt='.2f',
                    ax=ax_slot_attn)
        ax_slot_attn.set_title('Slot Importance\n(Gumbel-Softmax normalized)')
        
        # 3. 블록-슬롯 활성화 (라인 플롯)
        ax_block_slot = fig.add_subplot(gs[1:3, 1])
        block_activity_np = block_activity.cpu().numpy()
        
        for slot_idx in range(args.num_slots):
            ax_block_slot.plot(range(1, args.num_blocks + 1), 
                             block_activity_np[slot_idx], 
                             marker='o',
                             label=f'Slot {slot_idx + 1}',
                             alpha=0.7)
        
        ax_block_slot.set_xlabel('Block Index')
        ax_block_slot.set_ylabel('Activity Level')
        ax_block_slot.set_title('Block-Slot Activity\n(Cosine similarity with semantic features)')
        ax_block_slot.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_block_slot.grid(True)
        
        # 4. 토큰-블록 어텐션 (히트맵)
        if token_attention is not None:
            ax_token_block = fig.add_subplot(gs[3, 1])
            sns.heatmap(token_attention.unsqueeze(0).cpu().numpy(),
                        cmap='viridis',
                        xticklabels=range(1, args.num_blocks + 1),
                        yticklabels=['Attention'],
                        vmin=0,
                        annot=True,
                        fmt='.2f',
                        ax=ax_token_block)
            ax_token_block.set_title('Token-Block Attention\n(Top-k filtered, temperature scaled)')
            
            # 토큰 정렬 (개별 샘플의 경우에만)
            if not is_average and hasattr(text_embedding_model, 'tokenizer'):
                ax_text = ax_token_block.twiny()
                ax_text.set_xlim(ax_token_block.get_xlim())
                tokens = text_embedding_model.tokenizer.tokenize(text)
                token_positions = np.linspace(0, args.num_blocks-1, len(tokens))
                ax_text.set_xticks(token_positions)
                ax_text.set_xticklabels(tokens, rotation=45, ha='left')
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        print(f"Error in create_visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        plt.close('all')
        
        fig = plt.figure(figsize=(10, 5))
        plt.text(0.5, 0.5, f"Error in visualization: {str(e)}", 
                ha='center', va='center', wrap=True)
        plt.axis('off')
        return fig

def visualize_text_token_relationships(text, slots_data, attns_data, text_embedding_model, args, emotion_label=None, model=None, is_average=False):
    try:
        # 1. Slot Importance 계산
        slot_attn = attns_data.squeeze()
        temperature = 0.5
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(slot_attn)))
        slot_importance = F.softmax((slot_attn + gumbel_noise) / temperature, dim=0)
        slot_importance = F.normalize(slot_importance, p=2, dim=0)
        
        # 2. Block-Slot Activity 계산
        block_activity = calculate_block_slot_activity_improved(slots_data, slot_importance, args)
        
        # 3. Token-Block Attention 계산
        token_block_attn_norm = None
        if model is not None:
            mem_attn, _, _ = get_prototype_memory_attention(model, slots_data, args)
            if mem_attn is not None:
                token_block_attn = mem_attn.mean(dim=0)
                token_block_attn = token_block_attn.mean(dim=0)
                token_block_attn = token_block_attn.mean(dim=0)
                
                k = max(1, token_block_attn.size(0) // 2)
                top_k_values, _ = torch.topk(token_block_attn, k)
                threshold = top_k_values[-1]
                
                mask = token_block_attn >= threshold
                token_block_attn = torch.where(mask, token_block_attn, torch.zeros_like(token_block_attn))
                
                sharp_temperature = args.attention_temperature * 0.1
                token_block_attn_norm = F.softmax(token_block_attn / sharp_temperature, dim=0)
        
        # 4. 시각화
        main_fig = create_visualization(
            text=text,
            slot_importance=slot_importance.view(-1, 1),
            block_activity=block_activity,
            token_attention=token_block_attn_norm,
            text_embedding_model=text_embedding_model,
            args=args,
            emotion_label=emotion_label,
            is_average=is_average
        )
        
        return main_fig
        
    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        plt.close('all')
        
        error_fig = plt.figure(figsize=(10, 5))
        plt.text(0.5, 0.5, f"Error in visualization: {str(e)}", 
                ha='center', va='center', wrap=True)
        plt.axis('off')
        return error_fig

def plot_emotion_metrics(emotion_metrics, phase='train'):
    """
    감정별 메트릭스 시각화
    """
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
    
    if emotions:  # 데이터가 있는 경우에만 시각화
        # Bar plot with error bars
        ax1.bar(emotions, similarities, yerr=errors)
        ax1.set_title(f'{phase.upper()} Reconstruction Quality by Emotion')
        ax1.set_xlabel('Emotion')
        ax1.set_ylabel('Average Cosine Similarity')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # 2. 감정별 샘플 수
    counts = [metrics['counts'] for metrics in emotion_metrics.values()]
    if any(counts):  # 데이터가 있는 경우에만 시각화
        ax2.bar(emotion_classes, counts)
        ax2.set_title(f'{phase.upper()} Sample Count by Emotion')
        ax2.set_xlabel('Emotion')
        ax2.set_ylabel('Count')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    return fig

def plot_train_val_comparison(train_metrics, val_metrics, epoch):
    """
    학습/검증 데이터 비교 시각화
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 재구성 품질 비교
    ax = axes[0, 0]
    train_sims = []
    val_sims = []
    
    for emotion in emotion_classes:
        if train_metrics[emotion]['similarities']:
            train_sims.append(safe_mean(train_metrics[emotion]['similarities']))
        if val_metrics[emotion]['similarities']:
            val_sims.append(safe_mean(val_metrics[emotion]['similarities']))
    
    if train_sims and val_sims:  # 데이터가 있는 경우에만 박스플롯 생성
        ax.boxplot([train_sims, val_sims], labels=['Train', 'Val'])
        ax.set_title('Reconstruction Quality Distribution')
        ax.set_ylabel('Cosine Similarity')
    
    # 2. 감정별 샘플 수
    ax = axes[0, 1]
    train_counts = [train_metrics[e]['counts'] for e in emotion_classes]
    val_counts = [val_metrics[e]['counts'] for e in emotion_classes]
    
    if any(train_counts) or any(val_counts):  # 데이터가 있는 경우에만 시각화
        x = np.arange(len(emotion_classes))
        width = 0.35
        
        ax.bar(x - width/2, train_counts, width, label='Train')
        ax.bar(x + width/2, val_counts, width, label='Val')
        ax.set_xticks(x)
        ax.set_xticklabels(emotion_classes, rotation=45)
        ax.legend()
        ax.set_title('Sample Distribution by Emotion')
    
    # 3. 감정별 평균 유사도 비교
    ax = axes[1, 0]
    train_means = []
    val_means = []
    emotions_with_data = []
    
    for emotion in emotion_classes:
        train_sim = train_metrics[emotion]['similarities']
        val_sim = val_metrics[emotion]['similarities']
        
        if train_sim or val_sim:
            emotions_with_data.append(emotion)
            train_means.append(safe_mean(train_sim))
            val_means.append(safe_mean(val_sim))
    
    if emotions_with_data:
        x = np.arange(len(emotions_with_data))
        width = 0.35
        
        ax.bar(x - width/2, train_means, width, label='Train')
        ax.bar(x + width/2, val_means, width, label='Val')
        ax.set_xticks(x)
        ax.set_xticklabels(emotions_with_data, rotation=45)
        ax.legend()
        ax.set_title('Average Similarity by Emotion')
        ax.set_ylabel('Cosine Similarity')
    
    plt.tight_layout()
    return fig

def process_batch_visualization(batch, model, text_embedding_model, phase, tau):
    """배치 데이터 시각화를 위한 메트릭스 수집"""
    emotion_metrics = {emotion: {'similarities': [], 'counts': 0} for emotion in emotion_classes}
    
    text_data = batch['text']
    emotion_labels = batch['emotion']
    text_embeddings = text_embedding_model.encode(text_data)
    
    with torch.no_grad():
        recon_dvae, _, _, attns = model(text_embeddings, tau)
        
        for i in range(len(text_data)):
            emotion_idx = torch.argmax(emotion_labels[i]).item()
            emotion = emotion_classes[emotion_idx]
            
            # 코사인 유사도 계산
            sim = F.cosine_similarity(
                text_embeddings[i].view(1, -1),
                recon_dvae[i].view(1, -1)
            ).item()
            
            emotion_metrics[emotion]['similarities'].append(sim)
            emotion_metrics[emotion]['counts'] += 1
    
    return {
        'emotion_metrics': emotion_metrics,
        'embeddings': text_embeddings,
        'reconstructions': recon_dvae,
        'attention': attns
    }

def visualize_image_relationships(image, slots_data, attns_data, args, model=None, is_average=False):
    try:
        # 1. Slot Importance 계산
        slot_attn = attns_data.squeeze()
        temperature = 0.5
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(slot_attn)))
        slot_importance = F.softmax((slot_attn + gumbel_noise) / temperature, dim=0)
        slot_importance = F.normalize(slot_importance, p=2, dim=0)
        
        # 2. Block-Slot Activity 계산
        block_activity = calculate_block_slot_activity_improved(slots_data, slot_importance, args)
        
        # 3. 이미지-블록 어텐션 계산
        image_block_attn_norm = None
        if model is not None:
            mem_attn, _, _ = get_prototype_memory_attention(model, slots_data, args)
            if mem_attn is not None:
                image_block_attn = mem_attn.mean(dim=0)
                image_block_attn = image_block_attn.mean(dim=0)
                image_block_attn = image_block_attn.mean(dim=0)
                
                k = max(1, image_block_attn.size(0) // 2)
                top_k_values, _ = torch.topk(image_block_attn, k)
                threshold = top_k_values[-1]
                
                mask = image_block_attn >= threshold
                image_block_attn = torch.where(mask, image_block_attn, torch.zeros_like(image_block_attn))
                
                sharp_temperature = args.attention_temperature * 0.1
                image_block_attn_norm = F.softmax(image_block_attn / sharp_temperature, dim=0)
        
        # 4. 시각화
        fig = plt.figure(figsize=(20, 15))
        gs = plt.GridSpec(4, 2, figure=fig, height_ratios=[1, 2, 2, 2], width_ratios=[1, 4])
        
        # 4.1 원본 이미지 표시
        ax_image = fig.add_subplot(gs[0, :])
        if not is_average:
            ax_image.imshow(image.permute(1, 2, 0).cpu().numpy())
        ax_image.axis('off')
        ax_image.set_title('Original Image' if not is_average else 'Average Visualization')
        
        # 4.2 슬롯 중요도 (히트맵)
        ax_slot_attn = fig.add_subplot(gs[1:3, 0])
        sns.heatmap(slot_importance.reshape(-1, 1).cpu().numpy(),
                    cmap='viridis',
                    yticklabels=range(1, args.num_slots + 1),
                    xticklabels=['Importance'],
                    vmin=0,
                    annot=True,
                    fmt='.2f',
                    ax=ax_slot_attn)
        ax_slot_attn.set_title('Slot Importance\n(Gumbel-Softmax normalized)')
        
        # 4.3 블록-슬롯 활성화 (라인 플롯)
        ax_block_slot = fig.add_subplot(gs[1:3, 1])
        block_activity_np = block_activity.cpu().numpy()
        
        for slot_idx in range(args.num_slots):
            ax_block_slot.plot(range(1, args.num_blocks + 1), 
                                block_activity_np[slot_idx], 
                                marker='o',
                                label=f'Slot {slot_idx + 1}',
                                alpha=0.7)
        
        ax_block_slot.set_xlabel('Block Index')
        ax_block_slot.set_ylabel('Activity Level')
        ax_block_slot.set_title('Block-Slot Activity\n(Cosine similarity with spatial features)')
        ax_block_slot.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_block_slot.grid(True)
        
        # 4.4 이미지-블록 어텐션 (히트맵)
        if image_block_attn_norm is not None:
            ax_image_block = fig.add_subplot(gs[3, 1])
            sns.heatmap(image_block_attn_norm.unsqueeze(0).cpu().numpy(),
                        cmap='viridis',
                        xticklabels=range(1, args.num_blocks + 1),
                        yticklabels=['Attention'],
                        vmin=0,
                        annot=True,
                        fmt='.2f',
                        ax=ax_image_block)
            ax_image_block.set_title('Image-Block Attention\n(Top-k filtered, temperature scaled)')
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        plt.close('all')
        
        error_fig = plt.figure(figsize=(10, 5))
        plt.text(0.5, 0.5, f"Error in visualization: {str(e)}", 
                ha='center', va='center', wrap=True)
        plt.axis('off')
        return error_fig

def process_image_batch_visualization(batch, model, phase, tau):
    """이미지 배치 데이터 시각화를 위한 메트릭스 수집"""
    batch_metrics = {
        'similarities': [],
        'spatial_attention': [],
        'reconstruction_quality': []
    }
    
    with torch.no_grad():
        recon_dvae, _, _, attns = model(batch, tau)
        
        # 재구성 품질 계산
        for i in range(len(batch)):
            # 1. 코사인 유사도
            sim = F.cosine_similarity(
                batch[i].view(1, -1),
                recon_dvae[i].view(1, -1)
            ).item()
            
            # 2. SSIM (Structural Similarity Index)
            ssim = torch.nn.functional.mse_loss(
                batch[i],
                recon_dvae[i]
            ).item()
            
            batch_metrics['similarities'].append(sim)
            batch_metrics['reconstruction_quality'].append(ssim)
        
        # 공간 어텐션 정보 저장
        batch_metrics['spatial_attention'] = attns.detach().cpu()
    
    return {
        'metrics': batch_metrics,
        'original': batch,
        'reconstructions': recon_dvae,
        'attention': attns
    }

def plot_image_metrics(metrics, phase='train'):
    """이미지 메트릭스 시각화"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. 재구성 품질 분포
    similarities = metrics['similarities']
    if similarities:
        sns.histplot(similarities, ax=ax1, bins=20)
        ax1.set_title(f'{phase.upper()} Reconstruction Quality Distribution')
        ax1.set_xlabel('Cosine Similarity')
        ax1.set_ylabel('Count')
    
    # 2. 공간 어텐션 히트맵
    if len(metrics['spatial_attention']) > 0:
        avg_attention = torch.stack(metrics['spatial_attention']).mean(0)
        sns.heatmap(avg_attention.numpy(),
                    cmap='viridis',
                    ax=ax2,
                    annot=True,
                    fmt='.2f')
        ax2.set_title(f'{phase.upper()} Average Spatial Attention')
    
    plt.tight_layout()
    return fig

def visualize_block_interactions(block_activity, block_interactions, args):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. 기존 라인 플롯
    for slot_idx in range(args.num_slots):
        ax1.plot(range(1, args.num_blocks + 1), 
                block_activity[slot_idx], 
                marker='o',
                label=f'Slot {slot_idx + 1}')
    
    # 2. 블록 간 상호작용 히트맵
    sns.heatmap(block_interactions.cpu().numpy(),
                cmap='viridis',
                xticklabels=range(1, args.num_blocks + 1),
                yticklabels=range(1, args.num_blocks + 1),
                ax=ax2)
    ax2.set_title('Block Interactions')
    
    return fig

def get_prototype_memory_attention(model, slots_data, args):
    """프로토타입 메모리 어텐션을 계산하는 함수"""
    if args.debug_print:
        print("\n=== Prototype Memory Analysis ===")
    B = slots_data.size(0) if len(slots_data.shape) > 2 else 1
    
    try:
        # DataParallel 여부 확인
        if isinstance(model, DP):
            sysbinder = model.module.text_encoder.sysbinder
        else:
            sysbinder = model.text_encoder.sysbinder
        
        # Prototype Memory 파라미터 확인
        mem_params = sysbinder.prototype_memory.mem_params
        if args.debug_print:
            print(f"1. Prototype Memory params shape: {mem_params.shape}")
        
        # Batch size 계산 및 slots_data 차원 조정
        if len(slots_data.shape) == 2:
            slots_data = slots_data.unsqueeze(0)  # [1, num_slots, slot_size]
        
        # slots_data를 블록으로 재구성
        d_block = args.slot_size // args.num_blocks
        slots_blocks = slots_data.reshape(B, -1, args.num_blocks, d_block)
        
        # 블록 특성화를 위한 정규화
        slots_blocks = F.normalize(slots_blocks, p=2, dim=-1)
        
        # Memory params 확장 및 정규화
        expanded_mem_params = mem_params.expand(B, -1, -1, -1)
        expanded_mem_params = F.normalize(expanded_mem_params, p=2, dim=-1)
        
        if args.debug_print:
            print(f"2. Shapes after reshape and normalization:")
            print(f"   - Slots blocks: {slots_blocks.shape}")
            print(f"   - Memory params: {expanded_mem_params.shape}")
        
        # slots_blocks를 prototype_memory와 같은 크기로 패딩
        if slots_blocks.size(1) < expanded_mem_params.size(1):
            padding_size = expanded_mem_params.size(1) - slots_blocks.size(1)
            padding = torch.zeros(B, padding_size, args.num_blocks, d_block, device=slots_blocks.device)
            slots_blocks = torch.cat([slots_blocks, padding], dim=1)
        elif slots_blocks.size(1) > expanded_mem_params.size(1):
            slots_blocks = slots_blocks[:, :expanded_mem_params.size(1), :, :]
        
        # Temperature scaling을 적용한 어텐션 계산
        attention_logits = torch.matmul(slots_blocks, expanded_mem_params.transpose(-2, -1))
        attention_logits = attention_logits / args.attention_temperature
        
        # Dropout 적용
        if args.attention_dropout > 0:
            attention_mask = torch.bernoulli(
                torch.ones_like(attention_logits) * (1 - args.attention_dropout)
            ).to(attention_logits.device)
            attention_logits = attention_logits * attention_mask
        
        # Softmax 적용
        mem_attn = F.softmax(attention_logits, dim=-1)
        
        # 블록 다양성 손실 계산 (유효한 슬롯만 사용)
        valid_slots = slots_blocks[:, :slots_data.size(1), :, :]  # 원래 슬롯 크기로 자르기
        block_similarities = torch.matmul(valid_slots, valid_slots.transpose(-2, -1))
        block_diversity_loss = torch.mean(torch.triu(block_similarities, diagonal=1))
        
        # 프로토타입 다양성 손실 계산
        prototype_similarities = torch.matmul(expanded_mem_params, expanded_mem_params.transpose(-2, -1))
        prototype_diversity_loss = torch.mean(torch.triu(prototype_similarities, diagonal=1))
        
        if args.debug_print:
            print(f"3. Memory attention stats:")
            print(f"   - Shape: {mem_attn.shape}")
            print(f"   - Min: {mem_attn.min().item():.4f}")
            print(f"   - Max: {mem_attn.max().item():.4f}")
            print(f"   - Mean: {mem_attn.mean().item():.4f}")
            print(f"4. Diversity metrics:")
            print(f"   - Block diversity loss: {block_diversity_loss.item():.4f}")
            print(f"   - Prototype diversity loss: {prototype_diversity_loss.item():.4f}")
        
        return mem_attn, block_diversity_loss, prototype_diversity_loss
        
    except Exception as e:
        if args.debug_print:
            print(f"Error in prototype memory attention: {str(e)}")
            import traceback
            traceback.print_exc()
        # 오류 발생 시 기본값 반환
        return (torch.zeros(B, args.num_prototypes, args.num_blocks, args.num_blocks).to(slots_data.device),
                torch.tensor(0.0).to(slots_data.device),
                torch.tensor(0.0).to(slots_data.device))

def calculate_block_slot_activity_improved(slots_data, slot_importance, args):
    d_block = args.slot_size // args.num_blocks
    slots_blocks = slots_data.reshape(-1, args.num_blocks, d_block)
    block_activity = torch.zeros(args.num_slots, args.num_blocks, device=slots_data.device)
    
    for slot_idx in range(args.num_slots):
        slot_blocks = slots_blocks[slot_idx]
        
        # 1. 블록별 특성 벡터 계산
        block_features = F.normalize(slot_blocks, dim=-1)
        
        # 2. 블록 간 주의 가중치 계산 (self-attention 방식)
        attention_weights = torch.matmul(block_features, block_features.transpose(-2, -1))
        attention_weights = F.softmax(attention_weights / math.sqrt(d_block), dim=-1)
        
        # 3. 의미적 클러스터링
        semantic_groups = torch.zeros(args.num_blocks, device=slots_data.device)
        for block_idx in range(args.num_blocks):
            # 가장 유사한 블록들과의 그룹 형성
            similar_blocks = attention_weights[block_idx] > attention_weights[block_idx].mean()
            group_features = block_features[similar_blocks]
            
            # 그룹 내 평균과의 차이 계산
            if len(group_features) > 0:
                group_mean = group_features.mean(dim=0)
                semantic_groups[block_idx] = F.cosine_similarity(
                    block_features[block_idx].unsqueeze(0),
                    group_mean.unsqueeze(0)
                )
        
        # 4. 블록별 고유성 계산
        uniqueness = 1 - attention_weights.mean(dim=1)  # 다른 블록과의 평균 유사도의 역수
        
        for block_idx in range(args.num_blocks):
            # 직접적인 특성
            direct_activation = torch.norm(block_features[block_idx])
            
            # 의미적 그룹 활성화
            semantic_activation = semantic_groups[block_idx]
            
            # 고유성
            block_uniqueness = uniqueness[block_idx]
            
            # Slot importance를 비선형 변환
            importance_weight = torch.sigmoid(slot_importance[slot_idx] * 5.0)
            
            # 최종 활성화 계산 (가중치 조정)
            block_activity[slot_idx, block_idx] = (
                0.3 * direct_activation +
                0.4 * semantic_activation +
                0.3 * block_uniqueness
            ) * importance_weight
        
        # 슬롯별 로컬 정규화 (전체가 아닌 각 슬롯 내에서만)
        block_activity[slot_idx] = F.normalize(block_activity[slot_idx], dim=-1)
    
    return block_activity