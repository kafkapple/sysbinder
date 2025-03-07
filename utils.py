import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd
import textwrap


def linear_warmup(step, start_value, final_value, start_step, final_step):

    assert start_value <= final_value
    assert start_step <= final_step

    if step < start_step:
        value = start_value
    elif step >= final_step:
        value = final_value
    else:
        a = final_value - start_value
        b = start_value
        progress = (step + 1 - start_step) / (final_step - start_step)
        value = a * progress + b

    return value


def cosine_anneal(step, start_value, final_value, start_step, final_step):

    assert start_value >= final_value
    assert start_step <= final_step

    if step < start_step:
        value = start_value
    elif step >= final_step:
        value = final_value
    else:
        a = 0.5 * (start_value - final_value)
        b = 0.5 * (start_value + final_value)
        progress = (step - start_step) / (final_step - start_step)
        value = a * math.cos(math.pi * progress) + b

    return value


def gumbel_softmax(logits, tau=1., hard=False, dim=-1):

    eps = torch.finfo(logits.dtype).tiny

    gumbels = -(torch.empty_like(logits).exponential_() + eps).log()
    gumbels = (logits + gumbels) / tau

    y_soft = F.softmax(gumbels, dim)

    if hard:
        index = y_soft.argmax(dim, keepdim=True)
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.)
        return y_hard - y_soft.detach() + y_soft
    else:
        return y_soft


def conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
           dilation=1, groups=1, bias=True, padding_mode='zeros',
           weight_init='xavier'):
    
    m = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                  dilation, groups, bias, padding_mode)
    
    if weight_init == 'kaiming':
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    else:
        nn.init.xavier_uniform_(m.weight)
    
    if bias:
        nn.init.zeros_(m.bias)
    
    return m


class Conv2dBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        
        self.m = conv2d(in_channels, out_channels, kernel_size, stride, padding,
                        bias=True, weight_init='kaiming')
    
    def forward(self, x):
        x = self.m(x)
        return F.relu(x)


def linear(in_features, out_features, bias=True, weight_init='xavier', gain=1.):
    
    m = nn.Linear(in_features, out_features, bias)
    
    if weight_init == 'kaiming':
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    else:
        nn.init.xavier_uniform_(m.weight, gain)
    
    if bias:
        nn.init.zeros_(m.bias)
    
    return m


class BlockGRU(nn.Module):
    """
        A GRU where the weight matrices have a block structure so that information flow is constrained
        Data is assumed to come in [block1, block2, ..., block_n].
    """

    def __init__(self, ninp, nhid, k):
        super(BlockGRU, self).__init__()

        assert ninp % k == 0
        assert nhid % k == 0

        self.k = k
        self.gru = nn.GRUCell(ninp, nhid)

        self.nhid = nhid
        self.ghid = self.nhid // k

        self.ninp = ninp
        self.ginp = self.ninp // k

        self.mask_hx = nn.Parameter(
            torch.eye(self.k, self.k)
                .repeat_interleave(self.ghid, dim=0)
                .repeat_interleave(self.ginp, dim=1)
                .repeat(3, 1),
            requires_grad=False
        )

        self.mask_hh = nn.Parameter(
            torch.eye(self.k, self.k)
                .repeat_interleave(self.ghid, dim=0)
                .repeat_interleave(self.ghid, dim=1)
                .repeat(3, 1),
            requires_grad=False
        )

    def blockify_params(self):
        for p in self.gru.parameters():
            p = p.data
            if p.shape == torch.Size([self.nhid * 3]):
                pass
            if p.shape == torch.Size([self.nhid * 3, self.nhid]):
                p.mul_(self.mask_hh)
            if p.shape == torch.Size([self.nhid * 3, self.ninp]):
                p.mul_(self.mask_hx)

    def forward(self, input, h):

        self.blockify_params()

        return self.gru(input, h)


class BlockLinear(nn.Module):
    def __init__(self, ninp, nout, k, bias=True):
        super(BlockLinear, self).__init__()

        assert ninp % k == 0
        assert nout % k == 0
        self.k = k

        self.w = nn.Parameter(torch.Tensor(self.k, ninp // k, nout // k))
        self.b = nn.Parameter(torch.Tensor(1, nout), requires_grad=bias)

        nn.init.xavier_uniform_(self.w)
        nn.init.zeros_(self.b)

    def forward(self, x):
        """

        :param x: Tensor, (B, D)
        :return:
        """

        *OTHER, D = x.shape
        x = x.reshape(np.prod(OTHER), self.k, -1)
        x = x.permute(1, 0, 2)
        x = torch.bmm(x, self.w)
        x = x.permute(1, 0, 2).reshape(*OTHER, -1)
        x += self.b
        return x


class BlockLayerNorm(nn.Module):
    def __init__(self, size, k):
        super(BlockLayerNorm, self).__init__()

        assert size % k == 0
        self.size = size
        self.k = k
        self.g = size // k
        self.norm = nn.LayerNorm(self.g, elementwise_affine=False)

    def forward(self, x):
        *OTHER, D = x.shape
        x = x.reshape(np.prod(OTHER), self.k, -1)
        x = self.norm(x)
        x = x.reshape(*OTHER, -1)
        return x


class BlockAttention(nn.Module):

    def __init__(self, d_model, num_blocks):
        super().__init__()

        assert d_model % num_blocks == 0, "d_model must be divisible by num_blocks"
        self.d_model = d_model
        self.num_blocks = num_blocks

    def forward(self, q, k, v):
        """
        q: batch_size x target_len x d_model
        k: batch_size x source_len x d_model
        v: batch_size x source_len x d_model
        attn_mask: target_len x source_len
        return: batch_size x target_len x d_model
        """

        B, T, _ = q.shape
        _, S, _ = k.shape

        q = q.view(B, T, self.num_blocks, -1).transpose(1, 2)
        k = k.view(B, S, self.num_blocks, -1).transpose(1, 2)
        v = v.view(B, S, self.num_blocks, -1).transpose(1, 2)

        q = q * (q.shape[-1] ** (-0.5))
        attn = torch.matmul(q, k.transpose(-1, -2))
        attn = F.softmax(attn, dim=-1)

        output = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, -1)

        return output


class LearnedPositionalEmbedding1D(nn.Module):

    def __init__(self, num_inputs, input_size, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Parameter(torch.zeros(1, num_inputs, input_size), requires_grad=True)
        nn.init.trunc_normal_(self.pe)

    def forward(self, input, offset=0):
        """
        input: batch_size x seq_len x d_model
        return: batch_size x seq_len x d_model
        """
        T = input.shape[1]
        return self.dropout(input + self.pe[:, offset:offset + T])


class CartesianPositionalEmbedding(nn.Module):

    def __init__(self, channels, image_size):
        super().__init__()

        self.projection = conv2d(4, channels, 1)
        self.pe = nn.Parameter(self.build_grid(image_size).unsqueeze(0), requires_grad=False)

    def build_grid(self, side_length):
        coords = torch.linspace(0., 1., side_length + 1)
        coords = 0.5 * (coords[:-1] + coords[1:])
        grid_y, grid_x = torch.meshgrid(coords, coords)
        return torch.stack((grid_x, grid_y, 1 - grid_x, 1 - grid_y), dim=0)

    def forward(self, inputs):
        # `inputs` has shape: [batch_size, out_channels, height, width].
        # `grid` has shape: [batch_size, in_channels, height, width].
        return inputs + self.projection(self.pe)


class OneHotDictionary(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.dictionary = nn.Embedding(vocab_size, emb_size)

    def forward(self, x):
        """
        x: B, N, vocab_size
        """

        tokens = torch.argmax(x, dim=-1)  # batch_size x N
        token_embs = self.dictionary(tokens)  # batch_size x N x emb_size
        return token_embs


def wrap_text(text, width=50):
    """Wrap text at specified width"""
    return textwrap.fill(text, width=width)


def normalize_values(values, min_val=None, max_val=None):
    """Normalize values to 0-1 range"""
    if min_val is None:
        min_val = np.min(values)
    if max_val is None:
        max_val = np.max(values)
    
    if min_val == max_val:
        return np.zeros_like(values)
    
    return (values - min_val) / (max_val - min_val)


def visualize_text_attention(text, text_embedding, slots, attns, emotion_label=None, num_blocks=8, 
                           fig_size=(15, 10), normalize=True):
    """
    Text data attention visualization with proper block structure and emotion class
    """
    # Set font for visualization
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    B, num_slots, slot_size = slots.size()
    
    # Create figure with adjusted ratio
    fig = plt.figure(figsize=fig_size)
    gs = plt.GridSpec(2, 2, width_ratios=[1, 1.5], height_ratios=[1, 1])
    
    # Add information to title
    fig.suptitle(f'Text Attention Analysis\nEmotion: {emotion_label or "N/A"} | Slots: {num_slots} | Blocks: {num_blocks}', 
                fontsize=12, y=0.95)
    
    # 1. Attention weights visualization
    ax1 = plt.subplot(gs[0, 0])
    attn_weights = attns.detach().cpu().numpy()
    if len(attn_weights.shape) == 3:
        attn_weights = attn_weights.squeeze(1)
    if len(attn_weights.shape) == 1:
        attn_weights = attn_weights.reshape(1, -1)
    
    if normalize:
        attn_weights = normalize_values(attn_weights)
        
    sns.heatmap(
        attn_weights, 
        cmap='YlOrRd',
        xticklabels=[f'Slot {i+1}' for i in range(num_slots)],
        yticklabels=['Text'],
        annot=True,
        fmt='.2f',
        square=True,
        vmin=0 if normalize else None,
        vmax=1 if normalize else None,
        ax=ax1
    )
    ax1.set_title(f'Slot Attention Weights\n{"(Normalized)" if normalize else ""}')
    
    # 2. Block value distribution visualization - wider ratio
    ax2 = plt.subplot(gs[0, 1])
    slot_values = slots.detach().cpu().numpy()
    
    block_size = slot_size // num_blocks
    reshaped_values = slot_values[0].reshape(num_slots, num_blocks, -1)
    block_means = np.mean(reshaped_values, axis=2)
    block_stds = np.std(reshaped_values, axis=2)
    
    scale_factor = 1e6
    scaled_means = block_means * scale_factor
    
    if normalize:
        scaled_means = normalize_values(scaled_means)
    
    hm = sns.heatmap(
        scaled_means,
        cmap='viridis',
        xticklabels=[f'B{i+1}' for i in range(num_blocks)],
        yticklabels=[f'S{i+1}' for i in range(num_slots)],
        annot=True,
        fmt='.2f',
        cbar_kws={'label': 'Normalized Value' if normalize else f'Mean Value (×10^-6)'},
        vmin=0 if normalize else None,
        vmax=1 if normalize else None,
        ax=ax2
    )
    # Adjust heatmap aspect ratio
    ax2.set_aspect('auto')
    ax2.set_title(f'Slot Block Values (Mean)\nBlock Size: {block_size}')
    
    # 3. Block value sequence by slot
    ax3 = plt.subplot(gs[1, :])
    x = np.arange(num_blocks)
    
    if normalize:
        block_stds = normalize_values(block_stds)
    else:
        block_stds = block_stds * scale_factor
        
    for i in range(num_slots):
        ax3.errorbar(x, scaled_means[i], 
                    yerr=block_stds[i],
                    label=f'Slot {i+1}', 
                    marker='o',
                    capsize=5)
    
    ax3.set_title(f'Block Values by Slot\nwith Standard Deviation')
    ax3.set_xlabel('Block Index')
    ax3.set_ylabel('Normalized Value' if normalize else f'Mean Value (×10^-6)')
    ax3.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.15))
    ax3.grid(True, alpha=0.3)
    if normalize:
        ax3.set_ylim(-0.1, 1.1)
    
    # 4. Display original text
    wrapped_text = wrap_text(text, width=80)
    text_box = fig.text(0.1, 0.02, f'Original Text:\n{wrapped_text}', 
                       wrap=True, fontsize=10,
                       bbox=dict(facecolor='white', alpha=0.8, 
                               edgecolor='lightgray', boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2, top=0.9)
    return fig


def visualize_text_reconstruction(original_text, reconstructed_embedding, original_embedding):
    """
    Compare original text with reconstructed embeddings
    Args:
        original_text: Original text (string)
        reconstructed_embedding: Reconstructed embedding (B, D)
        original_embedding: Original embedding (B, D)
    Returns:
        fig: matplotlib figure object
    """
    # Calculate cosine similarity
    cos_sim = F.cosine_similarity(reconstructed_embedding, original_embedding, dim=-1)
    
    fig = plt.figure(figsize=(8, 4))
    plt.text(0.1, 0.7, f'Original Text: {original_text}', fontsize=12)
    plt.text(0.1, 0.4, f'Reconstruction Similarity: {cos_sim.item():.4f}', fontsize=12)
    plt.axis('off')
    
    return fig


def visualize_class_comparison(texts, slots_list, attns_list, emotion_labels, 
                             num_blocks=8, fig_size=(20, 20), normalize=True):
    """
    Compare slot attention and block values across all emotion classes in a single visualization
    """
    # Set font for visualization
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    num_classes = len(emotion_labels)
    fig = plt.figure(figsize=fig_size)
    gs = plt.GridSpec(3, 1, height_ratios=[1, 1.5, 1.5])
    
    # Add main title
    fig.suptitle(f'Emotion Class Comparison Analysis\nTotal {num_classes} Classes | {len(texts)} Samples', 
                fontsize=14, y=0.95)
    
    # 1. Attention weights comparison
    ax1 = plt.subplot(gs[0])
    attention_data = []
    for i, (attns, label) in enumerate(zip(attns_list, emotion_labels)):
        attn_weights = attns.detach().cpu().numpy()
        if len(attn_weights.shape) == 3:
            attn_weights = attn_weights.squeeze(1)
        attention_data.append(attn_weights[0])
    attention_matrix = np.stack(attention_data)
    
    if normalize:
        attention_matrix = normalize_values(attention_matrix)
    
    sns.heatmap(
        attention_matrix,
        cmap='YlOrRd',
        xticklabels=[f'Slot {i+1}' for i in range(attention_matrix.shape[1])],
        yticklabels=emotion_labels,
        annot=True,
        fmt='.2f',
        vmin=0 if normalize else None,
        vmax=1 if normalize else None,
        ax=ax1
    )
    ax1.set_title(f'Attention Weights by Emotion\n{"(Normalized)" if normalize else ""}')
    
    # 2. Block values comparison - wider ratio
    ax2 = plt.subplot(gs[1])
    block_data = []
    for slots in slots_list:
        slot_values = slots.detach().cpu().numpy()
        block_size = slot_values.shape[-1] // num_blocks
        reshaped_values = slot_values[0].reshape(-1, num_blocks, block_size)
        block_means = np.mean(reshaped_values, axis=2)
        block_data.append(block_means)
    block_matrix = np.stack(block_data)
    
    if normalize:
        block_matrix = normalize_values(block_matrix)
    else:
        block_matrix = block_matrix * 1e6
    
    # Visualize block values for each emotion class
    for slot_idx in range(block_matrix.shape[1]):
        slot_data = block_matrix[:, slot_idx, :]
        ax2.plot([], [], label=f'Slot {slot_idx+1}', linestyle='-')
    
    for class_idx, emotion in enumerate(emotion_labels):
        for slot_idx in range(block_matrix.shape[1]):
            slot_values = block_matrix[class_idx, slot_idx, :]
            ax2.plot(range(num_blocks), slot_values, 
                    marker='o', linestyle='-', alpha=0.7,
                    label=f'{emotion} (S{slot_idx+1})')
    
    ax2.set_xlabel('Block Sequence')
    ax2.set_ylabel('Block Value' + (' (Normalized)' if normalize else ' (×10^-6)'))
    ax2.set_title('Sequential Block Values by Emotion and Slot')
    ax2.set_xticks(range(num_blocks))
    ax2.set_xticklabels([f'B{i+1}' for i in range(num_blocks)])
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 3. Temporal pattern analysis
    ax3 = plt.subplot(gs[2])
    mean_block_patterns = np.mean(block_matrix, axis=1)
    for class_idx, emotion in enumerate(emotion_labels):
        pattern = mean_block_patterns[class_idx]
        ax3.plot(range(num_blocks), pattern, 
                marker='o', label=emotion, linewidth=2)
    
    ax3.set_xlabel('Block Sequence')
    ax3.set_ylabel('Mean Block Value' + (' (Normalized)' if normalize else ' (×10^-6)'))
    ax3.set_title('Temporal Patterns by Emotion (All Slots Average)')
    ax3.set_xticks(range(num_blocks))
    ax3.set_xticklabels([f'B{i+1}' for i in range(num_blocks)])
    ax3.grid(True, alpha=0.3)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 4. Example texts
    text_y_pos = -0.3
    fig.text(0.02, text_y_pos + 0.02, 'Example Texts:', fontsize=10, weight='bold')
    for i, (text, label) in enumerate(zip(texts, emotion_labels)):
        wrapped_text = wrap_text(text, width=80)
        fig.text(0.02, text_y_pos - i*0.04, f'{label}: {wrapped_text}', 
                fontsize=9, wrap=True,
                bbox=dict(facecolor='white', alpha=0.8, 
                         edgecolor='lightgray', boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3, top=0.92)
    return fig


def analyze_emotion_relationships(slots_list, attns_list, emotion_labels, num_blocks=8):
    """
    Analyze relationships between emotion classes
    
    Returns:
        - attention_similarity: Similarity matrix based on attention patterns
        - block_pattern_similarity: Similarity matrix based on block patterns
        - emotion_distances: Distance matrix between emotions
    """
    num_classes = len(emotion_labels)
    
    # 1. Calculate attention pattern similarity
    attention_patterns = []
    for attns in attns_list:
        attn_weights = attns.detach().cpu().numpy()
        if len(attn_weights.shape) == 3:
            attn_weights = attn_weights.squeeze(1)
        attention_patterns.append(attn_weights[0])
    attention_patterns = np.stack(attention_patterns)
    
    attention_similarity = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            attention_similarity[i,j] = np.corrcoef(
                attention_patterns[i].flatten(), 
                attention_patterns[j].flatten()
            )[0,1]
    
    # 2. Calculate block pattern similarity
    block_patterns = []
    for slots in slots_list:
        slot_values = slots.detach().cpu().numpy()
        block_size = slot_values.shape[-1] // num_blocks
        reshaped_values = slot_values[0].reshape(-1, num_blocks, block_size)
        block_means = np.mean(reshaped_values, axis=2)
        block_patterns.append(block_means)
    block_patterns = np.stack(block_patterns)
    
    block_pattern_similarity = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            block_pattern_similarity[i,j] = np.corrcoef(
                block_patterns[i].flatten(), 
                block_patterns[j].flatten()
            )[0,1]
    
    # 3. Calculate overall feature-based distances
    combined_features = np.concatenate([
        attention_patterns.reshape(num_classes, -1),
        block_patterns.reshape(num_classes, -1)
    ], axis=1)
    
    emotion_distances = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            emotion_distances[i,j] = np.linalg.norm(
                combined_features[i] - combined_features[j]
            )
    
    return attention_similarity, block_pattern_similarity, emotion_distances

def visualize_emotion_relationships(attention_similarity, block_pattern_similarity, 
                                 emotion_distances, emotion_labels):
    """
    Visualize relationships between emotion classes
    """
    fig = plt.figure(figsize=(20, 6))
    
    # 1. Attention pattern similarity
    ax1 = plt.subplot(131)
    sns.heatmap(attention_similarity, 
                xticklabels=emotion_labels,
                yticklabels=emotion_labels,
                cmap='RdYlBu_r',
                center=0,
                annot=True,
                fmt='.2f',
                ax=ax1)
    ax1.set_title('Attention Pattern Similarity\n(Correlation Coefficient)')
    
    # 2. Block pattern similarity
    ax2 = plt.subplot(132)
    sns.heatmap(block_pattern_similarity,
                xticklabels=emotion_labels,
                yticklabels=emotion_labels,
                cmap='RdYlBu_r',
                center=0,
                annot=True,
                fmt='.2f',
                ax=ax2)
    ax2.set_title('Block Pattern Similarity\n(Correlation Coefficient)')
    
    # 3. Emotion distances
    ax3 = plt.subplot(133)
    # Normalize distances to 0-1
    normalized_distances = normalize_values(emotion_distances)
    sns.heatmap(normalized_distances,
                xticklabels=emotion_labels,
                yticklabels=emotion_labels,
                cmap='YlOrRd',
                annot=True,
                fmt='.2f',
                ax=ax3)
    ax3.set_title('Normalized Distances\n(Euclidean Distance in Feature Space)')
    
    plt.tight_layout()
    return fig

def log_text_visualizations(writer, epoch, text, text_embedding, slots, attns, 
                          reconstructed_embedding, emotion_label=None, num_blocks=8, 
                          tag_prefix='text', debug=False):
    """
    Log text-related visualizations to tensorboard
    """
    if debug:
        print(f"[DEBUG] Logging visualizations for epoch {epoch}")
        print(f"[DEBUG] Text: {text}")
        print(f"[DEBUG] Emotion: {emotion_label}")
        print(f"[DEBUG] Attention shape: {attns.shape}")
        print(f"[DEBUG] Slots shape: {slots.shape}")
    
    # 1. Individual visualizations
    attn_fig = visualize_text_attention(text, text_embedding, slots, attns, 
                                      emotion_label=emotion_label, num_blocks=num_blocks)
    writer.add_figure(f'{tag_prefix}/attention_{emotion_label}', attn_fig, epoch)
    
    # 2. Reconstruction visualization
    recon_fig = visualize_text_reconstruction(text, reconstructed_embedding, text_embedding)
    writer.add_figure(f'{tag_prefix}/reconstruction_{emotion_label}', recon_fig, epoch)
    
    # 3. Record numerical metrics
    cos_sim = F.cosine_similarity(reconstructed_embedding, text_embedding, dim=-1).mean()
    writer.add_scalar(f'{tag_prefix}/cosine_similarity_{emotion_label}', cos_sim, epoch)
