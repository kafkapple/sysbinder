import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd


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


def visualize_text_attention(text, text_embedding, slots, attns, fig_size=(10, 8)):
    """
    텍스트 데이터의 attention 시각화
    Args:
        text: 원본 텍스트 (문자열)
        text_embedding: 텍스트 임베딩 (B, D)
        slots: 슬롯 표현 (B, num_slots, slot_size)
        attns: attention weights (B, num_slots, 1)
    Returns:
        fig: matplotlib figure 객체
    """
    B, num_slots, _ = slots.size()
    
    # Figure 생성
    fig = plt.figure(figsize=fig_size)
    
    # 1. Attention weights 시각화
    plt.subplot(2, 1, 1)
    # 3D -> 2D로 변환
    attn_weights = attns.squeeze(-1).detach().cpu().numpy()  # (B, num_slots)
    if len(attn_weights.shape) == 3:  # (1, num_slots, 1) -> (num_slots, 1)
        attn_weights = attn_weights.squeeze(0)
    if len(attn_weights.shape) == 1:  # (num_slots,) -> (num_slots, 1)
        attn_weights = attn_weights[:, None]
        
    sns.heatmap(attn_weights, cmap='YlOrRd', 
                xticklabels=['Text'], yticklabels=[f'Slot {i+1}' for i in range(num_slots)])
    plt.title('Slot Attention Weights')
    
    # 2. 슬롯 값들의 분포 시각화
    plt.subplot(2, 1, 2)
    slot_values = slots.detach().cpu().numpy()
    if len(slot_values.shape) == 3:  # (B, num_slots, slot_size)
        slot_values = slot_values.reshape(-1, slot_values.shape[-1])
    sns.boxplot(data=pd.DataFrame(slot_values))
    plt.title('Slot Value Distributions')
    plt.xlabel('Slot Components')
    plt.ylabel('Value')
    
    plt.tight_layout()
    return fig


def visualize_text_reconstruction(original_text, reconstructed_embedding, original_embedding):
    """
    원본 텍스트와 재구성된 임베딩의 비교 시각화
    Args:
        original_text: 원본 텍스트 (문자열)
        reconstructed_embedding: 재구성된 임베딩 (B, D)
        original_embedding: 원본 임베딩 (B, D)
    Returns:
        fig: matplotlib figure 객체
    """
    # 코사인 유사도 계산
    cos_sim = F.cosine_similarity(reconstructed_embedding, original_embedding, dim=-1)
    
    fig = plt.figure(figsize=(8, 4))
    plt.text(0.1, 0.7, f'Original Text: {original_text}', fontsize=12)
    plt.text(0.1, 0.4, f'Reconstruction Similarity: {cos_sim.item():.4f}', fontsize=12)
    plt.axis('off')
    
    return fig


def log_text_visualizations(writer, epoch, text, text_embedding, slots, attns, 
                          reconstructed_embedding, tag_prefix='text'):
    """
    텍스트 관련 시각화를 tensorboard에 기록
    Args:
        writer: tensorboard SummaryWriter 객체
        epoch: 현재 에폭
        text: 원본 텍스트
        text_embedding: 텍스트 임베딩
        slots: 슬롯 표현
        attns: attention weights
        reconstructed_embedding: 재구성된 임베딩
        tag_prefix: tensorboard에 기록될 태그의 접두어
    """
    # 1. Attention 시각화
    attn_fig = visualize_text_attention(text, text_embedding, slots, attns)
    writer.add_figure(f'{tag_prefix}/attention', attn_fig, epoch)
    
    # 2. 재구성 결과 시각화
    recon_fig = visualize_text_reconstruction(text, reconstructed_embedding, text_embedding)
    writer.add_figure(f'{tag_prefix}/reconstruction', recon_fig, epoch)
    
    # 3. 수치 메트릭 기록
    cos_sim = F.cosine_similarity(reconstructed_embedding, text_embedding, dim=-1).mean()
    writer.add_scalar(f'{tag_prefix}/cosine_similarity', cos_sim, epoch)
