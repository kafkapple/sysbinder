from utils import *
from transformer import TransformerEncoder, TransformerDecoder
from dvae import dVAE, dVAE_text


class BlockPrototypeMemory(nn.Module):
    def __init__(self, num_prototypes, num_blocks, d_model):
        super().__init__()

        self.num_prototypes = num_prototypes
        self.num_blocks = num_blocks
        self.d_model = d_model
        self.d_block = self.d_model // self.num_blocks

        # block prototype memory
        self.mem_params = nn.Parameter(torch.zeros(1, num_prototypes, num_blocks, self.d_block), requires_grad=True)
        nn.init.trunc_normal_(self.mem_params)
        self.mem_proj = nn.Sequential(
            linear(self.d_block, 4 * self.d_block),
            nn.ReLU(),
            linear(4 * self.d_block, 4 * self.d_block),
            nn.ReLU(),
            linear(4 * self.d_block, 4 * self.d_block),
            nn.ReLU(),
            linear(4 * self.d_block, self.d_block)
        )

        # norms
        self.norm_mem = BlockLayerNorm(d_model, num_blocks)
        self.norm_query = BlockLayerNorm(d_model, num_blocks)

        # block attention
        self.attn = BlockAttention(d_model, num_blocks)

    def forward(self, queries):
        '''
        queries: B, N, d_model
        return: B, N, d_model
        '''

        B, N, _ = queries.shape

        # get memories
        mem = self.mem_proj(self.mem_params)  # 1, num_prototypes, num_blocks, d_block
        mem = mem.reshape(1, self.num_prototypes, -1)  # 1, num_prototypes, d_model

        # norms
        mem = self.norm_mem(mem)  # 1, num_prototypes, d_model
        queries = self.norm_query(queries)  # B, N, d_model

        # broadcast
        mem = mem.expand(B, -1, -1)  # B, num_prototypes, d_model

        # read
        return self.attn(queries, mem, mem)  # B, N, d_model


class SysBinder(nn.Module):
    
    def __init__(self, num_iterations, num_slots,
                 input_size, slot_size, mlp_hidden_size, num_prototypes, num_blocks,
                 epsilon=1e-8):
        super().__init__()
        
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.input_size = input_size
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.num_blocks = num_blocks
        self.epsilon = epsilon

        # parameters for Gaussian initialization (shared by all slots).
        self.slot_mu = nn.Parameter(torch.Tensor(1, 1, slot_size))
        self.slot_log_sigma = nn.Parameter(torch.Tensor(1, 1, slot_size))
        nn.init.xavier_uniform_(self.slot_mu)
        nn.init.xavier_uniform_(self.slot_log_sigma)

        # norms
        self.norm_inputs = nn.LayerNorm(input_size)
        self.norm_slots = nn.LayerNorm(slot_size)
        self.norm_mlp = BlockLayerNorm(slot_size, num_blocks)

        # linear maps for the attention module.
        self.project_q = linear(slot_size, slot_size, bias=False)
        self.project_k = linear(input_size, slot_size, bias=False)
        self.project_v = linear(input_size, slot_size, bias=False)
        
        # slot update functions.
        self.gru = BlockGRU(slot_size, slot_size, num_blocks)
        self.mlp = nn.Sequential(
            BlockLinear(slot_size, mlp_hidden_size, num_blocks),
            nn.ReLU(),
            BlockLinear(mlp_hidden_size, slot_size, num_blocks))
        self.prototype_memory = BlockPrototypeMemory(num_prototypes, num_blocks, slot_size)

    def forward(self, inputs):
        B, num_inputs, input_size = inputs.size()

        # initialize slots
        slots = inputs.new_empty(B, self.num_slots, self.slot_size).normal_()
        slots = self.slot_mu + torch.exp(self.slot_log_sigma) * slots

        # setup key and value
        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs)  # Shape: [batch_size, T, num_inputs, slot_size].
        v = self.project_v(inputs)  # Shape: [batch_size, T, num_inputs, slot_size].
        k = (self.slot_size ** (-0.5)) * k
        
        for i in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention.
            q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].
            attn_logits = torch.bmm(k, q.transpose(-1, -2))
            attn_vis = F.softmax(attn_logits, dim=-1)
            # `attn_vis` has shape: [batch_size, num_inputs, num_slots].

            # Weighted mean.
            attn = attn_vis + self.epsilon
            attn = attn / torch.sum(attn, dim=-2, keepdim=True)
            updates = torch.bmm(attn.transpose(-1, -2), v)
            # `updates` has shape: [batch_size, num_slots, slot_size].

            # Slot update.
            slots = self.gru(
                updates.view(-1, self.slot_size),
                slots_prev.view(-1, self.slot_size)
            )
            slots = slots.view(-1, self.num_slots, self.slot_size)
            slots = slots + self.mlp(self.norm_mlp(slots))
            slots = self.prototype_memory(slots)

        return slots, attn_vis
    

class ImageEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.cnn = nn.Sequential(
            Conv2dBlock(args.image_channels, args.cnn_hidden_size, 5, 1 if args.image_size == 64 else 2, 2),
            Conv2dBlock(args.cnn_hidden_size, args.cnn_hidden_size, 5, 1, 2),
            Conv2dBlock(args.cnn_hidden_size, args.cnn_hidden_size, 5, 1, 2),
            conv2d(args.cnn_hidden_size, args.d_model, 5, 1, 2),
        )

        self.pos = CartesianPositionalEmbedding(args.d_model, args.image_size if args.image_size == 64 else args.image_size // 2)

        self.layer_norm = nn.LayerNorm(args.d_model)

        self.mlp = nn.Sequential(
            linear(args.d_model, args.d_model, weight_init='kaiming'),
            nn.ReLU(),
            linear(args.d_model, args.d_model))

        self.sysbinder = SysBinder(
            args.num_iterations, args.num_slots,
            args.d_model, args.slot_size, args.mlp_hidden_size, args.num_prototypes, args.num_blocks)


class ImageDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.slot_proj = BlockLinear(args.slot_size, args.d_model * args.num_blocks, args.num_blocks)

        self.block_pos = nn.Parameter(torch.zeros(1, 1, args.d_model * args.num_blocks), requires_grad=True)
        self.block_pos_proj = nn.Sequential(
            BlockLinear(args.d_model * args.num_blocks, args.d_model * args.num_blocks, args.num_blocks),
            nn.ReLU(),
            BlockLinear(args.d_model * args.num_blocks, args.d_model * args.num_blocks, args.num_blocks)
        )

        self.block_coupler = TransformerEncoder(num_blocks=1, d_model=args.d_model, num_heads=4)

        self.dict = OneHotDictionary(args.vocab_size, args.d_model)

        self.bos = nn.Parameter(torch.Tensor(1, 1, args.d_model))
        nn.init.xavier_uniform_(self.bos)

        self.decoder_pos = LearnedPositionalEmbedding1D(1 + (args.image_size // 4) ** 2, args.d_model)

        self.tf = TransformerDecoder(
            args.num_decoder_layers, (args.image_size // 4) ** 2, args.d_model, args.num_decoder_heads, args.dropout)

        self.head = linear(args.d_model, args.vocab_size, bias=False)


class SysBinderImageAutoEncoder(nn.Module):
    
    def __init__(self, args):
        super().__init__()
        
        self.num_iterations = args.num_iterations
        self.num_slots = args.num_slots
        self.cnn_hidden_size = args.cnn_hidden_size
        self.slot_size = args.slot_size
        self.mlp_hidden_size = args.mlp_hidden_size
        self.num_prototypes = args.num_prototypes
        self.image_channels = args.image_channels
        self.image_size = args.image_size
        self.vocab_size = args.vocab_size
        self.d_model = args.d_model
        self.num_blocks = args.num_blocks

        # dvae
        self.dvae = dVAE(args.vocab_size, input_channels=args.image_channels)  # 이미지용 dVAE는 input_channels 키워드 인자 사용

        # encoder networks
        self.image_encoder = ImageEncoder(args)

        # decoder networks
        self.image_decoder = ImageDecoder(args)

    def forward(self, image, tau):
        """
        image: B, C, H, W (이미지의 경우) 또는 B, D (텍스트 임베딩의 경우)
        tau: float
        """
        if len(image.size()) == 4:  # 이미지 입력의 경우
            B, C, H, W = image.size()

            # dvae encode
            z_logits = F.log_softmax(self.dvae.encoder(image), dim=1)  # B, vocab_size, H_enc, W_enc
            z_soft = gumbel_softmax(z_logits, tau, False, dim=1)  # B, vocab_size, H_enc, W_enc
            z_hard = gumbel_softmax(z_logits, tau, True, dim=1).detach()  # B, vocab_size, H_enc, W_enc
            z_hard = z_hard.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)  # B, H_enc * W_enc, vocab_size
            z_emb = self.image_decoder.dict(z_hard)  # B, H_enc * W_enc, d_model
            z_emb = torch.cat([self.image_decoder.bos.expand(B, -1, -1), z_emb], dim=1)  # B, 1 + H_enc * W_enc, d_model
            z_emb = self.image_decoder.decoder_pos(z_emb)  # B, 1 + H_enc * W_enc, d_model

            # dvae recon
            dvae_recon = self.dvae.decoder(z_soft).reshape(B, C, H, W)  # B, C, H, W
            dvae_mse = ((image - dvae_recon) ** 2).sum() / B  # 1

            # sysbinder
            emb = self.image_encoder.cnn(image)  # B, cnn_hidden_size, H, W
            emb = self.image_encoder.pos(emb)  # B, cnn_hidden_size, H, W
            H_enc, W_enc = emb.shape[-2:]

            emb_set = emb.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)  # B, H * W, cnn_hidden_size
            emb_set = self.image_encoder.mlp(self.image_encoder.layer_norm(emb_set))  # B, H * W, cnn_hidden_size
            emb_set = emb_set.reshape(B, H_enc * W_enc, self.d_model)  # B, H * W, cnn_hidden_size

            slots, attns = self.image_encoder.sysbinder(emb_set)  # slots: B, num_slots, slot_size
                                                                  # attns: B, num_slots, num_inputs

            attns = attns\
                .transpose(-1, -2)\
                .reshape(B, self.num_slots, 1, H_enc, W_enc)\
                .repeat_interleave(H // H_enc, dim=-2)\
                .repeat_interleave(W // W_enc, dim=-1)  # B, num_slots, 1, H, W
            attns = image.unsqueeze(1) * attns + (1. - attns)  # B, num_slots, C, H, W

        else:  # 텍스트 임베딩의 경우
            B, D = image.size()
            
            # 텍스트 임베딩을 d_model 차원으로 변환
            emb_set = image.unsqueeze(1)  # B, 1, D
            
            # sysbinder를 통한 처리
            slots, attns = self.image_encoder.sysbinder(emb_set)  # slots: B, num_slots, slot_size
                                                                 # attns: B, num_slots, 1
            
            # 텍스트의 경우 reconstruction loss 대신 identity mapping loss 사용
            dvae_recon = image  # 원본 텍스트 임베딩 반환
            dvae_mse = torch.tensor(0.0, device=image.device)  # 텍스트의 경우 MSE 없음

        # block coupling (이미지와 텍스트 모두 동일하게 처리)
        slots = self.image_decoder.slot_proj(slots)  # B, num_slots, num_blocks * d_model
        slots = slots + self.image_decoder.block_pos_proj(self.image_decoder.block_pos)  # B, num_slots, num_blocks * d_model
        slots = slots.reshape(B, self.num_slots, self.num_blocks, -1)  # B, num_slots, num_blocks, d_model
        slots = self.image_decoder.block_coupler(slots.flatten(end_dim=1))  # B * num_slots, num_blocks, d_model
        slots = slots.reshape(B, self.num_slots * self.num_blocks, -1)  # B, num_slots * num_blocks, d_model

        if len(image.size()) == 4:  # 이미지 입력의 경우
            # decode
            pred = self.image_decoder.tf(z_emb[:, :-1], slots)   # B, H_enc * W_enc, d_model
            pred = self.image_decoder.head(pred)  # B, H_enc * W_enc, vocab_size
            cross_entropy = -(z_hard * torch.log_softmax(pred, dim=-1)).sum() / B  # 1
        else:  # 텍스트 임베딩의 경우
            # 텍스트의 경우 cross entropy 대신 L2 loss 사용
            pred = slots.mean(dim=1)  # B, d_model
            cross_entropy = F.mse_loss(pred, image)

        return (dvae_recon.clamp(0., 1.) if len(image.size()) == 4 else dvae_recon,
                cross_entropy,
                dvae_mse,
                attns)

    def encode(self, image):
        """
        image: B, C, H, W
        """
        B, C, H, W = image.size()

        # sysbinder
        emb = self.image_encoder.cnn(image)  # B, cnn_hidden_size, H, W
        emb = self.image_encoder.pos(emb)  # B, cnn_hidden_size, H, W
        H_enc, W_enc = emb.shape[-2:]

        emb_set = emb.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)  # B, H * W, cnn_hidden_size
        emb_set = self.image_encoder.mlp(self.image_encoder.layer_norm(emb_set))  # B, H * W, cnn_hidden_size
        emb_set = emb_set.reshape(B, H_enc * W_enc, self.d_model)  # B, H * W, cnn_hidden_size

        slots, attns = self.image_encoder.sysbinder(emb_set)  # slots: B, num_slots, slot_size
                                                              # attns: B, num_slots, num_inputs

        attns = attns\
            .transpose(-1, -2)\
            .reshape(B, self.num_slots, 1, H_enc, W_enc)\
            .repeat_interleave(H // H_enc, dim=-2)\
            .repeat_interleave(W // W_enc, dim=-1)  # B, num_slots, 1, H, W
        attns_vis = image.unsqueeze(1) * attns + (1. - attns)  # B, num_slots, C, H, W
        
        return slots, attns_vis, attns

    def decode(self, slots):
        """
        slots: B, N, slot_size
        """
        B, num_slots, slot_size = slots.size()
        H_enc, W_enc = (self.image_size // 4), (self.image_size // 4)
        gen_len = H_enc * W_enc

        # block coupling
        slots = self.image_decoder.slot_proj(slots)  # B, num_slots, num_blocks * d_model
        slots = slots + self.image_decoder.block_pos_proj(self.image_decoder.block_pos)  # B, num_slots, num_blocks * d_model
        slots = slots.reshape(B, num_slots, self.num_blocks, -1)  # B, num_slots, num_blocks, d_model
        slots = self.image_decoder.block_coupler(slots.flatten(end_dim=1))  # B * num_slots, num_blocks, d_model
        slots = slots.reshape(B, num_slots * self.num_blocks, -1)  # B, num_slots * num_blocks, d_model

        # generate image tokens auto-regressively
        z_gen = slots.new_zeros(0)
        input = self.image_decoder.bos.expand(B, 1, -1)
        for t in range(gen_len):
            decoder_output = self.image_decoder.tf(
                self.image_decoder.decoder_pos(input),
                slots
            )
            z_next = F.one_hot(self.image_decoder.head(decoder_output)[:, -1:].argmax(dim=-1), self.vocab_size)
            z_gen = torch.cat((z_gen, z_next), dim=1)
            input = torch.cat((input, self.image_decoder.dict(z_next)), dim=1)

        z_gen = z_gen.transpose(1, 2).float().reshape(B, -1, H_enc, W_enc)
        gen_transformer = self.dvae.decoder(z_gen)

        return gen_transformer.clamp(0., 1.)

    def reconstruct_autoregressive(self, image):
        """
        image: batch_size x image_channels x H x W
        """
        B, C, H, W = image.size()
        slots, attns, _ = self.encode(image)
        recon_transformer = self.decode(slots)
        recon_transformer = recon_transformer.reshape(B, C, H, W)

        return recon_transformer


class SysBinderTextAutoEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.num_iterations = args.num_iterations
        self.num_slots = args.num_slots
        self.slot_size = args.slot_size
        self.mlp_hidden_size = args.mlp_hidden_size
        self.num_prototypes = args.num_prototypes
        self.d_model = args.d_model
        self.num_blocks = args.num_blocks
        self.use_dvae = getattr(args, 'use_text_dvae', True)  # dVAE 기본값을 True로 변경
        
        # Loss balancing weights
        self.loss_weights = {
            'mse': 1.0,
            'cross_entropy': 0.1,
            'cosine': 0.5
        }
        
        if self.use_dvae:
            # Text dVAE - 텍스트 임베딩을 이산화된 토큰으로 변환
            self.dvae = dVAE_text(args.text_vocab_size, self.d_model)
            self.dict = OneHotDictionary(args.text_vocab_size, self.d_model)
            self.bos = nn.Parameter(torch.Tensor(1, 1, self.d_model))
            nn.init.xavier_uniform_(self.bos)

        # Text Encoder (sysbinder 기반)
        self.text_encoder = nn.ModuleDict({
            'layer_norm': nn.LayerNorm(args.d_model),
            'mlp': nn.Sequential(
                linear(args.d_model, args.d_model, weight_init='kaiming'),
                nn.ReLU(),
                linear(args.d_model, args.d_model)
            ),
            'sysbinder': SysBinder(
                args.num_iterations, args.num_slots,
                args.d_model, args.slot_size, args.mlp_hidden_size,
                args.num_prototypes, args.num_blocks
            )
        })

        # Text Decoder
        self.text_decoder = nn.ModuleDict({
            'slot_proj': BlockLinear(args.slot_size, args.d_model * args.num_blocks, args.num_blocks),
            'block_pos_proj': nn.Sequential(
                BlockLinear(args.d_model * args.num_blocks, args.d_model * args.num_blocks, args.num_blocks),
                nn.ReLU(),
                BlockLinear(args.d_model * args.num_blocks, args.d_model * args.num_blocks, args.num_blocks)
            ),
            'block_coupler': TransformerEncoder(num_blocks=1, d_model=args.d_model, num_heads=4)
        })
        
        # Parameters
        self.block_pos = nn.Parameter(torch.zeros(1, 1, args.d_model * args.num_blocks))
        nn.init.normal_(self.block_pos, std=0.02)

    def forward(self, text_embedding, tau=None):
        """
        text_embedding: B, D (텍스트 임베딩)
        tau: float (dVAE 사용 시 temperature parameter)
        """
        # 입력 텍스트 임베딩의 차원 처리
        if len(text_embedding.size()) > 2:
            B = text_embedding.size(0)
            text_embedding = text_embedding.view(B, -1)
        else:
            B = text_embedding.size(0)

        # 입력 임베딩 정규화
        text_embedding = F.normalize(text_embedding, p=2, dim=-1)

        if self.use_dvae:
            # dVAE를 통한 임베딩 처리
            z_logits = self.dvae.encoder(text_embedding)  # B, vocab_size
            z_soft = gumbel_softmax(z_logits, tau, False, dim=1)  # B, vocab_size
            z_hard = gumbel_softmax(z_logits, tau, True, dim=1).detach()  # B, vocab_size
            z_emb = self.dict(z_hard)  # B, d_model
            
            # dVAE reconstruction
            dvae_recon = self.dvae.decoder(z_soft)  # B, D
            dvae_mse = F.mse_loss(text_embedding, dvae_recon)
            
            # 처리된 임베딩으로 sysbinder 입력 생성
            emb_set = z_emb.unsqueeze(1)  # B, 1, D
        else:
            # 기존 방식대로 직접 임베딩 처리
            emb_set = text_embedding.unsqueeze(1)  # B, 1, D
            dvae_recon = text_embedding
            dvae_mse = torch.tensor(0.0, device=text_embedding.device)
        
        # 텍스트 임베딩 전처리
        emb_set = self.text_encoder['mlp'](self.text_encoder['layer_norm'](emb_set))
        
        # Sysbinder를 통한 처리
        slots, attns = self.text_encoder['sysbinder'](emb_set)
        
        # Block coupling
        slots = self.text_decoder['slot_proj'](slots)
        slots = slots + self.text_decoder['block_pos_proj'](self.block_pos)
        slots = slots.reshape(B, self.num_slots, self.num_blocks, -1)
        slots = self.text_decoder['block_coupler'](slots.flatten(end_dim=1))
        slots = slots.reshape(B, self.num_slots * self.num_blocks, -1)
        
        # 최종 출력 생성
        pred = slots.mean(dim=1)  # B, d_model
        pred = F.normalize(pred, p=2, dim=-1)  # L2 정규화
        
        # Loss 계산
        if self.use_dvae:
            # 1. Cross Entropy Loss (dVAE)
            cross_entropy = -(z_hard * F.log_softmax(z_logits, dim=-1)).sum(dim=-1).mean()
            
            # 2. MSE Loss (reconstruction)
            mse = F.mse_loss(pred, z_emb)
            
            # 3. Cosine Similarity Loss
            cosine_loss = 1 - F.cosine_similarity(pred, z_emb, dim=-1).mean()
            
            # Combined loss
            total_loss = (
                self.loss_weights['cross_entropy'] * cross_entropy +
                self.loss_weights['mse'] * mse +
                self.loss_weights['cosine'] * cosine_loss
            )
        else:
            # dVAE를 사용하지 않는 경우
            mse = F.mse_loss(pred, text_embedding)
            cosine_loss = 1 - F.cosine_similarity(pred, text_embedding, dim=-1).mean()
            total_loss = self.loss_weights['mse'] * mse + self.loss_weights['cosine'] * cosine_loss
            cross_entropy = torch.tensor(0.0, device=text_embedding.device)
        
        return dvae_recon, total_loss, dvae_mse, attns

    def encode(self, text_embedding):
        """
        text_embedding: B, D
        """
        B, D = text_embedding.size()
        
        # 텍스트 임베딩 전처리
        emb_set = text_embedding.unsqueeze(1)
        emb_set = self.text_encoder['mlp'](self.text_encoder['layer_norm'](emb_set))
        
        # Sysbinder를 통한 처리
        slots, attns = self.text_encoder['sysbinder'](emb_set)
        
        return slots, text_embedding.unsqueeze(1), attns  # attns_vis는 원본 임베딩 반환

    def decode(self, slots):
        """
        slots: B, N, slot_size
        """
        B, num_slots, slot_size = slots.size()
        
        # Block coupling
        slots = self.text_decoder['slot_proj'](slots)
        slots = slots + self.text_decoder['block_pos_proj'](self.block_pos)
        slots = slots.reshape(B, num_slots, self.num_blocks, -1)
        slots = self.text_decoder['block_coupler'](slots.flatten(end_dim=1))
        slots = slots.reshape(B, self.num_slots * self.num_blocks, -1)
        
        # 최종 출력 생성
        return slots.mean(dim=1)  # B, d_model
