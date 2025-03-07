from utils import *


class dVAE(nn.Module):
    
    def __init__(self, vocab_size, img_channels):
        super().__init__()
        
        self.encoder = nn.Sequential(
            Conv2dBlock(img_channels, 64, 4, 4),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            conv2d(64, vocab_size, 1)
        )
        
        self.decoder = nn.Sequential(
            Conv2dBlock(vocab_size, 64, 1),
            Conv2dBlock(64, 64, 3, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64 * 2 * 2, 1),
            nn.PixelShuffle(2),
            Conv2dBlock(64, 64, 3, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64 * 2 * 2, 1),
            nn.PixelShuffle(2),
            conv2d(64, img_channels, 1),
        )

class dVAE_text(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        
        # 인코더: 텍스트 임베딩을 이산화된 토큰으로 변환
        self.encoder = nn.Sequential(
            linear(embedding_dim, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.ReLU(),
            linear(embedding_dim * 2, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.ReLU(),
            linear(embedding_dim * 2, vocab_size)
        )
        
        # 디코더: 이산화된 토큰을 다시 텍스트 임베딩으로 변환
        self.decoder = nn.Sequential(
            linear(vocab_size, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.ReLU(),
            linear(embedding_dim * 2, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.ReLU(),
            linear(embedding_dim * 2, embedding_dim)
        )
        
    def forward(self, x):
        """
        x: B, D (텍스트 임베딩)
        return: B, D (재구성된 텍스트 임베딩)
        """
        # 인코딩
        z = self.encoder(x)  # B, vocab_size
        
        # 디코딩
        recon = self.decoder(z)  # B, D
        
        return recon
