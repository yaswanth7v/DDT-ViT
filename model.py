import torch
import torch.nn as nn
import numpy as np
from torch import nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, emb_dim, patch_size, num_patches, dropout):
        super().__init__()
        self.patcher = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=emb_dim,
                kernel_size=patch_size,
                stride=patch_size
            ),
            nn.Flatten(2)
        )
        
        self.cls_token = nn.Parameter(torch.randn(size=(1, 1, emb_dim)), requires_grad=True)  # Adjusted cls_token size
        self.position_embeddings = nn.Parameter(torch.randn(size=(1, num_patches + 1, emb_dim)), requires_grad=True)  # Adjusted position_embeddings size
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # Expand cls_token across batch dimension
        x = self.patcher(x)
        x = x.permute(0, 2, 1)  # Permute dimensions for concatenation
        x = torch.cat([cls_token, x], dim=1)  # Concatenate cls_token with patch embeddings
        
        # Ensure position_embeddings matches the shape of concatenated tensor x
        if x.shape[1] != self.position_embeddings.shape[1]:
            self.position_embeddings = nn.Parameter(self.position_embeddings.data[:, :x.shape[1], :])
        
        x += self.position_embeddings  # Add positional embeddings
        x = self.dropout(x)
        return x
    
class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable parameter

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
         # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class ResidualConnection(nn.Module):
    
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, dropout: nn.Dropout):
        d_k = query.shape[-1]
        
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v):
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, self.dropout)
        
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
 
        return self.w_o(x)

class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
    
class ViT(nn.Module):
    def __init__(self, input_channels, embed_dim, patch_size, num_patches, num_heads, num_encoders, num_classes, activation, dropout, hidden_dim):
        super().__init__()
        
        self.embeddings_block = PatchEmbedding(input_channels, embed_dim, patch_size, num_patches, dropout)
        
        encoder_blocks = []
        for _ in range(num_encoders):
            encoder_self_attention_block = MultiHeadAttentionBlock(embed_dim, num_heads, dropout)
            feed_forward_block = FeedForwardBlock(embed_dim, hidden_dim, dropout)
            encoder_block = EncoderBlock(embed_dim, encoder_self_attention_block, feed_forward_block, dropout)
            encoder_blocks.append(encoder_block)
        
        self.encoder = Encoder(embed_dim, nn.ModuleList(encoder_blocks))
        #------------------------- using inbuilt encoder functions --------------------------------
        # encoder_layer = nn.TransformerEncoderLayer(
        # d_model=embed_dim, 
        # nhead=num_heads, 
        # dropout=dropout, 
        # activation=activation, 
        # batch_first=True, 
        # norm_first=True
        # )
        #self.encoder_blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_encoders)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim),
            nn.Linear(in_features=embed_dim, out_features=num_classes)
        )
        
    def forward(self, x):
        x = self.embeddings_block(x)
        x = self.encoder(x)
        x = self.mlp_head(x[:, 0, :])
        return x
    