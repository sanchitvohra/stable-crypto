import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat

import math

class PriceFeatureExtractorATTN(nn.Module):
    def __init__(self, history, input_dim, hidden_dim, output_dim, n_head = 8, ff_dim = 2048, dropout=0.1):
        super(PriceFeatureExtractorATTN, self).__init__()

        # shape : [B, H, D]

        self.linear = nn.Linear(input_dim, hidden_dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, history + 1, hidden_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.dropout = nn.Dropout(0.1)

        self.transformer = nn.TransformerEncoderLayer(hidden_dim, n_head, ff_dim, dropout, batch_first=True)

        self.mlp_extractor = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, price):
        price = price.view(price.shape[0], price.shape[1], -1)

        # price : (B, N, D)

        embedding = self.linear(price)

        # embeddings : (B, N, H)

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = embedding.shape[0])

        x = torch.cat((cls_tokens, embedding), dim=1)

        pos_embedding = self.pos_embedding[:, :(embedding.shape[1] + 1)]

        x += pos_embedding
        x = self.dropout(x)

        # x : (B, N+1, H)

        x = self.transformer(x)

        x = x[:, 0] # take class tokens only

        return self.mlp_extractor(x)
