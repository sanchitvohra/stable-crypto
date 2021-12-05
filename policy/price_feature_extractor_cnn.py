import torch
import torch.nn as nn
from torch.nn.modules.activation import SiLU

class PriceFeatureExtractorCNN(nn.Module):
    def __init__(self, coins, dimensions, history, kernel_width, out_shape):
        super(PriceFeatureExtractorCNN, self).__init__()

        self.cnn_extractor = nn.Sequential(
            nn.Conv2d(coins, 128, (dimensions, kernel_width), padding=(0, 1)),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, (1, kernel_width), padding=(0, 1)),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )

        dummy = torch.rand((1, coins, dimensions, history))
        dummy_out = self.cnn_extractor(dummy)
        dummy_out = dummy_out.reshape(1, -1)
        dummy_out_shape = dummy_out.shape[1]

        self.mlp_extractor = nn.Sequential(
            nn.LayerNorm(dummy_out_shape),
            nn.Linear(dummy_out_shape, out_shape),
            nn.GELU(),
        )

    def forward(self, price):
        # price : (B, N, C, D)
        price = price.permute(0, 2, 3, 1)
        # price : (B, C, D, N)
        cnn_out = self.cnn_extractor(price)
        # cnn_out : (B, 64, D, N)
        return self.mlp_extractor(cnn_out.view(cnn_out.shape[0], -1))
