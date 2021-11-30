import torch
import torch.nn as nn
from torch.nn.modules.activation import SiLU

class PriceFeatureExtractor(nn.Module):
    def __init__(self, coins, dimensions, history, kernel_width, out_shape):
        super(PriceFeatureExtractor, self).__init__()

        self.cnn_extractor = nn.Sequential(
            nn.Conv2d(coins, 64, (dimensions, kernel_width), padding=(0, 1)),
            nn.SiLU(),
            nn.Conv2d(64, 32, (1, kernel_width), padding=(0, 1)),
            nn.SiLU()
        )

        dummy = torch.rand((1, coins, dimensions, history))
        dummy_out = self.cnn_extractor(dummy)
        dummy_out = dummy_out.reshape(1, -1)
        dummy_out_shape = dummy_out.shape[1]

        self.mlp_extractor = nn.Sequential(
            nn.Linear(dummy_out_shape, out_shape),
            nn.SiLU(),
        )

    def forward(self, price):
        price = price.permute(0, 2, 3, 1)
        cnn_out = self.cnn_extractor(price)
        return self.mlp_extractor(cnn_out.view(cnn_out.shape[0], -1))
