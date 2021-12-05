import gym
import torch
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from price_feature_extractor_cnn import PriceFeatureExtractorCNN


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}

        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == 'price':
                # We will just downsample one channel of the image by 4x4 and flatten.
                # Assume the image is single-channel (subspace.shape[0] == 0)
                extractors[key] = PriceFeatureExtractorCNN(subspace.shape[1], subspace.shape[2], subspace.shape[0], 3, 256)
            elif key == 'account':
                # Run through a simple MLP
                extractors[key] = nn.Sequential(
                    nn.Linear(subspace.shape[0], 32),
                    nn.GELU()
                )

        self.extractors = nn.ModuleDict(extractors)

        self.mlp = nn.Sequential(
            nn.LayerNorm(256 + 32),
            nn.Linear(256 + 32, features_dim),
            nn.GELU()
        )

        # Update the features dim manually
        self._features_dim = 256

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            extracted = extractor(observations[key])
            encoded_tensor_list.append(extracted)
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        features = torch.cat(encoded_tensor_list, dim=1)
        return self.mlp(features)