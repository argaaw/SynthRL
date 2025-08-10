import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianKernelConv(nn.Module):
    def __init__(self, k: int = 5, sigma: float = 0.02):
        super().__init__()
        self.k = k
        self.sigma = sigma
        self.cache = dict()

    def kernel(self, n):
        if n not in self.cache:
            self.cache[n] = torch.exp(-(torch.arange(-self.k, self.k + 1) / n / self.sigma) ** 2 / 2)
            self.cache[n] /= self.cache[n].sum()

        return self.cache[n]
    
    def forward(self, target):
        kernel = self.kernel(target.shape[-1]).unsqueeze(0).unsqueeze(1).type_as(target)
        
        with torch.no_grad():
            weights = F.conv1d(target.unsqueeze(1), kernel, padding='same')
            weights = F.normalize(weights.squeeze(1), p=1, dim=-1)
        
        return weights
    