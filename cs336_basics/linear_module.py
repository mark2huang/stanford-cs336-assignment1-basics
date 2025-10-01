import torch
import torch.nn as nn

class LinearModule(nn.Module):
    def __init__(self, d_in, d_out):
        super(LinearModule, self).__init__()
        self.weights = nn.Parameter(torch.randn(d_out, d_in))  # 初始化权重
        self.bias = nn.Parameter(torch.zeros(d_out))  # 初始化偏置

    def forward(self, x):
        # 线性变换：y = xW^T + b
        return torch.matmul(x, self.weights.T) + self.bias
    
