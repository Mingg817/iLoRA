import torch
from torch import nn
import torch.nn.functional as F

class Dense(nn.Module):
    def __init__(self, dim: int, num_moe: int) -> None:
        super().__init__()
        self.dim = 64
        self.num_moe = num_moe
        self.linear_layer = nn.Linear(self.dim, num_moe, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        logits = self.linear_layer(x)
        probs = self.softmax(logits)
        return probs

class topK(nn.Module):
    def __init__(self, dim: int, num_moe: int) -> None:
        super().__init__()
        self.dim = 64
        self.num_moe = num_moe
        self.linear_layer = nn.Linear(self.dim, num_moe, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, topk=1):
        logits = self.linear_layer(x)
        probs = self.softmax(logits)
        # 使用topk来选择最高的k个概率
        topk_values, topk_indices = torch.topk(probs, k=topk, dim=-1)
        # 创建一个初始值全为负无穷的张量，形状与probs相同
        topk_probs = torch.full_like(probs, float('-inf'))
        # 使用scatter填充topk的概率值
        topk_probs = topk_probs.scatter_(-1, topk_indices, topk_values)
        # 应用softmax确保top k值的和为1
        topk_probs = self.softmax(topk_probs)
        return topk_probs

class MLP(nn.Module):
    def __init__(self, dim: int, num_moe: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.dim = 64
        self.num_moe = num_moe
        # 添加多层感知机结构
        self.linear_layer1 = nn.Linear(self.dim, hidden_dim)
        self.activation = nn.GELU()  # 使用GELU激活函数
        self.linear_layer2 = nn.Linear(hidden_dim, self.num_moe)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.linear_layer1(x)
        x = self.activation(x)
        logits = self.linear_layer2(x)
        probs = self.softmax(logits)
        return probs
    
    
class Noise(nn.Module):
    def __init__(self, dim: int, num_moe: int, noise_std: float = 0.1) -> None:
        super().__init__()
        self.dim = 64
        self.num_moe = num_moe
        self.noise_std = noise_std
        self.linear_layer = nn.Linear(self.dim, num_moe, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        logits = self.linear_layer(x)
        
        # 添加噪声
        noise = torch.randn_like(logits) * self.noise_std
        logits = logits + noise
        
        probs = self.softmax(logits)
        return probs
    
class MLP_noise(nn.Module):
    def __init__(self, dim: int, num_moe: int, hidden_dim: int = 128, noise_std: float = 0.1) -> None:
        super().__init__()
        self.dim = 64
        self.num_moe = num_moe
        self.noise_std = noise_std
        self.linear1 = nn.Linear(self.dim, hidden_dim, bias=False)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, num_moe, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        hidden = self.linear1(x)
        hidden = self.relu(hidden)
        logits = self.linear2(hidden)
        
        # 添加噪声
        noise = torch.randn_like(logits) * self.noise_std
        logits = logits + noise
        
        probs = self.softmax(logits)
        return probs

    
class Drop(nn.Module):
    def __init__(self, dim: int, num_moe: int, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.dim = 64
        self.num_moe = num_moe
        self.linear_layer = nn.Linear(self.dim, num_moe, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        logits = self.linear_layer(x)
        # 添加Dropout
        logits = self.dropout(logits)
        probs = self.softmax(logits)
        return probs
    
GATING_TO_MODEL_MAPPING = {
    "Dense": Dense,
    "topK": topK,
    "MLP": MLP,
    "Drop": Drop,
    "MLP_noise": MLP_noise,
    "Noise": Noise,
}