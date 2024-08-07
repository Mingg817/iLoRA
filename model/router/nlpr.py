import torch
import torch.nn as nn
import torch.nn.functional as F


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class ResidualBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, conv_layer, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv_layer(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = conv_layer(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            diff = planes - in_planes
            self.shortcut = LambdaLayer(
                lambda x: F.pad(x[:, :, ::2], (0, 0, int(diff * 0.5), int((diff + 1) * 0.5)), "constant", 0))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class GateFunction(nn.Module):
    def __init__(self, input_size, output_size):
        super(GateFunction, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return F.softmax(self.fc(x), dim=-1)


class NLPRecommendationRouter(nn.Module):
    def __init__(self, block, num_blocks, input_size=64, num_experts=4):
        super(NLPRecommendationRouter, self).__init__()
        self.in_planes = 16
        self.conv_layer = nn.Conv1d

        self.conv1 = nn.Conv1d(1, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_planes)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # Gate function
        self.gate = GateFunction(input_size, num_experts)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, self.conv_layer, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.gate(out)
        return out.unsqueeze(1)


def build_router(**kwargs):
    return NLPRecommendationRouter(ResidualBlock, [3, 3, 3], input_size=64, num_experts=4)