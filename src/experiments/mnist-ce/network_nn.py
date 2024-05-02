import torch
import torch.nn as nn

class Net_nn(nn.Module):
    def __init__(self,N):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),  # 6 is the output chanel size; 5 is the kernal size; 1 (chanel) 28 28 -> 6 24 24
            nn.MaxPool2d(2, 2),  # kernal size 2; stride size 2; 6 24 24 -> 6 12 12
            nn.ReLU(True),       # inplace=True means that it will modify the input directly thus save memory
            nn.Conv2d(6, 16, 5), # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            nn.ReLU(True) 
        )
        self.classifier =  nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, N),
            nn.Softmax(1)
        )

    def forward(self, x, marg_idx=None, type=1):
        
        assert type == 1, "only posterior computations are available for this network"

        # If the list of the pixel numbers to be marginalised is given,
        # then genarate a marginalisation mask from it and apply to the
        # tensor 'x'
        if marg_idx:
            batch_size = x.shape[0]
            with torch.no_grad():
                marg_mask = torch.ones_like(x, device=x.device).reshape(batch_size, 1, -1)
                marg_mask[:, :, marg_idx] = 0
                marg_mask = marg_mask.reshape_as(x)
                marg_mask.requires_grad_(False)
            x = torch.einsum('ijkl,ijkl->ijkl', x, marg_mask)
        x = self.encoder(x)
        x = x.view(-1, 16 * 4 * 4)
        x = self.classifier(x)
        return x


class Net_nn_simple(nn.Module):
    def __init__(self,N):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),  # 6 is the output chanel size; 5 is the kernal size; 1 (chanel) 28 28 -> 6 24 24
            nn.MaxPool2d(2, 2),  # kernal size 2; stride size 2; 6 24 24 -> 6 12 12
            nn.ReLU(True),       # inplace=True means that it will modify the input directly thus save memory
            # nn.Conv2d(6, 16, 5), # 6 12 12 -> 16 8 8
            # nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            # nn.ReLU(True) 
        )
        self.classifier =  nn.Sequential(
            nn.Linear(6 * 12 * 12, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, N),
            nn.Softmax(1)
        )

    def forward(self, x, marg_idx=None, type=1):
        
        assert type == 1, "only posterior computations are available for this network"

        # If the list of the pixel numbers to be marginalised is given,
        # then genarate a marginalisation mask from it and apply to the
        # tensor 'x'
        if marg_idx:
            batch_size = x.shape[0]
            with torch.no_grad():
                marg_mask = torch.ones_like(x, device=x.device).reshape(batch_size, 1, -1)
                marg_mask[:, :, marg_idx] = 0
                marg_mask = marg_mask.reshape_as(x)
                marg_mask.requires_grad_(False)
            x = torch.einsum('ijkl,ijkl->ijkl', x, marg_mask)
        x = self.encoder(x)
        x = x.view(-1, 6 * 12 * 12)
        x = self.classifier(x)
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, marg_idx=None, type=1):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = F.softmax(out, dim=1)
        return out



def ResNet50(N):
    return ResNet(Bottleneck, [1, 4, 6, 3],N)

def test():
    net = ResNet50(4)
    y = net(torch.randn(1, 1, 28, 28))
    print(y,y.size())

test()