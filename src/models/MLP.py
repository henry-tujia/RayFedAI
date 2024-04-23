'''
Author: henry tanhao0606@outlook.com
Date: 2024-04-18 01:21:26
LastEditors: henry tanhao0606@outlook.com
LastEditTime: 2024-04-18 01:22:30
FilePath: /Multi-FL-Training-main/src/models/MLP.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# import fedplat as fp
import torch
import torch.nn.functional as F


class MLP(torch.nn.Module):
    def __init__(self, input_dim, num_classes, KD=False):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 1000)
        self.fc2 = torch.nn.Linear(1000, 2000)
        self.fc3 = torch.nn.Linear(2000, num_classes)
        self.KD = KD

    def forward(self, x):
        x: torch.tensor
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        logits = self.fc3(x)
        if self.KD:
            return x, logits
        return logits
