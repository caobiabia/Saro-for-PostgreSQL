from src.modules.TreeCBAM import TreeCBAM
from src.modules.model_tools import *
import torch.nn as nn
from src.TreeConvolution.tcnn import BinaryTreeConv, TreeLayerNorm
from src.TreeConvolution.tcnn import TreeActivation, DynamicPooling
from src.TreeConvolution.util import prepare_trees


def left_child(x):
    if len(x) != 3:
        return None
    return x[1]


def right_child(x):
    if len(x) != 3:
        return None
    return x[2]


def features(x):
    return x[0]


class TCNN(nn.Module):
    def __init__(self, in_channels):
        super(TCNN, self).__init__()
        self.__in_channels = in_channels
        self.__cuda = True

        self.tree_conv = nn.Sequential(
            BinaryTreeConv(self.__in_channels, 256),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(256, 128),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(128, 64),
            TreeLayerNorm(),
            DynamicPooling()
        )

    def in_channels(self):
        return self.__in_channels

    def forward(self, x):
        # 将树结构转成张量
        trees = prepare_trees(x, features, left_child, right_child, cuda=self.__cuda)

        # 确保在 GPU 上进行计算
        if self.__cuda:
            trees = [tree.to("cuda") for tree in trees]  # 将每个树张量转移到 CUDA 上

        plan_emb = self.tree_conv(trees)
        return plan_emb

    def cuda(self):
        self.__cuda = True
        self.tree_conv.to("cuda")  # 将整个 `tree_conv` 移动到 GPU 上
        return super().cuda()


class SATCNN_Extend(nn.Module):
    def __init__(self, in_channels):
        super(SATCNN_Extend, self).__init__()
        self.__in_channels = in_channels
        self.__cuda = True

        self.tree_conv = nn.Sequential(
            BinaryTreeConv(self.__in_channels, 512),
            TreeCBAM(512),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),

            BinaryTreeConv(512, 256),
            TreeCBAM(256),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),

            BinaryTreeConv(256, 128),
            TreeCBAM(128),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),

            BinaryTreeConv(128, 64),
            TreeCBAM(64),
            TreeLayerNorm(),
            DynamicPooling()
        )

    def in_channels(self):
        return self.__in_channels

    def forward(self, x):
        """
        x: trees
        output: scalar
        """
        # 将输入数据移动到与模型参数相同的设备上
        device = next(self.parameters()).device
        trees = prepare_trees(x, features, left_child, right_child, cuda=self.__cuda)
        trees = tuple(t.to(device) for t in trees)
        plan_emb = self.tree_conv(trees)
        return plan_emb


class MCDTCNN(nn.Module):
    def __init__(self, in_channels, dropout_prob=0.5):
        super(MCDTCNN, self).__init__()
        self.tcnn1 = TCNN(in_channels)
        self.tcnn2 = TCNN(in_channels)

        # 全连接层和激活层
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)

        self.leakyrelu = nn.LeakyReLU()
        self.silu = nn.SiLU()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # 定义 Dropout 层
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x1, x2, mc_dropout=False):
        x1 = self.leakyrelu(self.tcnn1(x1))
        x2 = self.leakyrelu(self.tcnn2(x2))

        # 根据 mc_dropout 控制是否应用 Dropout
        if mc_dropout:
            x1 = self.dropout(x1)
            x2 = self.dropout(x2)

        # 计算差异并得到最终得分
        diff = x1 - x2
        diff = self.leakyrelu(diff)
        diff = self.fc1(diff)
        diff = self.leakyrelu(diff)
        score = self.fc2(diff)

        return score
