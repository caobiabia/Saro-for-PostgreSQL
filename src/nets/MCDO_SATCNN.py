from src.TreeConvolution.util import prepare_trees
from src.TreeConvolution.tcnn import BinaryTreeConv, TreeLayerNorm
from src.TreeConvolution.tcnn import TreeActivation, DynamicPooling
import torch.nn as nn
from src.modules.TreeCBAM import TreeCBAM


class SATCNN_Extend(nn.Module):
    def __init__(self, in_channels, dropout_rate=0.2):
        super(SATCNN_Extend, self).__init__()
        self.__in_channels = in_channels
        self.__cuda = False
        self.dropout_rate = dropout_rate

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

            BinaryTreeConv(64, 32),
            TreeCBAM(32),
            TreeLayerNorm(),
            DynamicPooling()
        )

        # 加入 Dropout 层，确保在每次前向传播中随机丢弃一些神经元
        self.dropout = nn.Dropout(p=self.dropout_rate)

        # 定义全连接层
        self.fc = nn.Sequential(
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1)
        )

    def in_channels(self):
        return self.__in_channels

    def forward(self, x):
        """
        x: trees
        output: scalar
        """
        device = next(self.parameters()).device
        trees = prepare_trees(x, features, left_child, right_child, cuda=self.__cuda)
        trees = tuple(t.to(device) for t in trees)

        # 通过 tree_conv 获得 plan_emb 表示
        plan_emb = self.tree_conv(trees)

        # 应用 Dropout
        plan_emb = self.dropout(plan_emb)

        # 将 Dropout 后的嵌入传递给全连接层
        score = self.fc(plan_emb)
        return score


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
