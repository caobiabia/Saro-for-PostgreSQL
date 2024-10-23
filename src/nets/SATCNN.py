from src.TreeConvolution.util import prepare_trees
from src.TreeConvolution.tcnn import BinaryTreeConv, TreeLayerNorm
from src.TreeConvolution.tcnn import TreeActivation, DynamicPooling
import torch.nn as nn
from src.modules.TreeCBAM import TreeCBAM


class SATCNN(nn.Module):
    def __init__(self, in_channels):
        super(SATCNN, self).__init__()
        self.__in_channels = in_channels
        self.__cuda = False

        self.tree_conv = nn.Sequential(
            BinaryTreeConv(self.__in_channels, 256),
            TreeCBAM(256),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),

            BinaryTreeConv(256, 128),
            TreeCBAM(128),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),

            BinaryTreeConv(128, 64),
            TreeCBAM(64),
            TreeLayerNorm(), DynamicPooling()
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )


class SATCNN_Extend(nn.Module):
    def __init__(self, in_channels):
        super(SATCNN_Extend, self).__init__()
        self.__in_channels = in_channels
        self.__cuda = False

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
        # 将输入数据移动到与模型参数相同的设备上
        device = next(self.parameters()).device
        trees = prepare_trees(x, features, left_child, right_child, cuda=self.__cuda)
        trees = tuple(t.to(device) for t in trees)
        plan_emb = self.tree_conv(trees)
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
