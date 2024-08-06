import torch
import torch.nn.functional as F


def plackett_luce_loss(y_true, y_pred):
    """
    计算Plackett-Luce模型的损失。

    参数:
    y_true (Tensor): 真实值张量, 形状为 (n, 49)，其中n是样本数量，49是每个样本中计划的真实得分。
    y_pred (Tensor): 预测值张量, 形状为 (n, 49)，其中n是样本数量，49是每个样本中计划的预测得分。

    返回:
    Tensor: Plackett-Luce损失值。
    """
    # 对预测得分进行softmax操作以获得每个项目的概率
    y_pred_softmax = F.softmax(y_pred, dim=1)

    # 计算每个样本的损失
    total_loss = 0
    n = y_true.size(0)
    m = y_true.size(1)

    for i in range(n):
        true_scores = y_true[i]
        pred_probs = y_pred_softmax[i]

        # 按真实得分进行排序
        sorted_indices = torch.argsort(true_scores, descending=True)

        # 计算每个排序的对数似然
        sample_loss = 0
        remaining_probs = pred_probs.clone()
        mask = torch.ones_like(remaining_probs, dtype=bool)

        for j in range(m):
            idx = sorted_indices[j]

            if mask[idx] == 0:
                continue

            log_prob = torch.log(remaining_probs[idx])
            sample_loss += log_prob

            # 更新掩码和剩余概率
            mask[idx] = False
            remaining_probs = remaining_probs * mask.float()
            remaining_probs = remaining_probs / remaining_probs.sum()

        total_loss += sample_loss

    # 返回平均损失
    return -total_loss / n


def plackett_luce_loss_linear(y_true, y_pred):
    """
    计算带权重的Plackett-Luce模型损失。

    参数:
    y_true (Tensor): 真实值张量, 形状为 (n, 49)，其中n是样本数量，49是每个样本中计划的真实得分。
    y_pred (Tensor): 预测值张量, 形状为 (n, 49)，其中n是样本数量，49是每个样本中计划的预测得分。

    返回:
    Tensor: 带权重的Plackett-Luce损失值。
    """
    n, m = y_true.size()

    # 获取设备
    device = y_true.device

    # 对预测得分进行softmax操作以获得每个项目的概率
    y_pred_softmax = F.softmax(y_pred, dim=1)

    # 按真实得分进行排序
    sorted_indices = torch.argsort(y_true, descending=True, dim=1)

    # 计算排序后的对数似然
    log_probs = torch.gather(torch.log(y_pred_softmax), 1, sorted_indices)

    # 创建权重向量，前面的排名权重大，后面的排名权重小，并移动到相同的设备
    weights = torch.linspace(1.0, 0.1, steps=m, device=device)

    # 应用权重到对数概率
    weighted_log_probs = log_probs * weights

    # 计算累积和
    cumulative_sums = torch.cumsum(weighted_log_probs, dim=1)

    # 计算损失
    losses = torch.sum(cumulative_sums, dim=1)

    # 返回平均损失
    return -torch.mean(losses)


def plackett_luce_loss_exp(y_true, y_pred, alpha=0.1):
    """
    计算带权重的Plackett-Luce模型损失。

    参数:
    y_true (Tensor): 真实值张量, 形状为 (n, 49)，其中n是样本数量，49是每个样本中计划的真实得分。
    y_pred (Tensor): 预测值张量, 形状为 (n, 49)，其中n是样本数量，49是每个样本中计划的预测得分。
    alpha (float): 控制指数递减速度的参数。

    返回:
    Tensor: 带权重的Plackett-Luce损失值。
    """
    n, m = y_true.size()

    # 获取设备
    device = y_true.device

    # 对预测得分进行softmax操作以获得每个项目的概率
    y_pred_softmax = F.softmax(y_pred, dim=1)

    # 按真实得分进行排序
    sorted_indices = torch.argsort(y_true, descending=True, dim=1)

    # 计算排序后的对数似然
    log_probs = torch.gather(torch.log(y_pred_softmax), 1, sorted_indices)

    # 创建指数递减权重向量，并移动到相同的设备
    weights = torch.exp(-alpha * torch.arange(m, device=device).float())

    # 应用权重到对数概率
    weighted_log_probs = log_probs * weights

    # 计算累积和
    cumulative_sums = torch.cumsum(weighted_log_probs, dim=1)

    # 计算损失
    losses = torch.sum(cumulative_sums, dim=1)

    # 返回平均损失
    return -torch.mean(losses)
