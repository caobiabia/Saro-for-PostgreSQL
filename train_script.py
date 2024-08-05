import os
import numpy as np
import torch.optim as optim
from utils import pre_evaluate_process, obtain_speedup, individual_query_performance, show_per_query_speedup, \
    split_TPCH, split_JOB
from datetime import datetime
from src.TreeConvolution.util import prepare_trees
from src.argument import args
from src.TreeConvolution.tcnn import BinaryTreeConv, TreeLayerNorm
from src.TreeConvolution.tcnn import TreeActivation, DynamicPooling
import logging
import torch.nn as nn
import torch
import torch.nn.functional as F
from src.modules.TreeCBAM import TreeCBAM


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
        self.__cuda = False

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
        self.layer1 = nn.Linear(64, 32)
        self.activation = nn.LeakyReLU()
        self.layer2 = nn.Linear(32, 1)

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
        latent = self.layer1(plan_emb)
        latent = self.activation(latent)
        score = self.layer2(latent)
        return score


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


def fit(copy_train, train_trees, train_latency_fit, train_latency_evaluate, test_trees, test_latency, epochs=None,
        lr=0.01, eval_interval=None):
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SATCNN_Extend(9).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_latency = train_latency_fit.to(device)
    for epoch in range(epochs):
        optimizer.zero_grad()
        scores = model(train_trees).to(device)
        scores = torch.reshape(scores, (-1, 49))
        loss = plackett_luce_loss_linear(train_latency, scores)
        loss.backward()
        optimizer.step()

        if epoch % eval_interval == 0:
            # train_accuracy = evaluate_accuracy(model, train_trees, train_latency)
            # test_accuracy = evaluate_accuracy(model, test_trees, test_latency)
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')
            logging.info(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

            # evaluate_model_exp(model, args)
            evaluate_model_repeat_test(model, copy_train, train_latency_evaluate, test_trees, test_latency, args)

        # Save the model every 50 epochs
        if (epoch + 1) % 20 == 0:
            current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"{model.__class__.__name__}_{current_datetime}_{epoch + 1}.pt"

            # 确保保存路径存在
            save_path = r'E:\COOOL_main\outputs'
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            full_model_path = os.path.join(save_path, model_filename)
            torch.save(model.state_dict(), full_model_path)
            print(f"Saved model as {full_model_path}")


def main():
    # 创建日志目录
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)

    # 获取当前日期和时间
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    log_file = os.path.join(log_dir, f'train_{current_time}.log')

    # 设置日志配置
    logging.basicConfig(filename=log_file,
                        format='%(message)s',
                        level=logging.INFO)

    # 检查GPU是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_plan_JOB, train_latency_JOB, test_plan_JOB, test_latency_JOB, validation_plan_JOB, validation_latency_JOB = split_JOB()
    (train_plan_TPCH, train_latency_TPCH, test_plan_TPCH, test_latency_TPCH, validation_plan_TPCH,
     validation_latency_TPCH) = split_TPCH()

    # 将相同类型的数据进行拼接
    train_plan = np.concatenate((train_plan_JOB, train_plan_TPCH), axis=0)
    train_latency = np.concatenate((train_latency_JOB, train_latency_TPCH), axis=0)
    test_plan = np.concatenate((test_plan_JOB, test_plan_TPCH), axis=0)
    test_latency = np.concatenate((test_latency_JOB, test_latency_TPCH), axis=0)
    validation_plan = np.concatenate((validation_plan_JOB, validation_plan_TPCH), axis=0)
    validation_latency = np.concatenate((validation_latency_JOB, validation_latency_TPCH), axis=0)

    train_plan_copy = train_plan
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # [#query, #hints]
    # print(f'load JOB succ')
    # print(train_qid[0], '\n', train_plan[0], '\n', train_latency[0])
    # print(train_qid.shape, '\n', train_plan.shape, '\n', train_latency.shape)
    # print(train_latency.shape, '\n', train_latency[0])
    train_latency = torch.tensor(train_latency)
    train_latency_log = torch.log(train_latency)
    # negated_train_latency = train_latency
    # test_latency = torch.tensor(test_latency)
    # negated_test_latency = test_latency
    # print(train_plan[0], '\n', train_latency[0])
    train_plan_vec = pre_evaluate_process(train_plan)
    # train_plan_vec = split_list(train_plan_vec, 49)
    # print(type(train_plan_vec))
    # print(train_plan_vec[0][0])
    # print(len(train_plan_vec), '\n', train_plan_vec[0], '\n', train_latency[0])
    # *****************************************************************************
    # n = len(train_plan_vec)
    # m = len(train_plan_vec[0]) if n > 0 else 0  # 假设所有子列表长度相同
    #
    # # 初始化一个形状为 (n, 49) 的空列表
    # train_trees = [[None for _ in range(m)] for _ in range(n)]

    # 填充 train_trees 列表
    # for i in range(n):
    # train_trees = util.prepare_trees(train_plan_vec, features, left_child, right_child, cuda=True)
    # print(type(train_trees[0]))

    # *****************************************************************************
    # print(len(train_trees[0]))  # (2, 3381)
    # trans_trees = torch.reshape(train_trees[0], (-1, 49))
    # print(trans_trees[0])
    # test_plan_vec = pre_evaluate_process(test_plan)
    # test_trees = util.prepare_trees(test_plan_vec, features, left_child, right_child, cuda=True)
    # print(len(train_plan_vec))  # 78*49个训练用的物理查询计划
    # 初始定义每个树节点的维度是9
    # scores = model(train_plan_vec)  # 输出78*49个score
    # scores = torch.reshape(scores, (-1, 49))  # 捏成（78, 49）
    # sorted_, indices = torch.sort(scores, dim=-1, descending=True)
    # indices = torch.reshape(indices, (-1, 49))  #
    # index = indices[:, 0].cpu().numpy()
    # print(scores.shape, scores[0])
    # # print(indices[0])
    # print(train_latency.shape, train_latency[0])

    fit(train_plan_copy, train_plan_vec, train_latency_log, train_latency, test_plan, test_latency, epochs=200, lr=0.005
        , eval_interval=1)


def evaluate_model_v1(model, model_path, test_plan, test_latency, args):
    # 加载模型
    model.load_state_dict(torch.load(model_path))  # model load
    model.eval()

    test_plan = [test_plan]
    test_latency = [test_latency]
    # print(model_path[21:27])
    # 对每个测试计划进行评估
    for i in range(len(test_plan)):
        # 获取当前测试计划和其相应的延迟
        current_test_plan = test_plan[i]
        current_test_latency = test_latency[i]

        # 预处理测试计划
        test_plan_vec = pre_evaluate_process(current_test_plan)

        # 使用模型进行预测
        scores = model(test_plan_vec)
        scores = torch.reshape(scores, (-1, args.NUM_HINT_SET))  # [num_sql, 49]
        sorted_, indices = torch.sort(scores, dim=-1, descending=False)
        indices = torch.reshape(indices, (-1, args.NUM_HINT_SET))
        index = indices[:, 0].cpu().numpy()  # 最优计划的索引

        # 计算速度提升
        pg, model, test_speedup = obtain_speedup(current_test_latency, index)
        print(f'pg runtimes {pg}, model runtimes {model}, total query execution latency speedup {test_speedup} x')
        logging.info(f'total query execution latency speedup {test_speedup} x')

        # 计算每个查询的性能
        test_query_model_pg = individual_query_performance(current_test_latency, index)
        show_per_query_speedup(test_query_model_pg)


def evaluate_model_random_test(model, args):
    # Load the model (if needed)
    # model = torch.load(model_path)

    # Split the data (assuming you have a function called split_JOB())
    (train_plan, train_latency, test_plan, test_latency, validation_plan, validation_latency,
     train_qid, cv_qid, test_qid) = split_JOB()

    # Prepare the draws and train data
    test_plan = [test_plan]
    test_latency = [test_latency]
    train_plan = [train_plan]
    train_latency = [train_latency]

    # Evaluate both draws and train plans
    for i in range(len(test_plan)):
        # Get the current draws plan and its corresponding latency
        current_test_plan = test_plan[i]
        current_test_latency = test_latency[i]

        # Preprocess the draws plan
        test_plan_vec = pre_evaluate_process(current_test_plan)

        # Prepare the draws tree (if needed)

        # Use the model for draws prediction
        test_scores = model(test_plan_vec)
        test_scores = torch.reshape(test_scores, (-1, args.NUM_HINT_SET))
        test_sorted, test_indices = torch.sort(test_scores, dim=-1, descending=False)
        test_indices = torch.reshape(test_indices, (-1, args.NUM_HINT_SET))
        test_index = test_indices[:, 0].cpu().numpy()  # Index of the best plan

        # Calculate speedup for draws data
        test_speedup = obtain_speedup(current_test_latency, test_index)
        print(f'Total speedup (test): {test_speedup}x')
        logging.info(f'Total speedup (test): {test_speedup}x')

        # Calculate performance for each query in draws data
        test_query_model_pg = individual_query_performance(current_test_latency, test_index)
        show_per_query_speedup(test_query_model_pg)

    # Now evaluate the train plans
    for i in range(len(train_plan)):
        # Get the current train plan and its corresponding latency
        current_train_plan = train_plan[i]
        current_train_latency = train_latency[i]

        # Preprocess the train plan
        train_plan_vec = pre_evaluate_process(current_train_plan)

        # Prepare the train tree (if needed)

        # Use the model for train prediction
        train_scores = model(train_plan_vec)
        train_scores = torch.reshape(train_scores, (-1, args.NUM_HINT_SET))
        train_sorted, train_indices = torch.sort(train_scores, dim=-1, descending=False)
        train_indices = torch.reshape(train_indices, (-1, args.NUM_HINT_SET))
        train_index = train_indices[:, 0].cpu().numpy()  # Index of the best plan

        # Calculate speedup for train data
        pg_time, model_time, train_speedup = obtain_speedup(current_train_latency, train_index)
        print(f'Total speedup (train): {train_speedup}x')
        logging.info(f'Total speedup (train): {train_speedup}x')

        # Calculate performance for each query in train data
        train_query_model_pg = individual_query_performance(current_train_latency, train_index)
        show_per_query_speedup(train_query_model_pg)


def evaluate_model_repeat_test(model, train_plan, train_latency, test_plan, test_latency, args):
    # Load the model (if needed)
    # model = torch.load(model_path)

    # Split the data (assuming you have a function called split_JOB())
    # Prepare the draws and train data
    test_plan = [test_plan]
    test_latency = [test_latency]
    train_plan = [train_plan]
    train_latency = [train_latency]

    # Evaluate both draws and train plans
    for i in range(len(test_plan)):
        # Get the current draws plan and its corresponding latency
        current_test_plan = test_plan[i]
        current_test_latency = test_latency[i]

        # Preprocess the draws plan
        test_plan_vec = pre_evaluate_process(current_test_plan)

        # Prepare the draws tree (if needed)

        # Use the model for draws prediction
        test_scores = model(test_plan_vec)
        test_scores = torch.reshape(test_scores, (-1, args.NUM_HINT_SET))
        test_sorted, test_indices = torch.sort(test_scores, dim=-1, descending=False)
        test_indices = torch.reshape(test_indices, (-1, args.NUM_HINT_SET))
        test_index = test_indices[:, 0].cpu().numpy()  # Index of the best plan

        # Calculate speedup for draws data
        pg_time, model_time, test_speedup = obtain_speedup(current_test_latency, test_index)
        print(f'Total speedup (test): {test_speedup}x')
        logging.info(f'Total speedup (test): {test_speedup}x')

        # Calculate performance for each query in draws data
        test_query_model_pg = individual_query_performance(current_test_latency, test_index)
        show_per_query_speedup(test_query_model_pg)

    # Now evaluate the train plans
    for i in range(len(train_plan)):
        # Get the current train plan and its corresponding latency
        current_train_plan = train_plan[i]
        current_train_latency = train_latency[i]

        # Preprocess the train plan
        train_plan_vec = pre_evaluate_process(current_train_plan)

        # Prepare the train tree (if needed)

        # Use the model for train prediction
        train_scores = model(train_plan_vec)
        train_scores = torch.reshape(train_scores, (-1, args.NUM_HINT_SET))
        train_sorted, train_indices = torch.sort(train_scores, dim=-1, descending=False)
        train_indices = torch.reshape(train_indices, (-1, args.NUM_HINT_SET))
        train_index = train_indices[:, 0].cpu().numpy()  # Index of the best plan

        # Calculate speedup for train data
        pg_time, model_time, train_speedup = obtain_speedup(current_train_latency, train_index)
        print(f'Total speedup (train): {train_speedup}x')
        logging.info(f'Total speedup (train): {train_speedup}x')

        # Calculate performance for each query in train data
        # train_query_model_pg = individual_query_performance(current_train_latency, train_index)
        # show_per_query_speedup(train_query_model_pg)


def evaluate_accuracy(model, model_path, test_plan, test_latency, args):
    # 加载模型
    model.load_state_dict(torch.load(model_path))  # model load
    model.eval()

    test_plan = [test_plan]
    test_latency = [test_latency]
    # print(model_path[21:27])
    # 对每个测试计划进行评估
    for i in range(len(test_plan)):
        # 获取当前测试计划和其相应的延迟
        current_test_plan = test_plan[i]
        current_test_latency = test_latency[i]

        # 预处理测试计划
        test_plan_vec = pre_evaluate_process(current_test_plan)

        # 使用模型进行预测
        scores = model(test_plan_vec)
        scores = torch.reshape(scores, (-1, args.NUM_HINT_SET))
        pred_indices = torch.argmin(scores, dim=1).cpu().numpy()

        # print(pred_indices)
        max_latency_indices = torch.argmax(torch.tensor(current_test_latency), dim=1).cpu().numpy()
        # print(min_latency_indices)
        correct = np.sum(pred_indices == max_latency_indices)
        accuracy = correct.item() / len(current_test_latency)
        return accuracy


def evaluate_my_model(arguments):
    # train_plan, train_latency, test_plan, test_latency, validation_plan, validation_latency, train_qid, cv_qid,
    # test_qid = split_JOB() train_plan_TPCH, train_latency_TPCH, test_plan, test_latency, validation_plan_TPCH,
    # validation_latency_TPCH = split_TPCH()
    train_plan_JOB, train_latency_JOB, test_plan_JOB, test_latency_JOB, validation_plan_JOB, validation_latency_JOB = split_JOB()
    train_plan_TPCH, train_latency_TPCH, test_plan_TPCH, test_latency_TPCH, validation_plan_TPCH, validation_latency_TPCH = split_TPCH()
    # 数据拼接
    test_plan = np.concatenate((test_plan_JOB, test_plan_TPCH), axis=0)
    test_latency = np.concatenate((test_latency_JOB, test_latency_TPCH), axis=0)
    models_folder = r'E:\COOOL_main\outputs'
    model = SATCNN_Extend(9)
    # 获取文件夹下所有文件
    for root, dirs, files in os.walk(models_folder):
        i = 1
        for file in files:
            print(f"{i}:{file}")
            i += 1
            # 获取模型文件的完整路径
            model_path = os.path.join(root, file)
            # 调用 evaluate_model_v1 函数
            evaluate_model_v1(model, model_path, test_plan, test_latency, arguments)
            # acc = evaluate_accuracy(model, model_path, test_plan, test_latency, args)
            # print("acc rate:", acc)


if __name__ == '__main__':
    # main()  # 训练
    evaluate_my_model(args)  # 测试我的模型延迟,加速度,  （准确度(有bug)注释掉了）
