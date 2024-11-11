import os
import pickle
import torch.optim as optim
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import copy

from src.Dataset import PlansDataset, custom_collate_fn
from src.featurizer import pre_evaluate_process
from src.nets.Sub_MCD_net import MCDTCNN


# 删除包含 'Plan Not Available' 的字典项
def remove_unavailable_plans(plans_dict):
    for key, plan_list in plans_dict.items():
        plans_dict[key] = [item for item in plan_list if item.get("plan") != "Plan Not Available"]


def calculate_accuracy(predictions, labels):
    predictions_sign = torch.sign(predictions)
    labels_sign = torch.sign(labels)
    correct = (predictions_sign == labels_sign).sum().item()
    return correct / len(labels)


def train_model(model, train_loader, val_loader, device, num_epochs=10, learning_rate=1e-4, mc_dropout=False,
                checkpoint_path=None):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    start_epoch = 0
    best_val_loss = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())

    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    # 加载 checkpoint（如果存在）
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        print("Checkpoint content:", checkpoint.keys())  # 输出 checkpoint 文件中的所有键
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            train_losses = checkpoint.get('train_losses', [])
            val_losses = checkpoint.get('val_losses', [])
            train_accuracies = checkpoint.get('train_accuracies', [])
            val_accuracies = checkpoint.get('val_accuracies', [])
            print(f"Checkpoint loaded. Resuming from epoch {start_epoch + 1}")
        except KeyError as e:
            print(f"Missing key in checkpoint file: {e}")
            print("Starting training from scratch due to incomplete checkpoint.")
            # 若关键键缺失，从头开始训练，或可选择抛出异常以提示
            start_epoch = 0

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        batch_count = 0

        for batch_idx, batch in enumerate(train_loader):
            plan1, plan2, label = batch
            label = label.view(-1, 1).to(device)
            v_plans1 = pre_evaluate_process(plan1)
            v_plans2 = pre_evaluate_process(plan2)

            optimizer.zero_grad()

            score = model(v_plans1, v_plans2, mc_dropout=mc_dropout).to(device)
            loss = criterion(score, label).to(device)
            running_loss += loss.item()
            batch_count += 1

            correct_predictions += calculate_accuracy(score, label) * len(label)

            loss.backward()
            optimizer.step()

        average_train_loss = running_loss / batch_count
        train_losses.append(average_train_loss)
        train_accuracy = correct_predictions / len(train_loader.dataset)
        train_accuracies.append(train_accuracy)
        print(f"Epoch {epoch + 1}/{num_epochs} Training Loss: {average_train_loss:.4f}, Accuracy: {train_accuracy:.4f}")

        model.eval()
        val_loss = 0.0
        correct_val_predictions = 0

        with torch.no_grad():
            for val_batch in val_loader:
                plan1, plan2, label = val_batch
                label = label.view(-1, 1).to(device)

                v_plans1 = pre_evaluate_process(plan1)
                v_plans2 = pre_evaluate_process(plan2)

                score = model(v_plans1, v_plans2, mc_dropout=mc_dropout).to(device)
                loss = criterion(score, label)
                val_loss += loss.item()

                correct_val_predictions += calculate_accuracy(score, label) * len(label)

        average_val_loss = val_loss / len(val_loader)
        val_losses.append(average_val_loss)
        val_accuracy = correct_val_predictions / len(val_loader.dataset)
        val_accuracies.append(val_accuracy)
        print(f"Epoch {epoch + 1}/{num_epochs} Validation Loss: {average_val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(best_model_weights, r'D:\Saro\outputs\MCD\MCD_TCNN.pth')
            print(f"Best model saved with validation loss: {best_val_loss:.4f}")
        print(f"Model is currently on device: {next(model.parameters()).device}")
        # 保存 checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
        }, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch + 1}")

    print("Training completed.")
    model.load_state_dict(best_model_weights)

    # 绘制图表
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Epoch')
    plt.legend()

    plt.show()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据
    with open(r"D:\Saro\records\plans_dict_train_STATS.pkl", "rb") as f:
        plans_dict_STATS = pickle.load(f)
    with open(r"D:\Saro\records\plans_dict_train_JOB.pkl", "rb") as f:
        plans_dict_JOB = pickle.load(f)

    # 过滤掉 'Plan Not Available' 的键值对
    remove_unavailable_plans(plans_dict_STATS)
    remove_unavailable_plans(plans_dict_JOB)

    # 合并两个字典
    plans_dict = plans_dict_STATS | plans_dict_JOB

    plans_dataset = PlansDataset(plans_dict)
    train_size = int(0.8 * len(plans_dataset))
    val_size = len(plans_dataset) - train_size
    train_dataset, val_dataset = random_split(plans_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=custom_collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=custom_collate_fn, num_workers=0)

    in_channels = 10
    model = MCDTCNN(in_channels=in_channels, dropout_prob=0.3).to(device)

    # 加载 checkpoint 或从头开始训练
    checkpoint_path = r'D:\Saro\outputs\MCD\MCD_TCNN.pth'
    train_model(model, train_loader, val_loader, device, num_epochs=1000, learning_rate=0.01, mc_dropout=False,
                checkpoint_path=checkpoint_path)
