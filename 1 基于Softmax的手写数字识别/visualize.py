import matplotlib.pyplot as plt
import numpy as np

def plot_loss_and_acc(loss_and_acc_dict):
    # 可视化损失曲线
    plt.figure()

    min_loss, max_loss = 100.0, 0.0

    # 遍历字典中的每个键值对
    for key, (loss_list, acc_list) in loss_and_acc_dict.items():
        # 根据当前 loss_list 更新 min_loss 和 max_loss
        min_loss = min(loss_list) if min(loss_list) < min_loss else min_loss
        max_loss = max(loss_list) if max(loss_list) > max_loss else max_loss

        # 获取 epoch 数
        num_epoch = len(loss_list)

        # 使用方块 ('-s') 绘制损失曲线，并用键名作为标签
        plt.plot(range(1, 1 + num_epoch), loss_list, '-s', label=key)

    # 设置损失曲线的标签和图例
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.xticks(range(0, num_epoch + 1, 2))
    plt.axis([0, num_epoch + 1, min_loss - 0.1, max_loss + 0.1])
    plt.show()

    # 可视化准确度曲线
    plt.figure()

    min_acc, max_acc = 1.0, 0.0
    # 遍历字典中的每个键值对
    for key, (loss_list, acc_list) in loss_and_acc_dict.items():
        # 根据当前 acc_list 更新 min_acc 和 max_acc
        min_acc = min(acc_list) if min(acc_list) < min_acc else min_acc
        max_acc = max(acc_list) if max(acc_list) > max_acc else max_acc

        # 获取 epoch 数量
        num_epoch = len(acc_list)

        # 使用方块 ('-s') 绘制准确度曲线，并用键名作为标签
        plt.plot(range(1, 1 + num_epoch), acc_list, '-s', label=key)

    # 设置准确度曲线的标签和图例
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.xticks(range(0, num_epoch + 1, 2))
    plt.axis([0, num_epoch + 1, min_acc, 1.0])
    plt.show()
