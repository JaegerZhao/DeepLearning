import numpy as np
from dataloader import build_dataloader
from optimizer import SGD
from loss import SoftmaxCrossEntropyLoss
from visualize import plot_loss_and_acc

class Solver(object):
    '''
        这是一个 基于Softmax的手写数字识别模型 训练和评估的 Solver 类。

        该类封装了模型训练的主要功能，包括构建数据加载器、模型、优化器，以及执行训练、验证和测试等操作。

        Attributes:
            cfg (dict): 包含配置信息的字典，指导 Solver 的行为。
            train_loader (Dataloader): 训练数据加载器。
            val_loader (Dataloader): 验证数据加载器。
            test_loader (Dataloader): 测试数据加载器。
            model (SoftmaxCrossEntropyLoss): 评估模型，基于 Softmax 和 CrossEntropyLoss。
            optimizer (SGD): 优化器，使用随机梯度下降进行模型参数更新。

        Methods:
            build_loader(cfg): 构建训练、验证和测试数据加载器的静态方法。
            build_optimizer(model, cfg): 构建优化器的静态方法。
            train(): 执行模型的训练过程。
            validate(): 在验证集上评估模型性能。
            test(): 在测试集上评估模型性能。
    '''
    def __init__(self, cfg):
        """
        构造函数，初始化 Solver 类的实例。

        Args:
            cfg (dict): 包含配置信息的字典，指导 Solver 的行为。
        """
        self.cfg = cfg

        # 构建数据加载器
        train_loader, val_loader, test_loader = self.build_loader(cfg)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # 构建评估模型
        self.model = SoftmaxCrossEntropyLoss(784, 10)

        # 构建优化器
        self.optimizer = self.build_optimizer(self.model, cfg)

    @staticmethod
    def build_loader(cfg):
        """
        构建训练、验证和测试数据加载器的静态方法。

        Args:
            cfg (dict): 包含配置信息的字典，指导数据加载器的构建。

        Returns:
            train_loader (Dataloader): 训练数据加载器。
            val_loader (Dataloader): 验证数据加载器。
            test_loader (Dataloader): 测试数据加载器。
        """
        # 训练数据加载器
        train_loader = build_dataloader(
            cfg['data_root'], cfg['max_epoch'], cfg['batch_size'], shuffle=True, mode='train')

        # 验证数据加载器
        val_loader = build_dataloader(
            cfg['data_root'], 1, cfg['batch_size'], shuffle=False, mode='val')

        # 测试数据加载器
        test_loader = build_dataloader(
            cfg['data_root'], 1, cfg['batch_size'], shuffle=False, mode='test')

        return train_loader, val_loader, test_loader

    @staticmethod
    def build_optimizer(model, cfg):
        """
        构建优化器的静态方法。

        Args:
            model (SoftmaxCrossEntropyLoss): 要优化的模型。
            cfg (dict): 包含配置信息的字典，指导优化器的构建。

        Returns:
            optimizer (SGD): 构建好的优化器。
        """
        return SGD(model, cfg['learning_rate'], cfg['momentum'])

    def train(self):
        """
        执行模型的训练过程。

        Returns:
            epoch_train_loss (list): 每个 epoch 的平均训练损失。
            epoch_train_acc (list): 每个 epoch 的平均训练准确率。
        """
        # 训练轮次
        max_epoch = self.cfg['max_epoch']

        # 每轮训练的 损失loss，准确率acc
        epoch_train_loss, epoch_train_acc = [], []
        for epoch in range(max_epoch):

            iteration_train_loss, iteration_train_acc = [], []
            for iteration, (images, labels) in enumerate(self.train_loader):
                # 前向传播
                loss, acc = self.model.forward(images, labels)

                # 计算梯度
                self.model.gradient_computing(images, labels)

                # 更新模型权重
                self.optimizer.step()

                # 保存损失和准确率
                iteration_train_loss.append(loss)
                iteration_train_acc.append(acc)

                # 显示迭代训练信息
                if iteration % self.cfg['display_freq'] == 0:
                    print("Epoch [{}][{}]\t Batch [{}][{}]\t Training Loss {:.4f}\t Accuracy {:.4f}".format(
                        epoch, max_epoch, iteration, len(self.train_loader), loss, acc))

            avg_train_loss, avg_train_acc = np.mean(iteration_train_loss), np.mean(iteration_train_acc)
            epoch_train_loss.append(avg_train_loss)
            epoch_train_acc.append(avg_train_acc)

            # 验证
            avg_val_loss, avg_val_acc = self.validate()

            # 显示每个 epoch 的训练信息
            print('\nEpoch [{}]\t Average training loss {:.4f}\t Average training accuracy {:.4f}'.format(
                epoch, avg_train_loss, avg_train_acc))

            # 显示每个 epoch 的验证信息
            print('Epoch [{}]\t Average validation loss {:.4f}\t Average validation accuracy {:.4f}\n'.format(
                epoch, avg_val_loss, avg_val_acc))

        return epoch_train_loss, epoch_train_acc

    def validate(self):
        """
        在验证集上评估模型性能。

        Returns:
            avg_val_loss (float): 验证集上的平均损失。
            avg_val_acc (float): 验证集上的平均准确率。
        """
        loss_set, acc_set = [], []
        for images, labels in self.val_loader:
            # 验证集上的前向传播
            loss, acc = self.model.forward(images, labels)
            loss_set.append(loss)
            acc_set.append(acc)

        loss = np.mean(loss_set)
        acc = np.mean(acc_set)
        return loss, acc

    def test(self):
        """
        在测试集上评估模型性能。

        Returns:
            avg_test_loss (float): 测试集上的平均损失。
            avg_test_acc (float): 测试集上的平均准确率。
        """
        loss_set, acc_set = [], []
        for images, labels in self.test_loader:
            # 测试集上的前向传播
            loss, acc = self.model.forward(images, labels)
            loss_set.append(loss)
            acc_set.append(acc)

        loss = np.mean(loss_set)
        acc = np.mean(acc_set)
        return loss, acc

