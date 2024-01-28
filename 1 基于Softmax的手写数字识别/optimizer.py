import numpy as np

class SGD(object):
    def __init__(self, model, learning_rate, momentum=0.0):
        """
        随机梯度下降SGD（Stochastic Gradient Descent）优化器的初始化函数。

        Args:
            model: 待优化的模型对象。
            learning_rate (float): 学习率，控制更新步长。
            momentum (float): 动量参数，控制之前梯度的权重，默认为0.0。
        """
        self.model = model
        self.learning_rate = learning_rate
        self.momentum = momentum

    def step(self):
        """
        执行一步更新，更新模型的权重。

        Notes:
            在训练过程中调用此方法来更新模型的权重。
            使用动量的权重更新方法。
        """

        layer = self.model
        if layer.trainable:
            if not hasattr(layer, 'diff_W'):
                layer.diff_W = 0.0
            if not hasattr(layer, 'diff_b'):
                layer.diff_b = 0.0

            # 使用动量更新权重 v=av'- ϵΔ(a：动量参数，ϵ：学习率)
            layer.diff_W = self.momentum * layer.diff_W - self.learning_rate * layer.grad_W
            layer.diff_b = self.momentum * layer.diff_b - self.learning_rate * layer.grad_b
            
            # 更新权重 θ=θ+v
            layer.W += layer.diff_W
            layer.b += layer.diff_b
