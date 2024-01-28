import numpy as np

# 极小的数，用于防止除以零
EPS = 1e-11

class SoftmaxCrossEntropyLoss(object):

    def __init__(self, num_input, num_output, trainable=True):
        """
        对输入数据应用线性变换：y = Wx + b
        Args:
            num_input: 每个输入样本的大小
            num_output: 每个输出样本的大小
            trainable: 是否可训练的标志
        """

        self.num_input = num_input
        self.num_output = num_output
        self.trainable = trainable
        self.XavierInit()

    def forward(self, Input, labels):
        """
        前向传播函数，计算 Softmax 交叉熵损失与精确度
        
        Args：
            Input: 输入图像矩阵，形状为(batch_size, 784)
            labels: 手写数字标签，形状为 (batch_size, 10)
        """
        # 计算输出矩阵
        z = np.dot(Input, self.W) + self.b

        # 计算 softmax
        softmax_probs = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

        # 计算交叉熵损失
        batch_size = Input.shape[0]
        loss = -np.sum(np.log(softmax_probs[np.arange(batch_size), labels] + EPS)) / batch_size

        # 计算准确度
        predicted_labels = np.argmax(softmax_probs, axis=1)
        acc = np.mean(predicted_labels == labels)
        return loss, acc

    def gradient_computing(self, Input, labels):
        """
        梯度计算函数，计算 W 和 b 的梯度
        
        Args：
            Input: 输入图像矩阵，形状为(batch_size, 784)
            labels: 手写数字标签，形状为 (batch_size, 10)
        """
        # 计算输出矩阵
        z = np.dot(Input, self.W) + self.b
        
        # 计算 softmax
        softmax_probs = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

        # 计算梯度 Δ=a-y(a:预测向量，y：one-hot标签向量)
        softmax_grad = softmax_probs.copy()
        softmax_grad[np.arange(Input.shape[0]), labels] -= 1
        softmax_grad /= Input.shape[0]

        # W 和 b 的梯度
        self.grad_W = np.dot(Input.T, softmax_grad)
        self.grad_b = np.sum(softmax_grad, axis=0, keepdims=True)

    def XavierInit(self):
        """
        初始化权重
        """
        raw_std = (2 / (self.num_input + self.num_output))**0.5
        init_std = raw_std * (2**0.5)
        self.W = np.random.normal(0, init_std, (self.num_input, self.num_output))
        self.b = np.random.normal(0, init_std, (1, self.num_output))
