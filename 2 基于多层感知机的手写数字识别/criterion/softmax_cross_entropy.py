""" Softmax交叉熵损失层 """

import numpy as np

# 为了防止分母为零，必要时可在分母加上一个极小项EPS
EPS = 1e-11

class SoftmaxCrossEntropyLossLayer():
	def __init__(self):
		self.acc = 0.
		self.loss = np.zeros(1, dtype='f')

	def forward(self, logit, gt):
		"""
	      输入: (minibatch)
	      - logit: 最后一个全连接层的输出结果, 尺寸(batch_size, 10)
	      - gt: 真实标签, 尺寸(batch_size, 10)
	    """
		# 计算softmax激活函数
		exp_logit = np.exp(logit - np.max(logit, axis=1, keepdims=True))
		self.softmax_output = exp_logit / np.sum(exp_logit, axis=1, keepdims=True)

		# 计算交叉熵损失
		self.loss = -np.sum(gt * np.log(self.softmax_output + EPS)) / logit.shape[0]

		# 计算平均准确率
		self.acc = np.sum(np.argmax(logit, axis=1) == np.argmax(gt, axis=1)) / logit.shape[0]

		# 保存真实标签，用于反向传播
		self.gt = gt
		return self.loss
	
	def backward(self):
		# 计算梯度
		return np.subtract(self.softmax_output, self.gt) / self.gt.shape[0]
