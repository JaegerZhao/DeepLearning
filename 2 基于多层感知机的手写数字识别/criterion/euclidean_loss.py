""" 欧式距离损失层 """

import numpy as np

class EuclideanLossLayer():
	def __init__(self):
		self.acc = 0.
		self.loss = 0.

	def forward(self, logit, gt):
		"""
	      输入: (minibatch)
	      - logit: 最后一个全连接层的输出结果, 尺寸(batch_size, 10)
	      - gt: 真实标签, 尺寸(batch_size, 10)
	    """
		# 计算欧式距离损失
		self.logit = logit
		self.diff = logit - gt
		self.loss = 0.5 * np.sum(self.diff ** 2) / logit.shape[0]  # 计算平均损失
		self.acc = np.sum(np.argmax(logit, axis=1) == np.argmax(gt, axis=1)) / logit.shape[0]  # 计算平均准确率

		return self.loss

	def backward(self):
		# 欧式距离损失的梯度即为(logit - gt) / batch_size
		return self.diff / self.logit.shape[0]
		