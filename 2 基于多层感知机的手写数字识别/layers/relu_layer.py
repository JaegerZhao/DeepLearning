""" ReLU激活层 """

import numpy as np

class ReLULayer():
	def __init__(self):
		"""
		ReLU激活函数: relu(x) = max(x, 0)
		"""
		self.trainable = False # 没有可训练的参数
		self.input = None

	def forward(self, Input):
		"""
		对输入应用ReLU激活函数并返回结果
		"""
		self.input = Input
		return np.maximum(0, Input)


	def backward(self, delta):
		"""
		根据delta计算梯度
		"""
		return delta * (self.input > 0)