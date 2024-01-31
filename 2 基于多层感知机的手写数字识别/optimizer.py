""" SGD优化器 """

import numpy as np

class SGD():
	def __init__(self, learningRate, weightDecay):
		self.learningRate = learningRate
		self.weightDecay = weightDecay

	# 一步反向传播，逐层更新参数
	def step(self, model):
		layers = model.layerList
		for layer in layers:
			if layer.trainable:
				# 计算梯度更新的变化量
				layer.diff_W = - self.learningRate * (layer.grad_W + self.weightDecay * layer.W)
				layer.diff_b = - self.learningRate * layer.grad_b

				# 更新权重和偏置项
				layer.W += layer.diff_W
				layer.b += layer.diff_b
