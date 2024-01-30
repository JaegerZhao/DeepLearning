""" 全连接层 """

import numpy as np

class FCLayer():
	def __init__(self, num_input, num_output, actFunction='relu', trainable=True):
		"""
		对输入进行线性变换: y = Wx + b
		参数简介:
			num_input: 输入大小
			num_output: 输出大小
			actFunction: 激活函数类型(无需修改)
			trainable: 是否具有可训练的参数
		"""
		self.num_input = num_input
		self.num_output = num_output
		self.trainable = trainable
		self.actFunction = actFunction
		assert actFunction in ['relu', 'sigmoid']

		self.XavierInit()

		self.grad_W = np.zeros((num_input, num_output))
		self.grad_b = np.zeros((1, num_output))


	def forward(self, Input):

		############################################################################
	    # TODO: 
		# 对输入计算Wx+b并返回结果.


	    ############################################################################


	def backward(self, delta):
		# 输入的delta由下一层计算得到
		############################################################################
	    # TODO: 
		# 根据delta计算梯度


	    ############################################################################


	def XavierInit(self):
		# 初始化，无需了解.
		raw_std = (2 / (self.num_input + self.num_output))**0.5
		if 'relu' == self.actFunction:
			init_std = raw_std * (2**0.5)
		elif 'sigmoid' == self.actFunction:
			init_std = raw_std
		else:
			init_std = raw_std # * 4

		self.W = np.random.normal(0, init_std, (self.num_input, self.num_output))
		self.b = np.random.normal(0, init_std, (1, self.num_output))
