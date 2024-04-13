import torch
from torch import nn
import torchvision
import math


class ConvolutionalBlock(nn.Module):
    """
    一个卷积块，包括卷积层、批量归一化层和激活层。
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, batch_norm=False, activation=None):
        """
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param kernel_size: 卷积核大小
        :param stride: 步长
        :param batch_norm: 是否包含批量归一化层？
        :param activation: 激活函数类型；如果没有则为None
        """
        super(ConvolutionalBlock, self).__init__()

        if activation is not None:
            activation = activation.lower()
            assert activation in {'prelu', 'leakyrelu', 'tanh'}

        # 用于保存这个卷积块中的层的容器
        layers = list()

        # 一个卷积层
        layers.append(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2))

        # 如果需要，加入批量归一化（BN）层
        if batch_norm is True:
            layers.append(nn.BatchNorm2d(num_features=out_channels))

        # 如果需要，加入激活层
        if activation == 'prelu':
            layers.append(nn.PReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.2))
        elif activation == 'tanh':
            layers.append(nn.Tanh())

        # 将卷积块的层按顺序组合起来
        self.conv_block = nn.Sequential(*layers)

    def forward(self, input):
        """
        前向传播。

        :param input: 输入图像，尺寸为 (N, in_channels, w, h) 的张量
        :return: 输出图像，尺寸为 (N, out_channels, w, h) 的张量
        """
        output = self.conv_block(input)  # (N, out_channels, w, h)

        return output


class SubPixelConvolutionalBlock(nn.Module):
    """
    一个子像素卷积块，包括卷积层、像素重排层和PReLU激活层。
    """

    def __init__(self, kernel_size=3, n_channels=64, scaling_factor=2):
        """
        :param kernel_size: 卷积核大小
        :param n_channels: 输入和输出通道数
        :param scaling_factor: 沿两个维度缩放输入图像的因子
        """
        super(SubPixelConvolutionalBlock, self).__init__()

        # 一个卷积层，通过缩放因子的平方增加通道数，然后是像素重排和PReLU
        self.conv = nn.Conv2d(in_channels=n_channels, out_channels=n_channels * (scaling_factor ** 2),
                              kernel_size=kernel_size, padding=kernel_size // 2)
        # 这些额外的通道被重排以形成额外的像素，每个维度通过缩放因子进行上采样
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scaling_factor)
        self.prelu = nn.PReLU()

    def forward(self, input):
        """
        前向传播。

        :param input: 输入图像，尺寸为 (N, n_channels, w, h) 的张量
        :return: 缩放后的输出图像，尺寸为 (N, n_channels, w * 缩放因子, h * 缩放因子) 的张量
        """
        output = self.conv(input)  # (N, n_channels * 缩放因子的平方, w, h)
        output = self.pixel_shuffle(output)  # (N, n_channels, w * 缩放因子, h * 缩放因子)
        output = self.prelu(output)  # (N, n_channels, w * 缩放因子, h * 缩放因子)

        return output


class ResidualBlock(nn.Module):
    """
    一个残差块，包含两个卷积块并通过残差连接。
    """

    def __init__(self, kernel_size=3, n_channels=64):
        """
        :param kernel_size: 卷积核大小
        :param n_channels: 输入和输出通道数（相同，因为输入必须加到输出上）
        """
        super(ResidualBlock, self).__init__()

        # 第一个卷积块
        self.conv_block1 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                                              batch_norm=True, activation='PReLu')

        # 第二个卷积块
        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                                              batch_norm=True, activation=None)

    def forward(self, input):
        """
        前向传播。

        :param input: 输入图像，尺寸为 (N, n_channels, w, h) 的张量
        :return: 输出图像，尺寸为 (N, n_channels, w, h) 的张量
        """
        residual = input  # (N, n_channels, w, h)
        output = self.conv_block1(input)  # (N, n_channels, w, h)
        output = self.conv_block2(output)  # (N, n_channels, w, h)
        output = output + residual  # (N, n_channels, w, h)

        return output


class SRResNet(nn.Module):
    """
    SRResNet，如论文中定义的。
    """

    def __init__(self, large_kernel_size=9, small_kernel_size=3, n_channels=64, n_blocks=16, scaling_factor=4):
        """
        :param large_kernel_size: 输入和输出转换时的第一个和最后一个卷积的卷积核大小
        :param small_kernel_size: 所有中间卷积的卷积核大小，即残差块和小像素卷积块中的卷积
        :param n_channels: 中间通道数，即残差块和小像素卷积块的输入和输出通道数
        :param n_blocks: 残差块的数量
        :param scaling_factor: 在小像素卷积块中沿两个维度缩放输入图像的因子
        """
        super(SRResNet, self).__init__()

        # 缩放因子必须是2、4或8
        scaling_factor = int(scaling_factor)
        assert scaling_factor in {2, 4, 8}, "缩放因子必须是2、4或8！"

        # 第一个卷积块
        self.conv_block1 = ConvolutionalBlock(in_channels=3, out_channels=n_channels, kernel_size=large_kernel_size,
                                              batch_norm=False, activation='PReLu')

        # 一系列残差块，每个块中包含一个跨块的跳跃连接
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(kernel_size=small_kernel_size, n_channels=n_channels) for i in range(n_blocks)])

        # 另一个卷积块
        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels,
                                              kernel_size=small_kernel_size,
                                              batch_norm=True, activation=None)

        # 通过子像素卷积进行上采样，每个这样的块上采样因子为2
        n_subpixel_convolution_blocks = int(math.log2(scaling_factor))
        self.subpixel_convolutional_blocks = nn.Sequential(
            *[SubPixelConvolutionalBlock(kernel_size=small_kernel_size, n_channels=n_channels, scaling_factor=2) for i
              in range(n_subpixel_convolution_blocks)])

        # 最后一个卷积块
        self.conv_block3 = ConvolutionalBlock(in_channels=n_channels, out_channels=3, kernel_size=large_kernel_size,
                                              batch_norm=False, activation='Tanh')

    def forward(self, lr_imgs):
        """
        前向传播。

        :param lr_imgs: 低分辨率输入图像，尺寸为 (N, 3, w, h) 的张量
        :return: 超分辨率输出图像，尺寸为 (N, 3, w * 缩放因子, h * 缩放因子) 的张量
        """
        output = self.conv_block1(lr_imgs)  # (N, 3, w, h)
        residual = output  # (N, n_channels, w, h)
        output = self.residual_blocks(output)  # (N, n_channels, w, h)
        output = self.conv_block2(output)  # (N, n_channels, w, h)
        output = output + residual  # (N, n_channels, w, h)
        output = self.subpixel_convolutional_blocks(output)  # (N, n_channels, w * 缩放因子, h * 缩放因子)
        sr_imgs = self.conv_block3(output)  # (N, 3, w * 缩放因子, h * 缩放因子)

        return sr_imgs

class Generator(nn.Module):
    """
    SRGAN中的生成器，与论文中定义的SRResNet架构相同。
    """

    def __init__(self, config):
        # 初始化函数
        super(Generator, self).__init__()
        # 从配置中获取生成器的参数
        n_channels = config.G.n_channels  # 生成器中的通道数
        n_blocks = config.G.n_blocks  # 残差块的数量
        large_kernel_size = config.G.large_kernel_size  # 生成器中卷积层的大核大小
        small_kernel_size = config.G.small_kernel_size  # 生成器中卷积层的小核大小
        scaling_factor = config.scaling_factor  # 上采样因子

        # 生成器实际上是一个SRResNet
        self.net = SRResNet(large_kernel_size=large_kernel_size, small_kernel_size=small_kernel_size,
                            n_channels=n_channels, n_blocks=n_blocks, scaling_factor=scaling_factor)

    def initialize_with_srresnet(self, srresnet_checkpoint):
        """
        初始化生成器，使用预训练的SRResNet权重。

        :param srresnet_checkpoint: 预训练模型的检查点文件路径
        """
        # 加载预训练模型
        srresnet = torch.load(srresnet_checkpoint)['model']
        # 将预训练权重加载到生成器中
        self.net.load_state_dict(srresnet.state_dict())
        # 打印加载权重的信息
        print("\n已加载预训练SRResNet的权重。\n")

    def forward(self, lr_imgs):
        """
        前向传播。

        :param lr_imgs: 低分辨率输入图像，尺寸为 (N, 3, w, h) 的张量
        :return: 超分辨率输出图像，尺寸为 (N, 3, w * 上采样因子, h * 上采样因子) 的张量
        """
        # 通过生成器网络进行前向传播
        sr_imgs = self.net(lr_imgs)  # (N, n_channels, w * 上采样因子, h * 上采样因子)
        return sr_imgs  # 返回超分辨率图像


class Discriminator(nn.Module):
    """
    SRGAN中的鉴别器，与论文中定义的架构相同。
    """

    def __init__(self, config):
        # 初始化函数
        super(Discriminator, self).__init__()
        # 从配置中获取鉴别器的参数
        kernel_size = config.D.kernel_size  # 卷积核大小
        n_channels = config.D.n_channels  # 通道数
        n_blocks = config.D.n_blocks  # 卷积块的数量
        fc_size = config.D.fc_size  # 全连接层的大小
        in_channels = 3  # 输入通道数

        # 一系列卷积块
        # 奇数卷积块增加通道数但保持图像大小不变
        # 偶数卷积块保持通道数不变但将图像大小减半
        # 第一个卷积块特殊，因为它不使用批量归一化
        conv_blocks = list()
        for i in range(n_blocks):
            out_channels = (n_channels if i is 0 else in_channels * 2) if i % 2 is 0 else in_channels
            conv_blocks.append(
                ConvolutionalBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=1 if i % 2 is 0 else 2, batch_norm=i is not 0, activation='LeakyReLu'))
            in_channels = out_channels  # 更新通道数
        self.conv_blocks = nn.Sequential(*conv_blocks)  # 将卷积块序列化为一个模型

        # 一个自适应池化层，将输出调整为标准大小
        # 对于默认输入大小96和8个卷积块，这不会有任何效果
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))

        self.fc1 = nn.Linear(out_channels * 6 * 6, fc_size)  # 全连接层

        self.leaky_relu = nn.LeakyReLU(0.2)  # LeakyReLU激活函数

        self.fc2 = nn.Linear(1024, 1)  # 最后一个全连接层

        # 不需要sigmoid层，因为sigmoid操作由PyTorch的nn.BCEWithLogitsLoss()执行()

    def forward(self, imgs):
        """
        前向传播。

        :param imgs: 需要被分类为高分辨率或超分辨率图像的张量，尺寸为 (N, 3, w * 上采样因子, h * 上采样因子)
        :return: 高分辨率图像的得分（logit），尺寸为 (N,)
        """
        batch_size = imgs.size(0)  # 获取批量大小
        output = self.conv_blocks(imgs)  # 通过卷积块进行前向传播
        output = self.adaptive_pool(output)  # 通过自适应池化层
        output = self.fc1(output.view(batch_size, -1))  # 通过第一个全连接层
        output = self.leaky_relu(output)  # 通过LeakyReLU激活函数
        logit = self.fc2(output)  # 通过第二个全连接层获取logit

        return logit  # 返回logit


class TruncatedVGG19(nn.Module):
    """
    截断的VGG19模型，用于生成器和鉴别器之间的感知损失计算。
    """

    def __init__(self, i, j, weights_path):
        # 初始化函数
        super(TruncatedVGG19, self).__init__()

        # 加载预训练的VGG19模型
        vgg19 = torchvision.models.vgg19(pretrained=False)

        maxpool_counter = 0  # 最大池化层计数器
        conv_counter = 0  # 卷积层计数器
        truncate_at = 0  # 截断点

        # 遍历VGG19的卷积部分（"features"）
        for layer in vgg19.features.children():
            truncate_at += 1

            # 计算每个最大池化层后的最大池化层和卷积层的数量
            if isinstance(layer, nn.Conv2d):
                conv_counter += 1
            if isinstance(layer, nn.MaxPool2d):
                maxpool_counter += 1
                conv_counter = 0

            # 如果达到第i个最大池化后的第j个卷积层，则中断
            if maxpool_counter == i - 1 and conv_counter == j:
                break

        # 检查条件是否满足
        assert maxpool_counter == i - 1 and conv_counter == j, "VGG19的i=%d和j=%d的选择无效！" % (i, j)

        # 截断到第i个最大池化前的第j个卷积层（+激活）
        self.truncated_vgg19 = nn.Sequential(*list(vgg19.features.children())[:truncate_at + 1])

        # 加载本地预训练权重
        state_dict = torch.load(weights_path, map_location=torch.device('cuda'))
        self.truncated_vgg19.load_state_dict(state_dict, strict=False)

        # 冻结截断VGG19的参数
        for param in self.truncated_vgg19.parameters():
            param.requires_grad = False

    def forward(self, input):
        """
        前向传播。

        :param input: 输入图像，尺寸为 (N, feature_map_channels, feature_map_w, feature_map_h) 的张量
        :return: 输出特征图
        """
        # 通过截断的VGG19模型进行前向传播
        output = self.truncated_vgg19(input)  # (N, feature_map_channels, feature_map_w, feature_map_h)
        return output  # 返回输出特征图



