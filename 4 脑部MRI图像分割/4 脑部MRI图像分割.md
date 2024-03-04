# 案例4：脑部MRI图像分割

 

相关知识点：语义分割、医学图像处理（skimage, medpy）、可视化（matplotlib）



## 1 任务目标

### 1.1 任务简介

   本次案例将使用深度学习技术来完成脑部MRI(磁共振)图像分割任务，即对于处理好的一张MRI图像，通过神经网络分割出其中病变的区域。本次案例使用的数据集来自Kaggle^[1]^，共包含110位病人的MRI数据，每位病人对应多张通道数为3的.tif格式图像，其对应的分割结果为单通道黑白图像(白色为病变区域)，示例如下。

![image.png](https://rain-oplat.xuetangx.com/ue_i/20221221/b04dd438-7484-4f10-a3f9-861d4eb5d1ef.png) 

> 第一行: MRI图像；第二行: 对应的分割标签
>

   更详细的背景介绍请参考文献^[2]^.

### 1.2 参考程序

   本次案例提供了完整、可供运行的参考程序，来源于Kaggle^[3]^和GitHub^[4]^，建议在参考程序的基础上进行修改来完成本案例。各个程序简介如下：

- `train.ipynb`用来完成模型训练

- `inference.ipynb`用来对训练后的模型进行推理

- `unet.py`定义了U-Net网络结构，参考资料^[5]^

- `loss.py`定义了损失函数(Dice Loss)，参考资料^[6]^

- `dataset.py`用来定义和读取数据集

- `transform.py`用来预处理数据

- `utils.py`定义了若干辅助函数

- `logger.py`用来记录训练过程(使用TensorBoard^[7]^功能)，包括损失函数曲线等



   参考程序对运行环境的要求如下，请自行调整环境至适配，否则可能无法运行：

> torch==2.0.*
>
> torchvision==0.15.*
>
> ipykernel==6.26.*
>
> matplotlib==3.8.*
>
> medpy==0.4.*
>
> scipy==1.11.*
>
> numpy==1.23.* (1.24+版本无法运行，需要先降级)
>
> scikit-image==0.22.*
>
> imageio==2.31.*
>
> tensorboard==2.15.*
>
> tqdm==4.*

   其它细节以及示例运行结果可直接参考Kaggle^[3]^和GitHub^[4]^。

### 1.3 要求和建议

   在参考程序的基础上，使用深度学习技术，尝试提升该模型在脑部MRI图像上的分割效果，以程序最终输出的validation mean DSC值作为评价标准(参考程序约为90%)。可从网络结构(替换U-Net)、损失函数(替换Dice Loss)、训练过程(如优化器)等角度尝试改进，还可参考通用图像分割的一些技巧^[8]^。

### 1.4 注意事项

- 提交所有代码和一份案例报告；

- 案例报告应详细介绍所有改进尝试及对应的结果(包括DSC值和若干分割结果示例)，无论是否成功提升模型效果，并对结果作出分析；

- 禁止任何形式的抄袭，借鉴开源程序务必加以说明。

### 1.5 参考资料

> [1] Brain MRI数据集: https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation
>
> [2] Buda et al. Association of genomic subtypes of lower-grade gliomas with shape  features automatically extracted by a deep learning algorithm. Computers in Biology and Medicine 2019.
>
> [3] 示例程序: https://www.kaggle.com/mateuszbuda/brain-segmentation-pytorch
>
> [4] 示例程序: https://github.com/mateuszbuda/brain-segmentation-pytorch
>
> [5] Ronneberger et al. U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI 2015.
>
> [6] Dice Loss: https://zhuanlan.zhihu.com/p/86704421
>
> [7] TensorBoard参考资料：https://www.tensorflow.org/tensorboard 
>
> [8] Minaee et al. Image Segmentation Using Deep Learning: A Survey. arXiv 2020.



## 2 通过云平台训练基础代码

​	本次实验由于运行时间较长，占用内存较大，所以选择了用学堂在线的 **和鲸云平台** 来进行训练。下面介绍是如何使用云平台训练基础代码的。

### 2.1 云平台环境配置

1. 数据集接入

   本次实验采用Brain MRI数据集，该数据集已被上传到了云平台的共享数据中，可以直接调用。

   ![image-20240228231424167](https://raw.githubusercontent.com/ZzDarker/figure/main/img/image-20240228231424167.png)

2. 项目创建

   此处没有采用直接fork作业中的文件，因为作业中的文件与本地有所不同，经对比，采用从学堂在线下载文件训练效果更好。

   在我的空间里，点击新建，创建项目，输入项目名称，选择数据源，完成项目的创建。

   ![image-20240228231837429](https://raw.githubusercontent.com/ZzDarker/figure/main/img/image-20240228231837429.png)

3. 环境配置

   创建完成后，点击右上角齿轮按钮，配置项目环境。

   ![image-20240228232023797](https://raw.githubusercontent.com/ZzDarker/figure/main/img/image-20240228232023797.png)

   选择 `T4 GPU` ，基础环境选择 `Pytorch 2.0.1 Cuda11.7 Python3.10` 的版本。

   ![image-20240228232312501](https://raw.githubusercontent.com/ZzDarker/figure/main/img/image-20240228232312501.png)

   点击，运行完成基础环境配置。然后导入从学堂在线下载的实验四文件。

   按照其中 `train.ipynb` 的内容安装其他依赖库。即在notebook里的代码格输入以下内容，注意前面要加 `!` 。

   ```cmd
   !pip install ipykernel==6.26.* matplotlib==3.8.* medpy==0.4.* scipy==1.11.* numpy==1.23.* scikit-image==0.22.* imageio==2.31.* tensorboard==2.15.* tqdm==4.* -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

### 2.2 云平台项目训练

​	云平台支持在线训练和离线训练两种方式，其中在线训练要求网络保持通畅不能断网，离线训练最好在在线训练跑通后再进行训练。

1. 在线训练

  首先需要修改数据集地址，不然无法训练。在 `train.ipynb` 的 `args` 中，将 `images` 的路径替换成如下内容。

  ```py
  images = '../input/02039681/utf-8kaggle_3m/kaggle_3m'
  ```

  默认训练轮次是100轮，实际上训练100轮太多了，20轮足矣。所以我选择将轮次更改成20轮。

  ```py
  epochs = 20,
  ```

  在配置好后，在 `train.ipynb` 界面，点击任务栏的 `运行所有` 键，开始U-net模型的训练。

  ![8e9820121aa65ac6daa527de5a8c8c6](https://raw.githubusercontent.com/ZzDarker/figure/main/img/8e9820121aa65ac6daa527de5a8c8c6.png)

2. 训练过程

  训练结束后，会在 `project` 栏中，生成 `log`文件夹，存储训练日志，可以用TensorBoard查看训练过程，有loss、val_dsc、val_loss以下三个图表。

  ![image-20240302204417111](https://raw.githubusercontent.com/ZzDarker/figure/main/img/image-20240302204417111.png)

  ![image-20240302204444479](https://raw.githubusercontent.com/ZzDarker/figure/main/img/image-20240302204444479.png)

  ![image-20240302204509300](https://raw.githubusercontent.com/ZzDarker/figure/main/img/image-20240302204509300.png)

  可以看到20轮训练，有点欠拟合，不过训练结果还算不错。

3. 训练结果

  训练结束后，在最后会显示 `Best validation mean DSC` 值。

  ![image-20240302185734333](https://raw.githubusercontent.com/ZzDarker/figure/main/img/image-20240302185734333.png)

  可以看到，经过20轮的训练，在测试集得到最好的DSC达到了0.913458。

4. 离线训练

  离线训练首先需要将配置好的环境，保存成私有镜像。点击任务栏的 `镜像` →`保存当前环境` 等待配置后，保存成功当前环境。

  ![image-20240228234854084](https://raw.githubusercontent.com/ZzDarker/figure/main/img/image-20240228234854084.png)

  然后点击任务栏中的 `离线任务`，选择刚才配置好的镜像，即可进行离线训练。

  离线训练时，可以从云平台侧边栏的离线任务中，查看离线任务的运行状态，包括内存、CPU占用等。

  ![image-20240228235203999](https://raw.githubusercontent.com/ZzDarker/figure/main/img/image-20240228235203999.png)

  离线任务运行结束后，若运行的没有问题，则可以保存回原文件，得到在线任务提到的两个文件夹。

### 2.3 云平台项目测试

​	在训练结束后，选择 `inference.ipynb` 文件进行测试，按训练项目中的步骤，替换数据集路径。点击运行所有完成项目的测试。

在测试结束后，可以得到一个 `dsc.png` 图片记录了不同类别图像的DSC（迪斯相似系数）值。图中的红线为所有DSC值的均值，绿线为DSC值的中值。

![dsc_klab_2_upload](https://raw.githubusercontent.com/ZzDarker/figure/main/img/dsc_klab_2_upload.png)

> **Dice Similarity Coefficient (DSC)** 是通过比较模型预测的分割结果与地面真实分割的重叠部分来衡量相似度的指标。它的计算公式如下：
> $$
> DSC = \frac{2 \times | \text{Intersection} |}{ | \text{Prediction} | + | \text{Ground Truth} |} 
> $$
> 
>
> 其中：
>
> - $ \text{Intersection} $ 表示模型预测和背景真实分割的交集部分。
> - $ | \text{Prediction} | $ 表示模型预测的分割的总像素数。
> - $ | \text{Ground Truth} | $ 表示背景真实分割的总像素数。

并得到一个 `predictions` 文件夹，存储预测的一系列脑部MRI图像，其中红色为预测框，绿色为真实框。

![TCGA_CS_4944_20010208-08](https://raw.githubusercontent.com/ZzDarker/figure/main/img/TCGA_CS_4944_20010208-08.png)

云平台图片不能批量下载，只能一张一张下载查看图片，结果并不直观。

## 3 本地训练基础代码

​	由于云平台编辑代码不方便，图片需要下载才能查看等弊端，于是我又选择在本地训练。

### 3.1 本地环境配置

​	根据 `train.ipynb` 的步骤安装环境。

1. 新建python 3.10环境

   ```bash
   conda create -n hw4 python=3.10 -y
   conda activate hw4
   ```

2. 安装torch，注意cuda版本适配

   ```bash
   pip install torch==2.0.* torchvision==0.15.* --index-url https://download.pytorch.org/whl/cu117
   ```

3. 安装其他依赖库

   ```bash
   pip install ipykernel==6.26.* matplotlib==3.8.* medpy==0.4.* scipy==1.11.* numpy==1.23.* scikit-image==0.22.* imageio==2.31.* tensorboard==2.15.* tqdm==4.* -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

### 3.2 本地项目训练

1. 修改数据集路径

   将 `images` 数据集路径修改为你的路径。

   ```cmd
   images = './archive/kaggle_3m',
   ```

2. 项目训练

   全部运行 `train.ipynb` ，本次训练，训练100个epoch，最优结果如下。

   ```bash
   100%|██████████| 208/208 [01:19<00:00,  2.62it/s]
   100%|██████████| 21/21 [00:07<00:00,  2.78it/s]
   epoch 100 | val_loss: 0.21832050595964705
   epoch 100 | val_dsc: 0.9073074088508986
   Best validation mean DSC: 0.914025
   ```

   在100轮的训练后，最佳DSC值为0.914025。

3. 训练结果

   模型训练结束得到训练日志，通过Tensorboard打开，得到结果如下。

   ![image-20240302202001678](https://raw.githubusercontent.com/ZzDarker/figure/main/img/image-20240302202001678.png)

   ![image-20240302202032211](https://raw.githubusercontent.com/ZzDarker/figure/main/img/image-20240302202032211.png)

   

   ![image-20240302202120555](https://raw.githubusercontent.com/ZzDarker/figure/main/img/image-20240302202120555.png)

   由上图可知，100轮训练后，训练集和测试集的loss逐步降低，测试集精度逐步上升，训练结果还算不错。

### 3.3 本地项目测试

1. 项目测试

   修改好数据集路径后，全部运行 `inference.ipynb` 。

   ```bash
   ...
    70%|███████   | 7/10 [00:05<00:02,  1.07it/s]C:\Users\Administrator\AppData\Local\Temp\ipykernel_15188\1157483797.py:12: UserWarning: ./predictions\kaggle_3m\TCGA_DU_5851_19950428-34.png is a low contrast image
     imsave(filepath, image)
   100%|██████████| 10/10 [00:06<00:00,  1.52it/s]
   ```

2. DSC测试

   在测试结束后，会生成一张DSC图片。

   ![dsc](https://raw.githubusercontent.com/ZzDarker/figure/main/img/dsc.png)

   由上图可知，除了 `TCGA_HT_717` 文件，其他的测试DSC值都达到了0.9及以上，测试结果还算不错。

3. 预测图片

   本地运行后，可以得到一个 `predictions` 文件夹，存储所有预测图片。

   可以通过以下代码，将同一系列的图片转换为gif图片，可以更直观的查看测试效果。

   ```py
   from PIL import Image
   import os
   
   def create_gif(image_folder, output_gif_path, prefix):
       images = []
   
       # 获取文件夹中所有以 prefix 开头的图片文件
       image_files = [file for file in os.listdir(image_folder) if file.startswith(prefix)]
       image_files.sort()  # 按名称排序
   
       for image_file in image_files:
           image_path = os.path.join(image_folder, image_file)
           img = Image.open(image_path)
           images.append(img)
   
       # 保存为 GIF 图片
       images[0].save(output_gif_path+image_prefix+".gif", save_all=True, append_images=images[1:], duration=100, loop=0)
   ```

   通过代码，可以得到以下GIF图片。

   - TCGA_DU_6404

     通过DSC测试可知，该样本脑部MRI图像测试结果最好，如下图所示，绿色与红色框几乎重合。

     ![TCGA_DU_6404_19850629](https://raw.githubusercontent.com/ZzDarker/figure/main/img/TCGA_DU_6404_19850629.gif)

   - TCGA_CS_4944

     通过DSC测试可知，该样本脑部MRI图像测试结果较为不错，如下图所示，绿色与红色框差异不大。

     ![TCGA_CS_4944_20010208](https://raw.githubusercontent.com/ZzDarker/figure/main/img/TCGA_CS_4944_20010208.gif)

   - TCGA_HT_7616

     通过DSC测试可知，该样本脑部MRI图像测试结果最差，如下图所示，绿色与红色框差异较大。

     ![TCGA_HT_7616_19940813](https://raw.githubusercontent.com/ZzDarker/figure/main/img/TCGA_HT_7616_19940813.gif)

## 4 代码解析

​	本次实验采用U-Net架构，实现对脑部MRI图像的图像分割。

### 4.1 U-Net架构

U-Net 是一种用于图像分割的卷积神经网络，它由编码器和解码器两部分组成，形状类似字母 U，因此得名。

![image-20240302213716022](https://raw.githubusercontent.com/ZzDarker/figure/main/img/image-20240302213716022.png)

U-Net 的主要特点包括：

1. **编码器-解码器结构**：U-Net 由两部分组成，一个收缩路径（编码器）和一个对称的扩展路径（解码器）。编码器通过卷积和池化操作逐步减小特征图的尺寸，同时捕获图像的上下文信息。解码器通过上采样和卷积操作逐步恢复特征图的尺寸，同时保留和增强重要的空间信息。
2. **跳跃连接**：在编码器和解码器之间，U-Net 使用了跳跃连接（也称为长距离连接或残差连接），将编码阶段的特征图与解码阶段的对应特征图进行拼接。这种连接有助于解码器恢复更多的空间细节，从而提高分割精度。
3. **上采样**：在解码阶段，U-Net 使用了上采样操作（如转置卷积或上采样层）来逐步增大特征图的尺寸。这有助于恢复原始图像的分辨率，使得网络能够输出与输入图像尺寸相同的分割图。
4. **多尺度特征融合**：由于跳跃连接和编码器-解码器结构的结合，U-Net 能够有效地融合多尺度的特征信息。这对于处理具有不同尺寸和形状的目标非常重要。
5. **轻量级和高效**：尽管 U-Net 结构相对简单，但它在许多图像分割任务中表现出了出色的性能。此外，由于其轻量级的特性，U-Net 可以在有限的计算资源上实现高效的训练和推理。

### 4.2 U-Net代码解析

​	代码定义了一个UNet类，具体内容如下。

1. 构造函数（`__init__`）接受三个参数：

   - `in_channels`：输入通道的数量（默认为3，适用于RGB图像）。
   - `out_channels`：输出通道的数量（默认为1，用于二元分割）。
   - `init_features`：初始特征的数量（默认为32）。

   ```py
   def __init__(self, in_channels=3, out_channels=1, init_features=32):
           super(UNet, self).__init__()
   ```

2. 编码器架构

   - U-Net的编码器部分由四个块（`enc1`到`enc4`）组成，每个块包含两个卷积层、批归一化和ReLU激活。
   - 在每个编码器块后，应用最大池化层（`pool1`到`pool4`）以减小空间维度。

   ```py
           features = init_features
           self.encoder1 = UNet._block(in_channels, features, name="enc1")
           self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
           self.encoder2 = UNet._block(features, features * 2, name="enc2")
           self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
           self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
           self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
           self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
           self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
   ```

3. 中间层

   瓶颈块（`bottleneck`）是另一组包含两个卷积层、批归一化和ReLU激活的层，表示U-Net的中央层。

   ```py
   self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")
   ```

4. 解码器结构

   - 解码器部分包含四个块（`dec4`到`dec1`），每个块包含两个卷积层、批归一化和ReLU激活。
   - 在每个解码器块后，应用转置卷积（`upconv4`到`upconv1`）以上采样特征图。

   ```py
   		self.upconv4 = nn.ConvTranspose2d(
               features * 16, features * 8, kernel_size=2, stride=2
           )
           self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
           self.upconv3 = nn.ConvTranspose2d(
               features * 8, features * 4, kernel_size=2, stride=2
           )
           self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
           self.upconv2 = nn.ConvTranspose2d(
               features * 4, features * 2, kernel_size=2, stride=2
           )
           self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
           self.upconv1 = nn.ConvTranspose2d(
               features * 2, features, kernel_size=2, stride=2
           )
           self.decoder1 = UNet._block(features * 2, features, name="dec1")
   ```

5. 前向传播

   - 输入`x`通过编码器块和池化层。

     ```py
         def forward(self, x):
             enc1 = self.encoder1(x)
             enc2 = self.encoder2(self.pool1(enc1))
             enc3 = self.encoder3(self.pool2(enc2))
             enc4 = self.encoder4(self.pool3(enc3))
     ```

   - 瓶颈应用于下采样的特征图。

     ```py
             bottleneck = self.bottleneck(self.pool4(enc4))
     ```

   - 解码器块和转置卷积用于通过**跳跃连接** ( `torch.cat` ) 上采样特征。

     ```py
             dec4 = self.upconv4(bottleneck)
             dec4 = torch.cat((dec4, enc4), dim=1)
             dec4 = self.decoder4(dec4)
             dec3 = self.upconv3(dec4)
             dec3 = torch.cat((dec3, enc3), dim=1)
             dec3 = self.decoder3(dec3)
             dec2 = self.upconv2(dec3)
             dec2 = torch.cat((dec2, enc2), dim=1)
             dec2 = self.decoder2(dec2)
             dec1 = self.upconv1(dec2)
             dec1 = torch.cat((dec1, enc1), dim=1)
             dec1 = self.decoder1(dec1)
     ```

   - 最后一层应用Sigmoid激活进行二元分割。

     ```py
             return torch.sigmoid(self.conv(dec1))
     ```

6. 辅助方法 `_block`

   - 定义了一个静态方法`_block`，用于创建包含两个卷积层、批归一化和ReLU激活的基本块。
   - 该块被返回为`nn.Sequential`模块。

   ```py
       @staticmethod
       def _block(in_channels, features, name):
           return nn.Sequential(
               OrderedDict(
                   [
                       (
                           name + "conv1",
                           nn.Conv2d(
                               in_channels=in_channels,
                               out_channels=features,
                               kernel_size=3,
                               padding=1,
                               bias=False,
                           ),
                       ),
                       (name + "norm1", nn.BatchNorm2d(num_features=features)),
                       (name + "relu1", nn.ReLU(inplace=True)),
                       (
                           name + "conv2",
                           nn.Conv2d(
                               in_channels=features,
                               out_channels=features,
                               kernel_size=3,
                               padding=1,
                               bias=False,
                           ),
                       ),
                       (name + "norm2", nn.BatchNorm2d(num_features=features)),
                       (name + "relu2", nn.ReLU(inplace=True)),
                   ]
               )
           )
   ```

### 4.3 训练函数代码

1. 训练超参数

   ```py
   args = SimpleNamespace(
       device = 'cuda:0',
       batch_size = 16,
       epochs = 100,
       lr = 0.0001,
       workers = 0,
       vis_images = 200,
       vis_freq = 10,
       weights = './weights',
       logs = './logs',
       images = './archive/kaggle_3m',
       image_size = 256,
       aug_scale = 0.05,
       aug_angle = 15,
   )
   ```

   - `device`: 指定模型训练的设备，这里设置为 `cuda:0`，表示使用第一个 GPU。如果没有可用的 GPU，可以将其设置为 `cpu`。
   - `batch_size`: 每个训练批次中包含的样本数量，这里设置为 16。
   - `epochs`: 训练的总轮数，这里设置为 100。
   - `lr`: 学习率，即模型在每个训练步骤中权重更新的大小，这里设置为 0.0001。
   - `workers`: 数据加载时的并行工作数，这里设置为 0，表示不使用多线程加载数据。
   - `vis_images`: 每次可视化的图像数量，这里设置为 200。
   - `vis_freq`: 可视化的频率，即每训练多少个批次可视化一次，这里设置为 10。
   - `weights`: 模型权重的保存路径，这里设置为 `./weights`。
   - `logs`: 训练日志的保存路径，这里设置为 `./logs`。
   - `images`: 存储图像数据的路径，这里设置为 `./archive/kaggle_3m`。
   - `image_size`: 输入图像的大小，这里设置为 256。
   - `aug_scale`: 数据增强的缩放参数，这里设置为 0.05。
   - `aug_angle`: 数据增强的旋转角度参数，这里设置为 15。

2. 读取数据

   - **`worker_init` 函数：**

     这是一个用于多线程数据加载的初始化函数，确保每个线程有相同的随机种子。这里使用 `np.random.seed` 来设置随机种子。

     ```py
     # 读取数据
     def worker_init(worker_id):
         np.random.seed(42 + worker_id)
     ```

   - **`data_loaders` 函数：**

     该函数用于创建训练和验证数据加载器，并返回训练和验证数据集的对象。
     
     - `dataset_train, dataset_valid = datasets(args)` 调用 `datasets` 函数获取训练和验证数据集。
     - 然后使用 `DataLoader` 创建两个数据加载器 `loader_train` 和 `loader_valid`，分别用于训练和验证。这些加载器将数据集划分为批次，可以在模型训练时使用。
     - `worker_init_fn=worker_init` 用于设置每个数据加载线程的随机种子。
     
     ```py
     def data_loaders(args):
         dataset_train, dataset_valid = datasets(args)
     
         loader_train = DataLoader(
             dataset_train,
             batch_size=args.batch_size,
             shuffle=True,
             drop_last=True,
             num_workers=args.workers,
             worker_init_fn=worker_init,
         )
         loader_valid = DataLoader(
             dataset_valid,
             batch_size=args.batch_size,
             drop_last=False,
             num_workers=args.workers,
             worker_init_fn=worker_init,
         )
     
         return dataset_train, dataset_valid, loader_train, loader_valid
     ```

3. 数据集定义

   - **`datasets` 函数：**

     - 该函数用于定义训练和验证数据集，并返回它们的对象。

     - 调用 `Dataset` 类来创建训练和验证数据集，传递了一些参数如 `images_dir`、`subset`、`image_size`、`transform`等。

     ```py
     # 数据集定义
     def datasets(args):
         train = Dataset(
             images_dir=args.images,
             subset="train",
             image_size=args.image_size,
             transform=transforms(scale=args.aug_scale, angle=args.aug_angle, flip_prob=0.5),
         )
         valid = Dataset(
             images_dir=args.images,
             subset="validation",
             image_size=args.image_size,
             random_sampling=False,
         )
         return train, valid
     ```

4. 数据处理

   - **`dsc_per_volume` 函数：**

     - 这是一个用于计算 Dice Similarity Coefficient（DSC）的函数，其中 `validation_pred` 是预测的分割结果，`validation_true` 是真实的分割结果，`patient_slice_index` 是患者每个切片的索引。
     - 该函数对每个样本计算 DSC，并将结果存储在 `dsc_list` 中返回。

     ```py
     def dsc_per_volume(validation_pred, validation_true, patient_slice_index):
         dsc_list = []
         num_slices = np.bincount([p[0] for p in patient_slice_index])
         index = 0
         for p in range(len(num_slices)):
             y_pred = np.array(validation_pred[index : index + num_slices[p]])
             y_true = np.array(validation_true[index : index + num_slices[p]])
             dsc_list.append(dsc(y_pred, y_true))
             index += num_slices[p]
         return dsc_list
     ```

   - **`log_loss_summary` 函数：**

     - 该函数用于记录损失值到日志中，通常用于可视化和监控训练过程。这里使用了 `logger.scalar_summary` 函数来记录损失值。

     ```py
     def log_loss_summary(logger, loss, step, prefix=""):
         logger.scalar_summary(prefix + "loss", np.mean(loss), step)
     ```

   - **`makedirs` 函数：**

     - 用于创建存储模型权重和日志的目录。调用 `os.makedirs` 函数来创建目录，如果目录已存在则不会报错。

     ```py
     def makedirs(args):
         os.makedirs(args.weights, exist_ok=True)
         os.makedirs(args.logs, exist_ok=True)
     ```

   - **`snapshotargs` 函数：**

     - 该函数用于保存实验参数到一个 JSON 文件中，这样可以在后续的实验中追溯实验的设置。调用 `json.dump` 将参数写入 JSON 文件。

     ```py
     def snapshotargs(args):
         args_file = os.path.join(args.logs, "args.json")
         with open(args_file, "w") as fp:
             json.dump(vars(args), fp)
     ```

5. 加载数据集

   根据上面的函数，建立数据集，与训练日志，加载数据集并对数据集内容进行预处理。

   ```py
   makedirs(args)
   snapshotargs(args)
   device = torch.device("cpu" if not torch.cuda.is_available() else args.device)
   
   dataset_train, dataset_valid, loader_train, loader_valid = data_loaders(args)
   loaders = {"train": loader_train, "valid": loader_valid}
   ```

6. 初始化Unet模型

   - UNet模型初始化

     ```py
     unet = UNet(in_channels=Dataset.in_channels, out_channels=Dataset.out_channels)
     unet.to(device)
     ```

   - Dice Loss初始化

     创建了一个Dice Loss的实例 `dsc_loss`，用于度量分割模型的性能。

     ```py
     dsc_loss = DiceLoss()
     best_validation_dsc = 0.0
     ```

   - 定义优化器

     创建了一个Adam优化器，用于更新UNet模型的参数。

     ```py
     optimizer = optim.Adam(unet.parameters(), lr=args.lr)
     ```

   - 日志记录器初始化

     创建了一个日志记录器 `logger`，用于记录训练过程中的信息，如训练和验证损失。

     ```py
     logger = Logger(args.logs)
     ```

   - 训练参数初始化

     初始化训练和验证损失列表，创建了两个空列表 `loss_train` 和 `loss_valid`，用于存储每个训练和验证步骤的损失值。

     初始化训练步数，用于跟踪训练过程中的步数。

     ```py
     loss_train = []
     loss_valid = []
     
     step = 0
     ```

7. 模型训练

   - **循环训练**

     两个嵌套的循环，外层循环迭代训练的轮数，内层循环迭代训练和验证阶段。

     ```py
     for epoch in range(args.epochs):
         for phase in ["train", "valid"]:
             # ...
     ```

   - **模型模式设置**

     根据当前阶段（训练或验证）设置模型的模式，对于训练模式，启用 Batch Normalization 和 Dropout 等层的训练行为；对于验证模式，关闭这些层的训练行为以避免随机性。

     ```py
     if phase == "train":
         unet.train()
     else:
         unet.eval()
     ```

   - **损失和优化器初始化**

     初始化 Adam 优化器和 Dice Loss 损失函数。

     ```py
     optimizer = optim.Adam(unet.parameters(), lr=args.lr)
     dsc_loss = DiceLoss()
     best_validation_dsc = 0.0
     ```

   - **数据加载和处理**

     通过 `loaders` 加载训练或验证数据，并将数据移动到指定的计算设备。

     ```py
     for i, data in enumerate(tqdm.tqdm(loaders[phase])):
         x, y_true = data
         x, y_true = x.to(device), y_true.to(device)
     ```

   - **模型前向传播和损失计算**

     使用 UNet 模型进行前向传播，计算预测结果 `y_pred` 并计算 Dice Loss。

     ```py
     y_pred = unet(x)
     loss = dsc_loss(y_pred, y_true)
     ```

   - **反向传播和优化**

     如果是训练阶段，则执行反向传播和参数优化更新。

     ```py
     if phase == "train":
         loss_train.append(loss.item())
         loss.backward()
         optimizer.step()
     ```

   - **验证阶段损失和性能评估**

     在验证阶段，记录验证集的损失，计算 Dice 相似性系数，并保存模型权重如果当前性能更好。

     ```py
     if phase == "valid":
         loss_valid.append(loss.item())
         # ...
     ```

   - **可视化和日志记录**

     ```py
     if (epoch % args.vis_freq == 0) or (epoch == args.epochs - 1):
         # ...
     ```

   - **性能指标记录和保存最佳模型**

     如果当前验证性能更好，则保存当前的 UNet++ 模型权重。

     ```py
     if mean_dsc > best_validation_dsc:
         best_validation_dsc = mean_dsc
         torch.save(unet.state_dict(), os.path.join(args.weights, "unet.pt"))
     ```

   - **最终输出**

     在训练完成后，输出最佳验证集 Dice 相似性系数。

     ```py
     print("Best validation mean DSC: {:4f}".format(best_validation_dsc))
     ```



## 5 代码优化

### 5.1 优化loss



### 5.2 加入Attention

![image-20240304022934568](https://raw.githubusercontent.com/ZzDarker/figure/main/img/image-20240304022934568.png)

![image-20240304022954313](https://raw.githubusercontent.com/ZzDarker/figure/main/img/image-20240304022954313.png)

![image-20240304023022203](https://raw.githubusercontent.com/ZzDarker/figure/main/img/image-20240304023022203.png)

[(99+ 封私信 / 81 条消息) 周纵苇 - 知乎 (zhihu.com)](https://www.zhihu.com/people/zongweiz/posts)

### 5.3 Unet++

[Unet-Segmentation-Pytorch-Nest-of-Unets/Models.py at master · bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets (github.com)](https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets/blob/master/Models.py)

[ShawnBIT/UNet-family: Paper and implementation of UNet-related model. (github.com)](https://github.com/ShawnBIT/UNet-family/tree/master)
