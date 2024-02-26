# 案例4：脑部MRI图像分割

 

相关知识点：语义分割、医学图像处理（skimage, medpy）、可视化（matplotlib）

 

## 1 任务目标

### 1.1 任务简介

   本次案例将使用深度学习技术来完成脑部MRI(磁共振)图像分割任务，即对于处理好的一张MRI图像，通过神经网络分割出其中病变的区域。本次案例使用的数据集来自Kaggle^[1]^，共包含110位病人的MRI数据，每位病人对应多张通道数为3的.tif格式图像，其对应的分割结果为单通道黑白图像(白色为病变区域)，示例如下。

![image.png](https://rain-oplat.xuetangx.com/ue_i/20221221/b04dd438-7484-4f10-a3f9-861d4eb5d1ef.png) 

第一行: MRI图像；第二行: 对应的分割标签

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