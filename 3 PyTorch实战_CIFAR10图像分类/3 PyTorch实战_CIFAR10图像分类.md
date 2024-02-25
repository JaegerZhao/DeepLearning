# 案例3：PyTorch实战: CIFAR10图像分类



## 1 任务目标

### 1.1 用多层感知机(MLP)和卷积网络(ConvNet)完成CIFAR10分类

   使用PyTorch分别实现多层感知机(MLP)和卷积网络(ConvNet)，并完成CIFAR10数据集（http://www.cs.toronto.edu/~kriz/cifar.html）分类。本案例不提供初始代码，请自行配置网络和选取超参数，包括层数、卷积核数目、激活函数类型、损失函数类型、优化器等方面。

   提交所有代码和一份案例报告，要求如下：

- 详细介绍所使用的模型及其结果，至少包括超参数选取，损失函数、准确率及其曲线；
- 比较不同模型配置下的结果，<font color='red'>至少从三个方面</font>作比较和分析，例如层数、卷积核数目、激活函数类型、损失函数类型、优化器等。

### 1.2 学习PyTorch ImageNet分类示例

   请自行学习PyTorch官方提供的ImageNet分类示例代码，以便更好地完成后续案例(https://github.com/pytorch/examples/tree/master/imagenet)，这部分无需提交代码和报告。

### 1.3 注意事项

- 提交所有代码和一份案例报告；

- 禁止任何形式的抄袭。



## 2 代码设计

### 2.1 初始化及数据预处理

1. 设置设备

   让模型采用gpu `torch.cuda` 进行训练，若 `cuda` 不可用，则采用cpu进行训练。

   ```py
   # 设置设备
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   print('training on', device)
   ```

2. 数据预处理以及数据增强

   该部分负责在创建数据集时，对数据进行数据增强和预处理，具体操作如下：

       - 训练集数据处理：
         
           - 随机水平翻转`RandomHorizontalFlip()`：以0.5的概率随机水平翻转图像。
           
           - 随机旋转`RandomRotation(5)`：在-5度到5度之间随机旋转图像。
           
           - 颜色抖动`ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)`：对图像进行颜色抖动，包括亮度、对比度和饱和度的随机变化。
           
           - 随机裁剪`RandomResizedCrop(32, scale=(0.9, 1.0))`：对图像进行随机裁剪，然后重新调整大小到指定的尺寸（这里是32x32像素）。
           
           - 张量转换`ToTensor()`：将图像转换为PyTorch张量。

           - 图像标准化`Normalize`：对图像进行标准化，将像素值缩放到[-1, 1]的范围。
      - 测试集数据处理：
           - 张量转换`ToTensor()`：将图像转换为PyTorch张量。

           - 图像标准化`Normalize`：对图像进行标准化，将像素值缩放到[-1, 1]的范围。

   ```py
   # 数据增强和预处理
   transform_train = transforms.Compose([
       transforms.RandomHorizontalFlip(),
       transforms.RandomRotation(5),
       transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
       transforms.RandomResizedCrop(32, scale=(0.9, 1.0)),
       transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
   ])
   
   transform_test = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
   ])
   ```

3. 数据集创建

   ​    CIFAR-10 数据集是一个用于图像分类的常用数据集，该数据集包含了10个类别的彩色图像，每个类别有6,000张图像，共计60,000张图像。每张图像的尺寸是32x32像素。

   ​    该部分用于下载CIFAR-10数据集，并按指定预处理操作，创建训练集和测试集。
      - `torchvision.datasets.CIFAR10`: 这是PyTorch中专门用于加载CIFAR-10数据集的类。
      - `root='./data'`: 这指定了数据集将被下载到的本地目录。
      - `train=True`和`train=False`: 当train=True时，表示创建训练集；当train=False时，表示创建测试集。
   - `download=True`: 如果本地没有找到CIFAR-10数据集，设置为True时，PyTorch将自动下载并解压缩数据集。
   - `transform=transform`: 这里的transform参数指定了数据集中图像的预处理操作。

   ```py
   # 加载CIFAR-10数据集并进行划分
   train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
   test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
   ```

4. 创建数据加载器

   创建用于训练和测试的数据加载器（DataLoader）。`train_loader`用于训练，`test_loader`用于测试。

   ​    这些数据加载器在训练和测试过程中会循环提供每个批次的图像和标签，方便使用PyTorch的模型进行训练和评估。

      - `torch.utils.data.DataLoader`: 这是PyTorch中的一个类，用于从数据集中加载批量的数据。
      - `batch_size=64`: 这指定了每个批次加载的图像数量。
      - `shuffle=True`和`shuffle=False`: 这表示是否在每个epoch开始时随机打乱数据。在训练集中，通常希望打乱数据以防止模型学到数据的顺序性，而在测试集中则可以保持数据的原始顺序。
   - `num_workers=2`: 这指定了用于加载数据的子进程数量。
