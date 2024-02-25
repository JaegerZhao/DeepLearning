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

   这些数据加载器在训练和测试过程中会循环提供每个批次的图像和标签，方便使用PyTorch的模型进行训练和评估。

      - `torch.utils.data.DataLoader`: 这是PyTorch中的一个类，用于从数据集中加载批量的数据。
      - `batch_size=64`: 这指定了每个批次加载的图像数量。
      - `shuffle=True`和`shuffle=False`: 这表示是否在每个epoch开始时随机打乱数据。在训练集中，通常希望打乱数据以防止模型学到数据的顺序性，而在测试集中则可以保持数据的原始顺序。
   - `num_workers=2`: 这指定了用于加载数据的子进程数量。

   ```py
   # 创建数据加载器
   train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
   test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
   ```

5. 显示数据集图片

   将一个批次（64）的数据集图像及标签通过 `Matplotlib` 显示出来。

   ```py
   # 获取一个批次的图像和标签
   for images, labels in train_loader:
       break  # 获取第一个批次后就跳出循环
   
   # 将张量转换为NumPy数组
   images = images.numpy()
   
   # 反归一化
   images = (images * 0.5) + 0.5
   
   # 定义标签对应的类别名称
   class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
   
   # 显示图像和标签
   plt.figure(figsize=(20, 20))
   for i in range(64):
       plt.subplot(8, 8, i + 1)
       plt.imshow(np.transpose(images[i], (1, 2, 0)))
       plt.title(class_names[labels[i]])
       plt.axis('off')
   
   plt.show()
   ```

   ![output](https://raw.githubusercontent.com/ZzDarker/figure/main/img/output.png)

   从上图可知，图像以及经过了数据增强，有的图像经过了颜色抖动、翻转、旋转以及缩放。

### 2.2 模型建立

1. 建立MLP模型

   MLP模型由三个全连接层组成，通过ReLU激活函数进行非线性变换，最终输出一个10维的张量，表示对每个类别的预测得分。

   - **输入层**：
     - 输入层的大小是`32*32*3`，这对应于CIFAR-10图像的尺寸（32x32像素）和通道数（RGB图像，每个像素3个通道）。
   - **隐藏层1**：
     - 第一个全连接层 (`self.fc1`) 具有512个神经元。
     - 激活函数：ReLU（修正线性单元），通过 `x = torch.relu(self.fc1(x))` 应用。
   - **隐藏层2**：
     - 第二个全连接层 (`self.fc2`) 具有256个神经元。
     - 激活函数：ReLU，通过 `x = torch.relu(self.fc2(x))` 应用。
   - **输出层：**
     - 输出层 (`self.fc3`) 具有10个神经元，对应于CIFAR-10数据集中的10个类别。
     - 激活函数：Softmax，通过 `x = F.softmax(self.fc3(x), dim=1)` 应用。Softmax将网络的原始输出转换为概率分布，每个类别的输出值表示该类别的概率。

   ```py
   # 定义MLP模型
   class MLP(nn.Module):
       def __init__(self):
           super(MLP, self).__init__()
           self.fc1 = nn.Linear(32*32*3, 512)
           self.fc2 = nn.Linear(512, 256)
           self.fc3 = nn.Linear(256, 10)
   
       def forward(self, x):
           x = x.view(x.size(0), -1)
           x = torch.relu(self.fc1(x))
           x = torch.relu(self.fc2(x))
           x = self.fc3(x)
           x = F.softmax(x, dim=1)
           return x
   ```

2. 建立自定义MLP模型

   在这个 DefMLP 模型中，可以调整层数`num_layers`,激活函数 `activation_func`。

   - **输入层：**
     - 输入层的大小是`32*32*3`，这对应于CIFAR-10图像的尺寸（32x32像素）和通道数（RGB图像，每个像素3个通道）。
   - **隐藏层：**
     - 模型包含 `num_layers` 个隐藏层。每个隐藏层包含一个全连接层 (`nn.Linear`) 和一个激活函数。
     - 全连接层的输入大小在每个隐藏层中更新为512，这是因为每个隐藏层都有512个神经元。
     - 可以选择使用 'relu' 或 'sigmoid' 作为激活函数，通过指定 `activation_func` 参数来选择。
   - **输出层：**
     - 输出层 (`self.fc`) 具有10个神经元，对应于CIFAR-10数据集中的10个类别。
   - **前向传播：**
     - 输入通过 `view` 操作展平，然后通过隐藏层 (`self.layers`) 处理，最后通过输出层 (`self.fc`) 得到模型的最终输出。

   ```py
   class DefMLP(nn.Module):
       def __init__(self, num_layers, activation_func):
           super(DefMLP, self).__init__()
   
           layers = []
           input_size = 32 * 32 * 3
           for _ in range(num_layers):
               layers.append(nn.Linear(input_size, 512))
               input_size = 512  # 更新输入大小
               if activation_func == 'relu':
                   layers.append(nn.ReLU())
               elif activation_func == 'sigmoid':
                   layers.append(nn.Sigmoid())
               else:
                   raise ValueError("Invalid activation function")
   
           self.layers = nn.Sequential(*layers)
           self.fc = nn.Linear(512, 10)
   
       def forward(self, x):
           x = x.view(x.size(0), -1)
           x = self.layers(x)
           x = self.fc(x)
           return x
   ```

3. 简单卷积神经网络模型

   ConvNet模型包含了两个卷积层和两个全连接层，通过ReLU激活函数进行非线性变换，并通过最大池化进行下采样。

   - **卷积层1 (`self.conv1`):**
     - 输入通道数为3（RGB图像），输出通道数为64。
     - 使用3x3的卷积核进行卷积操作。
     - 使用ReLU激活函数进行非线性变换。
     - 使用1个像素的填充（padding=1）来保持特征图的尺寸。
   - **最大池化层 (`self.pool`):**
     - 使用2x2的最大池化核进行池化操作。
     - 步长为2，以减小特征图的尺寸。
   - **卷积层2 (`self.conv2`):**
     - 输入通道数为64，输出通道数为128。
     - 使用3x3的卷积核进行卷积操作。
     - 使用ReLU激活函数进行非线性变换。
     - 使用1个像素的填充（padding=1）来保持特征图的尺寸。
   - **全连接层1 (`self.fc1`):**
     - 输入特征的大小为128x8x8，因为经过两次最大池化，每次都减小了图像尺寸。
     - 输出大小为512。
     - 使用ReLU激活函数进行非线性变换。
   - **全连接层2 (`self.fc2`):**
     - 输入大小为512，输出大小为10，对应于CIFAR-10数据集中的10个类别。
   - **前向传播 (`forward` 方法):**
     - 输入通过卷积层、池化层和全连接层进行前向传播。
     - 最后一层输出未经过激活函数，因为在训练时一般会使用交叉熵损失函数，该函数包含了softmax操作。

   ```py
   # 定义ConvNet模型
   class ConvNet(nn.Module):
       def __init__(self):
           super(ConvNet, self).__init__()
           self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
           self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
           self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
           self.fc1 = nn.Linear(128 * 8 * 8, 512)
           self.fc2 = nn.Linear(512, 10)
   
       def forward(self, x):
           x = self.pool(torch.relu(self.conv1(x)))
           x = self.pool(torch.relu(self.conv2(x)))
           x = x.view(x.size(0), -1)
           x = torch.relu(self.fc1(x))
           x = self.fc2(x)
           return x
   ```

4. 多层卷积神经网络模型

   该模型相比上一个模型，多了一个卷积层，拥有3个卷基层。

   - **卷积层1 (`self.conv1`):**
     - 输入通道数为3（RGB图像），输出通道数为64。
     - 使用3x3的卷积核进行卷积操作。
     - 使用ReLU激活函数进行非线性变换。
     - 使用1个像素的填充（padding=1）来保持特征图的尺寸。
   - **卷积层2 (`self.conv2`):**
     - 输入通道数为64，输出通道数为128。
     - 使用3x3的卷积核进行卷积操作。
     - 使用ReLU激活函数进行非线性变换。
     - 使用1个像素的填充（padding=1）来保持特征图的尺寸。
   - **卷积层3 (`self.conv3`):**
     - 输入通道数为128，输出通道数为256。
     - 使用3x3的卷积核进行卷积操作。
     - 使用ReLU激活函数进行非线性变换。
     - 使用1个像素的填充（padding=1）来保持特征图的尺寸。
   - **最大池化层 (`self.pool`):**
     - 使用2x2的最大池化核进行池化操作。
     - 步长为2，以减小特征图的尺寸。
   - **全连接层1 (`self.fc1`):**
     - 输入特征的大小为4096，因为经过了三次最大池化，每次都减小了图像尺寸。
     - 输出大小为512。
     - 使用ReLU激活函数进行非线性变换。
   - **全连接层2 (`self.fc2`):**
     - 输入大小为512，输出大小为10，对应于CIFAR-10数据集中的10个类别。
   - **前向传播 (`forward` 方法):**
     - 输入通过卷积层、池化层和全连接层进行前向传播。
     - 最后一层输出未经过激活函数，因为在训练时一般会使用交叉熵损失函数，该函数包含了softmax操作。

   ```py
   class ConvNet_3layers(nn.Module):
       def __init__(self):
           super(ConvNet_3layers, self).__init__()
           self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
           self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
           self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
           self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
           self.fc1 = nn.Linear(4096, 512)
           self.fc2 = nn.Linear(512, 10)
   
       def forward(self, x):
           x = self.pool(torch.relu(self.conv1(x)))
           x = self.pool(torch.relu(self.conv2(x)))
           x = self.pool(torch.relu(self.conv3(x)))
           x = x.view(x.size(0), -1)
           x = torch.relu(self.fc1(x))
           x = self.fc2(x)
           return x
   ```

5. 少核卷积神经网络模型

   该模型相比以上卷积神经网络，卷积核数减半，从`3-64-128-256`变为`3-32-64-128`。

   - **卷积层1 (`self.conv1`):**
     - 输入通道数为3（RGB图像），输出通道数为32。
     - 使用3x3的卷积核进行卷积操作。
     - 使用ReLU激活函数进行非线性变换。
     - 使用1个像素的填充（padding=1）来保持特征图的尺寸。
   - **卷积层2 (`self.conv2`):**
     - 输入通道数为32，输出通道数为64。
     - 使用3x3的卷积核进行卷积操作。
     - 使用ReLU激活函数进行非线性变换。
     - 使用1个像素的填充（padding=1）来保持特征图的尺寸。
   - **卷积层3 (`self.conv3`):**
     - 输入通道数为64，输出通道数为128。
     - 使用3x3的卷积核进行卷积操作。
     - 使用ReLU激活函数进行非线性变换。
     - 使用1个像素的填充（padding=1）来保持特征图的尺寸。
   - **最大池化层 (`self.pool`):**
     - 使用2x2的最大池化核进行池化操作。
     - 步长为2，以减小特征图的尺寸。
   - **全连接层1 (`self.fc1`):**
     - 输入特征的大小为2048，因为经过了三次最大池化，每次都减小了图像尺寸。
     - 输出大小为512。
     - 使用ReLU激活函数进行非线性变换。
   - **全连接层2 (`self.fc2`):**
     - 输入大小为512，输出大小为10，对应于CIFAR-10数据集中的10个类别。
   - **前向传播 (`forward` 方法):**
     - 输入通过卷积层、池化层和全连接层进行前向传播。
     - 最后一层输出未经过激活函数，因为在训练时一般会使用交叉熵损失函数，该函数包含了softmax操作。

   ```py
   class ConvNet_4(nn.Module):
       def __init__(self):
           super(ConvNet_4, self).__init__()
           self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
           self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
           self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
           self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
           self.fc1 = nn.Linear(2048, 512)
           self.fc2 = nn.Linear(512, 10)
   
       def forward(self, x):
           x = self.pool(torch.relu(self.conv1(x)))
           x = self.pool(torch.relu(self.conv2(x)))
           x = self.pool(torch.relu(self.conv3(x)))
           x = x.view(x.size(0), -1)
           x = torch.relu(self.fc1(x))
           x = self.fc2(x)
           return x
   ```

6. 建立包含BN层的卷积神经网络模型

   该模型在2层卷积层后，添加了层批量归一化层。

   - **卷积层1 (`self.conv1`):**
     - 输入通道数为3（RGB图像），输出通道数为64。
     - 使用3x3的卷积核进行卷积操作。
     - 使用1个像素的填充（padding=1）来保持特征图的尺寸。
     - 卷积后通过批归一化层 (`self.bn1`) 进行规范化。
     - 使用ReLU激活函数进行非线性变换。
   - **最大池化层 (`self.pool`):**
     - 使用2x2的最大池化核进行池化操作。
     - 步长为2，以减小特征图的尺寸。
   - **卷积层2 (`self.conv2`):**
     - 输入通道数为64，输出通道数为128。
     - 使用3x3的卷积核进行卷积操作。
     - 使用1个像素的填充（padding=1）来保持特征图的尺寸。
     - 卷积后通过批归一化层 (`self.bn2`) 进行规范化。
     - 使用ReLU激活函数进行非线性变换。
   - **全连接层1 (`self.fc1`):**
     - 输入特征的大小为128 * 8 * 8，因为经过了两次最大池化，每次都减小了图像尺寸。
     - 输出大小为512。
     - 全连接层后通过批归一化层 (`self.bn3`) 进行规范化。
     - 使用ReLU激活函数进行非线性变换。
   - **全连接层2 (`self.fc2`):**
     - 输入大小为512，输出大小为10，对应于CIFAR-10数据集中的10个类别。
   - **前向传播 (`forward` 方法):**
     - 输入通过卷积层、池化层和全连接层进行前向传播。
     - 批归一化层被嵌入在激活函数之前，有助于提高训练稳定性和加速收敛。
     - 最后一层输出未经过激活函数，因为在训练时一般会使用交叉熵损失函数，该函数包含了softmax操作。

   ```py
   # 定义卷积神经网络模型（包含BN层）
   class ConvNetBN(nn.Module):
       def __init__(self):
           super(ConvNetBN, self).__init__()
           self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
           self.bn1 = nn.BatchNorm2d(64)
           self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
           self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
           self.bn2 = nn.BatchNorm2d(128)
           self.fc1 = nn.Linear(128 * 8 * 8, 512)
           self.bn3 = nn.BatchNorm1d(512)
           self.fc2 = nn.Linear(512, 10)
   
       def forward(self, x):
           x = self.pool(torch.relu(self.bn1(self.conv1(x))))
           x = self.pool(torch.relu(self.bn2(self.conv2(x))))
           x = x.view(x.size(0), -1)
           x = torch.relu(self.bn3(self.fc1(x)))
           x = self.fc2(x)
           return x
   ```

### 2.3 模型训练

1. 建立训练函数

   本次实验采用了两种训练函数，一种是普通的训练函数，另一种是采用早停法的训练函数。

   - `train(model, optimizer, criterion, num_epochs)`：训练&测试函数，根据输入的模型`model`、优化器`optimizer`以及损失函数`criterion`进行指定轮次`num_epochs`的训练。训练后，在测试集上进行测试，并将训练集、测试集上该轮次的准确率和损失记录，并返回。

     ```py
     # 训练函数
     def train(model, optimizer, criterion, num_epochs):
         train_losses = []
         train_accuracies = []
         test_losses = []
         test_accuracies = []
     
         model.to(device)
         
         for epoch in range(num_epochs):
             model.train()
             running_loss = 0.0
             correct_train = 0
             total_train = 0
     
             for inputs, labels in train_loader:
                 inputs, labels = inputs.to(device), labels.to(device)
     
                 optimizer.zero_grad()
                 outputs = model(inputs)
                 loss = criterion(outputs, labels)
                 loss.backward()
                 optimizer.step()
     
                 running_loss += loss.item()
     
                 _, predicted = torch.max(outputs.data, 1)
                 total_train += labels.size(0)
                 correct_train += (predicted == labels).sum().item()
     
             train_accuracy = correct_train / total_train
             train_losses.append(running_loss / len(train_loader))
             train_accuracies.append(train_accuracy)
     
             # 在测试集上评估模型
             model.eval()
             running_loss = 0.0
             correct_test = 0
             total_test = 0
     
             with torch.no_grad():
                 for inputs, labels in test_loader:
                     inputs, labels = inputs.to(device), labels.to(device)
     
                     outputs = model(inputs)
                     loss = criterion(outputs, labels)
                     running_loss += loss.item()
     
                     _, predicted = torch.max(outputs.data, 1)
                     total_test += labels.size(0)
                     correct_test += (predicted == labels).sum().item()
     
             test_accuracy = correct_test / total_test
             test_losses.append(running_loss / len(test_loader))
             test_accuracies.append(test_accuracy)
     
             print(f'Epoch {epoch + 1}/{num_epochs}, '
                   f'Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracy:.4f}, '
                   f'Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_accuracy:.4f}')
     
         return train_losses, train_accuracies, test_losses, test_accuracies
     ```

   - `train_with_early_stopping(model, optimizer, criterion, num_epochs, patience=3)`：采用<font color='red'>早停法</font>的训练函数，输出参数里比以上函数多了个`patience`，代表若训练时若测试集上的loss连续3次都未降低则停止训练。

     ```py
     def train_with_early_stopping(model, optimizer, criterion, num_epochs, patience=3):
         train_losses = []
         train_accuracies = []
         test_losses = []
         test_accuracies = []
     
         model.to(device)
     
         best_validation_loss = float('inf')  # 初始最佳验证集损失为正无穷
         early_stopping_counter = 0  # 连续未减小的epoch计数器
     
         for epoch in range(num_epochs):
             model.train()
             running_loss = 0.0
             correct_train = 0
             total_train = 0
     
             for inputs, labels in train_loader:
                 inputs, labels = inputs.to(device), labels.to(device)
     
                 optimizer.zero_grad()
                 outputs = model(inputs)
                 loss = criterion(outputs, labels)
                 loss.backward()
                 optimizer.step()
     
                 running_loss += loss.item()
     
                 _, predicted = torch.max(outputs.data, 1)
                 total_train += labels.size(0)
                 correct_train += (predicted == labels).sum().item()
     
             train_accuracy = correct_train / total_train
             train_losses.append(running_loss / len(train_loader))
             train_accuracies.append(train_accuracy)
     
             # 在验证集上评估模型
             model.eval()
             running_loss = 0.0
             correct_test = 0
             total_test = 0
     
             with torch.no_grad():
                 for inputs, labels in test_loader:
                     inputs, labels = inputs.to(device), labels.to(device)
     
                     outputs = model(inputs)
                     loss = criterion(outputs, labels)
                     running_loss += loss.item()
     
                     _, predicted = torch.max(outputs.data, 1)
                     total_test += labels.size(0)
                     correct_test += (predicted == labels).sum().item()
     
                 test_accuracy = correct_test / total_test
                 test_losses.append(running_loss / len(test_loader))
                 test_accuracies.append(test_accuracy)
     
                 print(f'Epoch {epoch + 1}/{num_epochs}, '
                       f'Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracy:.4f}, '
                       f'Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_accuracy:.4f}')
     
                 # 判断是否进行早停
                 if test_losses[-1] < best_validation_loss:
                     best_validation_loss = test_losses[-1]
                     early_stopping_counter = 0
                 else:
                     early_stopping_counter += 1
     
                 if early_stopping_counter >= patience:
                     print(f'Validation loss has not decreased for {patience} consecutive epochs. Early stopping...')
                     break
     
         return train_losses, train_accuracies, test_losses, test_accuracies
     ```

2. 绘制曲线函数

   用于模型训练后，生成模型在训练集、测试集的损失和准确率曲线。

   ```py
   def drawlines(train_losses, train_accuracies, test_losses, test_accuracies):
       # 绘制损失和准确率曲线
       plt.figure(figsize=(10, 4))
       plt.subplot(1, 2, 1)
       plt.plot(train_losses, label='Train Loss')
       plt.plot(test_losses, label='Test Loss')
       plt.title(f'Loss Curve')
       plt.xlabel('Epochs')
       plt.ylabel('Loss')
       plt.legend()
   
       plt.subplot(1, 2, 2)
       plt.plot(train_accuracies, label='Train Accuracy')
       plt.plot(test_accuracies, label='Test Accuracy')
       plt.title(f'Accuracy Curve')
       plt.xlabel('Epochs')
       plt.ylabel('Accuracy')
       plt.legend()
   ```

3. 各类别测试函数

   用于测试模型再各个类别上的准确率，并返回分类准确率列表。

   ```py
   def ClassTest(model):
       model.to(device)
       model.eval()
   
       class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
   
       class_correct = list(0. for i in range(10))
       class_total = list(0. for i in range(10))
       for imgs, labels in test_loader:
           imgs, labels = imgs.to(device), labels.to(device)
           outputs = model(imgs)
           _, preds = torch.max(outputs, 1)
           c = (preds == labels)
           c = c.squeeze()
           for i in range(4):
               label = labels[i]
               class_correct[label] += c[i]
               class_total[label] += 1
   
       class_accuarcy=[]     
       for i in range(10):
           print(f"Accuracy of {class_names[i]:>10} : {np.round(100 * class_correct[i].detach().cpu().numpy() / class_total[i], 2)}%")
           class_accuarcy.append(np.round(100 * class_correct[i].detach().cpu().numpy() / class_total[i], 2))
       
       return class_accuarcy
   ```

#### 2.3.2 模型训练

1. **MLP模型训练**

   采用交叉熵损失函数训练，优化器采用SGD优化器，学习率为0.01，动量为0.9，训练20轮。

   ```py
   model = MLP().to(device)
   
   # 定义损失函数和优化器
   criterion = nn.CrossEntropyLoss()
   optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
   
   # 训练模型
   num_epochs = 20
   train_losses_mlp, train_accuracies_mlp, test_losses_mlp, test_accuracies_mlp = train(model, optimizer, criterion, num_epochs)
   
   drawlines(train_losses_mlp, train_accuracies_mlp, test_losses_mlp, test_accuracies_mlp)
   ```

   ![CNN_35_1](https://raw.githubusercontent.com/ZzDarker/figure/main/img/CNN_35_1.png)

   由上图对比可知，MLP的训练结果明显差于同层数的卷积神经网络。

2. **MLP对比训练**

   对比不同MLP层数、激活函数、优化器对MLP网络训练的影响。以下对比实验是在未采用数据增强的数据集训练得到的，可以看出出现了明显的过拟合。

   ![png](https://raw.githubusercontent.com/ZzDarker/figure/main/img/CNN_38_1.png)

   由上图可知，虽然出现了过拟合，但是可以判断出4层，relu为激活函数，adam优化器的MLP训练结果最好。

3. **CNN模型训练**

   对比学习率对CNN模型的影响：

   - 采用交叉熵为损失函数，SGD作为优化器，学习率为0.01，动量为0.9。

     ```py
     model = ConvNet().to(device)
     
     # 定义损失函数和优化器
     criterion = nn.CrossEntropyLoss()
     optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
     
     # 训练模型
     num_epochs = 20
     train_losses_cnn1, train_accuracies_cnn1, test_losses_cnn1, test_accuracies_cnn1 = train(model, optimizer, criterion, num_epochs)
     
     drawlines(train_losses_cnn1, train_accuracies_cnn1, test_losses_cnn1, test_accuracies_cnn1)
     ```

     ![png](https://raw.githubusercontent.com/ZzDarker/figure/main/img/CNN_40_1.png)

   - 采用交叉熵为损失函数，SGD作为优化器，学习率为0.001，动量为0.9。

     ```py
     model = ConvNet().to(device)
     
     # 定义损失函数和优化器
     criterion = nn.CrossEntropyLoss()
     optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
     
     # 训练模型
     num_epochs = 20
     train_losses_cnn2, train_accuracies_cnn2, test_losses_cnn2, test_accuracies_cnn2 = train(model, optimizer, criterion, num_epochs)
     
     drawlines(train_losses_cnn2, train_accuracies_cnn2, test_losses_cnn2, test_accuracies_cnn2)
     ```

     ![png](https://raw.githubusercontent.com/ZzDarker/figure/main/img/CNN_42_1.png)

   由上面两组结果对比可知，`lr=0.01`的模型完成了拟合，并有些许过拟合；而`lr=0.001` 的模型欠拟合，前者测试集的准确率也更高。

4. **多层CNN模型训练**

   采用3层的CNN模型训练，交叉熵为损失函数，SGD作为优化器，学习率为0.01，动量为0.9；与2层的进行对比。

   ```py
   model = ConvNet_3layers().to(device)
   
   # 定义损失函数和优化器
   criterion = nn.CrossEntropyLoss()
   optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
   
   # 训练模型
   num_epochs = 20
   train_losses_cnn3, train_accuracies_cnn3, test_losses_cnn3, test_accuracies_cnn3 = train(model, optimizer, criterion, num_epochs)
   
   drawlines(train_losses_cnn3, train_accuracies_cnn3, test_losses_cnn3, test_accuracies_cnn3)
   ```

   ![png](https://raw.githubusercontent.com/ZzDarker/figure/main/img/CNN_44_1.png)

   对比可知，相比于2层卷积层，3层卷积层的卷积神经网络训练准确率更高。

5. **少核CNN模型训练**

   采用卷积核核数更少的模型训练，交叉熵为损失函数，SGD作为优化器，学习率为0.01，动量为0.9；与核数多的模型进行对比。

   ```py
   model = ConvNet_4().to(device)
   
   # 定义损失函数和优化器
   criterion = nn.CrossEntropyLoss()
   optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
   
   # 训练模型
   num_epochs = 20
   train_losses_cnn4, train_accuracies_cnn4, test_losses_cnn4, test_accuracies_cnn4 = train(model, optimizer, criterion, num_epochs)
   
   drawlines(train_losses_cnn4, train_accuracies_cnn4, test_losses_cnn4, test_accuracies_cnn4)
   ```

   ![png](https://raw.githubusercontent.com/ZzDarker/figure/main/img/CNN_46_1.png)

   相比于卷积核更多的模型`3-64-128-256`，卷积核更少的模型 `3-32-64-128` 训练得到的准确率更低。

6. **采用批归一化的卷积网络模型训练**

   采用添加批量归一化的卷积网络进行训练，交叉熵为损失函数，SGD作为优化器，学习率为0.01，动量为0.9。

   ```py
   # 批归一化
   
   # 选择模型
   model = ConvNetBN().to(device)
   
   # 定义损失函数和优化器
   criterion = nn.CrossEntropyLoss()
   optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
   
   # 训练模型
   num_epochs = 20
   train_losses_CnnBN, train_accuracies_CnnBN, test_losses_CnnBN, test_accuracies_CnnBN = train(model, optimizer, criterion, num_epochs)
   
   # 绘制曲线
   drawlines(train_losses_CnnBN, train_accuracies_CnnBN, test_losses_CnnBN, test_accuracies_CnnBN)
   ```

   ![png](https://raw.githubusercontent.com/ZzDarker/figure/main/img/CNN_49_1.png)

   相比于未进行批量归一化（BN）的卷积神经网络，该模型训练得到的准确率更高。

7. **采用早停法的ConvNetBN模型训练**

   采用了交叉熵为损失函数，SGD为优化器，学习率为0.01，动量为0.9进行训练，并设置早停步数为3。即如果3步内测试集loss未下降，则停止训练。

   ```py
   # 早停法
   
   # 选择模型
   model = ConvNetBN().to(device)
   
   # 定义损失函数和优化器
   criterion = nn.CrossEntropyLoss()
   optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
   
   # 训练模型
   num_epochs = 20
   
   # 调用训练函数
   train_losses_CnnBN_Estop, train_accuracies_CnnBN_Estop, test_losses_CnnBN_Estop, test_accuracies_CnnBN_Estop = train_with_early_stopping(model, optimizer, criterion, num_epochs)
   
   # 绘制曲线
   drawlines(train_losses_CnnBN_Estop, train_accuracies_CnnBN_Estop, test_losses_CnnBN_Estop, test_accuracies_CnnBN_Estop)
   ```

   ![png](https://raw.githubusercontent.com/ZzDarker/figure/main/img/CNN_52_1.png)

   采用早停法时，进行到12轮时就停止了，但是模型明显还是处于欠拟合的状态，应适当调高限制的步数。

8. **采用早停法+Adam优化器的ConvNetBN模型训练**

   采用了交叉熵为损失函数，Adam为优化器，学习率为0.01进行训练，并设置早停步数为3。

   ```py
   # 早停法+Adam优化器
   
   # 选择模型
   model = ConvNetBN().to(device)
   
   # 定义损失函数和优化器
   criterion = nn.CrossEntropyLoss()
   optimizer = optim.Adam(model.parameters(), lr=0.01)
   
   # 训练模型
   num_epochs = 20
   
   # 调用训练函数
   train_losses_CnnBN_Estop_Adam, train_accuracies_CnnBN_Estop_Adam, test_losses_CnnBN_Estop_Adam, test_accuracies_CnnBN_Estop_Adam = train_with_early_stopping(model, optimizer, criterion, num_epochs)
   
   # 绘制曲线
   drawlines(train_losses_CnnBN_Estop_Adam, train_accuracies_CnnBN_Estop_Adam, test_losses_CnnBN_Estop_Adam, test_accuracies_CnnBN_Estop_Adam)
   ```

   ![png](https://raw.githubusercontent.com/ZzDarker/figure/main/img/CNN_55_1.png)

   对比可知，Adam优化器在该神经网络的表现不如SGD优化器。

9. **ResNet模型训练**

   采用预训练模型 `ResNet18` 进行训练，交叉熵为损失函数，SGD为优化器，学习率为0.01，动量为0.9进行训练，训练20轮。

   ```py
   # 选择模型
   model = torchvision.models.resnet18(pretrained=True)
   num_features = model.fc.in_features
   model.fc = nn.Linear(num_features, 10)
   
   # 定义损失函数和优化器
   criterion = nn.CrossEntropyLoss()
   optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
   
   # 训练模型
   num_epochs = 20
   
   # 调用训练函数
   train_losses, train_accuracies, test_losses, test_accuracies = train(model, optimizer, criterion, num_epochs)
   
   # 绘制曲线
   drawlines(train_losses, train_accuracies, test_losses, test_accuracies)
   ```

   ![png](https://raw.githubusercontent.com/ZzDarker/figure/main/img/CNN_58_2.png)

   由上图可知，ResNet训练得到的结果明显好于自己构建的卷积网络，准确率有着大幅度的提升。

10. **添加BN层的ResNet模型训练**

    这个模型是在预训练的ResNet18基础上进行修改的，用于适应CIFAR-10图像分类任务。

    - **预训练的ResNet18:**
      - 从torchvision库中加载ResNet18的预训练模型 (`resnet18_pretrained`)。
      - 预训练模型包含卷积层、批归一化层、残差块（residual blocks）和全连接层。
    - **修改全连接层:**
      - 获取ResNet18最后一个全连接层的输入特征数量 (`num_features`)。
      - 将原始全连接层替换为适应CIFAR-10类别数的新全连接层 (`nn.Linear(num_features, 10)`)。
    - **添加Batch Normalization层:**
      - 使用`nn.Sequential`定义模型。
      - 通过`nn.AdaptiveAvgPool2d(1)`进行全局平均池化，将特征图大小调整为1x1。
      - 使用`nn.Flatten()`将特征张量展平为一维。
      - 添加`nn.BatchNorm1d(num_features)`，对全连接层的输入进行批归一化。
      - 最后添加新的全连接层 (`nn.Linear(num_features, 10)`)，用于输出CIFAR-10的类别分数。

    ```py
    # 加载预训练的ResNet18模型
    resnet18_pretrained = torchvision.models.resnet18(pretrained=True)
    
    # 修改最后的全连接层以适应CIFAR-10的类别数（10类）
    num_features = resnet18_pretrained.fc.in_features
    resnet18_pretrained.fc = nn.Linear(num_features, 10)
    
    # 添加Batch Normalization层
    model = nn.Sequential(
        resnet18_pretrained.conv1,
        resnet18_pretrained.bn1,
        resnet18_pretrained.relu,
        resnet18_pretrained.layer1,
        resnet18_pretrained.layer2,
        resnet18_pretrained.layer3,
        resnet18_pretrained.layer4,
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),  # 将特征张量展平
        nn.BatchNorm1d(num_features),  # Batch Normalization层
        nn.Linear(num_features, 10),  # 新的全连接层
    )
    
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer_1 = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer_2 = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # 调用训练函数
    train_losses_1, train_accuracies_1, test_losses_1, test_accuracies_1 = train(model, optimizer_1, criterion, num_epochs=10)
    train_losses_2, train_accuracies_2, test_losses_2, test_accuracies_2 = train(model, optimizer_2, criterion, num_epochs=10)
    
    # 绘制曲线
    drawlines(train_losses_1+train_losses_2, train_accuracies_1+train_accuracies_2, test_losses_1+test_losses_2, test_accuracies_1+test_accuracies_2)
    ```

    该模型训练分成两个部分，前10个轮次，采用学习率为0.01的SGD进行训练；后10个轮次，采用学习率为0.001的SGD进行训练。

    ![png](https://raw.githubusercontent.com/ZzDarker/figure/main/img/CNN_60_1.png)

    由图可知，改模型训练准确率有着明显提升，测试集准确率首次上了90%。

11. **采用Adam作为优化器的ResNet_BN模型训练**

    将SGD优化器，替换为Adam优化器，损失函数仍未交叉熵损失。

    ```py
    # 加载预训练的ResNet18模型
    resnet18_pretrained = torchvision.models.resnet18(pretrained=True)
    
    # 修改最后的全连接层以适应CIFAR-10的类别数（10类）
    num_features = resnet18_pretrained.fc.in_features
    resnet18_pretrained.fc = nn.Linear(num_features, 10)
    
    # 添加Batch Normalization层
    model = nn.Sequential(
        resnet18_pretrained.conv1,
        resnet18_pretrained.bn1,
        resnet18_pretrained.relu,
        resnet18_pretrained.layer1,
        resnet18_pretrained.layer2,
        resnet18_pretrained.layer3,
        resnet18_pretrained.layer4,
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),  # 将特征张量展平
        nn.BatchNorm1d(num_features),  # Batch Normalization层
        nn.Linear(num_features, 10),  # 新的全连接层
    )
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    # 训练模型
    num_epochs = 20
    
    # 调用训练函数
    train_losses, train_accuracies, test_losses, test_accuracies = train(model, optimizer, criterion, num_epochs)
    
    # 绘制曲线
    drawlines(train_losses, train_accuracies, test_losses, test_accuracies)
    ```

    ![png](https://raw.githubusercontent.com/ZzDarker/figure/main/img/CNN_62_1.png)

    采用Adam作为优化器的训练结果，不如采用SGD作为优化器的训练结果。

### 2.4 模型对比

1. 对比各个模型的能力

   分别对比了以下上面训练果的模型的 **<font color='red'>训练集与测试集</font>** 上的<font color='red'> **损失和准确率**</font> 。

   - MLP模型：SGD优化器
   - CNN模型：2层，SGD优化器，lr=0.01
   - CNN模型：2层，SGD优化器，lr=0.001
   - CNN模型：3层，SGD优化器，lr=0.01
   - CNN模型：3层，SGD优化器，lr=0.01，卷积核减半
   - CNN+BN模型：SGD优化器，lr=0.01
   - CNN+BN模型：SGD优化器，lr=0.01，早停法
   - CNN+BN模型：Adam优化器，早停法
   - ResNet模型：SGD优化器
   - ResNet+BN模型：SGD优化器
   - ResNet+BN模型：Adam优化器

   ```py
   epochs = list(range(1,21))
   labels=['MLP+SGD','CNN+SGD,lr=0.01','CNN+SGD,lr=0.001','CNN+SGD,3层','CNN+SGD,3层，卷积核减半',
           'CNN_BN+SGD','CNN_BN+SGD,早停法','CNN_BN+Adam,早停法','ResNet+SGD','ResNet_BN+SGD','ResNet_BN+Adam']
   
   plt.rcParams['font.family'] = ['sans-serif']
   plt.rcParams['font.sans-serif'] = ['SimHei']
   
   # 绘制 Train Loss 和 Test Loss 对比图
   plt.figure(figsize=(12, 10))
   plt.subplot(2, 2, 1)
   for i, train_loss in enumerate(train_losses_all):
       plt.plot(epochs[:len(train_loss)], train_loss, label=labels[i])
   plt.xticks(range(0, 21, 2), range(0, 21, 2))
   plt.title('Train Losses Comparison')
   plt.xlabel('Epochs')
   plt.ylabel('Loss')
   
   plt.subplot(2, 2, 2)
   for i, test_loss in enumerate(test_losses_all):
       plt.plot(epochs[:len(test_loss)], test_loss, label=labels[i])
   plt.xticks(range(0, 21, 2), range(0, 21, 2))
   plt.title('Test Losses Comparison')
   plt.xlabel('Epochs')
   plt.ylabel('Loss')
   
   # 绘制 Train Accuracy 和 Test Accuracy 对比图
   plt.subplot(2, 2, 3)
   for i, train_accuracy in enumerate(train_accuracies_all):
       plt.plot(epochs[:len(train_accuracy)], train_accuracy, label=labels[i])
   plt.xticks(range(0, 21, 2), range(0, 21, 2))
   plt.title('Train Accuracy Comparison')
   plt.xlabel('Epochs')
   plt.ylabel('Accuracy')
   
   plt.subplot(2, 2, 4)
   for i, test_accuracy in enumerate(test_accuracies_all):
       plt.plot(epochs[:len(test_accuracy)], test_accuracy, label=labels[i])
   plt.xticks(range(0, 21, 2), range(0, 21, 2))
   plt.title('Test Accuracy Comparison')
   plt.xlabel('Epochs')
   plt.ylabel('Accuracy')
   plt.legend()
   
   plt.tight_layout()
   plt.show()
   ```

   ![png](https://raw.githubusercontent.com/ZzDarker/figure/main/img/CNN_65_0.png)

   结果如上图所示，明显可知`ResNet+BN模型：SGD优化器`的结果最好，而`MLP`的结果最差。

2. 对比模型在各个类别上面的准确率

   ```py
   # 类别名称
   class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
   
   # 绘制折线图
   plt.figure(figsize=(10, 6))
   
   for i in range(len(class_accuarcy_all)):
       plt.plot(class_names, class_accuarcy_all[i], label=labels[i])
   
   plt.xlabel('Categories')
   plt.ylabel('Accuracy(%)')
   plt.title('Classification Accuracy for Different Models')
   
   plt.legend()
   plt.show()
   ```

   ![png](https://raw.githubusercontent.com/ZzDarker/figure/main/img/CNN_66_0.png)

   由上图可知，模型普遍对猫、狗的识别准确率较低，而对飞机、汽车以及青蛙的识别准确率较高。表现最好的模型仍是`ResNet_BN+SGD`，最差的仍是`MLP`模型。



