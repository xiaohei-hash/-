# -基于神经网络的图像分类实验
项目简介
本项目实现了一个完整的基于卷积神经网络（CNN）的图像分类实验系统。通过构建、训练和评估深度学习模型，对手写数字（MNIST）和彩色图像（CIFAR-10）进行自动分类识别。

实验环境
硬件要求
CPU：支持AVX指令集（推荐）

GPU：NVIDIA GPU + CUDA（可选，用于加速训练）

内存：建议8GB以上

软件要求
Python 3.8+

TensorFlow 2.10+

NumPy

Matplotlib

Scikit-learn

Seaborn

安装依赖
bash
pip install tensorflow==2.10 numpy matplotlib scikit-learn seaborn
数据集说明
MNIST 手写数字数据集
内容：0-9的手写数字灰度图像

图像尺寸：28×28 像素

训练集：60,000张

测试集：10,000张

颜色通道：1（灰度）

CIFAR-10 彩色图像数据集
类别：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车

图像尺寸：32×32 像素

训练集：50,000张

测试集：10,000张

颜色通道：3（RGB）

模型架构
卷积神经网络结构
text
输入层 (28×28×1 或 32×32×3)
    ↓
Conv2D (32个3×3卷积核, ReLU) + BatchNormalization
    ↓
MaxPooling2D (2×2)
    ↓
Conv2D (64个3×3卷积核, ReLU) + BatchNormalization
    ↓
MaxPooling2D (2×2)
    ↓
Conv2D (64个3×3卷积核, ReLU) + BatchNormalization
    ↓
Flatten
    ↓
Dense (128, ReLU)
    ↓
Dropout (0.5)
    ↓
Dense (10, Softmax)
模型参数
优化器：Adam

损失函数：稀疏分类交叉熵

评估指标：准确率

学习率：0.001（可调）

实验流程
阶段1：环境验证
检查TensorFlow和NumPy版本

检测GPU可用性

阶段2：数据加载与预处理
数据加载与划分（训练/验证/测试）

像素值归一化 [0,1]

维度调整适配CNN输入

阶段3：模型构建与编译
搭建CNN架构

配置优化器和损失函数

阶段4：模型训练
批量大小：64

训练轮次：20

回调函数：

早停（EarlyStopping）

学习率衰减（ReduceLROnPlateau）

模型检查点（ModelCheckpoint）

阶段5：性能评估
测试集准确率评估

混淆矩阵分析

错误样本可视化

分类报告生成

运行方法
基础运行（使用MNIST数据集）
bash
python image_classification.py
切换至CIFAR-10数据集
在 main() 函数中修改：

python
DATASET = 'cifar10'  # 改为'cifar10'
运行优化对比实验
取消 main() 函数中最后几行的注释：

python
compare_optimizations(x_train, y_train, x_val, y_val, x_test, y_test,
                      input_shape, num_classes, class_names)
实验结果
MNIST 预期结果
测试准确率：>99%

训练时间：约5-10分钟（CPU）

CIFAR-10 预期结果
测试准确率：约70-75%

训练时间：约15-30分钟（CPU）

输出文件
文件名	说明
{dataset}_training_history.png	训练/验证准确率和损失曲线
{dataset}_confusion_matrix.png	分类结果混淆矩阵
{dataset}_error_samples.png	分类错误样本可视化
best_model.h5	验证集最佳模型
{dataset}_final_model.h5	最终训练完成的模型
优化实验
程序支持三种配置对比实验：

无Dropout：基础模型，观察过拟合情况

Dropout(0.5)：添加Dropout正则化

学习率0.0001：较低学习率对收敛的影响

代码结构
text
image_classification.py
├── check_environment()          # 环境检查
├── load_and_preprocess_data()   # 数据加载与预处理
├── create_cnn_model()           # 模型构建
├── compile_model()              # 模型编译
├── train_model()                # 模型训练
├── plot_training_history()      # 训练曲线绘制
├── evaluate_model()             # 模型评估
├── plot_confusion_matrix()      # 混淆矩阵绘制
├── visualize_errors()           # 错误样本可视化
├── compare_optimizations()      # 优化对比实验
└── main()                       # 主函数
常见问题
Q1: 提示缺少模块怎么办？
A: 使用 pip install 模块名 安装缺失的依赖包。

Q2: 训练速度很慢？
A:

检查GPU是否可用

减小 BATCH_SIZE

减少 EPOCHS

Q3: 中文显示异常？
A: 系统缺少中文字体，可以：

安装SimHei字体

或修改 plt.rcParams 中的字体设置

Q4: CIFAR-10准确率低？
A:

增加训练轮次

调整学习率

尝试更深的网络结构

扩展建议
数据增强：添加随机旋转、平移、翻转等操作

网络改进：尝试ResNet、VGG等预训练模型

超参数调优：使用Grid Search或Random Search

可视化：添加特征图可视化、注意力机制
