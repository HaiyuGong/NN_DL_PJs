# 期中作业-任务 1：微调在 ImageNet 上预训练的卷积神经网络实现鸟类识别


## 环境配置
以下为项目所运行的环境：
```{bash}
python==3.8.10
torch==1.13.1+cu116
torchvision==0.14.1+cu116
pandas==2.0.2
matplotlib==3.7.1
tensorboard==2.13.0
```
## 数据准备
下载[CUB-200-2011]( https://data.caltech.edu/records/65de6-vp158)并解压到指定目录。

## 模型训练
在`train_val.py`文件中配置好相应的数据集路径和超参数设置，直接运行即可开始训练。

## 权重下载
不同实验设置的最优模型的权重下载链接：https://pan.baidu.com/s/12Ko4vBuW6bOp_uq_NJxWpQ?pwd=icou，下载后将权重文件放在目录`saved_models`。

## 模型测试
在`test.py`文件中配置好相应的数据集路径和需要加载的模型权重的路径，直接运行即可开始测试。

## 训练过程可视化
运行一下代码，即可运行tensorboard服务。
```bash
sh tb.bash
```