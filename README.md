# DCGAN

Pytorch implementation of [Deep Convolutional Generative Adversarial Networks](http://arxiv.org/abs/1511.06434) which is a stabilize Generative Adversarial Networks.

![DCGAN结构图](DCGAN.png)

## 环境准备

- 运行`pip install -r requirements.txt`安装其他依赖

## 数据准备

更好的图片生成效果更好

- 可以自己写爬虫爬取[Danbooru](http://link.zhihu.com/?target=http%3A//safebooru.donmai.us/)或者[konachan](http://konachan.net/)
- 如果你不想从头开始爬图片，可以直接使用爬好的头像数据（275M，约5万多张图片）：https://pan.baidu.com/s/1eSifHcA 提取码：g5qa
感谢知乎用户[何之源](https://www.zhihu.com/people/he-zhi-yuan-16)爬取的数据。

请把所有的图片保存于data/faces/目录下，形如

```
data/
└── faces/
    ├── 0000fdee4208b8b7e12074c920bc6166-0.jpg
    ├── 0001a0fca4e9d2193afea712421693be-0.jpg
    ├── 0001d9ed32d932d298e1ff9cc5b7a2ab-0.jpg
    ├── 0001d9ed32d932d298e1ff9cc5b7a2ab-1.jpg
    ├── 00028d3882ec183e0f55ff29827527d3-0.jpg
    ├── 00028d3882ec183e0f55ff29827527d3-1.jpg
    ├── 000333906d04217408bb0d501f298448-0.jpg
    ├── 0005027ac1dcc32835a37be806f226cb-0.jpg
```
即data目录下只有一个文件夹，文件夹中有所有的图片

## 用法

基本用法：

```
Usage: python main.py FUNCTION --key=value,--key2=value2 ..
```

- 训练

```bash
python main.py train 
```

- 生成图片

[点此](http://pytorch-1252820389.cosbj.myqcloud.com/netg_200.pth)可下载预训练好的生成模型，如果想要下载预训练的判别模型，请[点此](http://pytorch-1252820389.cosbj.myqcloud.com/netd_200.pth)

```bash
python main.py generate --netd-path=checkpoints/netd_200.pth \
		--netg-path=checkpoints/netg_200.pth
		--gen-num=64
```

#### 完整的选项及默认值:

```python
    # Data params	
    data_path = 'data/' # 数据集存放路径
    image_size = 96     # 图片尺寸
    
    # Model params
    nz = 100    # 噪声维度
    ngf = 64    # 生成器feature map数
    ndf = 64    # 判别器feature map数
    
    # Train params
    num_workers = 4                     # 多进程加载数据所用的进程数
    batch_size = 256     
    num_epochs = 200    
    d_learning_rate = 2e-4              # 判别器的学习率
    g_learning_rate = 2e-4              # 生成器的学习率
    optim_beta1=0.5                     # Adam优化器的beta1参数
    d_every = 1                         # 每1个batch训练一次判别器
    g_every = 1                         # 每1个batch训练一次生成器
    epoch_every = 10                    # 每10个epoch保存一次模型
    save_img_path = 'imgs/'             # 可视化训练过程中图片的保存路径
    checkpoints_path = 'checkpoints/'   # 训练持久模型的保存路径
    netd_path = None                    # 'checkpoints/netd_.pth' 预训练模型
    netg_path = None                    # 'checkpoints/netg_211.pth'
    

    # Generate params
    # 只测试不训练
    gen_img = 'result.png'
    # 从512张生成的图片中保存最好的64张
    gen_search_num = 512 
    gen_num = 64 
    gen_mean = 0  # 噪声的均值
    gen_std = 1   # 噪声的方差
```

#### 生成的部分图片

![result.png](result.png)

### 兼容性测试

train

- [x] GPU
- [x] CPU
- [x] Python2
- [x] Python3

test

- [x] GPU
- [x] CPU
- [x] Python2
- [x] Python3
