#coding:utf8
"""
"""

import torch
from torch.autograd import Variable
import torchvision
import tqdm

from models import NetG, NetD


class Config(object):

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
    


opt = Config()
def train(**kwargs):
    for k_,v_ in kwargs.items():
        setattr(opt, k_, v_)
    
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(opt.image_size),
        torchvision.transforms.CenterCrop(opt.image_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
    dataset = torchvision.datasets.ImageFolder(opt.data_path, transform=transforms)

    dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size = opt.batch_size,
                                         shuffle = True,
                                         num_workers= opt.num_workers,
                                         drop_last=True)


    # 1、定义神经网络
    D = NetD(opt)
    G = NetG(opt)

    map_location=lambda storage, loc: storage
    if opt.netd_path:
        D.load_state_dict(torch.load(opt.netd_path, map_location = map_location)) 
    if opt.netg_path:
        G.load_state_dict(torch.load(opt.netg_path, map_location = map_location))


    # 2、定义优化器和损失
    d_optim = torch.optim.Adam(D.parameters(), opt.d_learning_rate, betas=(opt.optim_beta1, 0.999))
    g_optim = torch.optim.Adam(G.parameters(), opt.g_learning_rate, betas=(opt.optim_beta1, 0.999))
    criterion = torch.nn.BCELoss()

    # 真图片label为1，假图片label为0
    real_labels = Variable(torch.ones(opt.batch_size))
    fake_labels = Variable(torch.zeros(opt.batch_size))

    if torch.cuda.is_available():
        D.cuda()
        G.cuda()
        criterion.cuda()
        real_labels, fake_labels = real_labels.cuda(), fake_labels.cuda()
        
    # 3、可视化训练过程     
    for epoch in range(opt.num_epochs):
        for step, (images, _) in tqdm.tqdm(enumerate(dataloader)):
            
            if step % opt.d_every == 0:
                # 1、训练判别器
                d_optim.zero_grad()

                ## 尽可能的把真图片判别为正确
                d_real_data = Variable(images)
                d_real_data = d_real_data.cuda() if torch.cuda.is_available() else d_real_data
                d_real_decision = D(d_real_data)
                d_real_error = criterion(d_real_decision, real_labels)
                d_real_error.backward()

                ## 尽可能把假图片判别为错误
                d_gen_input = Variable(torch.randn(opt.batch_size, opt.nz, 1, 1))
                d_gen_input = d_gen_input.cuda() if torch.cuda.is_available() else d_gen_input
                d_fake_data = G(d_gen_input).detach()
                d_fake_decision = D(d_fake_data)
                d_fake_error = criterion(d_fake_decision, fake_labels)
                d_fake_error.backward()
                d_optim.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()

            if step % opt.g_every == 0:
                # 2、训练生成器
                g_optim.zero_grad()

                ## 尽可能让判别器把假图片判别为正确
                g_gen_input = Variable(torch.randn(opt.batch_size, opt.nz, 1, 1))
                g_gen_input = g_gen_input.cuda() if torch.cuda.is_available() else g_gen_input
                g_fake_data = G(g_gen_input)
                g_fake_decision = D(g_fake_data)
                g_fake_error = criterion(g_fake_decision, real_labels)
                g_fake_error.backward()

                g_optim.step()  

        if step % opt.epoch_every == 0:
            print("%s, %s, D: %s/%s G: %s" % (step, g_fake_decision.cpu().data.numpy().mean(), d_real_error.cpu().data[0], d_fake_error.cpu().data[0], g_fake_error.cpu().data[0]))

            # 保存模型、图片
            torchvision.utils.save_image(g_fake_data.data[:36], '%s/%s.png' %(opt.save_img_path, epoch), normalize=True, range=(-1,1))
            torch.save(D.state_dict(), '%s/netd_%s.pth' % (opt.checkpoints_path, epoch))
            torch.save(G.state_dict(), '%s/netg_%s.pth' % (opt.checkpoints_path, epoch))
            
            
def generate(**kwargs):
    '''
    随机生成动漫头像，并根据netd的分数选择较好的
    '''
    for k_,v_ in kwargs.items():
        setattr(opt, k_, v_)
    

    D = NetD(opt)
    G = NetG(opt)

    noises = torch.randn(opt.gen_search_num,opt.nz,1,1).normal_(opt.gen_mean,opt.gen_std)
    noises = Variable(noises, volatile=True)

    map_location=lambda storage, loc: storage
    D.load_state_dict(torch.load(opt.netd_path, map_location = map_location))
    G.load_state_dict(torch.load(opt.netg_path, map_location = map_location))
    
    if torch.cuda.is_available():
        D.cuda()
        G.cuda()
        noises = noises.cuda()
        
    # 生成图片，并计算图片在判别器的分数
    fake_img = G(noises)
    scores = D(fake_img).data

    # 挑选最好的某几张
    indexs = scores.topk(opt.gen_num)[1]
    result = []
    for idx in indexs:
        result.append(fake_img.data[idx])
    # 保存图片
    torchvision.utils.save_image(torch.stack(result),opt.gen_img,normalize=True,range=(-1,1))



if __name__ == '__main__':
    import fire
    fire.Fire()


