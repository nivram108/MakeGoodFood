
import argparse
import torch
import torchvision
import torchvision.utils as vutils
import torch.nn as nn
from torch.autograd import Variable
 
from model import NetD, NetG
 
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=100)
parser.add_argument('--imageSize', type=int, default=96)
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--epoch', type=int, default=999, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--data_path', default='data2/', help='folder to train data')
parser.add_argument('--outf', default='imgs4', help='folder to output images and model checkpoints')
opt = parser.parse_args()

# 214  215  219
# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# 图像读入与预处理
transforms = torchvision.transforms.Compose([
    torchvision.transforms.Scale((opt.imageSize,opt.imageSize)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])
 
dataset = torchvision.datasets.ImageFolder(opt.data_path, transform=transforms)
 
dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    drop_last=True,
)
 
netG = NetG(opt.ngf, opt.nz).to(device)
netD = NetD(opt.ndf).to(device)
 
criterion = nn.BCELoss()
optimizerG = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerD = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
 
for epoch in range(1, opt.epoch + 1):
    for i, (imgs, _) in enumerate(dataloader):  # 每次epoch，遍历所有图片，共800个batch
 
        # 1,固定生成器G，训练鉴别器D
        real_label = Variable(torch.ones(opt.batchSize)).cuda()
        fake_label = Variable(torch.zeros(opt.batchSize)).cuda()
 
        netD.zero_grad()
 
        # 让D尽可能的把真图片判别为1
        real_imgs = Variable(imgs.to(device))
        real_output = netD(real_imgs)
        d_real_loss = criterion(real_output, real_label)
        real_scores = real_output
        # d_real_loss.backward()  # compute/store gradients, but don't change params
 
        # 让D尽可能把假图片判别为0
        noise = Variable(torch.randn(opt.batchSize, opt.nz, 1, 1)).to(device)
        noise = noise.to(device)
        fake_imgs = netG(noise)  # 生成假图
        fake_output = netD(fake_imgs.detach())  # 避免梯度传到G，因为G不用更新, detach分离
        d_fake_loss = criterion(fake_output, fake_label)
        fake_scores = fake_output
        # d_fake_loss.backward()
 
        d_total_loss = d_fake_loss + d_real_loss
        netG.zero_grad()
        d_total_loss.backward()  # 反向传播，计算梯度
        optimizerD.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()
 
        # 2,固定鉴别器D，训练生成器G
        fake_output = netD(fake_imgs)
        g_fake_loss = criterion(fake_output, real_label)
        g_fake_loss.backward()  # 反向传播，计算梯度
        optimizerG.step()  # 梯度信息来更新网络的参数，Only optimizes G's parameters
 
        print('[%d/%d][%d/%d] real_scores: %.3f fake_scores %.3f'
              % (epoch, opt.epoch, i, len(dataloader), real_scores.data.mean(), fake_scores.data.mean()))
        if i % 100 == 0:
            vutils.save_image(fake_imgs.data,
                              '%s/fake_samples_epoch_%03d_batch_i_%03d.png' % (opt.outf, epoch, i),
                              normalize=True)
 
    # vutils.save_image(fake.data,
    #                   '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
    #                   normalize=True)
    torch.save(netG.state_dict(), '%s/netG_%03d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_%03d.pth' % (opt.outf, epoch))