import torch
from torch.optim import lr_scheduler
from network import Generator, NLayerDiscriminator
import torch.nn.functional as F
import numpy as np
from torch.nn import init

def correlation(x, y):
    x = x - torch.mean(x, 1, keepdim=True)
    y = y - torch.mean(y, 1, keepdim=True)
    x = F.normalize(x)
    y = F.normalize(y)
    return torch.mean(torch.sum(x * y, 1, keepdim=True))


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

class GANModel():
    def __init__(self, bsize):
        self.bsize = bsize
        self.device = 'cuda:0'
        self.G = Generator().cuda()
        self.D = NLayerDiscriminator(3).cuda()
        init_weights(self.G)
        init_weights(self.D)
        #self.G = torch.nn.DataParallel(self.G).cuda()
        #self.D = torch.nn.DataParallel(self.D).cuda()
        #self.zmap = torch.nn.DataParallel(self.zmap).cuda()
        self.optimizers = []
        self.optimizer_G = torch.optim.Adam(self.G.parameters(),
                                            lr=2e-4, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.D.parameters(),
                                            lr=2e-4, betas=(0.5, 0.999))

        
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)

        self.metric = None
        self.reg_param = 10.
        def lambda_rule(iteration):
            if iteration < 100000:
                lr_l = 1
            if 100000 <= iteration and iteration < 500000:
                lr_l = 0.5
            if iteration >= 500000:
                lr_l = 0.5 * (1 - (iteration - 500000)/ float(1000000 - 500000 + 1))
            return lr_l
        self.schedulers = [lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule) for optimizer in self.optimizers]
        

    def set_input(self, input, test=False):
       
        
        data, label = input
        if test:
            data = torch.cat([data, data], dim=0)
        bs, _, _, _ = data.shape
        #z = np.random.uniform(-1, 1, (bs, 8)).astype(np.float32)
        #z = torch.tensor(z)
        z = torch.randn(bs, 8)
        #print(z.shape)
        self.z = z.to(self.device)
        self.real = data.to(self.device)
        self.label = label.to(self.device)

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
                    

    def forward(self, test=False):

        self.fake, mu1, mu2, mu3 = self.G(self.real, self.z)
        if not test:
            with torch.no_grad():
                mu1 = mu1.mean(dim=0, keepdim=True)
                mu2 = mu2.mean(dim=0, keepdim=True)
                mu3 = mu3.mean(dim=0, keepdim=True)
                momentum = 0.9
                self.G.G.emu1.mu *= momentum
                self.G.G.emu1.mu += mu1 * (1 - momentum)
                self.G.G.emu2.mu *= momentum
                self.G.G.emu2.mu += mu2 * (1 - momentum)
                self.G.G.emu3.mu *= momentum
                self.G.G.emu3.mu += mu3 * (1 - momentum)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def backward_D(self):
        
        real = self.label
        outs1= self.D(real)

        fake = self.fake.detach()
        outs0 = self.D(fake)
        

        # alpha = torch.rand(real.size(0), 1, 1, 1).to(self.device)
        # x_hat = (alpha * real.data + (1 - alpha) * fake.data).requires_grad_(True)
        # out_src= self.D(x_hat)
        # d_loss_gp = self.gradient_penalty(out_src, x_hat)

        self.loss_D = torch.mean((outs1 - 0.9)**2) + torch.mean(outs0**2) #+ 10 * d_loss_gp
        self.loss_D.backward()



    def backward_G(self):
        fake = self.fake
        outs0 = self.D(fake)
        self.loss_G = torch.mean((outs0 - 0.9)**2)
        self.loss_G.backward()


    def optimize_parametersD(self):
        self.forward()
        
        self.set_requires_grad(self.D, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        
    def optimize_parametersG(self):
        self.forward()
        self.set_requires_grad(self.D, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
               
        

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step(self.metric)
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)
    def test(self, input):
        z = torch.randn(self.bsize, 10)
        self.z = z.to(self.device)
        data = input
        self.real = data.to(self.device)


if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim import lr_scheduler
    from torchvision.models import AlexNet
    import matplotlib.pyplot as plt

    params = nn.Parameter(torch.randn(1))
    optimizer = optim.SGD(params=[params], lr=2e-4)
    def lambda_rule(iteration):
        if iteration < 100000:
            lr_l = 1
        if 100000 <= iteration and iteration < 500000:
            lr_l = 0.5
        if iteration >= 500000:
            lr_l = 0.5 * (1 - (iteration - 500000)/ float(1000000 - 500000 + 1))
        return lr_l
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda= lambda_rule)
    plt.figure()
    x = list(range(1000000))
    y = []
    for epoch in range(1000000):
        scheduler.step()
        lr = scheduler.get_lr()
        y.append(scheduler.get_lr()[0])

    plt.plot(x, y)
    plt.savefig("./lr.jpg")
