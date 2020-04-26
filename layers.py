import torch
import torch.nn as nn
import math
from functools import partial
import torch.nn.functional as F
norm_layer = partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)

class Adap_ChannelWise_EMAU(nn.Module):
    '''The Adaptive Channel-Wise Expectation-Maximization Attention Unit (EMAU).
    Arguments:
        c (int): The input and output channel number.
        k (int): The number of the bases.
        stage_num (int): The iteration number for EM.
    '''
    def __init__(self, c, k, stage_num=3):
        super(Adap_ChannelWise_EMAU, self).__init__()
        self.stage_num = stage_num

        mu = torch.Tensor(1, k, 64*64) # Init k*n, N=w*h, w=h=64
        mu.normal_(0, math.sqrt(2. / k))    # Init with Kaiming Norm.
        mu = self._l2norm(mu, dim=2)
        self.register_buffer('mu', mu)

        self.conv1 = nn.Conv2d(c, c, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            norm_layer(c))     

        self.scale_weight = None # style code
        self.num_features = k
        self.k = k


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

 

    def forward(self, x):
        assert self.scale_weight is not None
        idn = x
        # The first 1x1 conv
        x = self.conv1(x)

        # The EM Attention
        b, c, h, w = x.size()
        x = x.view(b, c, h*w)               # b * c * n
        mu = self.mu.repeat(b, 1, 1)        # b * k * n
        with torch.no_grad():
            for i in range(self.stage_num):
                x_t = x.permute(0, 2, 1)    # b * n * c
                z = torch.bmm(mu, x_t)      # b * k * c
                z = F.softmax(z, dim=1)     # b * k * c
                z_ = z / (1e-6 + z.sum(dim=2, keepdim=True))
                mu = torch.bmm(z_, x)       # b * k * n
                mu = self._l2norm(mu, dim=2)

        z_t = z.permute(0, 2, 1)            # b * c * k
        self.scale_weight = self.scale_weight.view(-1, self.k,1)
        mu = mu * self.scale_weight         #base mu scaling with style code
        x = z_t.matmul(mu)                  # b * c * n
        x = x.view(b, c, h, w)              # b * c * h * w
        x = F.relu(x, inplace=True)

        # The second 1x1 conv
        x = self.conv2(x)
        x = x + idn
        x = F.relu(x, inplace=True)

        return x, mu

    def _l2norm(self, inp, dim):
        '''Normlize the inp tensor with l2-norm.
        Returns a tensor where each sub-tensor of input along the given dim is 
        normalized such that the 2-norm of the sub-tensor is equal to 1.
        Arguments:
            inp (tensor): The input tensor.
            dim (int): The dimension to slice over to get the ssub-tensors.
        Returns:
            (tensor) The normalized tensor.
        '''
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))

'''

SCConv2d, SCConvTranspose2d are implementations of Latent Filter Scaling

'''



class SCConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(SCConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups,
                 bias, padding_mode)
        self.scale_weight = None #style code
        self.num_features = out_channels


    def forward(self, x):
        #print(self.kernel_size, self.padding)
        assert self.scale_weight is not None
        #print('conv: ', self.scale_weight.shape)
        b, c = self.scale_weight.shape
        bx, _, _, _ = x.shape
        self.scale_weight = self.scale_weight[:bx,:]
        self.scale_weight = self.scale_weight.view(bx,c,1,1)
        h = super().forward(x)
        h = h * self.scale_weight
        return h

class SCConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1, padding_mode='zeros'):
        super(SCConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, output_padding, groups, bias,
                 dilation, padding_mode='zeros')
        self.scale_weight = None #style code
        self.num_features = out_channels
    def forward(self, x, output_size=None):
        assert self.scale_weight is not None
        b, c = self.scale_weight.shape
        self.scale_weight = self.scale_weight.view(b,c,1,1)
        h = super().forward(x, output_size)
        h = h * self.scale_weight
        return h