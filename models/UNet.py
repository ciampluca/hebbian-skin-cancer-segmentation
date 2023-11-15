import math 

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms.functional import resize

from utils import get_init_param_by_name


class UNet(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(UNet, self).__init__()
        self.net = UNetModel(
            in_channels=get_init_param_by_name('in_channels', kwargs, cfg.model, 3),
            out_channels=get_init_param_by_name('out_channels', kwargs, cfg.model, 1),
            depth=get_init_param_by_name('depth', kwargs, cfg.model, 5),
            wf=get_init_param_by_name('wf', kwargs, cfg.model, 6),
            padding=get_init_param_by_name('padding', kwargs, cfg.model, True),
            batch_norm=get_init_param_by_name('batch_norm', kwargs, cfg.model, True),
            up_mode=get_init_param_by_name('up_mode', kwargs, cfg.model, 'upconv'),
            last_bias=get_init_param_by_name('last_bias', kwargs, cfg.model, True),
            latent_sampling=get_init_param_by_name('latent_sampling', kwargs, cfg.model, False),
        )
    
    def forward(self, x):
        return self.net(x)
    
    def state_dict(self):
        return self.net.state_dict()
    
    def load_state_dict(self, state_dict, strict = ...):
        return self.net.load_state_dict(state_dict, strict)
    
    def reset_clf(self, out_channels):
        self.net.reset_clf(out_channels)
    
class UNetModel(nn.Module):

    def __init__(
            self,
            in_channels=3,
            out_channels=1,
            depth=5,
            wf=6,
            padding=True,
            batch_norm=True,
            up_mode='upconv',
            last_bias=True,
            latent_sampling=False,
    ):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation 
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        Using the default arguments will yield the exact version used
        in the original paper
        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNetModel, self).__init__()
        
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)
        
        self.latent_sampling = latent_sampling
        if self.latent_sampling:
            self.mu = nn.Conv2d(prev_channels, prev_channels, kernel_size=1)
            self.var = nn.Conv2d(prev_channels, prev_channels, kernel_size=1)
        
        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)
        
        self.last_bias = last_bias
        self.last = nn.Conv2d(prev_channels, out_channels, kernel_size=1, bias=self.last_bias)
    
    def reset_clf(self, out_channels):
        device = self.last.weight.device
        self.last = nn.Conv2d(self.last.in_channels, out_channels, kernel_size=1, bias=self.last_bias).to(device)
    
    def forward(self, x):
        h, w = x.shape[-2:]
        need_resize = (h % 32) or (w % 32)

        if need_resize:
            newH = math.ceil(h / 32) * 32
            newW = math.ceil(w / 32) * 32
            x = resize(x, (newH, newW))

        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)
        
        if self.latent_sampling:
            # Sampling
            mu = self.mu(x)
            log_var = self.var(x)
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            x =  eps * std + mu
        
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        output = self.last(x)

        if need_resize:
            output = resize(output, (h, w))
        
        if self.latent_sampling:
            output = {'mu': mu, 'log_var': log_var, 'reconstr': output}
        
        return output


class UNetConvBlock(nn.Module):
    
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        
        return out


class UNetUpBlock(nn.Module):
    
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        
        return layer[:, :, diff_y: (diff_y + target_size[0]), diff_x: (diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out



# Test code
def main():
    device = 'cpu'
    
    input = torch.rand(8, 3, 256, 256).to(device)
    model = UNetModel(out_channels=1, last_bias=False).to(device)
    print(model)
    output = model(input)

    print("Input Shape: {}, Output Shape: {}".format(input.shape, output.shape))
    print("Output: {}".format(output))


if __name__ == "__main__":
    main()
