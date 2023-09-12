import math 

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms.functional import resize

from hebb.legacy import *


default_hebb_params = dict(w_nrm=True, mode='swta', k=0.02, patchwise=True, contrast=1., uniformity=False, alpha=0)

class HUNet(nn.Module):
    def __init__(self, cfg):
        super(HUNet, self).__init__()
        hebb_params = default_hebb_params
        if hasattr(cfg.model, 'hebb'):
            hebb_params = dict(
                w_nrm=getattr(cfg.model.hebb, 'w_nrm', True),
                mode=getattr(cfg.model.hebb, 'mode', 'swta'),
                k=getattr(cfg.model.hebb, 'k', 0.02),
                patchwise=getattr(cfg.model.hebb, 'patchwise', True),
                contrast=getattr(cfg.model.hebb, 'contrast', 1.),
                uniformity=getattr(cfg.model.hebb, 'uniformity', False),
                alpha=getattr(cfg.model.hebb, 'alpha', 0)
            )
        
        self.net = HUNetModel(
            in_channels=getattr(cfg.model, 'in_channels', 3),
            out_channels=getattr(cfg.model, 'out_channels', 1),
            depth=getattr(cfg.model, 'depth', 5),
            wf=getattr(cfg.model, 'wf', 6),
            padding=getattr(cfg.model, 'padding', True),
            batch_norm=getattr(cfg.model, 'batch_norm', True),
            up_mode=getattr(cfg.model, 'up_mode', 'upconv'),
            hebb_params=hebb_params,
            last_bias=getattr(cfg.model, 'last_bias', True),
        )
    
    def forward(self, x):
        return self.net(x)
    
    def local_update(self):
        for m in self.modules():
            if hasattr(m, 'local_update'): m.local_update()
    
    def state_dict(self, destination, prefix, keep_vars):
        return self.net.state_dict(destination, prefix, keep_vars)
    
    def load_state_dict(self, state_dict, strict):
        return self.net.load_state_dict(state_dict, strict)

class HUNetModel(nn.Module):
    
    def __init__(
            self,
            in_channels=3,
            out_channels=1,
            depth=5,
            wf=6,
            padding=True,
            batch_norm=True,
            up_mode='upconv',
            hebb_params=None,
            last_bias=True,
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
        super(HUNetModel, self).__init__()
        
        assert up_mode in ('upconv', 'upsample')
        self.hebb_params = hebb_params if hebb_params is not None else default_hebb_params
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                HUNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm, hebb_params=self.hebb_params)
            )
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                HUNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm, hebb_params=self.hebb_params)
            )
            prev_channels = 2 ** (wf + i)

        self.last = HebbianConv2d(prev_channels, out_channels, kernel_size=1, **self.hebb_params)
        if not last_bias: self.last.bias.requires_grad = False

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

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        output = self.last(x)

        if need_resize:
            output = resize(output, (h, w))
        
        return output


class HUNetConvBlock(nn.Module):
    
    def __init__(self, in_size, out_size, padding, batch_norm, hebb_params=None):
        super(HUNetConvBlock, self).__init__()
        block = []
        hebb_params = hebb_params if hebb_params is not None else default_hebb_params
        
        padding = int(padding)
        block.append(nn.ZeroPad2d(padding))
        block.append(HebbianConv2d(in_size, out_size, kernel_size=3, act=nn.ReLU(), **hebb_params))
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))
        block.append(nn.ZeroPad2d(padding))
        block.append(HebbianConv2d(out_size, out_size, kernel_size=3, act=nn.ReLU(), **hebb_params))
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        
        return out


class HUNetUpBlock(nn.Module):
    
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm, hebb_params=None):
        super(HUNetUpBlock, self).__init__()
        hebb_params = hebb_params if hebb_params is not None else default_hebb_params
        
        if up_mode == 'upconv':
            self.up = HebbianConvTranspose2D(in_size, out_size, kernel_size=2, stride=2, **hebb_params)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                HebbianConv2d(in_size, out_size, kernel_size=1, **hebb_params),
            )

        self.conv_block = HUNetConvBlock(in_size, out_size, padding, batch_norm, hebb_params=hebb_params)

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
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    input = torch.rand(1, 3, 32, 32).to(device)
    model = HUNetModel(out_channels=1, last_bias=False).to(device)
    print(model)
    output = model(input)

    print("Input Shape: {}, Output Shape: {}".format(input.shape, output.shape))
    print("Output: {}".format(output))


if __name__ == "__main__":
    main()
