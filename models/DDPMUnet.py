import torch
from torch import nn

from denoising_diffusion_pytorch import Unet


class DDPMUNet(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(DDPMUNet, self).__init__()

        self.net = Unet(
            dim = 64,
            dim_mults = (1, 2, 4, 8),
            flash_attn = True,
            out_dim = 3,
        )
    
    def forward(self, x):
        return self.net(x)
    
    def state_dict(self):
        return self.net.state_dict()
    
    def load_state_dict(self, state_dict, strict = ...):
        return self.net.load_state_dict(state_dict, strict)
    
    def reset_clf(self, out_channels):
        self.net.final_conv = nn.Conv2d(self.final_conv.weight.shape[1], out_channels, 1)
        
    # TODO
    #def reset_internal_grads(self):
    #    self.net.reset_internal_grads()