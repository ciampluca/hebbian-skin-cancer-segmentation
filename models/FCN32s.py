import numpy as np

import torch
import torch.nn as nn

from utils import get_init_param_by_name


class FCN32s(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(FCN32s, self).__init__()
        self.net = FCN32sModel(
            in_channels=get_init_param_by_name('in_channels', kwargs, cfg.model, 3),
            out_channels=get_init_param_by_name('out_channels', kwargs, cfg.model, 1),
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
        
    def reset_internal_grads(self):
        self.net.reset_internal_grads()

class FCN32sModel(nn.Module):
    def __init__(
            self,
            in_channels=3, 
            out_channels=1,
            latent_sampling=False,
    ):
        """
        Implementation of Fully Convolutional Networks for Semantic Segmentation (Long et al., 2015)
        (Ronneberger et al., 2015)
        https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Long_Fully_Convolutional_Networks_2015_CVPR_paper.html
        inspired from https://github.com/wkentaro/pytorch-fcn/tree/main
        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
        """

        super(FCN32sModel, self).__init__()
        # conv1
        self.conv1_1 = self._get_conv_layer(in_channels, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = self._get_conv_layer(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = self._get_conv_layer(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = self._get_conv_layer(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = self._get_conv_layer(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = self._get_conv_layer(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = self._get_conv_layer(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = self._get_conv_layer(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = self._get_conv_layer(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = self._get_conv_layer(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = self._get_conv_layer(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = self._get_conv_layer(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = self._get_conv_layer(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32
        
        self.latent_sampling = latent_sampling
        if self.latent_sampling:
            self.mu = nn.Conv2d(512, 512, kernel_size=1)
            self.var = nn.Conv2d(512, 512, kernel_size=1)
        
        # fc6
        self.fc6 = self._get_conv_layer(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = self._get_conv_layer(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = self._get_conv_layer(4096, 256, 1)
        self.upscore = nn.ConvTranspose2d(256, out_channels, 64, stride=32,
                                          bias=False)
        self.reconstr = None
        if self.latent_sampling: self.reconstr = nn.ConvTranspose2d(256, in_channels, 64, stride=32,
                                          bias=False)

        #self._initialize_weights()
    
    def _get_conv_layer(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels)
        )
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.kaiming_norm_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = self._get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    # https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
    def _get_upsampling_weight(self, in_channels, out_channels, kernel_size):
        """Make a 2D bilinear kernel suitable for upsampling"""
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:kernel_size, :kernel_size]
        filt = (1 - abs(og[0] - center) / factor) * \
            (1 - abs(og[1] - center) / factor)
        weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                        dtype=np.float64)
        weight[range(in_channels), range(out_channels), :, :] = filt
        return torch.from_numpy(weight).float()
    
    def reset_clf(self, out_channels):
        device = self.upscore.weight.device
        self.upscore = nn.ConvTranspose2d(256, out_channels, 64, stride=32, bias=False).to(device)
    
    def reset_internal_grads(self):
        for m in self.modules():
            if m != self.upscore: m.zero_grad()
    
    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)
        
        if self.latent_sampling:
            # Sampling
            mu = self.mu(h)
            log_var = self.var(h)
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            h =  eps * std + mu

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)
        reconstr = None
        if self.reconstr is not None: reconstr = self.reconstr(h)
        h = self.upscore(h)
        
        
        crop_start_x = h.shape[2] // 2 - x.shape[2] // 2
        crop_start_y = h.shape[3] // 2 - x.shape[3] // 2
        output = h[:, :, crop_start_x:crop_start_x + x.shape[2], crop_start_y:crop_start_y + x.shape[3]]
        if reconstr is not None: reconstr = reconstr[:, :, crop_start_x:crop_start_x + x.shape[2], crop_start_y:crop_start_y + x.shape[3]]
        
        if self.latent_sampling:
            output = {'output': output, 'mu': mu, 'log_var': log_var, 'reconstr': reconstr}
        
        return output



# Test code
def main():
    device = 'cpu'
    
    input = torch.rand(8, 3, 256, 256).to(device)
    model = FCN32sModel(out_channels=1).to(device)
    print(model)
    output = model(input)

    print("Input Shape: {}, Output Shape: {}".format(input.shape, output.shape))
    print("Output: {}".format(output))


if __name__ == "__main__":
    main()