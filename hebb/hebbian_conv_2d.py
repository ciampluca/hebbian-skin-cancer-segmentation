import torch
import torch.nn as nn

from hebb.hebbian_update_rule import SoftWinnerTakesAll
from hebb.unit_types import DotUnit
from hebb.hebbian_layer_helpers import to_2dvector


class HebbianConv2d(nn.Module):
    """
    A 2d convolutional layer that learns through Hebbian plasticity
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, groups=1,
                 bias=True,
                 dilation=1, padding_mode='zeros', device=None, dtype=None,
                 unit_type=DotUnit(),
                 hebbian_update_rule=SoftWinnerTakesAll(0.02),
                 patchwise=True,
                 alpha=0):
        """

        :param out_channels: output channels of the convolutional kernel
        :param in_channels: input channels of the convolutional kernel
        :param kernel_size: size of the convolutional kernel (int or tuple)
        :param stride: stride of the convolutional kernel (int or tuple)
        :param patchwise: whether updates for each convolutional patch should be computed separately,
        and then aggregated
        :param alpha: weighting coefficient between hebbian and backprop updates (0 means fully backprop, 1 means fully hebbian).
		

        """

        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels

        self.__kernel_size = to_2dvector(kernel_size, 'kernel_size')
        self.__stride = to_2dvector(stride, 'stride')
        self.__dilation = to_2dvector(dilation, 'dilation')
        self.__padding = to_2dvector(padding, 'padding')

        if groups != 1:
            raise NotImplementedError("Groups different from 1 not implemented")

        if padding_mode != 'zeros':
            raise NotImplemented("Only padding mode zeros is supported")

        self.__weight = torch.nn.Parameter(
            data=torch.nn.init.xavier_normal_(
                torch.empty(
                    torch.Size([self.out_channels, self.in_channels * self.kernel_size[0] * self.kernel_size[1]]),
                    dtype=dtype,
                    device=device)
            ),
            requires_grad=True
        )
        self.__bias = torch.nn.Parameter(
            torch.zeros((out_channels, 1), dtype=dtype, device=device),
            requires_grad=True
        ) if bias else None

        self.register_buffer('delta_w', torch.zeros_like(self.__weight))

        self.__unit_type = unit_type
        self.__hebbian_update_rule = hebbian_update_rule
        self.__unfold = torch.nn.Unfold(self.kernel_size,
                                        self.dilation,
                                        padding=self.padding,
                                        stride=self.stride)

        self.patchwise = patchwise
        self.alpha = alpha

    def __calc_output_size(self, input_size: tuple):
        # ignore output_padding
        return torch.floor(
            (
                    torch.tensor(input_size) +
                    2 * self.__padding -
                    self.__dilation * (self.__kernel_size - 1) - 1
            ).type(torch.float32)/self.__stride + 1
        ).type(torch.int32)

    def forward(self, x):
        input_size = x.size()
        tensor_dim = len(input_size)
        if tensor_dim != 3 and tensor_dim != 4:
            raise RuntimeError(f'Expected 3D (unbatched) or 4D (batched) input to HebbianConvTransposed2D but got size'
                               f' {input_size}')
        if tensor_dim == 3:
            x = torch.unsqueeze(x, 0)

        # Calculate output shape for 3D and 4D tensors
        output_shape = torch.Size((
            x.size()[0],
            self.out_channels,
            *self.__calc_output_size(input_size[2:])
        ))

        unfolded_x = self.__unfold(x)
        if self.alpha != 0:
            unfolded_y = self.__unit_type(unfolded_x, self.__weight)

            if self.training:
                self.__update_delta_w(unfolded_x, unfolded_y)
            return unfolded_y.reshape(output_shape)
        return self.__unit_type(unfolded_x, self.__weight, self.__bias).reshape(output_shape)
    
    @ torch.no_grad()
    def __update_delta_w(self, x, y):
        if self.patchwise:
            self.delta_w[:, :] += self.__hebbian_update_rule(x, y, self.__weight)
        else:
            raise NotImplementedError("Non-patchwise learning is not implemented")
    
    @torch.no_grad()
    def local_update(self):
        """
        
        This function transfers a previously computed weight update, stored in buffer self.delta_w, to the gradient
        self.weight.grad of the weigth parameter.
        
        This function should be called before optimizer.step(), so that the optimizer will use the locally computed
        update as optimization direction. Local updates can also be combined with end-to-end updates by calling this
        function between loss.backward() and optimizer.step(). loss.backward will store the end-to-end gradient in
        self.weight.grad, and this function combines this value with self.delta_w as
        self.weight.grad = (1 - alpha) * self.weight.grad - alpha * self.delta_w
        Parameter alpha determines the scale of the local update compared to the end-to-end gradient in the combination.
        
        """
        
        if self.alpha == 0:
            return
        
        if self.__weight.grad is None:
            self.__weight.grad = - self.alpha * self.delta_w
        else:
            self.__weight.grad = (1 - self.alpha) * self.__weight.grad - self.alpha * self.delta_w

        self.delta_w.zero_()

    @property
    def kernel_size(self):
        return tuple(self.__kernel_size.tolist())

    @property
    def stride(self):
        return tuple(self.__stride.tolist())

    @property
    def dilation(self):
        return tuple(self.__dilation.tolist())

    @property
    def padding(self):
        return tuple(self.__padding.tolist())

    @property
    def weight(self):
        return torch.nn.Parameter(
            self.__weight.data.reshape(self.out_channels, self.in_channels, *self.kernel_size).clone()
        )

    @weight.setter
    def weight(self, w):
        self.__weight.data[:, :] = w.reshape(self.out_channels, self.in_channels * torch.prod(self.__kernel_size))
