import torch
import torch.nn as nn
import torch.nn.init as init

from src.modules.base_generator import GeneratorAbstract
from src.utils.torch_utils import Activation, autopad

class Fire(nn.Module):
    '''A class as base module for squeezenet

    https://github.com/pytorch/vision/blob/main/torchvision/models/squeezenet.py

    '''
    def __init__(self,
                 inplanes: int,
                 squeeze_planes: int,
                 expand1x1_planes: int,
                 expand3x3_planes: int) -> None:
        super().__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat(
            [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        )

class FireGenerator(GeneratorAbstract):
    '''A class as base module generator for squeezenet

    https://github.com/pytorch/vision/blob/main/torchvision/models/squeezenet.py

    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) # args example [16, 64, 64]

    @property
    def out_channel(self) -> int:
        return int(self.args[1] + self.args[2]) # caution => out_channel = expand1x1 channel + expand3x3 channel

    @property
    def base_module(self) -> nn.Module:
        return getattr(__import__("src.modules", fromlist=[""]), self.name)

    def __call__(self, repeat: int = 1):
        args = [self.in_channel, *self.args] # (e.g) (32, 64, 112, 112)

        if repeat > 1:
            module = []
            for i in range(repeat):
                module.append(self.base_module(*args))
                args[0] = self.out_channel
        else:
            module = self.base_module(*args)
        return self._get_module(module)