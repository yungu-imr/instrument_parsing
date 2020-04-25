import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.segmentation_models_pytorch.base import modules as md


class PoolContourHead(nn.Module):
    '''
        2 parameters:
            if ignore the background => this is in the validation
            if do the softmax to the final result
    '''

    def __init__(self,
                 input_channel=4,
                 output_channel=7,
                 kernel_size=5,
                 activation=None,
                 onehot_convert=True,
    ):
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = 5
        self.onehot_convert = onehot_convert

        # module can not be use twice
        self.pools = []
        for i in range(1, input_channel):
            self.pools.append(nn.MaxPool2d(kernel_size=self.kernel_size, stride=1, padding=2))

    def convert2onehot(self, tensor, num_classes):
        n, h, w = tensor.size()
        one_hot = torch.zeros(n, num_classes, h, w).cuda().scatter_(1, tensor.view(n, 1, h, w), 1)
        return one_hot

    def forward(self, x):
        #
        if len(x.size()) < 4:
            x = self.convert2onehot(x, self.input_channel)
        # convert to one hot
        elif self.onehot_convert:
            x = torch.argmax(x, dim=1)
            x = self.convert2onehot(x, self.input_channel)
        c_shape = [x.size(0), self.output_channel, x.size(2), x.size(3)]

        # ignore the background to get the value
        pool_x = []
        for i in range(1, self.input_channel):
            pool_x.append(self.pools[i-1](x[:, i, :, :]))

        res_x = torch.zeros(c_shape, dtype=torch.float).cuda()
        idx = 0
        for j in range(1, self.input_channel):
            for k in range(j+1, self.input_channel):
                res_x[:, idx, :, :] = pool_x[j-1] * pool_x[k-1]
                idx += 1

        return res_x
