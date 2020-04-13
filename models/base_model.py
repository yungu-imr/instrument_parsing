# @Author: ygu
# @Date: 5/3/19

import os
import torch


class BaseModel(object):
    def __init__(self):
        pass
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain

        # torch.cuda.set_device(opt.gpu_ids[0])

        self.DataTensor = torch.cuda.FloatTensor if self.gpu_ids else torch.FloatTensor
        self.LabelTensor = torch.cuda.LongTensor if self.gpu_ids else torch.LongTensor

        self.save_dir = os.path.join(opt.save_model_dir, opt.name)

    def set_input(self, data):
        input,output = data
        self.input = input
        self.output = output

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        pass

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda()

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.opt.checkpoints_dir, self.opt.load_name, save_filename)
        network.load_state_dict(torch.load(save_path))

    def update_learning_rate(self, epoch):
        pass