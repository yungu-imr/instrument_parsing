from collections import OrderedDict
from .base_model import BaseModel
from utils.measure import DiceLoss, CEDiceLoss, general_dice, general_jaccard, ICNetLoss,BoundaryLoss
from models import get_model
from data.utils.prepare_data import class_num
import torch
import torch.nn as nn
import utils.simple_util as simple_util


class SimpleModel(BaseModel):
    def __init__(self, opt):
        super(SimpleModel, self).__init__()
        BaseModel.initialize(self, opt)

        # data
        self.input_im = None
        self.real_mask = None
        self.est_mask = None
        self.loss = None
        self.num_classes = class_num[opt.problem_type]

        # model
        self.net = get_model(opt.model)(in_channels=3, classes=class_num[opt.problem_type])
        if len(opt.gpu_ids) > 0:
            assert (torch.cuda.is_available())
            self.net = nn.DataParallel(self.net).cuda()

        self.isTrain = opt.isTrain
        self.state = 'train'

        if not self.isTrain or opt.continue_train:
            self.load_network(self.net, 'onestage_parse_net_%s' % opt.model, opt.which_epoch)

        # learning parameters
        if self.isTrain:
            self.old_lr = opt.lr
            # self.criterion = nn.MSELoss()
            if self.opt.model == 'ICNet':
                self.criterion = ICNetLoss()
            elif self.opt.problem_type == 'binary':
                self.criterion = DiceLoss(num_classes=class_num[opt.problem_type])
            elif self.opt.problem_type == 'parts':
                self.criterion = CEDiceLoss(num_classes=class_num[opt.problem_type])
                self.criterion_b = BoundaryLoss()

            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(0.5, 0.999))
            # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.opt.decay_step)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.opt.decay_step * 2, self.opt.lr * 0.01)

    def name(self):
        return 'SimpleModel'

    def set_train(self):
        self.net.train()
        self.state = 'train'

    def set_eval(self):
        self.net.eval()
        self.state = 'val'

    def initialize(self, opt):
        pass

    def set_input(self, data):
        input_im, input_mask = data
        self.input_im = input_im.to(self.opt.device)
        self.real_mask = input_mask.to(self.opt.device)

    def forward(self):
        self.est_mask = self.net.forward(self.input_im)

    def calc_loss(self):
        self.loss = self.criterion(self.est_mask, self.real_mask) + self.criterion(self.est_mask, self.real_mask)

    def get_loss(self):
        return self.loss

    def get_acc(self):
        acc_dice = general_dice(self.real_mask.data.cpu().numpy(), self.est_mask.data.cpu().numpy(),
                                num_classes=self.num_classes)
        acc_jaccard = general_jaccard(self.real_mask.data.cpu().numpy(), self.est_mask.data.cpu().numpy(),
                                      num_classes=self.num_classes)

        return {'DICE': acc_dice, 'IoU': acc_jaccard}

    def get_val(self):
        return self.get_acc(), self.get_loss()

    def backward(self):
        self.calc_loss()
        self.loss.backward()

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.forward()
        self.backward()
        self.optimizer.step()

    def get_current_visuals(self):
        if self.opt.model == 'ICNet':
            return OrderedDict([('img_%s' % self.state, simple_util.tensor2image(self.input_im.data)),
                                ('seg_est_%s' % self.state, simple_util.tensor2mask(self.est_mask[0].data,
                                                                                    num_classes=self.num_classes)),
                                ('seg_real_%s' % self.state, simple_util.tensor2mask(self.real_mask.data,
                                                                                     num_classes=self.num_classes)),
                                ])

        else:
            return OrderedDict([('img_%s'%self.state, simple_util.tensor2image(self.input_im.data)),
                            ('seg_est_%s'%self.state, simple_util.tensor2mask(self.est_mask.data,
                                                                              num_classes=self.num_classes)),
                            ('seg_real_%s'%self.state, simple_util.tensor2mask(self.real_mask.data,
                                                                               num_classes=self.num_classes)),
                            ])

    def save(self, label):
        self.save_network(self.net, 'onestage_parse_net_%s'% self.opt.model, label, self.gpu_ids)
