import argparse
from models import model_list
import os
import torch
import utils.simple_util as simple_util
from datetime import datetime


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--jaccard-weight', default=0.5, type=float)
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='For example 0,1 to run on two GPUs')
        self.parser.add_argument('--fold', type=int, help='fold', default=0)
        self.parser.add_argument('--batch_size', type=int, default=2)
        self.parser.add_argument('--niter', type=int, default=100)
        self.parser.add_argument('--decay_step', type=int, default=5)
        self.parser.add_argument('--lr', type=float, default=0.0001)
        self.parser.add_argument('--workers', type=int, default=12)
        self.parser.add_argument('--train_crop_height', type=int, default=1024)
        self.parser.add_argument('--train_crop_width', type=int, default=1280)
        self.parser.add_argument('--val_crop_height', type=int, default=1024)
        self.parser.add_argument('--val_crop_width', type=int, default=1280)
        self.parser.add_argument('--problem_type', type=str, default='parts', choices=['binary', 'parts', 'instruments'])
        self.parser.add_argument('--model', type=str, default='UNetM', choices=model_list.keys())
        self.parser.add_argument('--seed', type=int, default=1132)
        self.parser.add_argument('--isTrain', type=bool, default=True)
        self.parser.add_argument('--continue_train', type=bool, default=False)
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
        self.parser.add_argument('--evaluate_freq', type=int, default=1)
        self.parser.add_argument('--save_epoch_freq', type=int, default=5)
        self.opt = None
        self.parse()

    def parse(self):
        self.opt = self.parser.parse_args()
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        self.opt.device = torch.device('cuda:%d' % self.opt.gpu_ids[0]
                                       if torch.cuda.is_available() and len(self.opt.gpu_ids) > 0 else 'cpu')

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        self.opt.name = self.opt.model + '_' + self.opt.problem_type + '_' + datetime.now().strftime("%Y%m%d_%H%M%S")
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        simple_util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
