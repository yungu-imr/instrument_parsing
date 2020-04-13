# This is a new script for ICRA 2019 Train_Test

import torch
import torchnet as tnt
import random
import numpy as np
import models.simple_model as simple_model
import time
from tqdm import tqdm
from utils.visualizer import Visualizer


class TrainEngine:
    def __init__(self, opt):
        self.opt = opt
        self.cudnn_init()
        self.model = simple_model.SimpleModel(opt)
        self.train_dataloader = None
        self.val_dataloader = None
        self.visualizer = Visualizer(opt, False)

    def set_data(self, train_dataloader, val_dataloader):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

    def cudnn_init(self):
        # Set seed
        torch.backends.cudnn.benchmark = True
        random.seed(self.opt.seed)
        torch.manual_seed(self.opt.seed)
        torch.cuda.manual_seed(self.opt.seed)
        np.random.seed(self.opt.seed)

    def evaluate(self, dataloader, epoch=0):
        epoch_acc = {'DICE': tnt.meter.AverageValueMeter(),
                     'IoU': tnt.meter.AverageValueMeter()}

        self.model.set_eval()
        with torch.no_grad():
            t = tqdm(dataloader)
            for batch_itr, data in enumerate(t):
                self.model.set_input(data)
                self.model.forward()
                acc, loss = self.model.get_val()
                for key in epoch_acc.keys():
                    epoch_acc[key].add(acc[key])
                t.set_description('[Testing]')
                t.set_postfix(DICE=epoch_acc['DICE'].mean, IoU=epoch_acc['IoU'].mean)
        self.visualizer.add_log('[Testing]：DICE:%f, IoU:%f' % (epoch_acc['DICE'].mean, epoch_acc['IoU'].mean))
        self.visualizer.save_images(self.model.get_current_visuals(), epoch)
        self.model.set_train()
        return epoch_acc

    # This function is used for saving the image picture
    def worse_case(self, dataloader, initial_IoU):
        epoch_acc = {'DICE': tnt.meter.AverageValueMeter(),
                     'IoU': tnt.meter.AverageValueMeter()}

        self.model.set_eval()
        with torch.no_grad():
            t = tqdm(dataloader)
            for batch_itr, data in enumerate(t):
                self.model.set_input(data)
                self.model.forward()
                acc, loss = self.model.get_val()
                if acc['IoU'] < initial_IoU:
                    self.visualizer.save__worse_images(self.model.get_current_visuals(), acc['IoU'])
        self.model.set_train()
        return epoch_acc


    def train_model(self):

        training_time = 0.0
        final_avg_IOU = 0
        for cur_iter in range(0, self.opt.niter):
            running_loss = 0.0
            tic = time.time()

            t = tqdm(self.train_dataloader)
            batch_accum = 0
            for batch_itr, data in enumerate(t):
                self.model.set_input(data)
                self.model.optimize_parameters()
                running_loss += self.model.get_loss().item()
                batch_accum += data[0].size(0)
                t.set_description('[Training Epoch %d/%d]' % (cur_iter, self.opt.niter))
                t.set_postfix(loss=running_loss/batch_accum)
            self.model.scheduler.step()
            self.visualizer.plot_errors({'train': running_loss / len(self.train_dataloader.dataset)},
                                        main_fig_title='err')
            self.visualizer.add_log('[Training Epoch %d/%d]：%f' % (cur_iter, self.opt.niter,
                                                                   running_loss / len(self.train_dataloader.dataset)))
            training_time += time.time() - tic

            if cur_iter % self.opt.evaluate_freq == 0:
                acc_metric = self.evaluate(self.val_dataloader, epoch=cur_iter)
                final_avg_IOU = acc_metric['DICE'].mean
                for key in acc_metric.keys():
                    self.visualizer.plot_errors({'test': acc_metric[key].mean}, main_fig_title=key)

            if cur_iter % self.opt.save_epoch_freq == 0:
                print('saving the model at the end of epoch %d' % cur_iter)
                self.model.save('latest')
                self.model.save(cur_iter)

            # if cur_iter == self.opt.niter - 1:
            #     print('find the worst picture')
            #     number = self.worse_case()
