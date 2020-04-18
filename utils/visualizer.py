'''
    This is a wrapper of tensorboardX
    url: https://github.com/lanpa/tensorboard-pytorch

'''
from tensorboardX import SummaryWriter
import os.path
import os
from utils.writer import Writer
from subprocess import Popen
import utils.simple_util as simple_util

# TODO: Current limitation is that visulization does not support specify the value of X-AXIS


def launchTensorBoard(tensorBoardPath, port):
    import os
    # os.system('~/my_anaconda/bin/python ~/my_anaconda/lib/python2.7/site-packages/tensorboard/main.py --bind_all --logdir='  + tensorBoardPath + ' --port %d'%port)
    Popen(
        '~/my_anaconda/bin/python ~/my_anaconda/lib/python2.7/site-packages/tensorboard/main.py --bind_all --logdir=' + tensorBoardPath + ' --port %d' % port,
        shell=True)
    return


class Visualizer:
    def __init__(self, opt, tensorboard=True):
        # opt is the global options shared in the coding env
        self.opt = opt
        self.tensorboard = tensorboard
        if tensorboard:
            import threading
            # self.t = threading.Thread(target=launchTensorBoard, args=([os.path.join(opt.checkpoints_dir, opt.name), opt.port]))
            # self.t.daemon = True
            launchTensorBoard(os.path.join(opt.checkpoints_dir, opt.name), opt.port)
            self.t.start()
            # self.t.join(20)
            self.writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name))
        else:
            self.writer = Writer(os.path.join(opt.checkpoints_dir, opt.name))
        log_filename = os.path.join(opt.checkpoints_dir, '%s_log.txt' % opt.name)
        self.log = open(log_filename, 'w')
        self.write_opt(opt)
        self.save_image_root = os.path.join(opt.checkpoints_dir, opt.name)
        self.save_worse_root = os.path.join(opt.save_worse_dir, opt.name)

    def add_text(self, tag, text):
        self.writer.add_text(tag, text)

    def write_opt(self, opt):
        args = vars(self.opt)

        self.write_log_rightnow('------------ Options -------------\n')
        for k, v in sorted(args.items()):
            self.write_log_rightnow('%s: %s\n' % (str(k), str(v)))
        self.write_log_rightnow('-------------- End ----------------\n')

    def write_acc(self, acc):
        self.write_log_rightnow('Acc: %f\n' % acc)

    def write_log_rightnow(self, str):
        self.log.write(str)
        self.log.flush()
        os.fsync(self.log)

    def write_confmat(self, conf_mat):
        self.write_log_rightnow('Conf Mat:\n')
        for line in conf_mat:
            for e in line:
                self.write_log_rightnow('%f ' % e)
            self.write_log_rightnow('\n')

    def add_log(self, str):
        # print str
        self.write_log_rightnow('%s\n' % str)

    def write_epoch(self, epoch):
        self.write_log_rightnow('Epoch %d\n' % epoch)

    def plot_errors(self, errors, main_fig_title='errors', time_step=None):
        # example:  main_fig_title = 'GAN_depth_to_image', sub_fig_title = 'train_loss'
        # scalars_dict = {'item': value, 'item2':value},
        # when time_step is None, it uses internal global_time_step
        # for k, v in errors.items():
        tag = main_fig_title + '/'
        self.writer.add_scalars(tag, errors, time_step)

    def kill(self):
        # print self.t.getName()
        self.log.close()

    def display_image(self, images, time_step=None):
        # images : {'real': real_im,'fake':fake_im}
        for item in images.keys():
            self.writer.add_image(item, images[item], time_step)

    def save_images(self, images, epoch, acc):
        for item in images.keys():
            cur_image_fullpath = os.path.join(self.save_image_root, 'Epoch_%06d_%s_%s.png' % (epoch, item, acc))

            simple_util.save_image(images[item], cur_image_fullpath)

    def save__worse_images(self, images, score):
        for item in images.keys():
            cur_image_fullpath = os.path.join(self.save_worse_root, 'Epoch_%06d_%s.png' % (score, item))

            simple_util.save_image(images[item], cur_image_fullpath)

    def print_errors(self, epoch, batch, batch_all, errors):
        message = '(epoch: %d, (%d/%d)) ' % (epoch, batch, batch_all)
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)
        print(message)



