# @Author: ygu
# @Date: 5/29/19

import os
class Writer:
    def __init__(self, path):
        self.scalars = {}
        self.images = {}
        self.text = {}
        self.path = path

        if not os.path.exists(path):
            os.mkdir(path)

    def add_scalars(self, tag, errors, time_step):
        # if tag not in self.scalars.keys():
        #     self.scalars[tag] = {'data': {}, 'time_step': []}
        # for err_key in errors.keys():
        #     if err_key not in self.scalars[tag]['data'].keys():
        #         self.scalars[tag]['data'][err_key] = []
        # self.scalars[tag]['time_step'].append(time_step)
        # for err_key in errors.keys():
        #     self.scalars[tag]['data'][err_key].append(errors[err_key])
        pass

    def add_image(self, tag, images, time_step):
        if tag not in self.images.keys(): # No images
            self.images[tag] = {'data':{},'time_step':[]}
        pass