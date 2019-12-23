from __future__ import absolute_import, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from collections import namedtuple
from show_frame import show_frame
from PIL import Image
import time


class Tracker(object):

    def __init__(self, name, is_deterministic=False):
        self.name = name
        self.is_deterministic = is_deterministic

    def init(self, image, box):
        raise NotImplementedError()

    def update(self, image):
        raise NotImplementedError()

    def track(self, img_files, box, visualize=False):
        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box
        times = np.zeros(frame_num)

        for f, img_file in enumerate(img_files):
            image = Image.open(img_file)
            if not image.mode == 'RGB':
                image = image.convert('RGB')

            start_time = time.time()
            if f == 0:
                self.init(image, box)
            else:
                boxes[f, :] = self.update(image)
            times[f] = time.time() - start_time

            if visualize:
                show_frame(image, boxes[f, :])

        return boxes, times

class SiamRPN(nn.Module):

    def __init__(self, anchor_num=5):
        super(SiamRPN, self).__init__()
        self.anchor_num = anchor_num
        self.feature = nn.Sequential(
            # conv1
            nn.Conv2d(3, 192, 11, 2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            # conv2
            nn.Conv2d(192, 512, 5, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            # conv3
            nn.Conv2d(512, 768, 3, 1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            # conv4
            nn.Conv2d(768, 768, 3, 1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            # conv5
            nn.Conv2d(768, 512, 3, 1),
            nn.BatchNorm2d(512))
        
        self.conv_reg_z = nn.Conv2d(512, 512 * 4 * anchor_num, 3, 1)
        self.conv_reg_x = nn.Conv2d(512, 512, 3)
        self.conv_cls_z = nn.Conv2d(512, 512 * 2 * anchor_num, 3, 1)
        self.conv_cls_x = nn.Conv2d(512, 512, 3)
        self.adjust_reg = nn.Conv2d(4 * anchor_num, 4 * anchor_num, 1)

    def forward(self, z, x):
        return self.inference(x, **self.learn(z))

    def learn(self, z, number):
        z = self.feature(z)
        kernel_reg = self.conv_reg_z(z)
        kernel_cls = self.conv_cls_z(z)

        k = kernel_reg.size()[-1]
        kernel_reg = kernel_reg.view(number, 4 * self.anchor_num, 512, k, k)
        kernel_cls = kernel_cls.view(number, 2 * self.anchor_num, 512, k, k)

        return kernel_reg, kernel_cls

    def inference(self, x, kernel_reg, kernel_cls, number = 1):
        x = self.feature(x)
        x_reg = self.conv_reg_x(x) ####### b,c,h,w
        x_cls = self.conv_cls_x(x)
        ########## llj
        out_reg = []
        out_cls = []
        for x in range(number):
            out_reg.append(self.adjust_reg(F.conv2d(x_reg[[x],:], kernel_reg[x])).squeeze(0))
            out_cls.append(F.conv2d(x_cls[[x],:], kernel_cls[x]).squeeze(0))
        out_reg = torch.stack(out_reg)
        out_cls = torch.stack(out_cls)
        # out_reg = self.adjust_reg(F.conv2d(x_reg, kernel_reg))
        # out_cls = F.conv2d(x_cls, kernel_cls)

        return out_reg, out_cls


class TrackerSiamRPN(Tracker):

    def __init__(self, net_path=None, **kargs):
        super(TrackerSiamRPN, self).__init__(
            name='SiamRPN', is_deterministic=True)
        self.parse_args(**kargs)

        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        # setup model
        self.net = SiamRPN()
        if net_path is not None:
            self.net.load_state_dict(torch.load(
                net_path, map_location=lambda storage, loc: storage))
        self.net = self.net.to(self.device)
        self.box_num = 1

    def parse_args(self, **kargs):
        self.cfg = {
            'exemplar_sz': 127,
            'instance_sz': 271,
            'total_stride': 8,
            'context': 0.5,
            'ratios': [0.33, 0.5, 1, 2, 3],
            'scales': [8,],
            'penalty_k': 0.055,
            'window_influence': 0.42,
            'lr': 0.295}

        for key, val in kargs.items():
            self.cfg.update({key: val})
        self.cfg = namedtuple('GenericDict', self.cfg.keys())(**self.cfg)

    def init(self, image, box):
        image = np.asarray(image)

        # convert box to 0-indexed and center based [y, x, h, w]
        ########### llj
        self.box_num = len(box)
        box = np.array(box)
        temp = np.ones(box.shape)
        temp[:,0] = box[:,1] - 1 + (box[:,3] - 1) / 2
        temp[:,1] = box[:,0] - 1 + (box[:,2] - 1) / 2
        temp[:,2] = box[:,3]
        temp[:,3] = box[:,2]
        box = temp

        self.center, self.target_sz = box[:,:2], box[:,2:] ####### (number,2)

        # box = np.array([
        #     box[1] - 1 + (box[3] - 1) / 2,
        #     box[0] - 1 + (box[2] - 1) / 2,
        #     box[3], box[2]], dtype=np.float32)
        # self.center, self.target_sz = box[:2], box[2:]

        # for small target, use larger search region
        if np.prod(self.target_sz[0]) / np.prod(image.shape[:2]) < 0.004:
        # if np.prod(self.target_sz) / np.prod(image.shape[:2]) < 0.004:
            self.cfg = self.cfg._replace(instance_sz=287)     ########### llj change instance_sz to modify search region

        # generate anchors
        self.response_sz = (self.cfg.instance_sz - \
            self.cfg.exemplar_sz) // self.cfg.total_stride + 1
        self.anchors = self._create_anchors(self.response_sz)

        # create hanning window
        self.hann_window = np.outer(
            np.hanning(self.response_sz),
            np.hanning(self.response_sz))
        self.hann_window = np.tile(
            self.hann_window.flatten(),
            len(self.cfg.ratios) * len(self.cfg.scales))

        # exemplar and search sizes
        context = self.cfg.context * np.sum(self.target_sz, axis=1,keepdims=True)
        ########## llj
        self.z_sz = np.sqrt(np.prod(self.target_sz + context, axis= 1))
        # self.z_sz = np.sqrt(np.prod(self.target_sz + context))

        self.x_sz = self.z_sz * \
            self.cfg.instance_sz / self.cfg.exemplar_sz

        # exemplar image
        self.avg_color = np.mean(image, axis=(0, 1))

        ############ llj
        exemplar_image = []
        for i, x in enumerate(self.z_sz):
            exemplar_one = self._crop_and_resize(
                image, self.center[i], self.z_sz[i],
                self.cfg.exemplar_sz, self.avg_color)
            exemplar_image.append(exemplar_one)
        exemplar_image = np.array(exemplar_image) ######## b,h,w,c
        # exemplar_image = self._crop_and_resize(
        #     image, self.center, self.z_sz,
        #     self.cfg.exemplar_sz, self.avg_color)

        # classification and regression kernels
        ##############llj
        exemplar_image = torch.from_numpy(exemplar_image).to(self.device).permute([0,3,1,2]).float()
        # exemplar_image = torch.from_numpy(exemplar_image).to(
        #     self.device).permute([2, 0, 1]).unsqueeze(0).float()
        with torch.set_grad_enabled(False):
            self.net.eval()
            # self.kernel_reg, self.kernel_cls = self.net.learn(exemplar_image)
            self.kernel_reg, self.kernel_cls = self.net.learn(exemplar_image, self.box_num)

    def update(self, image):
        image = np.asarray(image)
        
        # search image
        #### llj
        instance_image = []
        for i, x in enumerate(self.x_sz):
            instance_one = self._crop_and_resize(
                image, self.center[i], x,
                self.cfg.instance_sz, self.avg_color)
            instance_image.append(instance_one)
        instance_image = np.array(instance_image)
        # instance_image = self._crop_and_resize(
        #     image, self.center, self.x_sz,
        #     self.cfg.instance_sz, self.avg_color)

        # classification and regression outputs
        ######## llj
        instance_image = torch.from_numpy(instance_image).to(self.device).permute(0,3,1,2).float()
        # instance_image = torch.from_numpy(instance_image).to(
        #     self.device).permute(2, 0, 1).unsqueeze(0).float()

        with torch.set_grad_enabled(False):
            self.net.eval()
            out_reg, out_cls = self.net.inference(
                instance_image, self.kernel_reg, self.kernel_cls, number=self.box_num) ### b,c,h,w  number, 20, 19,19
        
        # offsets
        ######### llj
        offsets = out_reg.reshape((self.box_num, 4, -1)).cpu().numpy()
        offsets[:,0] = offsets[:,0] * self.anchors[:, 2] + self.anchors[:, 0]
        offsets[:,1] = offsets[:,1] * self.anchors[:, 3] + self.anchors[:, 1]
        offsets[:,2] = np.exp(offsets[:,2]) * self.anchors[:, 2]
        offsets[:,3] = np.exp(offsets[:,3]) * self.anchors[:, 3]
        # offsets = out_reg.permute(
        #     1, 2, 3, 0).contiguous().view(4, -1).cpu().numpy()
        # offsets[0] = offsets[0] * self.anchors[:, 2] + self.anchors[:, 0]
        # offsets[1] = offsets[1] * self.anchors[:, 3] + self.anchors[:, 1]
        # offsets[2] = np.exp(offsets[2]) * self.anchors[:, 2]
        # offsets[3] = np.exp(offsets[3]) * self.anchors[:, 3]

        # scale and ratio penalty
        penalty = self._create_penalty(self.target_sz, offsets)

        # response
        ###### llj
        maxout = F.softmax(out_cls.reshape((self.box_num, 2, -1)), dim=1)
        response = maxout[:,1,:].cpu().numpy()
        # response = F.softmax(out_cls.permute(
        #     1, 2, 3, 0).contiguous().view(2, -1), dim=0).data[1].cpu().numpy()
        response = response * penalty
        response = (1 - self.cfg.window_influence) * response + \
            self.cfg.window_influence * self.hann_window
        
        # peak location
        ############## llj
        best_id = np.argmax(response, axis = 1) ### (number)
        offset = []
        print('best prob : {}'.format(response[:,best_id]))
        for i, x in enumerate(best_id):
            offset.append(offsets[i,:,x] * self.z_sz[i] / self.cfg.exemplar_sz) ## (number, 4)
        offset = np.array(offset)
        #offset = offsets[:, :, best_id] * self.z_sz / self.cfg.exemplar_sz ## (num, 4, number) * (num)

        # update center
        ########## llj
        self.center += offset[:,1::-1]
        self.center = np.clip(self.center, 0 , np.tile(image.shape[:2],(self.box_num,1)))
        # self.center += offset[:2][::-1]
        # self.center = np.clip(self.center, 0, image.shape[:2])

        # update scale
        pair_id = np.concatenate((np.arange(self.box_num).reshape((-1,1)), np.reshape(best_id, (-1,1))), axis = 1)
        lr = response[pair_id[:,0],pair_id[:,1]] * self.cfg.lr
        lr = np.reshape(lr,(-1,1))
        #lr = response[:,best_id] * self.cfg.lr
        ######### llj
        self.target_sz = (1 - lr) * self.target_sz + lr * offset[:,3:1:-1]
        self.target_sz = np.clip(self.target_sz, 10, np.tile(image.shape[:2],(self.box_num,1)))
        # self.target_sz = (1 - lr) * self.target_sz + lr * offset[2:][::-1]
        # self.target_sz = np.clip(self.target_sz, 10, image.shape[:2])

        # update exemplar and instance sizes
        context = self.cfg.context * np.sum(self.target_sz,axis=1).reshape((-1,1))
        self.z_sz = np.sqrt(np.prod(self.target_sz + context, axis=1))
        self.x_sz = self.z_sz * \
            self.cfg.instance_sz / self.cfg.exemplar_sz

        # return 1-indexed and left-top based bounding box
        ####### llj
        temp = np.ones((self.box_num, 4))
        temp[:,0] = self.center[:,1] + 1 - (self.target_sz[:,1] - 1) / 2
        temp[:,1] = self.center[:,0] + 1 - (self.target_sz[:,0] - 1) / 2
        temp[:,2] = self.target_sz[:,1]
        temp[:,3] = self.target_sz[:,0]
        box = temp

        # box = np.array([
        #     self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
        #     self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
        #     self.target_sz[1], self.target_sz[0]])

        return box

    def _create_anchors(self, response_sz):
        anchor_num = len(self.cfg.ratios) * len(self.cfg.scales)
        anchors = np.zeros((anchor_num, 4), dtype=np.float32)

        size = self.cfg.total_stride * self.cfg.total_stride
        ind = 0
        for ratio in self.cfg.ratios:
            w = int(np.sqrt(size / ratio))
            h = int(w * ratio)
            for scale in self.cfg.scales:
                anchors[ind, 0] = 0
                anchors[ind, 1] = 0
                anchors[ind, 2] = w * scale
                anchors[ind, 3] = h * scale
                ind += 1
        anchors = np.tile(
            anchors, response_sz * response_sz).reshape((-1, 4))

        begin = -(response_sz // 2) * self.cfg.total_stride
        xs, ys = np.meshgrid(
            begin + self.cfg.total_stride * np.arange(response_sz),
            begin + self.cfg.total_stride * np.arange(response_sz))
        xs = np.tile(xs.flatten(), (anchor_num, 1)).flatten()
        ys = np.tile(ys.flatten(), (anchor_num, 1)).flatten()
        anchors[:, 0] = xs.astype(np.float32)
        anchors[:, 1] = ys.astype(np.float32)

        return anchors

    def _create_penalty(self, target_sz, offsets):
        def padded_size(w, h):
            context = self.cfg.context * (w + h)
            return np.sqrt((w + context) * (h + context))

        def larger_ratio(r):
            return np.maximum(r, 1 / r)
        ###########     llj
        penalty = []
        for i, x in enumerate(target_sz):
            src_sz = padded_size(
                *(target_sz[i] * self.cfg.exemplar_sz / self.z_sz[i]))
            dst_sz = padded_size(offsets[i,2], offsets[i,3])
            change_sz = larger_ratio(dst_sz / src_sz)

            src_ratio = target_sz[i,1] / target_sz[i,0]
            dst_ratio = offsets[i,2] / offsets[i,3]
            change_ratio = larger_ratio(dst_ratio / src_ratio)

            penalty.append(np.exp(-(change_ratio * change_sz - 1) * \
                             self.cfg.penalty_k))
        penalty = np.array(penalty)
        # src_sz = padded_size(
        #     *(target_sz * self.cfg.exemplar_sz / self.z_sz))
        # dst_sz = padded_size(offsets[2], offsets[3])
        # change_sz = larger_ratio(dst_sz / src_sz)
        #
        # src_ratio = target_sz[1] / target_sz[0]
        # dst_ratio = offsets[2] / offsets[3]
        # change_ratio = larger_ratio(dst_ratio / src_ratio)
        #
        # penalty = np.exp(-(change_ratio * change_sz - 1) * \
        #     self.cfg.penalty_k)

        return penalty

    def _crop_and_resize(self, image, center, size, out_size, pad_color):
        # convert box to corners (0-indexed)
        size = round(size)
        corners = np.concatenate((
            np.round(center - (size - 1) / 2),
            np.round(center - (size - 1) / 2) + size))
        corners = np.round(corners).astype(int)

        # pad image if necessary
        pads = np.concatenate((
            -corners[:2], corners[2:] - image.shape[:2]))
        npad = max(0, int(pads.max()))
        if npad > 0:
            image = cv2.copyMakeBorder(
                image, npad, npad, npad, npad,
                cv2.BORDER_CONSTANT, value=pad_color)

        # crop image patch
        corners = (corners + npad).astype(int)
        patch = image[corners[0]:corners[2], corners[1]:corners[3]]

        # resize to out_size
        patch = cv2.resize(patch, (out_size, out_size))

        return patch
