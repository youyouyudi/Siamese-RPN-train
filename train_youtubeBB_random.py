# train by youtubeBB dataset
# get pairs randomly by videos, clips.
# the tow frames of a pair has a distance of 1sec
import os
import sys
import cv2  # imread
import torch
import numpy as np
from os.path import realpath, dirname, join
from net import SiamRPNBIG
from inference_SiamRPN import SiamRPN_init, SiamRPN_track
from utils.utils import get_axis_aligned_bbox, cxy_wh_2_rect, Get_annotation
from train_siamrpn import SiamRPNBIG_Lee, Firstframe_init, Secondframe_init
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
import logging
import random
import json
import torch.nn.init as init


def setup_logging(name):
    FORMAT = '%(levelname)s %(filename)s:%(lineno)4d: %(message)s'
    # Manually clear root loggers to prevent any module that may have called
    # logging.basicConfig() from blocking our logging setup
    logging.root.handlers = []
    logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
    logger = logging.getLogger(name)
    return logger


logger = setup_logging(__name__)
logging.getLogger('train_siamrpn').setLevel(logging.INFO)


def Get_imgs(inpath):
    import glob
    imgpaths = glob.glob(inpath)
    imgpaths.sort()
    return imgpaths


#net_file = join(realpath(dirname(__file__)), 'model\\SiamRPNBIG_epoch1_1000.model')
net_file = join(realpath(dirname(__file__)), 'SiamRPNBIG.model')
net = SiamRPNBIG()
net.id = 12345
net.load_state_dict(torch.load(net_file))

for key in net.state_dict():
    print("key", key)
    if key.split('.')[0] == 'featureExtract':
        continue
    elif key.split('.')[-1] == 'weight':
        init.normal_(net.state_dict()[key], std=0.01)
    elif key.split('.')[-1] == 'bias':
        net.state_dict()[key][...] = 0

# warm up
'''
for i in range(5):
    net.temple(torch.autograd.Variable(torch.FloatTensor(1, 3, 127, 127)).cuda())
    net(torch.autograd.Variable(torch.FloatTensor(1, 3, 255, 255)).cuda())
'''

params = []
params += list(net.conv_cls1.parameters())
params += list(net.conv_r1.parameters())
params += list(net.conv_cls2.parameters())
params += list(net.conv_r2.parameters())
params += list(net.regress_adjust.parameters())

optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0001)
net.train().cuda()
net.train(True)
optimizer.zero_grad()
for module in net.modules():
    if isinstance(module, torch.nn.modules.BatchNorm2d):
        module.eval().cuda()

# some transforma operation for images
transform = T.Compose([

])

frame_dir = "D:\\Yi+\\youtube_BB\\youtube-bb-master\\frame\\youtube_boundingboxes_detection_validation\\"
video_list = os.listdir(frame_dir)


class Youtube_track(Dataset):
    def __init__(self, image_dir, annFile):
        self.video_list = os.listdir(frame_dir)
        self.video_dic = {}
        for video in self.video_list:
            self.video_dic[video] = os.listdir(frame_dir + video)
        self.video_index = 0
        self.clips_index = 0
        self.frame_index = 0
        self.transform = transform
        self.if_first_frame = 1

    def getPair(self, index, annotation_id=-1):
        while True:
            self.video_index = random.randint(0, len(self.video_list)-1)
            video_name = self.video_list[self.video_index]
            json_dir = frame_dir + video_name + "\\" + "annotation.json"
            # print("json_dir", json_dir)
            with open(json_dir, 'r') as f:
                annotations = json.load(f)

            clip_names = self.video_dic[video_name]
            clip_names = [c for c in clip_names if c != "annotation.json"]
            self.clips_index = random.randint(0, len(clip_names)-1)
            print("self.clips_index, len(clip_names)", self.clips_index, len(clip_names))
            clip_name = clip_names[self.clips_index]

            clip_dir = frame_dir + video_name + "\\" + clip_name + "\\"
            frame_names = os.listdir(clip_dir)
            print("len(frame_names)-2",len(frame_names)-2)
            if len(frame_names)-2<0:
                continue
            self.frame_index = random.randint(0, len(frame_names)-2)

            time_start = int(clip_name.split("_")[-3])
            time_index1 = time_start + self.frame_index * 1000
            time_index2 = time_index1 + 1000
            annotation1 = [a for a in annotations if int(a[1]) == time_index1]
            annotation2 = [a for a in annotations if int(a[1]) == time_index2]
            if len(annotation1)==0 or len(annotation2)==0:
                continue
            annotation1 = annotation1[0]
            annotation2 = annotation2[0]
            if not annotation1[5] == "present":
                continue
            ispositive = 0
            if annotation2[5] == "present":
                ispositive = 1
            img_dir1 = clip_dir + str(self.frame_index) + ".jpg"
            im1 = cv2.imread(img_dir1)
            img_dir2 = clip_dir + str(self.frame_index + 1) + ".jpg"
            im2 = cv2.imread(img_dir2)
            data = [im1, im2]
            x1 = im1.shape[1] * (float(annotation1[6]) + float(annotation1[7])) / 2
            y1 = im1.shape[0] * (float(annotation1[8]) + float(annotation1[9])) / 2
            x2 = im2.shape[1] * (float(annotation2[6]) + float(annotation2[7])) / 2
            y2 = im2.shape[0] * (float(annotation2[8]) + float(annotation2[9])) / 2
            target_pos1 = np.array([x1, y1])
            target_pos2 = np.array([x2, y2])
            target_pos = [target_pos1, target_pos2]
            w1 = im1.shape[1] * (float(annotation1[7]) - float(annotation1[6]))
            h1 = im1.shape[0] * (float(annotation1[9]) - float(annotation1[8]))
            w2 = im2.shape[1] * (float(annotation2[7]) - float(annotation2[6]))
            h2 = im2.shape[0] * (float(annotation2[9]) - float(annotation2[8]))
            target_sz1 = np.array([w1, h1])
            target_sz2 = np.array([w2, h2])
            target_sz = [target_sz1, target_sz2]
            self.frame_index += 1
            if_first_frame = self.if_first_frame
            self.if_first_frame = 0
            return data, target_pos, target_sz, ispositive, if_first_frame

    def __getitem__(self, index0):
        data, target_pos, target_sz, ispositive, if_first_frame = self.getPair(index0)
        return data, [ispositive, target_pos, target_sz, if_first_frame]

    def __len__(self):
        return len(self.video_list)


dataset = Youtube_track()
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=False)

flag = 1
epochs = 15
state = 0
dataiter = iter(dataloader)
# num_samples = len(dataset)
num_samples = 5000
print("num_samples", num_samples)
dataset = Youtube_track()
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=False)
dataiter = iter(dataloader)
for epoch in range(epochs):
    net.train().cuda()
    net.train(True)
    print("epoch", epoch)
    for j in range(num_samples):
        print("index", j, len(dataset))
        optimizer.zero_grad()

        imgs, labels = next(dataiter)
        target_pos1, target_sz1 = labels[1][0].numpy()[0], labels[2][0].numpy()[0]
        target_pos2, target_sz2 = labels[1][1].numpy()[0], labels[2][1].numpy()[0]
        ispositive = labels[0].numpy()[0]
        print("target_pos1, target_sz1", target_pos1, target_sz1)
        print("target_pos2, target_sz2", target_pos2, target_sz2)
        img1, img2 = imgs
        img1 = img1.numpy()
        img1 = np.squeeze(img1)
        img2 = img2.numpy()
        img2 = np.squeeze(img2)
        """
        res = cxy_wh_2_rect(target_pos1, target_sz1)
        x1 = int(res[0])
        y1 = int(res[1])
        x2 = int(res[0] + res[2] - 1)
        y2 = int(res[1] + res[3] - 1)
        draw = cv2.rectangle(img1, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # output the track result
        cv2.imshow('result_final.jpg', draw)
        cv2.waitKey(0)
        """
        state = Firstframe_init(img1, target_pos1, target_sz1, net)
        Secondframe_init(ispositive, state, img2, target_pos2, target_sz2, optimizer)
        """
        state = SiamRPN_init(img1, target_pos1, target_sz1, net)
        state = SiamRPN_track(state, img2)  # track
        res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
        x1 = int(res[0])
        y1 = int(res[1])
        x2 = int(res[0] + res[2] - 1)
        y2 = int(res[1] + res[3] - 1)
        draw = cv2.rectangle(img2, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # output the track result
        cv2.imshow('result_final.jpg', draw)
        cv2.waitKey(0)
        """
        optimizer.step()
        if j > 0 and j+1 % 1000 == 0:
            torch.save(net.state_dict(), "model/SiamRPNBIG_epoch" + str(epoch) + "_" + str(j) + ".model")

    '''
    if epoch == 14 or True:
        # eval trainning set
        dataset = Youtube_track()
        dataloader2 = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=False)
        dataiter2 = iter(dataloader2)
        net.eval().cuda()
        net.eval()

        for j in range(30):
            imgs, labels = next(dataiter2)
            target_pos1, target_sz1 = labels[1][0].numpy()[0], labels[2][0].numpy()[0]
            target_pos2, target_sz2 = labels[1][1].numpy()[0], labels[2][1].numpy()[0]
            if_first_frame = labels[3].numpy()[0]
            print("if_first_frame", if_first_frame)
            img1, img2 = imgs
            img1 = img1.numpy()
            img1 = np.squeeze(img1)
            img2 = img2.numpy()
            img2 = np.squeeze(img2)

            if if_first_frame == 1:
                state = SiamRPN_init(img1, target_pos1, target_sz1, net)

            state = SiamRPN_track(state, img2)  # track
            res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            x1 = int(res[0])
            y1 = int(res[1])
            x2 = int(res[0] + res[2] - 1)
            y2 = int(res[1] + res[3] - 1)
            print(x1, y1, x2, y2)
            draw = cv2.rectangle(img2, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # output the track result
            cv2.imshow('result_final.jpg', draw)
            cv2.waitKey(0)
    '''

    # eval vot
    net.eval().cuda()
    net.eval()
    infile = 'vot/list.txt'
    listfiles = []
    for line in open(infile):
        val = line[0:-1]
        listfiles.append(val)
    listfiles_test = listfiles[10:20]
    if True:
        net.eval().cuda()
        net.eval()
        for j in range(len(listfiles_test)):
            val = listfiles_test[j]
            logger.info('classes = %s', val)
            inpath = 'vot/' + val + '/*.jpg'
            gtptah = 'vot/' + val + '/groundtruth.txt'
            imgpaths = Get_imgs(inpath)
            length = len(imgpaths)
            static_target_sz = 0
            state = 0
            for i in range(length-1):
                optimizer.zero_grad()
                target_pos1, target_sz1 = Get_annotation(i, gtptah)
                target_pos2, target_sz2 = Get_annotation(i+1, gtptah)
                imgname = imgpaths[i]
                im = cv2.imread(imgname)
                imgname2 = imgpaths[i + 1]
                im2 = cv2.imread(imgname2)
                if i==0:
                    static_target_sz = target_sz1
                    state = Firstframe_init(im, target_pos1, target_sz1, net)
                    #state = SiamRPN_init(im, target_pos, target_sz, net)
                    cv2.destroyAllWindows()

                # use the fixed target_sz
                #state['target_sz'] = static_target_sz
                state = SiamRPN_track(state, im2)  # track
                print("state['score']", state['score'])

                # reinitial by present frame if the score is high enough
                """
                if state['score']>0.99:
                    state = Firstframe_init(im2, state['target_pos'], state['target_sz'], net)
                """
                res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
                x1 = int(res[0])
                y1 = int(res[1])
                x2 = int(res[0] + res[2] - 1)
                y2 = int(res[1] + res[3] - 1)
                draw = cv2.rectangle(im2, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.imshow('result_final'+str(int(i/50)*0)+'.jpg', draw)
                cv2.waitKey(0)