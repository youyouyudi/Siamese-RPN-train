# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
#!/usr/bin/python

import vot
import sys
import cv2  # imread
import torch
import torch.nn.init as init
import numpy as np
import copy
from os.path import realpath, dirname, join
from net import SiamRPNBIG
from inference_SiamRPN import SiamRPN_init, SiamRPN_track
from utils.utils import get_axis_aligned_bbox, cxy_wh_2_rect, Get_annotation
from train_siamrpn import SiamRPNBIG_Lee,Firstframe_init,Secondframe_init


import logging
import random

print(torch.__version__)
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

# load net
net_file = join(realpath(dirname(__file__)), 'SiamRPNBIG.model')
#net_file = "model\\SiamRPNBIG_epoch200_VOT_0.2context.model"
net = SiamRPNBIG()
net.id = 12345

net.load_state_dict(torch.load(net_file))

#reinitial the layers except featureExtract layers
"""
for key in net.state_dict():
    print("key", key)
    if key.split('.')[0] == 'featureExtract':
       continue
    elif key.split('.')[-1] == 'weight':
       init.normal_(net.state_dict()[key], std=0.01)
    elif key.split('.')[-1] == 'bias':
       net.state_dict()[key][...] = 0
"""
net.eval().cuda()
#warm up
for i in range(5):
    net.temple(torch.autograd.Variable(torch.FloatTensor(1, 3, 127, 127)).cuda())
    net(torch.autograd.Variable(torch.FloatTensor(1, 3, 255, 255)).cuda())


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

infile = 'vot/list.txt'
listfiles = []
for line in open(infile):
    val = line[0:-1]
    listfiles.append(val)

listfiles_train = listfiles[:]
listfiles_test = listfiles[20:]
start_epoch = 31
epochs = 400
state = 0
listfiles = listfiles_train
epoch_size = 1000
# train using a random video. a pair consisits of row random frames, the template frame dosen't has to be a previous one on timeline.
for epoch in range(start_epoch, epochs+start_epoch):
    print("epoch", epoch)
    net.train().cuda()
    net.train(True)
    # BatchNorm paprameters fixed
    for module in net.modules():
        if isinstance(module, torch.nn.modules.BatchNorm2d):
            module.eval().cuda()
    for i in range(0, epoch_size):
        print("ith", i)
        optimizer.zero_grad()
        '''
        if val=="car1" or val=="car2" or val=="crossing":
            continue
        '''
        index_video = random.randint(0,len(listfiles)-1)
        val = listfiles[index_video]
        logger.info('classes = %s', val)
        inpath = 'vot/' + val + '/*.jpg'
        gtptah = 'vot/' + val + '/groundtruth.txt'
        imgpaths = Get_imgs(inpath)
        length = len(imgpaths)
        fir_index= random.randint(0, length - 1)
        sec_index = random.randint(0, length - 1)
        target_pos1, target_sz1 = Get_annotation(fir_index, gtptah)
        target_pos2, target_sz2 = Get_annotation(sec_index, gtptah)

        imgname = imgpaths[fir_index]
        im = cv2.imread(imgname)
        imgname2 = imgpaths[sec_index]
        im2 = cv2.imread(imgname2)

        state = Firstframe_init(im,target_pos1,target_sz1,net)
        # random shift and random search region scale, to push the model to recover from previous error.
        shift = np.array([target_sz2[0]*(random.random()-0.5)*0.75, target_sz2[1]*(random.random()-0.5)*0.75])
        state['target_pos'] = target_pos2 - shift
        scale = 1.0 + (random.random()-0.5)*0.5
        print("scale",scale)
        Secondframe_init(True, state,im2,target_pos2,target_sz2,optimizer, scale)
        optimizer.step()
    if epoch%10==0:
        torch.save(net.state_dict(), "model/SiamRPNBIG_epoch" + str(epoch) + "_VOT" + ".model")

    #evaluate
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


