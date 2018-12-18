# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
#!/usr/bin/python
import cv2  # imread
import torch
import numpy as np
from os.path import realpath, dirname, join
from net import SiamRPNBIG
from inference_SiamRPN import SiamRPN_init, SiamRPN_track
from utils import get_axis_aligned_bbox, cxy_wh_2_rect


def Get_imgs(inpath):
    import glob
    imgpaths = glob.glob(inpath)
    imgpaths.sort()
    return imgpaths

net_file = join(realpath(dirname(__file__)), 'SiamRPNBIG.model')
net = SiamRPNBIG()
net.load_state_dict(torch.load(net_file))
net.eval().cuda()

infile = 'vot/list.txt'
listfiles = []
for line in open(infile):
    val = line[0:-1]
    listfiles.append(val)
listfiles_test = listfiles[:]

for j in range(0, len(listfiles_test)):
    val = listfiles_test[j]
    inpath = 'vot/' + val + '/*.jpg'
    gtptah = 'vot/' + val + '/groundtruth.txt'
    imgpaths = Get_imgs(inpath)
    length = len(imgpaths)
    for i in range(length):
        imgname = imgpaths[i]
        im = cv2.imread(imgname)
        #get first frame
        if i == 0:
            gtfile = open(gtptah, 'r')
            gt_rect = gtfile.readlines()
            gt = gt_rect[i]
            g = gt[0:-1].split(',')
            for n in range(len(g)):
                g[n] = int(float(g[n]))
            x = [g[0], g[2], g[4], g[6]]
            y = [g[1], g[3], g[5], g[7]]
            x1 = min(x)
            y1 = min(y)
            x2 = max(x)
            y2 = max(y)
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            cx = int(0.5 * (x1 + x2))
            cy = int(0.5 * (y1 + y2))
            target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
            state = SiamRPN_init(im, target_pos, target_sz, net)
        # track
        else:
            state = SiamRPN_track(state, im)  # track
            res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            x1 = int(res[0])
            y1 = int(res[1])
            x2 = int(res[0] + res[2] - 1)
            y2 = int(res[1] + res[3] - 1)
            draw = cv2.rectangle(im, (x1, y1), (x2, y2), (255, 0, 0), 2)
            #output the track result
            print("score", state["score"] )
            cv2.imshow('track.jpg', draw)
            cv2.waitKey(0)