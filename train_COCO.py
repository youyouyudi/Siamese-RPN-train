
# train by COCO
# use pytorch Dataset to get pair
# template is an object, but the tracking frame may be a background, an object of another class, or an object of the same class but different instance
# As a result, there are positive pairs and negative pairs, negative pairs have no backprop from regression branch
import os
import sys
import cv2  # imread
import torch
import numpy as np
from os.path import realpath, dirname, join
from net import SiamRPNBIG
from inference_SiamRPN import SiamRPN_init, SiamRPN_track
from utils.utils import get_axis_aligned_bbox, cxy_wh_2_rect, Get_annotation
from train_siamrpn import SiamRPNBIG_Lee,Firstframe_init,Secondframe_init
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
import logging
import random
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

net_file = join(realpath(dirname(__file__)), 'SiamRPNBIG.model')
net = SiamRPNBIG()
net.id = 12345



net.load_state_dict(torch.load(net_file))

'''
for key in net.state_dict():
    print("key", key)
    if key.split('.')[0] == 'featureExtract':
       continue
    elif key.split('.')[-1] == 'weight':
       init.normal_(net.state_dict()[key], std=0.01)
    elif key.split('.')[-1] == 'bias':
       net.state_dict()[key][...] = 0
'''
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
for module in net.modules():
    if isinstance(module, torch.nn.modules.BatchNorm2d):
        module.eval().cuda()

#some transforma operation for images
transform = T.Compose([

])

class COCO_track(Dataset):
    def __init__(self, image_dir="D:/Yi+/COCO/val2017", annFile ="instances_val2017.json"):
        self.coco = COCO(annFile)
        self.image_dir = image_dir
        self.annotations = self.coco.getAnnIds()
        self.length = len(self.annotations)
        #self.images = os.listdir(image_dir)
        self.transform = transform

    def getSingle(self, index, annotation_id=-1):
        if index>=0:
            annotation_id = self.annotations[index]
        annotation = self.coco.loadAnns(annotation_id)[0]
        image_id = annotation["image_id"]
        image = self.coco.loadImgs(image_id)[0]
        img_path = os.path.join(self.image_dir, image["file_name"])
        # get images
        data = cv2.imread(img_path)
        print("data", data.shape, data)
        # generate label
        bbox = annotation["bbox"]
        cx = bbox[0] + bbox[2] / 2
        cy = bbox[1] + bbox[3] / 2
        w = bbox[2]
        h = bbox[3]
        # result is numpy in the function but torch.tensor when it returns
        target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
        return data, target_pos, target_sz

    def __getitem__(self, index0):
        '''
        if self.transform:
            data = self.transform(data)
        '''
        data1, target_pos1, target_sz1 = self.getSingle(index0)
        catid = self.coco.loadAnns(self.annotations[index0])[0]["category_id"]
        print("catid", catid)
        data2, target_pos2, target_sz2 = 0, 0, 0
        rand_num = random.random()
        ispositive = False
        if rand_num<0.1:#random background
            index = random.randint(0, self.length-1)
            data2, target_pos2, target_sz2 = self.getSingle(index)
            im_shape = data2.shape
            h, w = im_shape[0], im_shape[1]
            x1 = random.randint(0, int(w * 0.7))
            y1 = random.randint(0, int(h * 0.7))
            x2 = random.randint(int(x1+0.2*w), w-1)
            y2 = random.randint(int(y1+0.2*h), h-1)
            cx, cy = int((x1+x2)/2),int((y1+y2)/2)
            w, h = x2-x1, y2-y1
            data2, target_pos2, target_sz2 = data2, np.array([cx, cy]), np.array([w, h])

        elif rand_num<0.2:#different class
            catid2, index2 = -1, -1
            while True:
                index = random.randint(0, self.length-1)
                if not self.coco.loadAnns(self.annotations[index])[0]["category_id"] == catid:
                    break
            data2, target_pos2, target_sz2 = self.getSingle(index)

        elif rand_num<0.3:#same class different sample
            annos = self.coco.getAnnIds(catIds=catid)
            index = -1
            while True:
                index = random.randint(0, len(annos)-1)
                if annos[index] != self.annotations[index0]:
                    break
            data2, target_pos2, target_sz2 = self.getSingle(-1, annos[index])

        else:#positive
            ispositive = True
            data2, target_pos2, target_sz2 = data1, target_pos1, target_sz1

        #print("target_pos1", target_pos1)
        data = [data1, data2]
        target_pos = [target_pos1, target_pos2]
        target_sz = [target_sz1, target_sz2]
        return data, [ispositive, target_pos, target_sz]

    def __len__(self):
        return len(self.annotations)

dataset = COCO_track()
dataloader = DataLoader(dataset, batch_size = 1, shuffle = True, num_workers=0, drop_last=False)

flag = 1
epochs = 5
state = 0

net.train().cuda()
net.train(True)
num_samples = len(dataset)
print("num_samples", num_samples)
#num_samples = 0
for epoch in range(epochs):
    print("epoch", epoch)
    dataiter = iter(dataloader)
    '''
    net.train().cuda()
    net.train(True)
    for module in net.modules():
        if isinstance(module, torch.nn.modules.BatchNorm2d):
            module.eval().cuda()
    '''
    for j in range(num_samples):
        print("index", j, len(dataset))
        optimizer.zero_grad()
        imgs, labels = next(dataiter)
        target_pos1, target_sz1 = labels[1][0].numpy()[0], labels[2][0].numpy()[0]
        target_pos2, target_sz2 = labels[1][1].numpy()[0], labels[2][1].numpy()[0]
        ispositive = labels[0]
        print("ispositive", ispositive)
        print("target_pos1, target_sz1",target_pos1, target_sz1)
        img1, img2 = imgs
        img1 = img1.numpy()
        img1 = np.squeeze(img1)
        img2 = img2.numpy()
        img2 = np.squeeze(img2)

        state = Firstframe_init(img1,target_pos1,target_sz1,net,i)
        # random shift the initial position
        if ispositive:
            state['target_pos'] = state['target_pos'] + (random.random()-1)*2*0.2*state['target_sz']
        else:
            state['target_pos'] = target_pos2
            state['target_sz'] = target_sz2

        Secondframe_init(ispositive, state,img2,target_pos2,target_sz2,optimizer)

        optimizer.step()
        if j>0 and j%1000==0:
            torch.save(net.state_dict(), "model/COCO_SiamRPNBIG_epoch" + str(epoch) + "_" + str(j) + ".model")

    #eval
    net.eval().cuda()
    net.eval()
    infile = 'vot/list.txt'
    listfiles = []
    for line in open(infile):
        val = line[0:-1]
        listfiles.append(val)
    listfiles_test = listfiles[10:]
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
