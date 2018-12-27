# add ispositive
import random
import sys
import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from utils.utils import get_subwindow_tracking
from utils.utils import im_to_torch

import cv2  # imread

import logging

import math


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


def generate_anchor(total_stride, scales, ratios, score_size, instance_size):
    # total_stride = 18
    anchor_num = len(ratios) * len(scales)
    anchor = np.zeros((anchor_num, 4), dtype=np.float32)
    size = total_stride * total_stride
    count = 0
    for ratio in ratios:
        ws = int(np.sqrt(size / ratio))
        hs = int(ws * ratio)
        for scale in scales:
            wws = ws * scale
            hhs = hs * scale
            anchor[count, 0] = 0
            anchor[count, 1] = 0
            anchor[count, 2] = wws
            anchor[count, 3] = hhs
            count += 1

    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
    ori = - ((score_size - 1) / 2) * total_stride

    score_size = int(score_size)
    xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                         [ori + total_stride * dy for dy in range(score_size)])
    xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
             np.tile(yy.flatten(), (anchor_num, 1)).flatten()
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    # print("anchor[:, 0]",anchor[:, 0])
    all_anchors = np.zeros(anchor.shape, dtype=np.float32)
    all_anchors[:, 0] = anchor[:, 0] - 0.5 * anchor[:, 2]
    all_anchors[:, 1] = anchor[:, 1] - 0.5 * anchor[:, 3]
    all_anchors[:, 2] = anchor[:, 0] + 0.5 * anchor[:, 2]
    all_anchors[:, 3] = anchor[:, 1] + 0.5 * anchor[:, 3]
    # print("instance_size",instance_size,all_anchors[:, 0].shape,anchor[:, 0])
    inds_inside = np.where(
        (all_anchors[:, 0] > -0.5 * instance_size) &
        (all_anchors[:, 1] > -0.5 * instance_size) &
        (all_anchors[:, 2] < 0.5 * instance_size) &
        (all_anchors[:, 3] < 0.5 * instance_size))[0]
    anchor = anchor[inds_inside, :]
    return anchor


def bbox_transform_inv(boxes, label_pos, label_sz, weights=(1.0, 1.0, 1.0, 1.0)):
    # print("boxes",boxes)
    ex_widths = boxes[:, 2]
    ex_heights = boxes[:, 3]
    ex_ctr_x = boxes[:, 0]
    ex_ctr_y = boxes[:, 1]

    gt_widths = label_sz[0]
    gt_heights = label_sz[1]
    gt_ctr_x = label_pos[0]
    gt_ctr_y = label_pos[1]

    wx, wy, ww, wh = weights
    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = ww * np.log(gt_widths / ex_widths)
    targets_dh = wh * np.log(gt_heights / ex_heights)

    targets = np.vstack((targets_dx, targets_dy, targets_dw,
                         targets_dh)).transpose()
    return targets


def get_labels_bbox_targets(ispositive, label_pos, label_sz, anchors):
    num_inside = len(anchors)
    labels = np.empty((num_inside,), dtype=np.int32)
    labels.fill(-1)
    # print("labels.shape",labels.shape)
    bbox_targets = np.zeros((num_inside, 4), dtype=np.float32)
    ious = np.empty((num_inside,), dtype=np.float32)
    ious.fill(0.0)
    # print("*******************",label_pos[0], label_pos[1], label_sz[0],label_sz[1])
    x1 = label_pos[0] - 0.5 * label_sz[0]
    y1 = label_pos[1] - 0.5 * label_sz[1]
    x2 = label_pos[0] + 0.5 * label_sz[0]
    y2 = label_pos[1] + 0.5 * label_sz[1]
    # anchors_rect = np.zeros((1805,4),dtype=np.float)
    anchors_rect = np.zeros(anchors.shape, dtype=np.float)
    # print("anchors.shape",anchors.shape)
    anchors_rect[:, 0] = anchors[:, 0] - 0.5 * anchors[:, 2]
    anchors_rect[:, 1] = anchors[:, 1] - 0.5 * anchors[:, 3]
    anchors_rect[:, 2] = anchors[:, 0] + 0.5 * anchors[:, 2]
    anchors_rect[:, 3] = anchors[:, 1] + 0.5 * anchors[:, 3]
    for n in range(len(anchors_rect)):
        xx1, yy1, xx2, yy2 = anchors_rect[n, :]
        # print("xx1,yy1,xx2,yy2", xx1, yy1, xx2, yy2)
        maxx1 = max(x1, xx1)
        maxy1 = max(y1, yy1)
        minx2 = min(x2, xx2)
        miny2 = min(y2, yy2)
        width = minx2 - maxx1 + 1
        height = miny2 - maxy1 + 1
        if width <= 0 or height <= 0:
            ious[n] = 0
            continue
        area1 = (x2 - x1 + 1) * (y2 - y1 + 1)
        area2 = (xx2 - xx1 + 1) * (yy2 - yy1 + 1)
        iou = (width * height) / (area1 + area2 - width * height)
        ious[n] = iou
    inds_inside = np.where(ious > 0.6)[0]

    # logger.info('pos_cls = %d', len(inds_inside))
    # logger.info('neg_cls = %d', 100 - len(inds_inside))
    print("ispositive", ispositive)
    if ispositive == 1:
        print("inds_inside", len(inds_inside), len(labels))
        labels[inds_inside] = 1
        bbox_targets[inds_inside] = bbox_transform_inv(anchors[inds_inside], label_pos,
                                                       label_sz)  # label_pos,label_sz are for the patch
    bg_inds_inside = np.where(ious < 0.3)[0]
    labels[bg_inds_inside] = -2
    # print("labels shape", labels.shape)
    print("obj_labels_1_nums", len([x for x in labels if x == 1]))
    print("obj_labels_0_nums", len([x for x in labels if x == 0]))
    print("obj_labels_-1_nums", len([x for x in labels if x == -1]))
    # print("bbox_targets[inds_inside]", anchors[inds_inside, :])
    # print("len([m for m in labels if m==1])",len([m for m in labels if m==1 or m==0]))
    return labels, bbox_targets, inds_inside


def smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, beta=1.0):
    """
    SmoothL1(x) = 0.5 * x^2 / beta      if |x| < beta
                  |x| - 0.5 * beta      otherwise.
    1 / N * sum_i alpha_out[i] * SmoothL1(alpha_in[i] * (y_hat[i] - y[i])).
    N is the number of batch elements in the input predictions
    """
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < beta).detach().float()
    in_loss_box = smoothL1_sign * 0.5 * torch.pow(in_box_diff, 2) / beta + \
                  (1 - smoothL1_sign) * (abs_in_box_diff - (0.5 * beta))
    # in_loss_box = torch.pow(in_box_diff, 2)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    N = 1  # batch size
    loss_box = loss_box.view(-1).sum(0) / N
    return loss_box


class TrackerConfig(object):
    # These are the default hyper-params for DaSiamRPN 0.3827
    windowing = 'cosine'  # to penalize large displacements [cosine/uniform]
    # Params from the network architecture, have to be consistent with the training
    exemplar_size = 127
    # exemplar_size = 143  # input z size
    instance_size = 271  # input x size (search region)
    # instance_size = 287
    total_stride = 8
    score_size = (instance_size - exemplar_size) / total_stride + 1
    print("score_size", score_size)
    context_amount = 0.5
    #context_amount = 0.8  # context amount for the exemplar
    ratios = [0.33, 0.5, 1, 2, 3]
    scales = [8, ]
    # scales = [8, ]
    anchor_num = len(ratios) * len(scales)
    anchor = []
    penalty_k = 0.055
    window_influence = 0.42
    lr = 0.295
    #lr = 0.5


def Firstframe_init(im, target_pos, target_sz, net):
    state = dict()
    p = TrackerConfig()
    state['im_h'] = im.shape[0]
    state['im_w'] = im.shape[1]
    """
    if ((target_sz[0] * target_sz[1]) / float(state['im_h'] * state['im_w'])) < 0.004:
        p.instance_size = 287  # small object big search region
    else:
        p.instance_size = 271
    """

    p.score_size = (p.instance_size - p.exemplar_size) / p.total_stride + 1
    p.anchor = generate_anchor(p.total_stride, p.scales, p.ratios, p.score_size, p.instance_size)
    # print("print p.anchor", p.anchor)
    # print("im.shape", type(im), im.shape)
    avg_chans = np.mean(im, axis=(0, 1))

    wc_z = target_sz[0] + p.context_amount * sum(target_sz)
    hc_z = target_sz[1] + p.context_amount * sum(target_sz)
    s_z = round(np.sqrt(wc_z * hc_z))
    # initialize the exemplar
    im_patch = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)
    # cv2.destroyAllWindows()
    z_crop = im_to_torch(im_patch)
    # z_crop = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)

    z = Variable(z_crop.unsqueeze(0))
    net.temple(z.cuda())

    if p.windowing == 'cosine':
        window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))
    elif p.windowing == 'uniform':
        window = np.ones((p.score_size, p.score_size))
    window = np.tile(window.flatten(), p.anchor_num)

    state['p'] = p
    state['net'] = net
    state['avg_chans'] = avg_chans
    state['window'] = window
    state['target_pos'] = target_pos
    state['target_pos_pre'] = target_pos
    state['target_sz'] = target_sz
    return state


def Secondframe_init(ispositive, state, im, target_pos2, target_sz2, optimizer, patch_scale=1):
    p = state['p']
    net = state['net']
    # avg_chans = state['avg_chans']
    avg_chans = np.mean(im, axis=(0, 1))
    window = state['window']
    target_pos = state['target_pos']
    target_sz = state['target_sz']

    wc_z = target_sz[1] + p.context_amount * sum(target_sz)
    hc_z = target_sz[0] + p.context_amount * sum(target_sz)
    s_z = np.sqrt(wc_z * hc_z) * patch_scale
    scale_z = p.exemplar_size / s_z
    d_search = (p.instance_size - p.exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad
    s_x = s_x
    label_pos = target_pos2 - target_pos  # the offset of centerpoint
    label_sz = target_sz2
    # print("target_sz2", target_sz2)
    label_pos = scale_z * label_pos
    label_sz = scale_z * label_sz
    anchors = p.anchor
    # print("anchors outside", anchors.shape)
    # label_pos,label_sz,anchors is the position and size in patch
    # print("label_pos,label_sz",label_pos,label_sz)
    # print("anchors train", anchors)
    labels, bbox_targets, inds_inside = get_labels_bbox_targets(ispositive, label_pos, label_sz, anchors)
    if not ispositive:
        labels.fill(-1)
        enable_inds = random.sample(labels.tolist(), 25)
        labels[enable_inds] = 0
    print("labels,bbox_targets", labels.shape, bbox_targets.shape)
    # extract scaled crops for search region x at previous target position
    im_patch = get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans)
    # cv2.imshow('im_patch.jpg', im_patch)
    # cv2.waitKey(0)
    output_tmp = anchors[inds_inside]
    bbox_targets_tmp = bbox_targets[inds_inside]
    # print("output_tmp",output_tmp.shape, output_tmp)

    # print("label_pos, label_sz", label_pos, label_sz)
    x1 = int(label_pos[0] - 0.5 * label_sz[0] + p.instance_size / 2)
    y1 = int(label_pos[1] - 0.5 * label_sz[1] + p.instance_size / 2)
    x2 = int(label_pos[0] + 0.5 * label_sz[0] + p.instance_size / 2)
    y2 = int(label_pos[1] + 0.5 * label_sz[1] + p.instance_size / 2)
    draw = cv2.rectangle(im_patch, (x1, y1), (x2, y2), (255, 0, 0), 2)
    # cv2.imshow('result_final.jpg', draw)
    # cv2.waitKey(0)

    # cv2.destroyAllWindows()
    # outpath = '/home/lichong/tracking/DaSiamRPN/train/out/{}_{}_{}.jpg'.format(val,i,i)
    # cv2.imwrite(outpath,im_patch)
    z_crop = im_to_torch(im_patch)
    x_crop = Variable(z_crop.unsqueeze(0))
    # x_crop = Variable(get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans).unsqueeze(0))
    delta, score = net(x_crop.cuda())
    # print("score", score.shape)
    B, C, H, W = score.size()
    score_cls = score.view(B, 2, C // 2, H, W).permute(0, 2, 3, 4, 1).contiguous().view(-1, 2)
    # print("score_cls", score_cls.shape)
    obj_labels = torch.Tensor(labels).long()
    obj_labels = obj_labels.contiguous().view(-1).long()
    obj_labels = Variable(obj_labels.cuda())
    score_tmp = F.softmax(score.permute(1, 2, 3, 0).contiguous().view(2, -1), dim=0).data[1, :].cpu().numpy()
    order_index = np.argsort(-score_tmp)
    print("score_tmp", score_tmp[order_index])
    count = 0
    i = 0
    obj_labels_np_0 = obj_labels.cpu().numpy()
    print("score_tmp[order_index]", score_tmp[order_index])
    while True:
        if obj_labels[order_index[i]] == -2:
            obj_labels[order_index[i]] = 0
            count = count+1
            #print("negsco", score_tmp[order_index[i]])
            if count == 100:
                break
        """
        else:
            print("possco", score_tmp[order_index[i]])
        """
        i = i+1
        if order_index.shape[0] <= i:
            break
    obj_labels_np = obj_labels.cpu().numpy()
    print("obj_labels_np.shape", obj_labels_np.shape)
    for i in range(0, obj_labels_np.shape[0]):
        if obj_labels[i] == -2:
            obj_labels[i] = -1

    obj_labels_np = obj_labels.cpu().numpy()
    print("count_1", np.argwhere(obj_labels_np == 1).shape)
    print("count_-1", np.argwhere(obj_labels_np == -1).shape)
    print("count_0", np.argwhere(obj_labels_np == 0).shape)
    print("count_-2", np.argwhere(obj_labels_np == -2).shape)




    # print("obj_labels_nums", len([x for x in obj_labels if x!=-1]))
    loss_rpn_cls = F.cross_entropy(score_cls, obj_labels, ignore_index=-1)
    # print(obj_labels.cpu().detach().numpy())
    # print("ave(score_cls) train", score_cls.cpu().detach().numpy().shape, np.average(score_cls.cpu().detach().numpy()[:,0]), np.average(score_cls.cpu().detach().numpy()[:,1]))
    delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
    bbox_targets = bbox_targets.transpose()
    obj_bbox_targets = torch.Tensor(bbox_targets).float()
    obj_bbox_targets = Variable(obj_bbox_targets.cuda())

    bbox_inside_weights = np.zeros((len(labels), 4), dtype=np.float32)
    bbox_inside_weights[labels == 1, :] = [1.0, 1.0, 1.0, 1.0]
    bbox_inside_weights = bbox_inside_weights.transpose()
    bbox_outside_weights = np.zeros((len(labels), 4), dtype=np.float32)
    num_examples = np.sum(labels >= 0)
    # print("num_examples", num_examples)
    positive_weights = np.ones((1, 4)) * 1.0 / num_examples
    # negative_weights 0 or 1 ?
    negative_weights = np.zeros((1, 4)) * 1.0
    bbox_outside_weights[labels == 1, :] = positive_weights
    bbox_outside_weights[labels == 0, :] = negative_weights
    # print("bbox_outside_weights",bbox_outside_weights[labels == 1, :])
    bbox_outside_weights = bbox_outside_weights.transpose()

    bbox_inside_weights = torch.Tensor(bbox_inside_weights).float()
    bbox_inside_weights = Variable(bbox_inside_weights.cuda())
    bbox_outside_weights = torch.Tensor(bbox_outside_weights).float()
    bbox_outside_weights = Variable(bbox_outside_weights.cuda())

    # print("delta,obj_bbox_targets", delta,obj_bbox_targets)
    loss_rpn_bbox = smooth_l1_loss(delta, obj_bbox_targets, bbox_inside_weights, bbox_outside_weights, beta=1.0)
    # print("obj_bbox_targets", obj_bbox_targets)
    # print("delta[:,labels==1]", delta.cpu().detach().numpy()[:, labels == 1])
    x = obj_bbox_targets.cpu().detach().numpy()[:, labels == 1]
    y = delta.cpu().detach().numpy()[:, labels == 1]
    print("regression_label1", x[0, 5:15])
    print("regression_label2", y[0, 5:15])
    print("regression_label3", x[3, 5:15])
    print("regression_label4", y[3, 5:15])
    print("", np.average(x[3, :]), np.average(y[3, :]))
    for i in range(4):
        if len([label for label in labels if label == 1]) > 0:
            print("obj_bbox_targets[:,labels==1]", np.average(x[i, :]), np.max(x[i, :]), np.min(x[i, :]))
            print("delta[:,labels==1]", np.average(y[i, :]), np.max(y[i, :]), np.min(y[i, :]))
    if ispositive:
        loss = loss_rpn_cls + loss_rpn_bbox
    else:
        loss = loss_rpn_cls
    loss.backward()
    print("ispositive", ispositive)
    logger.info('loss_rpn_cls = %.6f', loss_rpn_cls.data.item())
    if ispositive:
        logger.info('loss_rpn_bbox = %.6f', loss_rpn_bbox.data.item())
    logger.info('loss = %.6f', loss.data.item())


    # if p.windowing == 'cosine':
    #     window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))
    # elif p.windowing == 'uniform':
    #     window = np.ones((p.score_size, p.score_size))
    # window = np.tile(window.flatten(), p.anchor_num)

    # state['p'] = p
    # state['net'] = net
    # state['avg_chans'] = avg_chans
    # state['window'] = window
    # state['target_pos'] = target_pos
    # state['target_sz'] = target_sz
    # return state


class SiamRPNBIG_Lee(nn.Module):
    def __init__(self, feat_in=512, feature_out=512, anchor=5):
        super(SiamRPNBIG_Lee, self).__init__()
        self.anchor = anchor
        self.feature_out = feature_out
        self.featureExtract = nn.Sequential(
            nn.Conv2d(3, 192, 11, stride=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(192, 512, 5),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(512, 768, 3),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            nn.Conv2d(768, 768, 3),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            nn.Conv2d(768, 512, 3),
            nn.BatchNorm2d(512),
        )
        self.conv_r1 = nn.Conv2d(feat_in, feature_out * 4 * anchor, 3)
        self.conv_r2 = nn.Conv2d(feat_in, feature_out, 3)
        self.conv_cls1 = nn.Conv2d(feat_in, feature_out * 2 * anchor, 3)
        self.conv_cls2 = nn.Conv2d(feat_in, feature_out, 3)
        # self.regress_adjust = nn.Conv2d(4*anchor, 4*anchor, 1)
        # self.r1_kernel = []
        # self.cls1_kernel = []
        self.cconv = nn.Conv2d(feature_out, self.anchor * 2, kernel_size=4, bias=False)
        self.rconv = nn.Conv2d(feature_out, self.anchor * 4, kernel_size=4, bias=False)

    def forward(self, x):
        x_f = self.featureExtract(x)
        cinput = self.conv_cls2(x_f)
        rinput = self.conv_r2(x_f)
        coutput = self.cconv(cinput)
        routput = self.rconv(rinput)
        return routput, coutput

        # return F.conv2d(self.conv_r2(x_f),self.r1_kernel), \
        #       F.conv2d(self.conv_cls2(x_f), self.cls1_kernel)

        # return self.regress_adjust( F.conv2d( self.conv_r2(x_f) , self.r1_kernel ) ), \
        #       F.conv2d(self.conv_cls2(x_f), self.cls1_kernel)

    def temple(self, z):
        z_f = self.featureExtract(z)
        r1_kernel_raw = self.conv_r1(z_f)
        cls1_kernel_raw = self.conv_cls1(z_f)
        kernel_size = r1_kernel_raw.data.size()[-1]
        # self.r1_kernel = r1_kernel_raw.view(self.anchor*4, self.feature_out, kernel_size, kernel_size)
        # self.cls1_kernel = cls1_kernel_raw.view(self.anchor*2, self.feature_out, kernel_size, kernel_size)

        r1_kernel = r1_kernel_raw.view(self.anchor * 4, self.feature_out, kernel_size, kernel_size)
        cls1_kernel = cls1_kernel_raw.view(self.anchor * 2, self.feature_out, kernel_size, kernel_size)
        self.cconv.weight = nn.Parameter(cls1_kernel)
        self.rconv.weight = nn.Parameter(r1_kernel)
        aa = 1




