# this is an example on how to use COCOAPI
from __future__ import print_function
import os, sys, zipfile
import numpy as np
import skimage.io as io
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import pylab
import cv2
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

annFile = "instances_val2017.json"
coco=COCO(annFile)

#category
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

#supercategory
nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))


catIds = coco.getCatIds(catNms=['person','dog','skateboard']);
imgIds = coco.getImgIds(catIds=catIds );
print("imgIds", imgIds)
#imgIds = coco.getImgIds(imgIds = [324158])
#img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
img = coco.loadImgs(imgIds)[0]

print("**********",img)
I = io.imread("D:\\Yi+\COCO\\val2017\\"+img["file_name"])

plt.imshow(I); plt.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
#annIds = coco.getAnnIds()
print("annIds",annIds)
anns = coco.loadAnns(annIds)
i = 1
print("anns", anns[i]["bbox"])
coco.showAnns([anns[i]])
plt.show()
print("anns[i]", anns[i])
anns = [int(x) for x in anns[i]["bbox"]]

draw = cv2.rectangle(I, (anns[0], anns[1]), (anns[0]+anns[2], anns[1]+anns[3]), (255, 255, 255), 2)
#target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
#draw = cv2.rectangle(I, (10, 10), (630, 420), (255, 255, 255), 2)
cv2.imshow('result_final.jpg', draw)
cv2.waitKey(0)