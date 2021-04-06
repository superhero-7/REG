import json
import time
import numpy as np
import random


class REFER:

    def __init__(self, cfg, train=False):  # data_root, image_dir, dataset, version):
        # provide data_root folder which contains refclef, refcoco, refcoco+ and refcocog
        # also provide dataset name and splitBy information
        # e.g., dataset = 'refcoco', splitBy = 'unc'
        print('loading dataset into memory...')
        if train:
            self.DATA_DIR = cfg.DATASET_TRAIN_DATA_ROOT
        else:
            self.DATA_DIR = cfg.DATASET_VAL_DATA_ROOT
        #self.IMAGE_DIR = cfg.DATASET_IMG_ROOT

        # load refs from data_root/refs(splitBy).json
        tic = time.time()

        with open(self.DATA_DIR, 'r') as f:
            self.data = json.load(f)

        # create index
        self.createIndex()
        print('DONE (t=%.2fs)' % (time.time() - tic))

    def createIndex(self):
        # create sets of mapping
        # 1)  Refs: 	 	{ref_id: ref} 本来就有
        # 2)  Anns: 	 	{ann_id: ann} 本来就有
        # 3)  Imgs:		 	{image_id: image} 本来就有
        # 4） annToRef:  	{ann_id: ref}
        # 5)  annToimg: {ann_id: img}
        print('creating index...')
        Refs = self.data['refexps']
        Anns = self.data['annotations']
        Imgs = self.data['images']

        annToRef, annToimg = {}, {}

        for ann_id, ann in Anns.items():
            #rdn_index = random.choice(ann['refexp_ids'])
            rdn_index = ann['refexp_ids'][0]  # 直接进行一个0的取，这样做会遭报应吧！！！
            ix = str(rdn_index)
            annToRef[ann_id] = Refs[ix]


        for ann_id, ann in Anns.items():
            ix = str(ann['image_id'])
            annToimg[ann_id] = Imgs[ix]

        # create class members
        self.Refs = Refs
        self.Anns = Anns
        self.Imgs = Imgs
        self.annToRef = annToRef
        self.annToimg = annToimg
        print('index created.')