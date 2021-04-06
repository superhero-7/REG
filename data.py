import os, random, json, math
from PIL import ImageDraw, Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL.ImageStat import Stat
import numpy as np


class ImageProcessing:
    def __init__(self, cfg, img_root=None, data_root=None):  # img_dir, depth_dir, data_dir,
        # disable_cuda=False, transform_size=224,
        # image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225], use_image=False,
        # depth_mean=19018.9, depth_std=18798.8, use_depth=False):

        if not cfg.MODEL_DISABLE_CUDA and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.use_image = cfg.IMG_PROCESSING_USE_IMAGE

        self.IMAGE_DIR = cfg.DATASET_IMG_ROOT
        #self.DATA_DIR = cfg.DATASET_DATA_ROOT

        if img_root is not None:
            self.IMAGE_DIR = img_root
        if data_root is not None:
            self.DATA_DIR = data_root

        self.toTensorTransform = transforms.ToTensor()

        image_mean = cfg.IMG_PROCESSING_IMG_MEAN
        image_std = cfg.IMG_PROCESSING_IMG_STD
        self.image_size = cfg.IMG_PROCESSING_TRANSFORM_SIZE
        self.img_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=image_mean,
                                 std=image_std)
        ])

        # Use these to visualize tensors
        rev_mean = [-1.0 * image_mean[0] / image_std[0], -1.0 * image_mean[1] / image_std[1],
                    -1.0 * image_mean[2] / image_std[2]]
        rev_std = [1.0 / image_std[0], 1.0 / image_std[1], 1.0 / image_std[2]]
        self.rev_img_normalize = transforms.Compose([
            transforms.Normalize(mean=rev_mean, std=rev_std),
            transforms.ToPILImage(mode='RGB')
        ])

    def standarizeImageFormat(self, image, bbox=None):
        image = image.copy()

        # Crop if bbox is smaller than image size
        if bbox is not None:
            width = int(bbox[2])
            height = int(bbox[3])
            bottom = int(bbox[1])
            left = int(bbox[0])

            image = image.crop((left, bottom, left + width, bottom + height))

        # Scale to self.image_size
        width, height = image.size
        ratio = float(self.image_size) / max([width, height])
        new_size = tuple([int(x * ratio) for x in [width, height]])
        image = image.resize(new_size, Image.ANTIALIAS)

        # Pad with mean value
        if image.mode == 'RGB':
            stat = Stat(image)
            median_val = tuple([int(x) for x in stat.median])
        elif image.mode == 'I':
            median_val = int(np.round(np.median(image, axis=0))[0])
        else:
            raise ValueError('Mode not supported')
        pad_image = Image.new(image.mode, (self.image_size, self.image_size), median_val)
        pad_image.paste(image, (int((self.image_size - new_size[0]) / 2), int((self.image_size - new_size[1]) / 2)))

        return pad_image

    def getAllImageFeatures(self, file_name):
        sample = dict()

        # Load the image
        img_name = os.path.join(self.IMAGE_DIR, file_name)
        raw_image = Image.open(img_name)

        w, h = raw_image.size

        if raw_image.mode != "RGB":
            raw_image = raw_image.convert("RGB")

        # Scale and normalize image
        image = self.img_normalize(self.standarizeImageFormat(raw_image))
        sample['image'] = image.to(self.device)

        return sample, raw_image


# Class to load referring expressions datasets
class ReferExpressionDataset(Dataset):

    def __init__(self, cfg, refer, img_root=None, data_root=None, test=False, MMI=False,
                 split=False):  # depth_dir, vocab, disable_cuda=False, transform_size=224,
        # image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225], use_image=False,
        # depth_mean=19018.9, depth_std=18798.8, use_depth=False, n_contrast_object=0):

        super(ReferExpressionDataset, self).__init__()

        if not cfg.MODEL_DISABLE_CUDA and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.image_process = ImageProcessing(cfg, img_root, data_root)
        self.refer = refer
        self.n_contrast_object = cfg.TRAINING_N_CONSTRAST_OBJECT
        self.test = test
        self.MMI = MMI

        #coco_category
        self.coco_id_name_map = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
                            6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
                            11: 'fire hydrant', 12: 'street sign', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
                            16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
                            22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 26: 'hat', 27: 'backpack',
                            28: 'umbrella', 29: 'shoe', 30: 'eye glasses', 31: 'handbag', 32: 'tie', 33: 'suitcase',
                            34: 'frisbee',
                            35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
                            40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
                            44: 'bottle', 45: 'plate', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife',
                            50: 'spoon',
                            51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
                            56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
                            61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
                            66: 'mirror', 67: 'dining table', 68: 'window', 69: 'desk', 70: 'toilet',
                            71: 'door', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
                            77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
                            82: 'refrigerator', 83: 'blender', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors',
                            88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}




        # Load the vocabulary
        with open(cfg.DATASET_VOCAB, 'r', encoding='utf-8') as f:
            vocab = f.read().split('#')

        # Add the start and end tokens
        vocab.extend(['<bos>', '<eos>', '<unk>'])

        self.max_sent_len = max(
            [len(sent['tokens']) for sent in self.refer.Refs.values()]) + 2  # For the begining and end tokens
        self.word2idx = dict(zip(vocab, range(1, len(vocab) + 1)))
        self.idx2word = {}
        for word, idx in self.word2idx.items():
            self.idx2word[idx] = word
        self.sent2vocab(self.word2idx)

        self.index = [str(ann_id) for ann_id in self.refer.Anns]

        if split:
            self.index = self.index[:int(0.2*len(self.index))]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        return self.getItem(item)

    def getItem(self, idx, display_image=False, ):
        sample = {}

        ann_idx = self.index[idx]

        sentence = self.refer.annToRef[ann_idx]['vocab_tensor']
        sample['vocab_tensor'] = sentence

        # 先不要下面这些东西，不然dataloader会出大问题，而且也没啥用啊我寻思着
        if self.test:
            sample['annID'] = [self.refer.Anns[ann_idx]['annotation_id']]
            sample['refID'] = [self.refer.Anns[ann_idx]['refexp_ids']]
            sample['imageID'] = [self.refer.Anns[ann_idx]['image_id']]
            sample['tokens'] = self.refer.annToRef[ann_idx]['tokens']
        # sample['objectClass'] = [self.refer.Anns[ann_idx]['category_id']]
        #         imageID = self.refer.Anns[ann_idx]['image_id']
        #         if self.comprehesion:
        #             #sample['contrast'] = random.sample(self.refer.Imgs[str(imageID)]['region_candidates'],k=3) # 随机选5个成不成呢？ 改成comprehension的时候有有这个就行了
        #             sample['region_candidates'] = self.refer.Imgs[str(imageID)]['region_candidates']

        if self.image_process.use_image or display_image:
            image_features, raw_image = self.getAllImageFeatures(ann_idx, display_image=display_image)
            sample.update(image_features)

        return sample

    # 把sentence数字化罢了，这个工作我也做了
    def sent2vocab(self, word2idx):
        begin_index = word2idx['<bos>']
        end_index = word2idx['<eos>']
        unk_index = word2idx['<unk>']

        for sentence in self.refer.Refs.values():
            sentence['vocab'] = [begin_index]
            for token in sentence['tokens']:
                if token in word2idx:
                    sentence['vocab'].append(word2idx[token])
                else:
                    sentence['vocab'].append(unk_index)
            sentence['vocab'].append(end_index)

            padding = [0] * (self.max_sent_len - len(sentence['vocab']))  # 这个地方改成0就不会warnning
            sentence['vocab_tensor'] = torch.tensor(padding + sentence['vocab'], dtype=torch.long, device=self.device)

    def getObject(self, bbox, image, depth=None):
        w, h = image.size

        pos = torch.tensor(
            [bbox[0] / w, bbox[1] / h, (bbox[0] + bbox[2]) / w, (bbox[1] + bbox[3]) / h, (bbox[2] * bbox[3]) / (w * h)],
            # 这地方我觉得不对啊
            dtype=torch.float, device=self.device)
        rgb_object = self.image_process.img_normalize(self.image_process.standarizeImageFormat(image, bbox=bbox))

        return rgb_object.to(self.device), pos

    def getAllImageFeatures(self, ann_idx, display_image=False):

        # Load the image
        file_name = self.refer.annToimg[ann_idx]['file_name']
        sample, raw_image = self.image_process.getAllImageFeatures(file_name)

        # Extract a crop of the target bounding box
        bbox = self.refer.Anns[ann_idx]['bbox']
        sample['object'], sample['pos'] = self.getObject(image=raw_image, bbox=bbox)

        # Extract crops of contrast objects 这块暂时没有用
        if self.test or self.MMI:
            sample['contrast'] = self.getContrastObjects(ann_idx, raw_image=raw_image)

        if display_image:
            bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            draw = ImageDraw.Draw(raw_image)
            draw.rectangle(bbox, fill=None, outline=(255, 0, 0, 255))
            del draw

        return sample, raw_image

    # 这个函数暂时不知道怎么用，好像也没有用，不过这个可以用来取区域好像，到后面理解那块可能可以用上
    def getContrastObjects(self, ann_idx, raw_image=None):

        if raw_image is None:
            # Load the image
            file_name = self.refer.annToimg[ann_idx]['file_name']
            sample, raw_image = self.image_process.getAllImageFeatures(file_name)

        imageID = self.refer.Anns[ann_idx]['image_id']
        CatID = self.refer.Anns[ann_idx]['category_id']
        CatName = self.coco_id_name_map[CatID]
        region_candidates = self.refer.Imgs[str(imageID)]['region_candidates']
        if self.test:
            bboxes = [region_candidate['bounding_box'] for region_candidate in region_candidates]
            # bboxes = random.sample(bboxes, min(n_contrast_object, len(bboxes))) 直接全要把
        if self.MMI:
            bboxes = []
            for region_candidate in region_candidates:
                if region_candidate['predicted_object_name'] == CatName:
                    bboxes.append(region_candidate['bounding_box'])
            if len(bboxes)==0:
                if len(region_candidates) != 0:
                    ran_region = random.choice(region_candidates)
                    bboxes.append(ran_region['bounding_box'])
                else:
                    bboxes.append(self.refer.Anns[ann_idx]['bbox'])  # 这地方小小地动了点手脚,这样做还是不太行
                    #bboxes = []
            else:
                bboxes = [random.choice(bboxes)]

        # Get the image crops for the selected bounding boxes
        contrast = []
        if self.test or self.MMI:
            for bbox in bboxes:
                object, pos = self.getObject(image=raw_image, bbox=bbox)
                contrast.append({'object': object, 'pos': pos})

        # if self.MMI:
        #     object, pos = self.getObject(image=raw_image, bbox=bbox)
        #     contrast.append({'object': object, 'pos': pos})
        #
        # tmp = []
        # tmp.append(self.refer.Anns[ann_idx]['bbox'])
        # tmp[1:] = bboxes[:]
        # bboxes = tmp

        return contrast