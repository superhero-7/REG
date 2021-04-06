import torch
import torch.nn as nn
import torchvision.models as models

from LSTM import LanguageModel
from ClassifierHelper import Classifier, SequenceLoss
from VGG import VGG,ResNet
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


# Network Definition
class LanguagePlusImage(Classifier):

    def __init__(self, cfg):
        super(LanguagePlusImage, self).__init__(cfg, loss_function=SequenceLoss(nn.CrossEntropyLoss()))
        #self.val_loss_function = SequenceLoss(nn.CrossEntropyLoss())
        # Text Embedding Network
        self.wordnet = LanguageModel(cfg)

        # Image Embedding Network
        # TODO Improve this hacky way to check if this is being called by another constructor i.e. MaoEtAl_depth

        self.imagenet = VGG(cfg, models.vgg16(pretrained=True), loss_function=None)
        #self.imagenet = ResNet(cfg, models.resnet50(pretrained=True), loss_function=None)

        self.to(self.device)
        self.lr = self.cfg.TRAINING_LEARNING_RATE
        self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()),
                                   lr=self.lr)

    def forward(self, ref):
        ref['feats'] = self.image_forward(ref)

        # Input to LanguageModel
        return self.wordnet(ref=ref)

    def image_forward(self, ref, comprehension=False):
        # Global feature
        image = ref['image']
        if self.use_cuda:
            image = image.cuda()
        image_out = self.imagenet(image)

        # Object feature
        object = ref['object']
        if self.use_cuda:
            object = object.cuda()
        object_out = self.imagenet(object)

        # Position features
        # [top_left_x / W, top_left_y/H, bottom_left_x/W, bottom_left_y/H, size_bbox/size_image]
        pos = ref['pos']

        if comprehension:
            feats = torch.cat([image_out, object_out, pos], 1)
            contrasts = ref['contrast']
            for contrast in contrasts:
                contrast_obj = contrast['object']
                contrast_pos = contrast['pos']
                if self.use_cuda:
                    contrast_obj = contrast_obj.cuda()
                contrast_out = self.imagenet(contrast_obj)
                feats = torch.cat([feats, torch.cat([contrast_out, object_out, contrast_pos], 1)], 0)
            return feats
        else:
            # Concatenate image representations
            return torch.cat([image_out, object_out, pos], 1)

    def trim_batch(self, instance):
        return self.wordnet.trim_batch(instance)

    def clear_gradients(self, batch_size):
        super(LanguagePlusImage, self).clear_gradients()
        self.wordnet.clear_gradients(batch_size)

    # def run_generate(self, refer_dataset, split=None):
    #     refer_dataset.active_split = split
    #     n = len(refer_dataset)
    #     dataloader = DataLoader(refer_dataset)
    #
    #     generated_exp = [0]*len(refer_dataset)
    #     for k, instance in enumerate(tqdm(dataloader, desc='Generation')):
    #         instances, targets = self.trim_batch(instance)
    #         generated_exp[k] = dict()
    #         generated_exp[k]['generated_sentence'] = ' '.join(self.generate("<bos>", instance=instances))
    #         generated_exp[k]['refID'] = instance['refID'].item()
    #         generated_exp[k]['imgID'] = instance['imageID'].item()
    #         generated_exp[k]['objID'] = instance['objectID'][0]
    #         generated_exp[k]['objClass'] = instance['objectClass'][0]
    #
    #     return generated_exp

    def generate(self, instance=None, feats=None):
        with torch.no_grad():
            if feats is None:
                feats = self.image_forward(instance)
            return self.wordnet.generate('<bos>', feats=feats)

    # 要用comprehesion的话要另外申请dataset
    def comprehension(self, instances):
        # 进来的instances是没有batch的吧！！！就是一张图
        with torch.no_grad():
            n_objects = len(instances['contrast']) + 1
            instances, targets = self.trim_batch(instances)
            targets = targets.repeat(n_objects, 1)

            self.clear_gradients(batch_size=n_objects)

            output = dict()
            feats = self.image_forward(instances, comprehension=True)
            label_scores = self.wordnet.generate_batch('<bos>', feats=feats,
                                                       max_len=instances['vocab_tensor'].shape[1] - 1)
            loss = self.loss_function(label_scores, targets, per_instance=True)  # 从这个地方入手改comprehension

            # 取loss最小的索引，是个list,如果sorted_loss[0]=0,说明ground_truth是loss最小的，否则不是
            sorted_loss = np.argsort(loss.cpu())
            # sorted_bboxes = []
            # for idx in sorted_loss:
            #     sorted_bboxes.append(instances['bboxes'][idx])
            # if sorted_loss[0] == 0:
            #     output['p@1'] = 1
            # else:
            #     output['p@1'] = 0
            # if sorted_loss[0] == 0 or sorted_loss[1] == 0:
            #     output['p@2'] = 1
            # else:
            #     output['p@2'] = 0

            # output['n_objects'] = n_objects
            output['sorted_bboxes_idx'] = sorted_loss

            return output

    def test(self, instance, targets):
        output = dict()
        # ID用来做可视化吧
        output['annID'] = instance['annID']
        output['refID'] = instance['refID']
        output['imgID'] = instance['imageID'][0]
        #         if isinstance(instance['objectID'][0], torch.Tensor):
        #             output['objID'] = instance['objectID'][0].item()
        #         else:
        #             output['objID'] = instance['objectID'][0]
        #         output['objClass'] = instance['objectClass'][0].item()
        # 这一块我有个想法，我觉得把test搞在model里面其实有时候很混乱，test单独拉出来写可能还方便去实现一些其他功能？
        output['gt_sentence'] = " ".join([t[0] for t in instance['tokens']]) # tokens在这里派上用场了，那数据集里又不能加怎么办
        output['gen_sentence_token'] = self.generate(instance=instance)
        output['gen_sentence'] = " ".join(self.generate(instance=instance))
        output.update(self.comprehension(instance))
        return output

    def run_metrics(self, output, refer_dataset):
        # refer = refer_dataset.refer
        # hypothesis = []
        # references = []
        metrics_dict = {}

        mp1 = 0.0
        mp2 = 0.0
        # mean_objects = 0.0
        total = 0.0

        for row in output:
            # ref_id = int(row['refID'])
            # gen_sentence = row['gen_sentence']
            # hypothesis.append(row['gen_sentence'])
            # references.append([s['sent'] for s in refer.Refs[ref_id]['sentences']])

            total += 1.0
            # mean_objects += row['n_objects']
            mp1 += row['p@1']
            mp2 += row['p@2']

        # references = list(zip(*references))
        # nlgeval = NLGEval(no_skipthoughts=True, no_glove=True, metrics_to_omit=['METEOR'])  # loads the models
        # metrics_dict = nlgeval.compute_metrics(references, hypothesis)

        metrics_dict['p@1'] = mp1 / total
        metrics_dict['p@2'] = mp2 / total

        return metrics_dict