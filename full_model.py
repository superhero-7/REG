import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline import LanguagePlusImage
from ClassifierHelper import SequenceLoss
import torch.optim as optim


# As described in "Generation and comprehension of unambiguous object descriptions."
# Mao, Junhua, et al.
# CVPR 2016
# Implemented by Cecilia Mauceri

#Network Definition
class LanguagePlusImage_Contrast(LanguagePlusImage):

    def __init__(self, cfg, training=True):
        super(LanguagePlusImage_Contrast, self).__init__(cfg)

        self.loss_function = MMI_MM_Loss()
        #self.val_loss_function = SequenceLoss(nn.CrossEntropyLoss())
        #self.training = training
        # self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),
        #                        lr=self.cfg.TRAINING_LEARNING_RATE, weight_decay=self.cfg.TRAINING_L2_FRACTION)

    def forward(self, ref):

        feats, contrast = self.image_forward(ref)

        #Input to LanguageModel
        ref['feats'] = feats
        embedding = F.softmax(self.wordnet(ref=ref), dim=2)  # (batch_size,max_len,vocab_size)

        #if self.training:
        for object in contrast:
            ref['feats'] = object
            embedding = torch.cat([embedding, F.softmax(self.wordnet(ref=ref), dim=2)], 0)

        return embedding


    def image_forward(self, ref, comprehension=False):

        #if self.training:
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

        contrast_out = []
        #if self.training:
        # Contrast objects
        for contrast in ref['contrast']:
            if self.use_cuda:
                contrast_item = contrast['object'].cuda()
            else:
                contrast_item = contrast['object']
            contrast_out.append(torch.cat([image_out, self.imagenet(contrast_item), contrast['pos']], 1))

        # Concatenate image representations
        return torch.cat([image_out, object_out, pos], 1), contrast_out

class MMI_MM_Loss(nn.Module):
    def __init__(self, disable_cuda=False):
        super(MMI_MM_Loss, self).__init__()
        self.Loss = SequenceLoss(nn.NLLLoss())
        self.Tanh = nn.Tanh()

        if not disable_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def forward(self, embeddings, targets):
        detal = torch.tensor(.1e-8, requires_grad=True)
        M = torch.tensor(0.5,requires_grad=True)
        dim = targets.size()[0]
        examples = embeddings[:dim, :]
        contrast = torch.zeros(examples.size(), device=self.device, dtype=torch.float)
        for i in range(dim, embeddings.size()[0]):
            contrast[i % dim, :] += embeddings[i, :]

        loss_positive = torch.zeros(dim, device=self.device, dtype=torch.float)
        loss_div = torch.zeros(dim, device=self.device, dtype=torch.float)
        loss_mm = torch.zeros(dim, device=self.device,dtype=torch.float)

        for instance in range(dim):
            for step in range(targets.size()[1]):
                div = self.Tanh(examples[instance, step, targets[instance, step]] / contrast[instance, step, targets[instance, step]])
                positive = self.Tanh(examples[instance, step, targets[instance, step]])
                loss_div[instance] += torch.log(div)
                loss_positive[instance] += torch.log(positive)
            if (loss_div[instance])<0:
                #print(loss_div[instance])
                loss_mm[instance] = -1*loss_positive[instance] -0.05*(self.Tanh(torch.log(contrast[instance, step, targets[instance, step]])))
            else:
                loss_mm[instance] = -1*loss_positive[instance]
        #torch.clamp(loss_div, -100, 0, out=None)
        #loss_mm = -1*loss_positive-0.5*loss_div

        loss = torch.mean(loss_mm)

        return loss

class MMI_softmax_Loss(nn.Module):
    def __init__(self, disable_cuda=False):
        super(MMI_softmax_Loss, self).__init__()
        self.Loss = SequenceLoss(nn.NLLLoss())
        self.Tanh = nn.Tanh()

        if not disable_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def forward(self, embeddings, targets):
        dim = targets.size()[0]
        examples = embeddings[:dim, :]
        contrast = torch.zeros(examples.size(), device=self.device, dtype=torch.float)
        for i in range(dim, embeddings.size()[0]):
            contrast[i % dim, :] += embeddings[i, :]

        #weighted = self.Tanh(torch.div(examples, contrast))
        #weighted_loss = self.Loss(weighted, targets)  # ?????????loss
        #print(loss)
        #
        # loss_ = torch.zeros(dim, device=self.device, dtype=torch.float)
        # for step in range(targets.size()[1]):
        #     for instance in range(dim):
        #         loss_[instance] += torch.log(self.Tanh(examples[instance, step, targets[instance, step]] / contrast[instance, step, targets[instance, step]]))
        # loss_ = -1 * torch.sum(loss_)
        # print(loss_)
        # #
        # contrast_loss = self.Loss(torch.log(contrast), targets)
        # print(contrast_loss)
        # # #
        weighted = torch.log(self.Tanh(torch.div(examples, contrast)))
        exampl_loss = torch.log(examples)
        l = exampl_loss + weighted
        loss = self.Loss(l, targets)
        # print(example_loss)


        return loss

    def image_forward(self, ref):
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

        # Concatenate image representations
        return torch.cat([image_out.repeat(object_out.size()[0], 1), object_out, pos], 1)

# if __name__ == "__main__":
#
#     parser = argparse.ArgumentParser(description='Classify missing words with LSTM.')
#     parser.add_argument('mode', help='train/test')
#     parser.add_argument('checkpoint_prefix',
#                         help='Filepath to save/load checkpoint. If file exists, checkpoint will be loaded')
#
#     parser.add_argument('--img_root', help='path to the image directory', default='pyutils/refer_python3/data/images/mscoco/train2014/')
#     parser.add_argument('--data_root', help='path to data directory', default='pyutils/refer_python3/data')
#     parser.add_argument('--dataset', help='dataset name', default='refcocog')
#     parser.add_argument('--splitBy', help='team that made the dataset splits', default='google')
#     parser.add_argument('--epochs', dest='epochs', type=int, default=1,
#                         help='Number of epochs to train (Default: 1)')
#     parser.add_argument('--hidden_dim', dest='hidden_dim', type=int, default=1024,
#                         help='Size of LSTM embedding (Default:100)')
#     parser.add_argument('--dropout', dest='dropout', type=float, default=0, help='Dropout probability')
#     parser.add_argument('--learningrate', dest='learningrate', type=float, default=0.001, help='Adam Optimizer Learning Rate')
#     parser.add_argument('--batch_size', dest='batch_size', type=int, default=16,
#                         help='Training batch size')
#
#
#     args = parser.parse_args()
#
#     with open('vocab_file.txt', 'r') as f:
#         vocab = f.read().split()
#     # Add the start and end tokens
#     vocab.extend(['<bos>', '<eos>', '<unk>'])
#
#     refer = ReferExpressionDataset(args.img_root, args.data_root, args.dataset, args.splitBy, vocab, use_image=True, n_contrast_object=2)
#
#     checkpt_file = LanguagePlusImage_Contrast.get_checkpt_file(args.checkpoint_prefix, args.hidden_dim, 2005, args.dropout)
#     if (os.path.isfile(checkpt_file)):
#         model = LanguagePlusImage_Contrast(checkpt_file=checkpt_file, vocab=vocab)
#     else:
#         model = LanguagePlusImage_Contrast(vocab=vocab, hidden_dim=args.hidden_dim, dropout=args.dropout)
#
#     if args.mode == 'train':
#         print("Start Training")
#         total_loss = model.run_training(args.epochs, refer, args.checkpoint_prefix, parameters={'use_image': True},
#                                         learning_rate=args.learningrate, batch_size=args.batch_size)
#         #total_loss = model.run_testing(refer, split='train', parameters={'use_image': True})
#
#     if args.mode == 'test':
#         print("Start Testing")
#         for i in range(10, 20):
#             item = refer.getItem(i, split='val', use_image=True, display_image=True)
#             item['PIL'].show()
#             print(model.generate("<bos>", item))
#             input('Any key to continue')
