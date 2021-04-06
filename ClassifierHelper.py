from tqdm import *
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os

# torch.manual_seed(1)


class Classifier(nn.Module):
    def __init__(self, cfg, loss_function):
        super(Classifier, self).__init__()
        self.cfg = cfg
        self.total_loss = []
        self.val_loss = []
        self.start_epoch = 0
        self.loss_function = loss_function


        if not cfg.MODEL_DISABLE_CUDA and torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.use_cuda = True
            print("Using cuda")
        else:
            self.device = torch.device('cpu')
            self.use_cuda = False
            print("Using cpu")

    # forward只是象征性地写了一下，没有什么实际作用
    def forward(self, instance, parameters):
        pass


    def load_model(self, checkpt_file):
        print("=> loading checkpoint '{}'".format(checkpt_file))
        #checkpoint = torch.load(checkpt_file, map_location=lambda storage, loc: storage) # GPU->CPU
        checkpoint = torch.load(checkpt_file)

        self.start_epoch = checkpoint['epoch'] + 1
        self.total_loss = checkpoint['total_loss']
        self.val_loss = checkpoint['val_loss']
        self.load_state_dict(checkpoint['state_dict'])
        #self.optimizer.load_state_dict(checkpoint['optimizer_dict'])

        print("=> loaded checkpoint '{}' (epoch {})"
              .format(checkpt_file, checkpoint['epoch']))

    def save_model(self, checkpt_prefix, params):
        print("=> saving '{}'".format(checkpt_prefix))
        torch.save(params, checkpt_prefix)

    @staticmethod
    def checkpt_file(cfg, epoch):
        return 'checkpoints/{}.mdl.checkpoint{}'.format(cfg.OUTPUT_CHECKPOINT_PREFIX, epoch)


    def run_training(self, refer_dataset, val_dataset):
        log_dir = os.path.join("output_")
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        writer = SummaryWriter(log_dir=log_dir)
        # TODO currently throwing error (DepthVGGorAlex object argument after * must be an iterable, not NoneType)
        # writer.add_graph(self.cpu(), images)



        #         if isinstance(refer_dataset, tuple):
        #             train_dataset = refer_dataset[0]
        #             test_dataset = refer_dataset[1]
        #             val_dataset = refer_dataset[2]
        #         else:
        #             train_dataset = refer_dataset
        #             test_dataset = refer_dataset
        #             val_dataset = refer_dataset

        # train_dataset.active_split = 'train'



        train_dataset = refer_dataset # 先不加validation set
        val_dataset = val_dataset # 用来做metrics的


        if self.use_cuda:
            dataloader = DataLoader(train_dataset, self.cfg.TRAINING_BATCH_SIZE, shuffle=True)
        else:
            dataloader = DataLoader(train_dataset, self.cfg.TRAINING_BATCH_SIZE, shuffle=True, num_workers=4)

        print("Before training")
        #train_loss = self.compute_average_loss(train_dataset, 'train', batch_size=self.cfg.TRAINING_BATCH_SIZE) ####改到这先跳去改函数
        #print('Average training loss:{}'.format(train_loss))
        #self.display_metrics(test_dataset, 'train')

        for epoch in range(self.start_epoch, self.cfg.TRAINING_N_EPOCH):
            self.train()
            # train_dataset.active_split = 'train'
            self.total_loss.append(0)

            # 一个epoch大概是2500个batch所以我觉得两个epoch差不多更新一次
            if epoch != 0 and epoch%30==0:
                self.lr = 0.5*self.lr
                print("Now learning rate change to {}".format(self.lr))

                self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()),
                                      lr=self.lr)


            #for i_batch, sample_batched in enumerate(tqdm(dataloader, desc='{}rd epoch'.format(epoch))):
            for i_batch,sample_batched in enumerate(dataloader):
                instances, targets = self.trim_batch(sample_batched)
                self.clear_gradients(batch_size=targets.size()[0])

                label_scores = self(instances)
                loss = self.loss_function(label_scores, targets)

                #assert torch.isnan(loss).sum() == 0, print(loss)
                print("Epoch [%d/%d], Iter [%d/%d] Loss: %.4f" % (epoch + 1, 300, i_batch + 1, 2802, loss.data.item()))

                loss.backward()
                # 下面这东西跟下面自己写的那个东西其实是一样的...
                torch.nn.utils.clip_grad_value_(self.parameters(), 10)

                # for param in self.optimizer.param_groups[0]["params"]:
                #     param.grad.data.clamp_(-10, 10)
                grad = [x.grad for x in self.optimizer.param_groups[0]['params']][0]
                grad = torch.tensor(grad)
                assert torch.isnan(grad).sum() == 0, print(grad)
                self.optimizer.step()

                self.total_loss[epoch] += loss.item()

                if self.cfg.DEBUG and i_batch == 5:
                    break

            if epoch % 5 == 0:
                val_loss = self.compute_average_loss(val_dataset, 'val', batch_size=self.cfg.TRAINING_BATCH_SIZE)
                self.val_loss.append(val_loss)
                writer.add_scalar('Average val loss', val_loss, global_step=epoch)
                print('\nAverage val loss:{}'.format(val_loss))


            self.total_loss[epoch] = self.total_loss[epoch] / float(i_batch + 1)
            writer.add_scalar('Average training loss', self.total_loss[epoch], global_step=epoch)

            print('\nAverage training loss:{}'.format(self.total_loss[epoch]))
            self.save_model(self.checkpt_file(self.cfg, epoch), {
                'epoch': epoch,
                'state_dict': self.state_dict(),
                'optimizer_dict': self.optimizer.state_dict(),
                'total_loss': self.total_loss,
                'val_loss': self.val_loss,
            })


        writer.close()
        return self.total_loss

    # 这个函数也基本不用动，用来计算整个dataset的平均loss（以batch为单位，因为没有除batch）
    def compute_average_loss(self, val_dataset, split=None, batch_size=16):
        self.eval()
        # refer_dataset.active_split = split
        dataloader = DataLoader(val_dataset, batch_size=batch_size)

        total_loss = 0
        for k, instance in enumerate(tqdm(dataloader, desc='Average loss on {}'.format(split))):
            with torch.no_grad():
                instances, targets = self.trim_batch(instance)
                self.clear_gradients(batch_size=targets.size()[0])

                label_scores = self(instances)
                total_loss += self.loss_function(label_scores, targets)

            if self.cfg.DEBUG and k == 5:
                break

        return total_loss /float(k)

    def run_test(self, test_dataset, split=None):
        self.eval()
        # refer_dataset.active_split = split

        # 还是没弄懂n_contrast_objecr到底是在干嘛的 用来做comprehension的，一会再来动这块，现在dataset里面没有
        # This is a hack to make sure that comprehension works in MaoEtAl_baseline regardless of whether contrast objects were used in training
        #         if hasattr(refer_dataset, 'n_contrast_object'):
        #             if self.cfg.TRAINING.N_CONSTRAST_OBJECT == 0:
        #                 refer_dataset.n_contrast_object = float('inf')

        dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

        output = list()
        for k, batch in enumerate(tqdm(dataloader, desc='Test {}'.format(split))):
            instances, targets = self.trim_batch(batch)
            output.append(self.test(instances, targets)) # 这一块跳去看test 这个函数改成这样就不用动了，去改baseline里的东西就行了

            # Large test sets can be very slow to process. Therefore, default only processes a random sample of 10000
            if not self.cfg.TEST_DO_ALL and k > 100:
                break

            if self.cfg.DEBUG and k == 5:
                break

        return output

    # 启动顺序是： disply_metrics-->run_test-->test-->run_metrics
    def display_metrics(self, test_dataset, split=None, verbose=False, writer=None, epoch=0):
        output = self.run_test(test_dataset, split) # 这个output是一个list是baseline里面的output的集合
        metrics = self.run_metrics(output, test_dataset) # metrics_dict contains p@1,p@2

        # 如果不去算meteor指标的话，其实下面这块都可以简化掉的
        #         for key, value in metrics.items():
        #             if isinstance(value, list) and verbose:
        #                 headers = value[0].keys()
        #                 print("\t".join(headers))
        #                 for entry in value:
        #                     print("\t".join(entry.values()))
        #             elif not isinstance(value, list):
        #                 print('{}:\t{:.3f}'.format(key, value))
        #                 if writer is not None:
        #                     writer.add_scalar('{}_{}'.format(split, key), value, global_step=epoch)
        for key, value in metrics.items():
            print('{}:\t{:.3f}'.format(key, value))

    def run_metrics(self, output, refer):
        pass

    def test(self, instance, targets):
        pass

    def trim_batch(self, instance):
        pass

    def clear_gradients(self, batch_size=None):
        self.zero_grad()  # 人家是optimizer.zero_grad()他这个怎么直接self了 好像说是等价的

# 搞不懂这个class的作用诶， 这个class就是用来输入loss_function的啦
class SequenceLoss(nn.Module):
    def __init__(self, loss_function, disable_cuda=False):
        super(SequenceLoss, self).__init__()
        self.Loss = loss_function

        if not disable_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def forward(self, embeddings, targets, per_instance=False):

        if per_instance:
            reduction_setting = self.Loss.reduction
            # If you use the per_instance setting, your Loss function must have reduction=='none'
            self.Loss.reduction = 'none'
            loss = torch.zeros(embeddings.size()[0], device=self.device)
            for step in range(targets.size()[1]):
                # TODO try the mean here instead
                loss += self.Loss(embeddings[:, step, :], targets[:, step])
            self.Loss.reduction = reduction_setting
        else:
            loss = 0.0
            for step in range(targets.size()[1]):
                loss += self.Loss(embeddings[:, step, :], targets[:, step])  # 确实loss就是一整个句子的loss!
        #loss = loss/(targets.size()[1]+1)

        return loss