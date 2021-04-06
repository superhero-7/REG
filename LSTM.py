import torch
import torch.nn as nn

# torch.manual_seed(1)

from ClassifierHelper import Classifier, SequenceLoss


# Network Definition
class LanguageModel(Classifier):

    def __init__(self, cfg):  # checkpt_file=None, vocab=None, hidden_dim=None, dropout=0, additional_feat=0):
        super(LanguageModel, self).__init__(cfg, loss_function=SequenceLoss(nn.CrossEntropyLoss()))

        # Word Embeddings
        with open(cfg.DATASET_VOCAB, 'r', encoding='utf-8') as f:
            vocab = f.read().split('#')  # 这要记得改成用#号间隔
        # Add the start and end tokens
        vocab.extend(['<bos>', '<eos>', '<unk>'])

        self.word2idx = dict(zip(vocab, range(1, len(vocab) + 1)))
        self.ind2word = ['<>'] + vocab
        self.vocab_dim = len(vocab) + 1
        self.embedding = torch.nn.Embedding(self.vocab_dim, self.cfg.LSTM_EMBED, padding_idx=0)

        # The LSTM takes word embeddings as inputs, and outputs hidden states with dimensionality hidden_dim
        self.dropout1 = nn.Dropout(p=self.cfg.TRAINING_DROPOUT)
        self.lstm = nn.LSTM(self.cfg.LSTM_EMBED + self.cfg.IMG_NET_FEATS, self.cfg.LSTM_HIDDEN, batch_first=True)
        self.dropout2 = nn.Dropout(p=self.cfg.TRAINING_DROPOUT)  # 这个地方dropout放的地方和用法都还不是很清楚
        self.hidden2vocab = nn.Linear(self.cfg.LSTM_HIDDEN, self.vocab_dim)
        self.hidden = self.init_hidden(1)

        self.to(self.device)

    # 这个init_hidden是个什么玩意？？ 终于知道了，用来初始化hidden的，不过为啥是两个一样的！？
    def init_hidden(self, batch_size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, batch_size, self.cfg.LSTM_HIDDEN, device=self.device, requires_grad=True),
                torch.zeros(1, batch_size, self.cfg.LSTM_HIDDEN, device=self.device, requires_grad=True))

    def forward(self, ref=None):
        sentence = ref['vocab_tensor'][:, :-1]  # 最后一个结束词没有取的呀
        embeds = self.embedding(sentence)
        embeds = self.dropout1(embeds)  # 为什么要用embeds
        n, m, b = embeds.size()  # n:batch_size, m:序列长度 b:embedding_dim

        if 'feats' in ref:
            feats = ref['feats'].repeat(m, 1, 1).permute(1, 0, 2)  # batch_size*m(序列长度)*2005

            # Concatenate text embedding and additional features
            # TODO fix for Maoetal_Full
            if embeds.size()[0] == 1:
                embeds = torch.cat([embeds.repeat(feats.size()[0], 1, 1), feats], 2)
            else:
                embeds = torch.cat([embeds, feats], 2)  # 按最后一维cat起来

        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = self.dropout2(lstm_out)  # 为毛线这dropout放在这个地方哦
        vocab_space = self.hidden2vocab(lstm_out)
        return vocab_space

    def trim_batch(self, ref):
        ref['vocab_tensor'] = ref['vocab_tensor'][:, torch.sum(ref['vocab_tensor'], 0) > 0]
        # 牛逼！！！ 上面这个操作是把很多padding是0的都给删了！！！！！！
        target = ref['vocab_tensor'][:, 1:].clone().detach()
        return ref, target

    def clear_gradients(self, batch_size):
        super(LanguageModel, self).clear_gradients()
        self.hidden = self.init_hidden(batch_size)

    def generate(self, start_word='<bos>', feats=None, max_len=30):
        sentence = []
        word_idx = self.word2idx[start_word]
        end_idx = self.word2idx['<eos>']

        self.clear_gradients(batch_size=1)

        idx = 0
        with torch.no_grad():
            while word_idx != end_idx and len(sentence) < max_len:
                ref = self.make_ref(word_idx, feats)
                output = self(ref)
                word_idx = torch.argmax(output)  # 这地方要不要+1呢

                if word_idx != end_idx:
                    sentence.append(self.ind2word[word_idx])
                idx += 1  # 这个idx有点废物

        return sentence

    def generate_batch(self, start_word='<bos>', feats=None, max_len=30):
        tensor = torch.zeros((feats.shape[0], max_len, self.vocab_dim), device=self.device)
        word_idx = self.word2idx[start_word]

        self.clear_gradients(batch_size=feats.shape[0])

        with torch.no_grad():
            for idx in range(max_len):
                ref = self.make_ref(word_idx, feats)
                output = self(ref)
                tensor[:, idx, :] = output.squeeze(1)

        return tensor

    def make_ref(self, word_idx, feats=None):
        ref = {'vocab_tensor': torch.tensor([word_idx, -1], dtype=torch.long, device=self.device).unsqueeze(0)}
        if feats is not None:
            ref['feats'] = feats
        return ref

    def test(self, instance):
        return self.generate(instance=instance)
