import pandas as pd
import numpy as np
import torch
import time
import random
import os
from torchtext import data
from torchtext.vocab import Vectors
from torch.nn import init
from tqdm import tqdm
from torchtext.data import Iterator, BucketIterator
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


tokenize = lambda x: x.split()
TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True, fix_length=200)
LABEL = data.Field(sequential=False, use_vocab=False)
train_path = 'data/train_one_label.csv'
valid_path = "data/valid_one_label.csv"
test_path = "data/test.csv"


# 定义Dataset
class MyDataset(data.Dataset):

    def __init__(self, path, text_field, label_field, test=False, aug=False, **kwargs):
        fields = [("id", None), # we won't be needing the id, so we pass in None as the field
                 ("comment_text", text_field), ("toxic", label_field)]
        
        examples = []
        csv_data = pd.read_csv(path)
        print('read data from {}'.format(path))

        if test:
            # 如果为测试集，则不加载label
            for text in tqdm(csv_data['comment_text']):
                examples.append(data.Example.fromlist([None, text, None], fields))
        else:
            for text, label in tqdm(zip(csv_data['comment_text'], csv_data['toxic'])):
                if aug:
                    # do augmentation
                    rate = random.random()
                    if rate > 0.5:
                        text = self.dropout(text)
                    else:
                        text = self.shuffle(text)
                # Example: Defines a single training or test example.Stores each column of the example as an attribute.
                examples.append(data.Example.fromlist([None, text, label], fields))
        # 之前是一些预处理操作，此处调用super调用父类构造方法，产生标准Dataset
        # super(MyDataset, self).__init__(examples, fields, **kwargs)
        super(MyDataset, self).__init__(examples, fields)

    def shuffle(self, text):
        text = np.random.permutation(text.strip().split())
        return ' '.join(text)

    def dropout(self, text, p=0.5):
        # random delete some text
        text = text.strip().split()
        len_ = len(text)
        indexs = np.random.choice(len_, int(len_ * p))
        for i in indexs:
            text[i] = ''
        return ' '.join(text)


class LSTM(nn.Module):

    def __init__(self, weight_matrix):
        super(LSTM, self).__init__()
        self.word_embeddings = nn.Embedding(len(TEXT.vocab), 300)  # embedding之后的shape: torch.Size([200, 8, 300])
        # 若使用预训练的词向量，需在此处指定预训练的权重
        embedding.weight.data.copy_(weight_matrix)
        self.lstm = nn.LSTM(input_size=300, hidden_size=128, num_layers=1)  # torch.Size([200, 8, 128])
        self.decoder = nn.Linear(128, 2)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out = self.lstm(embeds)[0]  # lstm_out:200x8x128
        # 取最后一个时间步
        final = lstm_out[-1]  # 8*128
        y = self.decoder(final)  # 8*2 
        return y


def data_iter(train_path, valid_path, test_path, TEXT, LABEL):
    train = MyDataset(train_path, text_field=TEXT, label_field=LABEL, test=False, aug=1)
    valid = MyDataset(valid_path, text_field=TEXT, label_field=LABEL, test=False, aug=1)
    # 因为test没有label,需要指定label_field为None
    test = MyDataset(test_path, text_field=TEXT, label_field=None, test=True, aug=1)

    TEXT.build_vocab(train)
    weight_matrix = TEXT.vocab.vectors
    # 若只针对训练集构造迭代器
    # train_iter = data.BucketIterator(dataset=train, batch_size=8, shuffle=True, sort_within_batch=False, repeat=False)
    train_iter, val_iter = BucketIterator.splits(
            (train, valid), # 构建数据集所需的数据集
            batch_sizes=(8, 8),
            # 如果使用gpu，此处将-1更换为GPU的编号
            device=-1,
            # the BucketIterator needs to be told what function it should use to group the data.
            sort_key=lambda x: len(x.comment_text),
            sort_within_batch=False,
            repeat=False
    )
    test_iter = Iterator(test, batch_size=8, device=-1, sort=False, sort_within_batch=False, repeat=False)
    return train_iter, val_iter, test_iter, weight_matrix


def main():
    train_iter, val_iter, test_iter, weight_matrix = data_iter(train_path, valid_path, test_path, TEXT, LABEL)

    model = LSTM(weight_matrix)
    model.train()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
    loss_funtion = F.cross_entropy

    for epoch, batch in enumerate(train_iter):
        optimizer.zero_grad()
        predicted = model(batch.comment_text)

        loss = loss_funtion(predicted, batch.toxic)
        loss.backward()
        optimizer.step()
        print(loss)


if __name__ == '__main__':
    main()
