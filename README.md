# torchtext的使用总结，并结合Pytorch实现LSTM

## 版本说明
- PyTorch版本：0.4.1
- torchtext：0.2.3
- python：3.6

## 文件说明
- Test-Dataset.ipynb Test-Dataset.py 使用torchtext进行文本预处理的notebook和py版。
- Test-Dataset2.ipynb 使用Keras和PyTorch构建数据集进行文本预处理。
- Language-Model.ipynb 使用gensim加载预训练的词向量，并使用PyTorch实现语言模型。

## 使用说明
- 分别提供了notebook版和标准py文件版。
- 从零开始逐步实现了torchtext文本预处理过程，包括截断补长，词表构建，使用预训练词向量，构建可用于pytorch的可迭代数据等。 

&nbsp;&nbsp;&nbsp;&nbsp;使用教程参考我的个人博客（第一个为github博客，图片显示有问题，以第二个为准）：

- http://www.nlpuser.com/pytorch/2018/10/30/useTorchText/
- https://blog.csdn.net/nlpuser/article/details/88067167

&nbsp;&nbsp;&nbsp;&nbsp;代码中在数据集中使用预训练词向量部分已注释为markdown格式，如下所示，若要使用预训练的词向量，例如glove开源的预训练词向量，需要在当前目录下创建mycache文件夹作为cache目录，并指定预训练词向量文件所在位置。glove词向量下载可参考此链接：https://pan.baidu.com/s/1i5XmTA9

    ###  通过预训练的词向量来构建词表的方式示例，以glove.6B.300d词向量为例
    cache = 'mycache'
    if not os.path.exists(cache):
        os.mkdir(cache)
    vectors = Vectors(name='/Users/wyw/Documents/vectors/glove/glove.6B.300d.txt', cache=cache)
    # 指定 Vector 缺失值的初始化方式，没有命中的token的初始化方式
    vectors.unk_init = init.xavier_uniform_ 
    TEXT.build_vocab(train, min_freq=5, vectors=vectors)
    # 查看词表元素
    TEXT.vocab.vectors
