import torch
import os
import pickle
import torch.nn as nn
import sys
sys.path.append('..')
import torch.nn.functional as F
import TextFooler.dataloader as dataloader
import TextFooler.modules as modules
import numpy as np
from core.bnn import BNN
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

class TextWrapper(object):
    def __init__(self, embedding_path=None, scd_path=None, cuda=True, max_length=400, pad=False,
                 tfidf_path=None, vote=None, soft=0):
        self.cuda = cuda
        self.scd_type = None
        self.max_length = max_length
        self.pad = pad
        self.tfidf = None
        self.vote = vote
        self.soft = soft
        if tfidf_path:
            with open(tfidf_path, 'rb') as f:
                self.tfidf = pickle.load(f)
        if embedding_path:
            self.emb_layer = modules.EmbeddingLayer(
                embs=dataloader.load_embedding(embedding_path)
            )
            self.word2id = self.emb_layer.word2id
            if self.cuda:
                self.emb_layer.cuda()
        if scd_path:
            if 'approx' in scd_path:
                if self.vote:
                    self.scd = BNN(['%s_%d.h5' % (scd_path, i) for i in range(self.vote)])
                else:
                    self.scd = BNN(['%s_%d.h5' % (scd_path, i) for i in range(8)])
            else:
                with open(scd_path, 'rb') as f:
                    self.scd = pickle.load(f)
                    if self.vote:
                        self.scd.round = self.vote
        self.oov_id = self.word2id['<oov>']
        if scd_path:
            if 'scd' in scd_path:
                self.scd_type = 'scd'
                if self.vote:
                    self.scd.round = self.vote

            elif 'approx' in scd_path:
                self.scd_type = 'bnn'
            elif 'mlp' in scd_path or 'rf' in scd_path:
                self.scd_type = 'mlp'




    def text_pred(self, text, batch_size=None):

        x = self.convert_features(text)
        if self.scd_type == 'mlp' or self.scd_type == 'bnn':
            yp = self.scd.predict_proba(x)

        elif self.scd_type == 'scd':
            yp = self.scd.predict(x, prob=True, soft=self.soft)
            yp = np.concatenate([1-yp, yp], axis=1)

        elif self.scd_type is None:
            return

        return torch.from_numpy(yp).cuda()

    def word2vector(self, text):
        x = [self.word2id.get(word, self.oov_id) for word in text]
        # ids = x
        # weights = torch.from_numpy(np.array(self.tfidf.transform([" ".join(text)]).todense()))[:, ids].float().T
        if self.pad:
            if len(x) < self.max_length:
                x += [self.word2id['<pad>']] * (self.max_length - len(x))
        x = torch.LongTensor(x)
        if self.cuda:
            x = x.cuda()
            # weights = weights.cuda()
        x_vectors = self.emb_layer(x)
        # x_tfidf = self.tfidf.transform(text)
        if self.pad:
            return x_vectors.reshape((1, -1))
        return x_vectors.mean(dim=0, keepdim=True)

    def convert_features(self, text):
        if isinstance(text[0], list):
            x = torch.cat([self.word2vector(samples) for samples in text], dim=0)
        else:
            x = self.word2vector(text)

        return x.cpu().numpy()

if __name__ == '__main__':
    np.random.seed(2018)
    torch.manual_seed(2018)

    path = '../data/ag'
    train_x, train_y = dataloader.read_corpus(os.path.join(path,
                                                           'train_tok.csv'),
                                              clean=False, MR=False, shuffle=False)
    test_x, test_y = dataloader.read_corpus(os.path.join(path,
                                                         'test_tok.csv'),
                                            clean=False, MR=False, shuffle=False)
    # path = '../data/mr'
    # train_x, train_y = dataloader.read_corpus(os.path.join(path, 'train.txt'))
    # test_x, test_y = dataloader.read_corpus(os.path.join(path, 'test.txt'))

    scd = TextWrapper(embedding_path='../TextFooler/glove.6B.200d.txt')

    # tfidf = TfidfVectorizer(sublinear_tf=True, vocabulary=scd.word2id)
    # train_x_ = [" ".join(i) for i in train_x]
    # tfidf.fit(train_x_)
    # scd.tfidf = tfidf

    # with open('../TextFooler/checkpoints/imdb_tfidf.pkl', 'wb') as f:
    #     pickle.dump(tfidf, f)
    # # train_tfidf = tfidf.transform(train_x)

    # scd = TextWrapper(embedding_path='../TextFooler/glove.6B.200d.txt')

    # rl = []
    # tl = []
    #
    # for i in range(len(train_x)):
    #     rl.append(len(train_x[i]))
    # rl = np.array(rl)
    #
    # for i in range(len(test_x)):
    #     tl.append(len(test_x[i]))
    # tl = np.array(tl)
    #
    # train_index = np.nonzero(rl < 400)[0]
    # test_index = np.nonzero(tl < 400)[0]
    #
    # np.save('../data/imdb400/train_index.npy', train_index)
    # np.save('../data/imdb400/test_index.npy', test_index)

    # train_x = np.array(train_x)[train_index].tolist()
    # train_y = np.array(train_y)[train_index].tolist()
    # test_x = np.array(test_x)[test_index].tolist()
    # test_y = np.array(test_y)[test_index].tolist()


    train_data = scd.convert_features(train_x)
    test_data = scd.convert_features(test_x)
    train_y = np.array(train_y)
    test_y = np.array(test_y)

    np.save(os.path.join(path, 'train_image.npy'), train_data[train_y < 2])
    np.save(os.path.join(path, 'test_image.npy'), test_data[test_y < 2])
    np.save(os.path.join(path, 'train_label.npy'), train_y[train_y < 2])
    np.save(os.path.join(path, 'test_label.npy'), test_y[test_y < 2])