import numpy as np
import pandas as pd
import os
from collections import Counter

from get_glove import get_glove, create_vocab

def load_dataset(dataset_path):

    sentences = []
    labels = []
    sen = []
    lab = []

    with open(dataset_path, 'r') as f:
        
        for line in f.readlines():
            if (line == ('-DOCSTART- -X- -X- O\n') or line == '\n'):
                if len(sen) > 0:
                    sentences.append(sen)
                    labels.append(lab)
                    sen = []
                    lab = []
            else:
                l = line.split(' ')
                sen.append(l[0])
                lab.append((l[3].strip('\n')))
    
    return sentences, labels


class Conll2003Dataset(object):
    def __init__(self, params):
        
        print("Init Conll2003Dataset")

        train_dataset_path = os.path.join(params.data_dir, 'train.txt')
        val_dataset_path = os.path.join(params.data_dir, 'valid.txt')
        test_dataset_path = os.path.join(params.data_dir, 'test.txt')
        error_msg = "{} file not found. ".format(train_dataset_path)
        assert os.path.isfile(train_dataset_path), error_msg
        error_msg = "{} file not found. ".format(val_dataset_path)
        assert os.path.isfile(val_dataset_path), error_msg
        error_msg = "{} file not found. ".format(test_dataset_path)
        assert os.path.isfile(test_dataset_path), error_msg

        self.words = Counter()

        self.train_sentences, self.train_labels = load_dataset(train_dataset_path)
        self.val_sentences, self.val_labels = load_dataset(val_dataset_path)
        self.test_sentences, self.test_labels = load_dataset(test_dataset_path)

        # Assert sentences and labels lengths:
        assert len(self.train_sentences) == len(self.train_labels)
        assert len(self.val_sentences) == len(self.val_labels)
        assert len(self.test_sentences) == len(self.test_labels)

        params.train_size = len(self.train_sentences)
        params.val_size = len(self.val_sentences)
        params.test_size = len(self.test_sentences)

        # Creating a number representation of labels.
        dataset_labels = self.train_labels + self.val_labels + self.test_labels
        list_labels = [l for lab in dataset_labels for l in lab]
        tags_vals = list(set(list_labels))
        #self.tags = {t: i for i, t in enumerate(tags_vals)}
        self.val2id = {t: i for i, t in enumerate(tags_vals)}
        self.id2val = {i: t for i, t in enumerate(tags_vals)}

        params.num_of_tags = len(self.val2id)
        params.max_sen_len = max([len(s) for s in dataset_labels])

        if params.wb_method.lower() == 'glove':
            create_vocab(self.train_sentences, self.val_sentences, self.test_sentences, params)
            get_glove(params)

        print("Init Conll2003Dataset done.")


    # def update_vocab(self, dataset, vocab):        
    #     for sen in dataset:
    #         vocab.update(sen)

    # def create_vocab(self, params):
    #     self.update_vocab(self.train_sentences, self.words)
    #     self.update_vocab(self.val_sentences, self.words)
    #     self.update_vocab(self.test_sentences, self.words)

    #     self.words = [tok for tok, count in self.words.items() if count >= 1]

    #     if params.pad_word not in self.words: self.words.append(params.pad_word)

    #     self.words.append(params.unk_word)

    #     params.vocab_size = len(self.words)

    #     # Saving vocab:
    #     with open(os.path.join(params.data_dir, 'words.txt'), "w") as f:
    #         for word in self.words:
    #             f.write(word + '\n')

    # def get_glove(self, params):
    #     vocab = {j.strip(): i for i, j in enumerate(open(os.path.join(params.data_dir, 'words.txt')), 0)}
    #     id2word = {vocab[i]: i for i in vocab}

    #     dim = 0
    #     w2v = {}
    #     for line in open(os.path.join(params.glove_dir, 'glove.6B.{}d.txt'.format(params.glove_dim))):
    #         line = line.strip().split()
    #         word = line[0]
    #         vec = list(map(float, line[1:]))
    #         dim = len(vec)
    #         w2v[word] = vec

    #     vecs = []
    #     vecs.append(np.random.uniform(low=-1.0, high=1.0, size=dim))

    #     for i in range(1, len(vocab) - 1):
    #         if id2word[i] in w2v:
    #             vecs.append(w2v[id2word[i]])
    #         else:
    #             vecs.append(vecs[0])
    #     vecs.append(np.zeros(dim))
    #     assert(len(vecs) == len(vocab))

    #     np.save(os.path.join(params.glove_dir, 'glove_{}d.npy'.format(dim)), np.array(vecs, dtype=np.float32))
    #     np.save(os.path.join(params.glove_dir, 'word2id.npy'), vocab)
    #     np.save(os.path.join(params.glove_dir, 'id2word.npy'), id2word)