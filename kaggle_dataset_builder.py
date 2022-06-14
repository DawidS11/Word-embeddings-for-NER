import numpy as np
import pandas as pd
import os
from collections import Counter

class GetSentences(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        function = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(function)
        self.sentences = [s for s in self.grouped]
    
    def retrieve(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


class KaggleDataset(object):
    def __init__(self, params):
        
        print("Init KaggleDataset")

        dataset_path = os.path.join(params.kaggle_dir, 'ner_dataset.csv')
        error_msg = "{} file not found. ".format(dataset_path)
        assert os.path.isfile(dataset_path), error_msg
        self.words = Counter()
        
        dataset_pd = pd.read_csv(dataset_path, encoding="latin1")
        dataset_pd = dataset_pd.fillna(method="ffill")

        sentences = GetSentences(dataset_pd)
        self.dataset_sentences = [" ".join([s[0] for s in sent]) for sent in sentences.sentences]
        self.dataset_labels  = [[s[2] for s in sent] for sent in sentences.sentences]
        
        # Creating sets of sentences and labels for train, val and test:
        self.train_sentences = self.dataset_sentences[:int(params.train_dataset_size*len(self.dataset_sentences))]
        self.train_labels = self.dataset_labels[:int(params.train_dataset_size*len(self.dataset_labels))]

        self.val_sentences = self.dataset_sentences[int(params.train_dataset_size*len(self.dataset_sentences)) \
            : int((params.train_dataset_size + params.val_dataset_size)*len(self.dataset_sentences))]
        self.val_labels = self.dataset_labels[int(params.train_dataset_size*len(self.dataset_labels)) \
            : int((params.train_dataset_size + params.val_dataset_size)*len(self.dataset_labels))]

        self.test_sentences = self.dataset_sentences[int((1.0 - params.val_dataset_size)*len(self.dataset_sentences)):]
        self.test_labels = self.dataset_labels[int((1.0 - params.val_dataset_size)*len(self.dataset_labels)):]

        # Assert sentences and labels lengths:
        assert len(self.train_sentences) == len(self.train_labels)
        assert len(self.val_sentences) == len(self.val_labels)
        assert len(self.test_sentences) == len(self.test_labels)

        params.train_size = len(self.train_sentences)
        params.val_size = len(self.val_sentences)
        params.test_size = len(self.test_sentences)

        # Creating a number representation of labels.
        tags_vals = list(set(dataset_pd["Tag"].values))
        self.tags = {t: i for i, t in enumerate(tags_vals)}
        # if params.pad_word not in tags_vals: 
        #     self.tags[params.pad_word] = params.pad_tag_num

        # if params.pad_tag not in tags_vals: tags_vals.append(params.pad_tag)
        # self.tags = {t: i for i, t in enumerate(tags_vals)}

        params.num_of_tags = len(self.tags)
        params.max_sen_len = max([len(s) for s in self.dataset_labels])

        if params.wb_method.lower() == 'glove':
            self.create_vocab(params)
            self.get_glove(params)

        print("Init KaggleDataset done.")


    def update_vocab(self, dataset, vocab):        
        for sen in dataset:
            vocab.update(sen.split(' '))

    def create_vocab(self, params):
        self.update_vocab(self.train_sentences, self.words)
        self.update_vocab(self.val_sentences, self.words)
        self.update_vocab(self.test_sentences, self.words)

        self.words = [tok for tok, count in self.words.items() if count >= 1]

        if params.pad_word not in self.words: self.words.append(params.pad_word)

        self.words.append(params.unk_word)

        params.vocab_size = len(self.words)

        # Saving vocab:
        with open(os.path.join(params.data_dir, 'words.txt'), "w") as f:
            for word in self.words:
                f.write(word + '\n')

    def get_glove(self, params):
        vocab = {j.strip(): i for i, j in enumerate(open(os.path.join(params.data_dir, 'words.txt')), 0)}
        id2word = {vocab[i]: i for i in vocab}

        dim = 0
        w2v = {}
        for line in open(os.path.join(params.glove_dir, 'glove.6B.{}d.txt'.format(params.glove_dim))):
            line = line.strip().split()
            word = line[0]
            vec = list(map(float, line[1:]))
            dim = len(vec)
            w2v[word] = vec

        vecs = []
        vecs.append(np.random.uniform(low=-1.0, high=1.0, size=dim))

        for i in range(1, len(vocab) - 1):
            if id2word[i] in w2v:
                vecs.append(w2v[id2word[i]])
            else:
                vecs.append(vecs[0])
        vecs.append(np.zeros(dim))
        assert(len(vecs) == len(vocab))

        np.save(os.path.join(params.glove_dir, 'glove_{}d.npy'.format(dim)), np.array(vecs, dtype=np.float32))
        np.save(os.path.join(params.glove_dir, 'word2id.npy'), vocab)
        np.save(os.path.join(params.glove_dir, 'id2word.npy'), id2word)