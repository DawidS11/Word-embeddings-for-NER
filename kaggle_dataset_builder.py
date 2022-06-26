import numpy as np
import pandas as pd
import os
from collections import Counter

from get_glove import get_glove, create_vocab
#from get_context import get_context_kaggle

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

        dataset_path = os.path.join(params.data_dir, 'ner_dataset.csv')
        error_msg = "{} file not found. ".format(dataset_path)
        assert os.path.isfile(dataset_path), error_msg
        self.words = Counter()
        
        dataset_pd = pd.read_csv(dataset_path, encoding="latin1")
        dataset_pd = dataset_pd.fillna(method="ffill")

        sentences = GetSentences(dataset_pd)

        self.dataset_sentences = [" ".join([s[0] for s in sent]) for sent in sentences.sentences]
        print(self.dataset_sentences[0])
        print(self.dataset_sentences[1])
        print(self.dataset_sentences[2])
        print(self.dataset_sentences[3])
        print(self.dataset_sentences[4])
        print(self.dataset_sentences[5])
        quit()
        self.dataset_sentences = [[s[0] for s in sen] for sen in sentences.sentences]
        self.dataset_labels  = [[s[2] for s in sen] for sen in sentences.sentences]
        
        # Creating sets of sentences and labels for train, val and test:
        self.train_sentences = self.dataset_sentences[:int(params.train_dataset_size*len(self.dataset_sentences))]
        self.train_labels = self.dataset_labels[:int(params.train_dataset_size*len(self.dataset_labels))]
        self.train_contexts = []

        self.val_sentences = self.dataset_sentences[int(params.train_dataset_size*len(self.dataset_sentences)) \
            : int((params.train_dataset_size + params.val_dataset_size)*len(self.dataset_sentences))]
        self.val_labels = self.dataset_labels[int(params.train_dataset_size*len(self.dataset_labels)) \
            : int((params.train_dataset_size + params.val_dataset_size)*len(self.dataset_labels))]
        self.val_contexts = []

        self.test_sentences = self.dataset_sentences[int((1.0 - params.val_dataset_size)*len(self.dataset_sentences)):]
        self.test_labels = self.dataset_labels[int((1.0 - params.val_dataset_size)*len(self.dataset_labels)):]
        self.test_contexts = []

        # Assert sentences and labels lengths:
        assert len(self.train_sentences) == len(self.train_labels)
        assert len(self.val_sentences) == len(self.val_labels)
        assert len(self.test_sentences) == len(self.test_labels)

        params.train_size = len(self.train_sentences)
        params.val_size = len(self.val_sentences)
        params.test_size = len(self.test_sentences)

        # Creating a number representation of labels.
        tags_vals = list(set(dataset_pd["Tag"].values))
        #self.tags = {t: i for i, t in enumerate(tags_vals)}
        self.val2id = {t: i for i, t in enumerate(tags_vals)}
        self.id2val = {i: t for i, t in enumerate(tags_vals)}

        params.num_of_tags = len(self.val2id)
        params.max_sen_len = max([len(s) for s in self.dataset_labels])

        if params.we_method.lower() == 'glove':
            create_vocab(self.train_sentences, self.val_sentences, self.test_sentences, params)
            get_glove(params)

        print("Init KaggleDataset done.")
