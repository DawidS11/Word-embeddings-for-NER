import numpy as np
import pandas as pd
import os

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

        #dataset_path = 'data/kaggle/ner_dataset.csv'
        dataset_path = '/content/ner_dataset.csv'
        error_msg = "{} file not found. ".format(dataset_path)
        assert os.path.isfile(dataset_path), error_msg
        
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
        if params.PAD_TAG not in tags_vals: tags_vals.append(params.PAD_TAG)
        self.tags = {t: i for i, t in enumerate(tags_vals)}

        params.num_of_tags = len(self.tags)
        params.max_sen_len = max([len(s) for s in self.dataset_labels])

        print("Init KaggleDataset done.")