import numpy as np
import random

from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import Dataset

from kaggle_dataset_builder import KaggleDataset


class KaggleDatasetLoader(object):

    def __init__(self, params):

        self.kaggle_dataset = KaggleDataset(params)  
        self.tags = self.kaggle_dataset.tags

    def load_data(self, case, params):
        data = {}
        sentences = []
        labels = []

        if case == "train":
            for sen in self.kaggle_dataset.train_sentences:
                s = [w for w in sen.split(' ')]
                sentences.append(s) 
            for sen in self.kaggle_dataset.train_labels:
                l = [self.tags[label] for label in sen]
                labels.append(l) 

        elif case == "val":
            for sen in self.kaggle_dataset.val_sentences:
                s = [w for w in sen.split(' ')]
                sentences.append(s) 
            for sen in self.kaggle_dataset.val_labels:
                l = [self.tags[label] for label in sen]
                labels.append(l)

        elif case == "test":
            for sen in self.kaggle_dataset.test_sentences:
                s = [w for w in sen.split(' ')]
                sentences.append(s) 
            for sen in self.kaggle_dataset.test_labels:
                l = [self.tags[label] for label in sen]
                labels.append(l) 

        else:
            print("Wrong case.")

        # padding:
        # sentences_padded = pad_sequences([[w for w in sen] for sen in sentences],
        #               maxlen=params.max_sen_len, value=params.pad_word, padding="post",
        #               dtype=object, truncating="post") 

        # labels_padded = pad_sequences([[l for l in lab] for lab in labels],
        #               maxlen=params.max_sen_len, value=self.tags[params.pad_tag], padding="post",       #self.tags[params.pad_tag] params.pad_tag_num
        #               dtype="long", truncating="post")

        data['sentences'] = sentences
        data['labels'] = labels
        return data


    def data_iterator(self, data, num_batches, params, shuffle=False):

        data_len = len(data['sentences'])
        order = list(range(data_len))      

        if shuffle:
            random.seed(params.seed)
            random.shuffle(order)

        for i in range(num_batches):
            batch_sentences = [data['sentences'][idx] for idx in order[i*params.batch_size : (i+1)*params.batch_size]]
            batch_tags = [data['labels'][idx] for idx in order[i*params.batch_size:(i+1)*params.batch_size]]
    
            yield batch_sentences, batch_tags



# class MyDataset(Dataset):
#     def __init__(self, case, kaggle_dataset, params):

#         if case == "train":
#             self.data = kaggle_dataset.load_data("train", params)
#         elif case == "val":
#             self.data = kaggle_dataset.load_data("val", params)
#         elif case == "test":
#             self.data = kaggle_dataset.load_data("test", params)
#         else:
#             print("Wrong case.")

#     def __getitem__(self, index):
#         return self.data['sentences'][index], self.data['labels'][index]

#     def __len__(self):
#         return len(self.data['sentences'])