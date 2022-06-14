import numpy as np
import random

from torch.utils.data import Dataset

from kaggle_dataset_builder import KaggleDataset
from conll2003_dataset_builder import Conll2003Dataset


class DatasetLoader(object):

    def __init__(self, params):
        
        if params.dataset_name == "kaggle":
            self.dataset = KaggleDataset(params)  
        elif params.dataset_name == "conll2003":
            self.dataset = Conll2003Dataset(params) 
        self.tags = self.dataset.tags

    def load_data(self, case, params):
        data = {}
        sentences = []
        labels = []

        if case == "train":
            sentences = self.dataset.train_sentences
            for sen in self.dataset.train_labels:
                l = [self.tags[label] for label in sen]
                labels.append(l) 

        elif case == "val":
            sentences = self.dataset.val_sentences
            for sen in self.dataset.val_labels:
                l = [self.tags[label] for label in sen]
                labels.append(l)

        elif case == "test":
            sentences = self.dataset.test_sentences
            for sen in self.dataset.test_labels:
                l = [self.tags[label] for label in sen]
                labels.append(l) 

        else:
            print("Wrong case.")

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
#     def __init__(self, case, dataset, params):

#         if case == "train":
#             self.data = dataset.load_data("train", params)
#         elif case == "val":
#             self.data = dataset.load_data("val", params)
#         elif case == "test":
#             self.data = dataset.load_data("test", params)
#         else:
#             print("Wrong case.")

#     def __getitem__(self, index):
#         return self.data['sentences'][index], self.data['labels'][index]

#     def __len__(self):
#         return len(self.data['sentences'])