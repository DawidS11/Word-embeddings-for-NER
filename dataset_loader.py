import numpy as np
import random
from keras_preprocessing.sequence import pad_sequences

from kaggle_dataset_builder import KaggleDataset
from conll2003_dataset_builder import Conll2003Dataset

import torch
from allennlp.modules.elmo import Elmo, batch_to_ids
from transformers import BertTokenizer, RobertaTokenizer


class DatasetLoader(object):

    def __init__(self, params):
        
        self.params = params

        if params.dataset_name == "kaggle":
            self.dataset = KaggleDataset(params)  
        elif params.dataset_name == "conll2003":
            self.dataset = Conll2003Dataset(params) 
        self.val2id = self.dataset.val2id
        self.id2val = self.dataset.id2val
        self.val2id_entity = self.dataset.val2id_entity
        self.id2val_entity = self.dataset.id2val_entity

        # if self.params.we_method == 'bert':
        #     self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        # elif self.params.we_method == 'roberta':
        #     self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    def load_data(self, case, params):
        data = {}
        sentences = []
        labels = []

        if case == "train":
            sentences = self.dataset.train_sentences
            for sen in self.dataset.train_labels:
                l = [self.val2id[label] for label in sen]
                labels.append(l) 
            contexts = self.dataset.train_contexts

        elif case == "val":
            sentences = self.dataset.val_sentences
            for sen in self.dataset.val_labels:
                l = [self.val2id[label] for label in sen]
                labels.append(l)
            contexts = self.dataset.val_contexts

        elif case == "test":
            sentences = self.dataset.test_sentences
            for sen in self.dataset.test_labels:
                l = [self.val2id[label] for label in sen]
                labels.append(l) 
            contexts = self.dataset.test_contexts

        else:
            print("Wrong case.")

        data['sentences'] = sentences
        data['labels'] = labels
        data['contexts'] = contexts
        
        # if self.params.we_method == 'bert' or self.params.we_method == 'roberta':
        #     data['prepared_context_text'], data['prepared_context_labels'] = self.prepare_bert_roberta(contexts)
        
        return data


    def data_iterator(self, data, dataset_size, batch_size, params, shuffle=False):

        data_len = len(data['contexts'])
        order = list(range(data_len))      

        if shuffle:
            random.seed(params.seed)
            random.shuffle(order)

        num_batches = (dataset_size + 1) // batch_size
        for i in range(num_batches):
            # if self.params.we_method == 'elmo':             # Padding w kazdej batch.
            #     batch_sentences = [data['sentences'][idx] for idx in order[i*batch_size : (i+1)*batch_size]]
            #     batch_labels = [data['labels'][idx] for idx in order[i*batch_size:(i+1)*batch_size]]
            # else:
            #     batch_sentences = np.array([data['prepared_context_text'][idx] for idx in order[i*batch_size : (i+1)*batch_size]])
            #     batch_labels = np.array([data['prepared_context_labels'][idx] for idx in order[i*batch_size:(i+1)*batch_size]])

            batch_sentences = [data['sentences'][idx] for idx in order[i*batch_size : (i+1)*batch_size]]
            batch_labels = [data['labels'][idx] for idx in order[i*batch_size:(i+1)*batch_size]]
            batch_contexts = [data['contexts'][idx] for idx in order[i*batch_size:(i+1)*batch_size]]

            yield batch_sentences, batch_labels, batch_contexts


def prepare_elmo(params, contexts):
    context_texts = [contexts[idx]['context_text'] for idx in range(len(contexts))]
    context_labels = [contexts[idx]['context_labels'] for idx in range(len(contexts))]
    max_num = max([len(s) for s in context_labels])

    padded_sentences = pad_sequences([[s for s in sen] for sen in context_texts],
            maxlen=max_num, value=params.pad_word, padding="post",      
            dtype=object, truncating="post")

    padded_sentences = batch_to_ids(padded_sentences)
    
    padded_labels = pad_sequences([[l for l in lab] for lab in context_labels],
        maxlen=max_num, value=params.pad_tag_num, padding="post",       
        dtype="long", truncating="post")

    return padded_sentences, padded_labels


def prepare_bert_roberta(params, tokenizer, contexts):
    context_texts = [contexts[idx]['context_text'] for idx in range(len(contexts))]
    context_labels = [contexts[idx]['context_labels'] for idx in range(len(contexts))]

    tokenized_sentences = []
    tokenized_sen = []
    tokenized_word = []
    
    tokenized_labels = []
    tokenized_sen_labels = []
    idx = -1
    is_first = True

    for sen, lab in zip(context_texts, context_labels):
        idx = -1
        for word in sen:
            idx += 1
            tokenized_word = tokenizer.tokenize(word)

            is_first = True
            for token in tokenized_word:
                tokenized_sen.append(token)
                if is_first:
                    tokenized_sen_labels.append(lab[idx])
                    is_first = False
                else:
                    tokenized_sen_labels.append(params.pad_tag_num)                

        tokenized_sentences.append(tokenized_sen)
        tokenized_labels.append(tokenized_sen_labels)
        tokenized_sen = []
        tokenized_sen_labels = []

    if params.we_method == 'bert':
        for sen, lab in zip(tokenized_sentences, tokenized_labels):
            if sen[0] != "[CLS]":
                sen.insert(0, "[CLS]")
                sen.append("[SEP]")
                lab.insert(0, params.pad_tag_num)
                lab.append(params.pad_tag_num)

    else:
        for sen, lab in zip(tokenized_sentences, tokenized_labels):
            if sen[0] != "<s>":
                sen.insert(0, "<s>")
                sen.append("</s>")
                lab.insert(0, params.pad_tag_num)
                lab.append(params.pad_tag_num)

    max_num = max([len(l) for l in tokenized_labels])

    tokenized_labels = pad_sequences([[l for l in lab] for lab in tokenized_labels],
        maxlen=max_num, value=params.pad_tag_num, padding="post",       
        dtype="long", truncating="post")

    tokenized_sentences = pad_sequences([tokenizer.convert_tokens_to_ids(sen) for sen in tokenized_sentences],
                        maxlen=max_num, dtype="long", truncating="post", padding="post")

    return tokenized_sentences, tokenized_labels