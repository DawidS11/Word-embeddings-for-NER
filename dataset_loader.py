import numpy as np
import random
from keras_preprocessing.sequence import pad_sequences
from allennlp.modules.elmo import batch_to_ids

from kaggle_dataset_builder import KaggleDataset
from conll2003_dataset_builder import Conll2003Dataset
from conll2003_dataset_builder_for_luke import Conll2003DatasetForLuke


class DatasetLoader(object):

    def __init__(self, params):
        
        self.params = params

        if params.dataset_name == "kaggle" or params.dataset_name == "kaggle_small":
            self.dataset = KaggleDataset(params)  
        elif params.dataset_name == "conll2003":
            if params.we_method == 'luke':
                self.dataset = Conll2003DatasetForLuke(params) 
            else:
                self.dataset = Conll2003Dataset(params) 
        self.val2id = self.dataset.val2id
        self.id2val = self.dataset.id2val
        self.val2id_entity = self.dataset.val2id_entity
        self.id2val_entity = self.dataset.id2val_entity


    def load_data(self, case):
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
        
        return data


    def data_iterator(self, data, dataset_size, batch_size, params, shuffle=False):

        data_len = len(data['contexts'])
        order = list(range(data_len))      

        if shuffle:
            random.seed(params.seed)
            random.shuffle(order)

        num_batches = (dataset_size + 1) // batch_size
        for i in range(num_batches):
            batch_sentences = [data['sentences'][idx] for idx in order[i*batch_size : (i+1)*batch_size]]
            batch_labels = [data['labels'][idx] for idx in order[i*batch_size:(i+1)*batch_size]]
            batch_contexts = [data['contexts'][idx] for idx in order[i*batch_size:(i+1)*batch_size]]

            yield batch_sentences, batch_labels, batch_contexts


def prepare_glove(params, word2id, contexts):
    context_texts = [contexts[idx]['context_text'] for idx in range(len(contexts))]
    context_labels = [contexts[idx]['context_labels'] for idx in range(len(contexts))]
    sentence_begs = [contexts[idx]['sentence_beg'] for idx in range(len(contexts))]
    sentence_ends = [contexts[idx]['sentence_end'] for idx in range(len(contexts))]

    max_len = max(map(lambda x: len(x), context_texts), default=0)                                   
    context_texts = list(map(lambda x: list(map(lambda w: word2id.get(w, 0), x)), context_texts))
    context_texts = list(map(lambda x: x + [params.vocab_size-1] * (max_len - len(x)), context_texts))

    max_num = max([len(s) for s in context_labels])

    padded_sentences = pad_sequences([[w for w in sen] for sen in context_texts],
        maxlen=max_num, dtype="long", truncating="post", padding="post")

    padded_labels = pad_sequences([[l for l in lab] for lab in context_labels],
        maxlen=max_num, value=params.pad_tag_num, padding="post",      
        dtype="long", truncating="post")

    return padded_sentences, padded_labels, sentence_begs, sentence_ends


def prepare_elmo(params, contexts):
    context_texts = [contexts[idx]['context_text'] for idx in range(len(contexts))]
    context_labels = [contexts[idx]['context_labels'] for idx in range(len(contexts))]
    sentence_begs = [contexts[idx]['sentence_beg'] for idx in range(len(contexts))]
    sentence_ends = [contexts[idx]['sentence_end'] for idx in range(len(contexts))]

    max_num = max([len(s) for s in context_labels])

    padded_sentences = batch_to_ids(context_texts)      # robi padding
    
    padded_labels = pad_sequences([[l for l in lab] for lab in context_labels],
        maxlen=max_num, value=params.pad_tag_num, padding="post",       
        dtype="long", truncating="post")

    return padded_sentences, padded_labels, sentence_begs, sentence_ends


def prepare_bert_roberta(params, tokenizer, contexts):
    context_texts = [contexts[idx]['context_text'] for idx in range(len(contexts))]       
    context_labels = [contexts[idx]['context_labels'] for idx in range(len(contexts))]
    sentence_begs = [contexts[idx]['sentence_beg'] for idx in range(len(contexts))]
    sentence_ends = [contexts[idx]['sentence_end'] for idx in range(len(contexts))]

    tokenized_sentences = []
    tokenized_sen = []
    tokenized_word = []
    tokenized_labels = []
    tokenized_sen_labels = []
    
    idx_sen = 0
    for sen, lab in zip(context_texts, context_labels):
        curr_token = 0
        beg_updated = False
        end_updated = False

        for idx_word, word in enumerate(sen):

            tokenized_word = tokenizer.tokenize(word)
            is_first = True

            if idx_word == sentence_begs[idx_sen]:
                if not beg_updated:
                    sentence_begs[idx_sen] = curr_token 
                    beg_updated = True     
            elif idx_word == sentence_ends[idx_sen]-1:
                if not end_updated:
                    sentence_ends[idx_sen] = curr_token+1
                    end_updated = True

            for token in tokenized_word:
                tokenized_sen.append(token)
                curr_token += 1 
                if is_first:
                    tokenized_sen_labels.append(lab[idx_word])    
                    is_first = False
                else:
                    tokenized_sen_labels.append(params.pad_tag_num)

        tokenized_sentences.append(tokenized_sen)
        tokenized_labels.append(tokenized_sen_labels)
        tokenized_sen = []
        tokenized_sen_labels = []
        idx_sen += 1

    idx_sen = 0
    if params.we_method == 'bert_base' or params.we_method == 'bert_large':
        for sen, lab in zip(tokenized_sentences, tokenized_labels):
            if sen[0] != "[CLS]":
                sen.insert(0, "[CLS]")
                sen.append("[SEP]")
                lab.insert(0, params.pad_tag_num)
                lab.append(params.pad_tag_num)
                sentence_begs[idx_sen] += 1
                sentence_ends[idx_sen] += 2
            idx_sen += 1

    else:
        for sen, lab in zip(tokenized_sentences, tokenized_labels):
            if sen[0] != "<s>":
                sen.insert(0, "<s>")
                sen.append("</s>")
                lab.insert(0, params.pad_tag_num)
                lab.append(params.pad_tag_num)
                sentence_begs[idx_sen] += 1
                sentence_ends[idx_sen] += 2
            idx_sen += 1

    max_num = max([len(l) for l in tokenized_labels])

    tokenized_labels = pad_sequences([[l for l in lab] for lab in tokenized_labels],
        maxlen=max_num, value=params.pad_tag_num, padding="post",       
        dtype="long", truncating="post")

    tokenized_sentences = pad_sequences([tokenizer.convert_tokens_to_ids(sen) for sen in tokenized_sentences],
                        maxlen=max_num, dtype="long", truncating="post", padding="post")

    return tokenized_sentences, tokenized_labels, sentence_begs, sentence_ends