import numpy as np
import random
from keras_preprocessing.sequence import pad_sequences
from allennlp.modules.elmo import batch_to_ids

from kaggle_dataset_builder import KaggleDataset
from conll2003_dataset_builder import Conll2003Dataset


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


def prepare_luke(params, contexts, tokenizer, id2val, val2id_entity):

    all_entities = calc_entity_spans(contexts, id2val)

    entities = []
    entity_spans = []
    entity_labels = []
    for context in all_entities:
        context_entities = []
        context_spans = []
        context_labels = []
        for entity in context:
            context_entities.append(entity['entity_text'])
            context_spans.append(entity['entity_span'])
            context_labels.append(val2id_entity[entity['entity_label']])
        entities.append(context_entities)
        entity_spans.append(context_spans)
        entity_labels.append(context_labels)

    context_texts = [contexts[idx]['context_text'] for idx in range(len(contexts))]         # tu nie bedzie sentence?
    texts = [" ".join(sen) for sen in context_texts]
    inputs = tokenizer(texts, entities=entities, entity_spans=entity_spans, return_tensors="pt", padding=True)

    max_num = max([len(l) for l in entity_labels])
    entity_labels = pad_sequences([[l for l in lab] for lab in entity_labels],
        maxlen=max_num, value=params.pad_tag_num, padding="post",
        dtype="long", truncating="post")

    return inputs, entity_labels


def calc_entity_spans(contexts, id2val):
    all_context_entities = []
    all_entities = []

    for context in contexts:

        beg = 0
        end = 0
        text = context['context_text']
        #labels = context['labels']
        labels = [id2val[l] for l in context['labels']]
        len_labels = len(labels)

        for i in range(context['sentence_beg']):            # przesuniecie beg na poczatek zdania w calym tekscie
            beg += len(text[i])
            
        # for idx in range(context['sentence_beg'], context['sentence_end']):
        for idx in range(len_labels):
            end = beg
            entity = text[idx]

            for idx2 in range(idx, len_labels):        # context['sentence_end']
                end += len(text[idx2])
                if idx != idx2:
                    entity += " "
                    entity += text[idx2]

                    # przypadki:
                    #     1) jest 'O' i nastepna to 'O'      
                    #     2) jest cos i nastepna to 'O'   
                    #     3) jest 'O' i nastepna to cos
                    #     4) jest ta sama koncowka ale rozne poczatki - sprawdzic poprawnosc
                    #     5) sa te same poczatki: jesli B- to bedzie jednoslowna, jesli I- to bedzie dluzsza encja
                    #     6) sa rozne poczatki i rozne koncowki
                    
                    if labels[idx2] == 'O':     # 1, 3
                        label = 'NIL'
                    elif idx2+1 < len_labels:
                        if labels[idx2+1] == 'O':       # 2
                            if labels[idx][:2] == 'I-':
                                label = 'NIL'
                                continue 
                            for idx3 in range(idx+1, idx2+1):
                                if labels[idx3][:2] != 'I-' or labels[idx3][2:] != labels[idx][2:]:
                                    label = 'NIL'
                                    continue

                        elif labels[idx2+1][2:] != labels[idx2][2:]:         # zaczyna się kolejna encja
                            if labels[idx][:2] == 'I-':
                                label = 'NIL'
                                continue 
                            for idx3 in range(idx+1, idx2+1):
                                if labels[idx3][:2] != 'I-' or labels[idx3][2:] != labels[idx][2:]:
                                    label = 'NIL'
                                    continue

                        elif labels[idx2+1][:2] != labels[idx2][:2]:
                            if labels[idx2+1][:2] == 'B-':          # poczatek kolejnej encji
                                if labels[idx][:2] == 'I-':
                                    label = 'NIL'
                                    continue 
                            for idx3 in range(idx+1, idx2+1):
                                if labels[idx3][:2] != 'I-' or labels[idx3][2:] != labels[idx][2:]:
                                    label = 'NIL'
                                    continue

                            else:   # nastepny label zaczyna sie od 'I-' wiec jest kontynuacja encji
                                label = 'NIL'
                        
                        else:   # nastepny label nie jest 'O', [2:] jest identyczne i [:2] jest identyczne
                            if labels[idx2+1][:2] == 'B-' and labels[idx2][:2] == 'B-':     # na pewno NIL, bo encja jest dluzsza niz jeden element wiec nie moze miec B- na koncu
                                label = 'NIL'
                            elif labels[idx2+1][:2] == 'I-' and labels[idx2][:2] == 'I-':
                                label = 'NIL'

                else:
                    if labels[idx2] == 'O':      
                        label = 'NIL'
                    elif labels[idx2][:2] == 'I-':
                        label = 'NIL'
                    elif idx2+1 < len_labels:                            # sprawdzenie czy nie jest poczatkiem dluzszej encji    
                        if labels[idx2+1][2:] == labels[idx2][2:] and labels[idx2+1][:2] == 'I-':
                            label = 'NIL'
                        else:
                            label = labels[idx2][2:]
                    else:
                        label = labels[idx2][2:]

                all_context_entities.append(dict(
                    entity_span=(beg, end), 
                    entity_text=entity,
                    entity_label=label,                 
                ))

                
            beg += len(text[idx])

        all_entities.append(all_context_entities)
        all_context_entities = []

    return all_entities