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
            if params.we_method == 'luke_conll':
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


def prepare_luke(params, contexts, tokenizer, id2val, val2id_entity):

    all_entities = calc_entity_spans(contexts, id2val)

    entities = []
    entity_spans = []
    entity_labels = []
    word_entity_spans = []
    for context in all_entities:
        context_entities = []
        context_spans = []
        context_labels = []
        context_word_spans = []
        for entity in context:
            context_entities.append(entity['entity_text'])
            context_spans.append(entity['entity_span'])
            context_labels.append(val2id_entity[entity['entity_label']])
            context_word_spans.append(entity['word_entity_span'])
        entities.append(context_entities)
        entity_spans.append(context_spans)
        entity_labels.append(context_labels)
        word_entity_spans.append(context_word_spans)

    context_texts = [contexts[idx]['context_text'] for idx in range(len(contexts))]       
    texts = [" ".join(sen) for sen in context_texts]
    inputs = tokenizer(texts, entity_spans=entity_spans, return_tensors="pt", padding=True)     #  entities=entities

    max_num = max([len(l) for l in entity_labels])
    entity_labels = pad_sequences([[l for l in lab] for lab in entity_labels],
        maxlen=max_num, value=params.pad_tag_num, padding="post",
        dtype="long", truncating="post")

    return inputs, entity_labels, word_entity_spans


def calc_entity_spans(contexts, id2val):
    all_context_entities = []
    all_entities = []

    for context in contexts:

        beg = 0
        end = 0
        text = context['context_text']
        #labels = context['labels']
        #labels = [id2val[l] for l in context['labels']] 
        labels = [id2val[l] for l in context['context_labels']] 
        word_a = 0
        word_b = 0

        for i in range(context['sentence_beg']):            # przesuniecie beg na poczatek zdania w calym tekscie
            beg += len(text[i])                     # lista słów
            word_a += 1
            
        for idx in range(context['sentence_beg'], context['sentence_end']):
        #for idx in range(len_labels):
            end = beg
            word_b = word_a
            entity = text[idx]

            #for idx2 in range(idx, len_labels):        # context['sentence_end']
            for idx2 in range(idx, context['sentence_end']):
                end += len(text[idx2])
                word_b += 1
                if idx != idx2:
                    entity += " "
                    end += 1
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
                    elif idx2+1 < context['sentence_end']:          #len_labels:
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
                    elif idx2+1 < context['sentence_end']:          #len_labels:                            # sprawdzenie czy nie jest poczatkiem dluzszej encji    
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
                    word_entity_span=(word_a, word_b),                 
                ))

            beg += len(text[idx]) + 1
            word_a += 1

        all_entities.append(all_context_entities)
        all_context_entities = []

    return all_entities