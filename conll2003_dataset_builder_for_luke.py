# Based on https://colab.research.google.com/github/studio-ousia/luke/blob/master/notebooks/huggingface_conll_2003.ipynb#scrollTo=q9bXAEPZp0ZT

import unicodedata

import numpy as np
import os
from collections import Counter
from tqdm import tqdm, trange
from transformers import BertTokenizer, RobertaTokenizer, LukeTokenizer

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


def load_documents(dataset_file):
    documents = []
    words = []
    labels = []
    sentence_boundaries = []
    with open(dataset_file) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("-DOCSTART"):
                if words:
                    documents.append(dict(
                        words=words,
                        labels=labels,
                        sentence_boundaries=sentence_boundaries
                    ))
                    words = []
                    labels = []
                    sentence_boundaries = []
                continue

            if not line:
                if not sentence_boundaries or len(words) != sentence_boundaries[-1]:
                    sentence_boundaries.append(len(words))
            else:
                items = line.split(" ")
                words.append(items[0])
                labels.append(items[-1])

    if words:
        documents.append(dict(
            words=words,
            labels=labels,
            sentence_boundaries=sentence_boundaries
        ))
        
    return documents


def load_examples(documents, params):
    examples = []
    max_mention_length = 30
    
    if params.we_method.lower() == 'bert_base':
        if params.bert_cased:
            tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        else:
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        params.max_context_len = 510
    elif params.we_method.lower() == 'bert_large':
        if params.bert_cased:
            tokenizer = BertTokenizer.from_pretrained("bert-large-cased")
        else:
            tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
        params.max_context_len = 510
    elif params.we_method.lower() == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        params.max_context_len = 510
    elif params.we_method.lower() == 'luke':
        tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base")
        params.max_context_len = 510
    elif params.we_method.lower() == 'luke_conll':
        tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-conll-2003")
        params.max_context_len = 510
    else:
        params.max_context_len = 2048  

    for document in tqdm(documents):
        words = document["words"]
        if params.we_method.lower() == 'glove' or params.we_method.lower() == 'elmo':
            subword_lengths = [len(w) for w in words]                          # liczba znaków
        else:
            subword_lengths = [len(tokenizer.tokenize(w)) for w in words]      # liczba tokenów
        total_subword_length = sum(subword_lengths)
        sentence_boundaries = document["sentence_boundaries"]
        document_labels = document['labels']

        for i in range(len(sentence_boundaries) - 1):
            sentence_start, sentence_end = sentence_boundaries[i:i+2]
            if total_subword_length <= params.max_context_len:
                # if the total sequence length of the document is shorter than the
                # maximum token length, we simply use all words to build the sequence
                context_start = 0
                context_end = len(words)
            else:
                # if the total sequence length is longer than the maximum length, we add
                # the surrounding words of the target sentence　to the sequence until it
                # reaches the maximum length
                context_start = sentence_start
                context_end = sentence_end
                cur_length = sum(subword_lengths[context_start:context_end])
                while True:
                    if context_start > 0:
                        if cur_length + subword_lengths[context_start - 1] <= params.max_context_len:
                            cur_length += subword_lengths[context_start - 1]
                            context_start -= 1
                        else:
                            break
                    if context_end < len(words):
                        if cur_length + subword_lengths[context_end] <= params.max_context_len:
                            cur_length += subword_lengths[context_end]
                            context_end += 1
                        else:
                            break

            text = ""
            for word in words[context_start:sentence_start]:
                text += word
                text += " "

            sentence_words = words[sentence_start:sentence_end]
            sentence_subword_lengths = subword_lengths[sentence_start:sentence_end]

            word_start_char_positions = []
            word_end_char_positions = []
            for word in sentence_words:
                word_start_char_positions.append(len(text))
                text += word
                word_end_char_positions.append(len(text))
                text += " "

            for word in words[sentence_end:context_end]:
                text += word
                text += " "
            text = text.rstrip()

            entity_spans = []
            original_word_spans = []
            for word_start in range(len(sentence_words)):
                for word_end in range(word_start, len(sentence_words)):
                    if sum(sentence_subword_lengths[word_start:word_end]) <= max_mention_length:
                        entity_spans.append(
                            (word_start_char_positions[word_start], word_end_char_positions[word_end])
                        )
                        original_word_spans.append(
                            (word_start, word_end + 1)
                        )

            examples.append(dict(
                text=text,
                words=sentence_words,
                entity_spans=entity_spans,
                original_word_spans=original_word_spans,
            ))

    return examples


class Conll2003DatasetForLuke(object):
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

        # Creating a number representation of labels:
        dataset_labels = self.train_labels + self.val_labels + self.test_labels
        list_labels = [l for lab in dataset_labels for l in lab]
        tags_vals = list(set(list_labels))
        tags_vals_entity = list(set([tag_val[2:] if tag_val != 'O' else 'NIL' for tag_val in tags_vals]))

        self.val2id = {t: i for i, t in enumerate(tags_vals)}
        self.id2val = {i: t for i, t in enumerate(tags_vals)}
        self.val2id_entity = {t: i for i, t in enumerate(tags_vals_entity)}
        self.id2val_entity = {i: t for i, t in enumerate(tags_vals_entity)}

        self.train_documents = load_documents(train_dataset_path)
        self.val_documents = load_documents(val_dataset_path)
        self.test_documents = load_documents(test_dataset_path)

        self.train_contexts = load_examples(self.train_documents, params)
        self.val_contexts = load_examples(self.val_documents, params)
        self.test_contexts = load_examples(self.test_documents, params)

        # Assert sentences and labels lengths:
        assert len(self.train_sentences) == len(self.train_labels)
        assert len(self.val_sentences) == len(self.val_labels)
        assert len(self.test_sentences) == len(self.test_labels)

        params.train_size = len(self.train_sentences)
        params.val_size = len(self.val_sentences)
        params.test_size = len(self.test_sentences)

        params.num_of_tags = len(self.val2id)
        params.num_of_tags_entity = len(self.val2id_entity)
        params.max_sen_len = max([len(s) for s in dataset_labels])
        max_entity_num = 0
        for i in range(params.max_sen_len):
            for j in range(i, params.max_sen_len):
                max_entity_num += 1
        params.max_entity_num = max_entity_num

        if params.we_method.lower() == 'glove':
            create_vocab(self.train_sentences, self.val_sentences, self.test_sentences, params)
            get_glove(params)

        print("Init Conll2003Dataset done.")