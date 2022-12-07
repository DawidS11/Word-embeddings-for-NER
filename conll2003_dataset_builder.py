import os
from collections import Counter

from get_context import get_context_conll2003
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
    sentences = []
    sentences_labels = []

    with open(dataset_file) as f:
        for line in f:
            line = line.rstrip()
            
            if line.startswith("-DOCSTART"):
                if sentences:
                    documents.append(dict(
                        sentences=sentences,
                        sentences_labels=sentences_labels,
                    ))
                    sentences = []
                    sentences_labels = []
                    words = []
                    labels = []

                continue

            if not line:
                if words:
                    sentences.append(words)
                    sentences_labels.append(labels)
                    words = []
                    labels = []
            else:
                items = line.split(" ")
                words.append(items[0])
                labels.append(items[-1])
            
    if sentences:
        documents.append(dict(
            sentences=sentences,
            sentences_labels=sentences_labels,
        ))

    return documents


class Conll2003Dataset(object):
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

        # Creating a number representation of labels.
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

        self.train_contexts = get_context_conll2003(self.train_documents, params, self.val2id)
        self.val_contexts = get_context_conll2003(self.val_documents, params, self.val2id)
        self.test_contexts = get_context_conll2003(self.test_documents, params, self.val2id)

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
